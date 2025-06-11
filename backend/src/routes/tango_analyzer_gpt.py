"""
Music Analysis Logic Layer

This module contains the core music analysis logic separated from the REST API layer.
The MusicAnalyzer class provides granular analysis methods for beat, melody, and section analysis.
"""

import os
import json
import time
import warnings
import numpy as np
import librosa
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from sklearn.mixture import GaussianMixture

# Optional heavy deps – import lazily / fallbacks handled inside
try:
    import madmom
except ImportError:
    warnings.warn("madmom not found, beat tracking will fall back to librosa.")
try:
    import vamp
except ImportError:
    warnings.warn("vamp not found, melody extraction will fall back to librosa.pyin.")


MIN_SEGMENT_DURATION = 1.0  # Minimum segment duration in seconds


class AnalysisMethod(Enum):
    """Enumeration of analysis methods"""
    LIBROSA = "librosa"
    MADMOM = "madmom"
    VAMP_MELODIA = "vamp_melodia"
    FALLBACK = "fallback"


class SegmentType(Enum):
    """Enumeration of melody segment types"""
    LEGATO = "legato"
    STACCATO = "staccato"


class SectionType(Enum):
    """Enumeration of musical section types"""
    A = "A"
    B = "B"
    C = "C"
    INTRO = "intro"
    OUTRO = "outro"
    UNKNOWN = "unknown"


class BeatType(Enum):
    """Enumeration of beat duration types"""
    SHORT = "short"
    LONG = "long"
    MARCATO2 = "marcato2"
    MARCATO4 = "marcato4"
    UNKNOWN = "unknown"


@dataclass
class BeatData:
    """Data class for individual beat information"""
    time: float
    confidence: float
    strength: float
    duration: float
    beat_type: BeatType


@dataclass
class BeatAnalysisResult:
    """Data class for beat analysis results"""
    bpm: float
    beats: List[BeatData]
    method: AnalysisMethod
    confidence: float
    total_beats: int
    downbeats: List[float] # Added from tango_analyzer_gpt
    marcato_type: BeatType # Added from tango_analyzer_gpt


@dataclass
class MelodySegment:
    """Data class for melody segment information"""
    start: float
    end: float
    segment_type: SegmentType
    confidence: float
    duration: float


@dataclass
class MelodyStatistics:
    """Data class for melody analysis statistics"""
    legato_percentage: float
    staccato_percentage: float
    total_segments: int
    average_segment_duration: float
    median_interval: float


@dataclass
class MelodyAnalysisResult:
    """Data class for melody analysis results"""
    segments: List[MelodySegment]
    statistics: MelodyStatistics
    f0_times: List[float] # Added from tango_analyzer_gpt
    f0_values: List[float] # Added from tango_analyzer_gpt


@dataclass
class MusicalSection:
    """Data class for musical section information"""
    section_type: SectionType
    start: float
    end: float
    duration: float
    description: str


@dataclass
class TangoPhrase:
    """Data class for tango musical phrase information"""
    start: float
    end: float
    duration: float
    pause_start: float
    pause_end: float
    pause_duration: float
    volume_drop_ratio: float # Kept for schema, but not calculated by tango_analyzer_gpt
    frequency_drop_ratio: float # Kept for schema, but not calculated by tango_analyzer_gpt
    confidence: float # Kept for schema, but not calculated by tango_analyzer_gpt


@dataclass
class PhraseStatistics:
    """Data class for phrase analysis statistics"""
    total_phrases: int
    average_phrase_duration: float
    average_pause_duration: float
    phrase_density: float  # phrases per minute
    total_pause_time: float


@dataclass
class SectionAnalysisResult:
    """Data class for section analysis results"""
    time_signature: str
    sections: List[MusicalSection]
    phrases: List[TangoPhrase]
    phrase_statistics: PhraseStatistics
    total_duration: float
    section_count: int
    section_boundaries: List[float] # Added from tango_analyzer_gpt


@dataclass
class CompleteAnalysisResult:
    """Data class for complete music analysis results"""
    beat_analysis: BeatAnalysisResult
    melody_analysis: MelodyAnalysisResult
    section_analysis: SectionAnalysisResult
    processing_duration: float
    file_info: Dict[str, Any]


@dataclass
class TangoConfig:
    """Central calibration for all algorithms (change once, propagate everywhere)."""

    # General toggles
    enabled: Dict[str, bool] = field(default_factory=lambda: {
        "beat": True,
        "melody": True,
        "pause": True,
        "phrase": True,
        "section": True,
    })

    # Beat / downbeat tracking
    use_madmom: bool = False  # Default to False as per previous attempts, can be overridden
    beat_transition_lambda: int = 100
    min_bpm: int = 40
    max_bpm: int = 240

    # Beat-type (marcato-2 vs marcato-4)
    marcato_energy_window_bars: int = 8  # analyse over this many bars for pattern

    # Melody extraction + articulation
    use_vamp_melodia: bool = False  # Default to False as per previous attempts, can be overridden
    gmm_components: int = 2  # legato/staccato clusters
    articulation_log: bool = True  # cluster on log(IOI)

    # Pause detection
    pause_rms_percentile: float = 25.0
    pause_min_duration: float = 0.35  # seconds

    # Phrase detection (energy dips + bar boundaries)
    phrase_min_bars: int = 4  # typical tango phrase length ~16 beats = 4 bars
    phrase_energy_drop_db: float = 10.0  # drop wrt local median considered phrase end

    # Section detection (Foote novelty)
    novelty_kernel: int = 32  # frames for checkerboard kernel


class MusicAnalyzer:
    """
    Main music analysis class that provides comprehensive analysis of audio files.
    
    This class follows the specified order: beat analysis first, then melody, then sections.
    All analysis methods are granular and can be used independently or together.
    """
    
    def __init__(self, audio_path: str, config: Optional[TangoConfig] = None):
        """
        Initialize the MusicAnalyzer with an audio file.
        
        Args:
            audio_path (str): Path to the audio file to analyze
            config (Optional[TangoConfig]): Configuration for the analysis.
        """
        self.audio_path = audio_path
        self.audio_data: Optional[np.ndarray] = None
        self.sample_rate: Optional[int] = None
        self.duration: Optional[float] = None
        self.cfg = config or TangoConfig()
        self._load_audio()
    
    def _load_audio(self) -> None:
        """
        Load audio file and extract basic information.
        
        This method loads the audio file using librosa, ensuring it's mono
        and preserving the original sample rate.
        """
        try:
            if not os.path.exists(self.audio_path):
                raise FileNotFoundError(f"Audio file not found: {self.audio_path}")
            
            # Load with sr=None to preserve original sample rate, mono=True for single channel
            self.audio_data, self.sample_rate = librosa.load(self.audio_path, sr=None, mono=True)
            self.duration = librosa.get_duration(y=self.audio_data, sr=self.sample_rate)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file: {str(e)}")

    # ==================== BEAT ANALYSIS (from tango_analyzer_gpt) ====================
    def _detect_beats_internal(self, y: np.ndarray, sr: int, audio_path: str) -> Tuple[float, np.ndarray, np.ndarray, str, AnalysisMethod]:
        cfg = self.cfg
        beat_times, downbeat_times = None, None
        method_used = AnalysisMethod.FALLBACK

        if cfg.use_madmom:
            try:
                from madmom.features.beats import (
                    RNNBeatProcessor, DBNBeatTrackingProcessor,
                    RNNDownBeatProcessor, DBNDownBeatTrackingProcessor,
                )
                proc_b = RNNBeatProcessor()(audio_path)
                beats = DBNBeatTrackingProcessor(
                    min_bpm=cfg.min_bpm,
                    max_bpm=cfg.max_bpm,
                    transition_lambda=cfg.beat_transition_lambda,
                )(proc_b)
                proc_db = RNNDownBeatProcessor()(audio_path)
                downbeats = DBNDownBeatTrackingProcessor(
                    beats_per_bar=[3, 4],
                    min_bpm=cfg.min_bpm,
                    max_bpm=cfg.max_bpm,
                )(proc_db)
                beat_times = beats
                downbeat_times = downbeats[:, 0]
                method_used = AnalysisMethod.MADMOM
            except Exception as e:
                warnings.warn(f"madmom unavailable or failed ({e}); falling back to librosa")

        if beat_times is None or len(beat_times) == 0: # Check length for empty madmom result
            tempo_val, beat_frames = librosa.beat.beat_track(y=y, sr=sr, start_bpm=120, trim=False)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            downbeat_times = beat_times[::4] # Crude fallback
            method_used = AnalysisMethod.LIBROSA
        
        if len(beat_times) > 1:
            tempo_val = 60.0 / np.median(np.diff(beat_times))
        elif beat_times is not None and len(beat_times) == 1: # if only one beat, tempo is undefined or use a default
             tempo_val = 0.0 # Or some default like 120.0
        else: # No beats detected
            tempo_val = 0.0
            beat_times = np.array([]) # Ensure it's an empty array
            downbeat_times = np.array([])

        marcato_type_str = self._classify_marcato(beat_times, y, sr)
        return float(tempo_val), beat_times, downbeat_times, marcato_type_str, method_used

    def _classify_marcato(self, beat_times: np.ndarray, y: np.ndarray, sr: int) -> str:
        cfg = self.cfg
        if len(beat_times) < 8:
            return "unknown"
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
        beat_frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=hop_length)
        beat_energy = []
        for i in range(len(beat_frames)-1):
            seg_start, seg_end = beat_frames[i], beat_frames[i+1]
            if seg_start >= len(rms) or seg_end > len(rms) or seg_start >= seg_end: # Boundary checks
                beat_energy.append(0)
                continue
            seg = rms[seg_start:seg_end]
            beat_energy.append(np.mean(seg) if len(seg) > 0 else 0)
        if len(beat_frames) > 0 and beat_frames[-1] < len(rms): # Handle last beat segment
            seg = rms[beat_frames[-1]:]
            beat_energy.append(np.mean(seg) if len(seg) > 0 else 0)
        else: # if beat_frames is empty or last beat is out of bounds
             if len(beat_frames) > 0: # only append if there was a last beat frame
                beat_energy.append(0)

        beat_energy = np.asarray(beat_energy)
        if len(beat_energy) < 4 : # Need at least one 4-beat bar
            return "unknown"

        # Evaluate pattern across 4-beat bars
        # Ensure indices are within bounds for beat_energy
        strong_beats_energy = []
        weak_beats_energy = []
        for i in range(0, len(beat_energy) - 3, 4):
            strong_beats_energy.extend([beat_energy[i], beat_energy[i+2]])
            weak_beats_energy.extend([beat_energy[i+1], beat_energy[i+3]])

        if not strong_beats_energy or not weak_beats_energy:
            return "unknown"

        strong_mean = np.mean(strong_beats_energy)
        weak_mean = np.mean(weak_beats_energy)

        if strong_mean > weak_mean * 1.2:
            return "marcato2"
        elif np.allclose(strong_mean, weak_mean, rtol=0.15):
            return "marcato4"
        else:
            return "unknown"

    def analyze_beats(self) -> BeatAnalysisResult:
        try:
            tempo, beat_times_arr, downbeat_times_arr, marcato_type_str, method_used = self._detect_beats_internal(
                self.audio_data, self.sample_rate, self.audio_path
            )
            
            # Map marcato_type_str to BeatType enum
            marcato_enum = BeatType.UNKNOWN
            if marcato_type_str == "marcato2":
                marcato_enum = BeatType.MARCATO2
            elif marcato_type_str == "marcato4":
                marcato_enum = BeatType.MARCATO4

            # Create BeatData objects (simplified, as tango_analyzer_gpt doesn't provide per-beat confidence/strength/duration)
            beats_data: List[BeatData] = []
            confidences = [] # Placeholder for confidences

            # Use librosa onset strength for placeholder confidence if method is librosa
            # For madmom, we don't have direct per-beat confidence from the DBN processor
            onset_envelope = None
            if method_used == AnalysisMethod.LIBROSA and len(beat_times_arr) > 0:
                onset_envelope = librosa.onset.onset_strength(y=self.audio_data, sr=self.sample_rate)
                beat_frames = librosa.time_to_frames(beat_times_arr, sr=self.sample_rate)

            for i, beat_time in enumerate(beat_times_arr):
                confidence = 0.7 # Default placeholder
                if onset_envelope is not None and i < len(beat_frames) and beat_frames[i] < len(onset_envelope):
                    confidence = float(onset_envelope[beat_frames[i]])
                elif method_used == AnalysisMethod.MADMOM:
                    confidence = 0.8 # Slightly higher default for madmom as it's generally more robust
                
                confidences.append(confidence)
                # Placeholder for duration and individual beat_type (could be refined)
                # Original music_analyzer_logic had complex duration calculation, which is not in tango_analyzer_gpt
                # For now, using a fixed short duration and the overall marcato_type
                beats_data.append(BeatData(
                    time=float(beat_time),
                    confidence=confidence, 
                    strength=confidence, # Using confidence as strength placeholder
                    duration=0.1, # Placeholder duration
                    beat_type=marcato_enum # Using overall marcato type for all beats for now
                ))
            
            # Normalize confidences if calculated
            if confidences:
                max_conf = max(confidences) if confidences else 1.0
                if max_conf > 0:
                    confidences = [c / max_conf for c in confidences]
                    for i, beat_obj in enumerate(beats_data):
                        beat_obj.confidence = confidences[i]
                        beat_obj.strength = confidences[i]

            overall_confidence = float(np.mean(confidences)) if confidences else 0.0
            if not beat_times_arr.any(): # Handle case of no beats detected
                overall_confidence = 0.0

            return BeatAnalysisResult(
                bpm=float(tempo),
                beats=beats_data,
                method=method_used,
                confidence=overall_confidence,
                total_beats=len(beats_data),
                downbeats=[float(db) for db in downbeat_times_arr],
                marcato_type=marcato_enum
            )
        except Exception as e:
            warnings.warn(f"Beat analysis failed: {e}")
            return BeatAnalysisResult(
                bpm=0.0, beats=[], method=AnalysisMethod.FALLBACK, confidence=0.0, total_beats=0, 
                downbeats=[], marcato_type=BeatType.UNKNOWN
            )

    # ==================== MELODY ANALYSIS (from tango_analyzer_gpt) ====================
    def _detect_melody_internal(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, List[MelodySegment], AnalysisMethod]:
        cfg = self.cfg
        times, f0 = None, None
        method_used = AnalysisMethod.FALLBACK

        if cfg.use_vamp_melodia:
            try:
                import vamp
                data = vamp.collect(y, sr, "mtg-melodia:melodia")
                hop_size = data["stepSize"]
                raw_frames, raw_values = zip(*data["vector"])
                # times = np.array([f * hop_size / sr for f in raw_frames]) # This seems to be how vamp returns it
                times = np.array(raw_frames) # if vamp returns actual time stamps
                f0 = np.array(raw_values)
                method_used = AnalysisMethod.VAMP_MELODIA
            except Exception as e:
                warnings.warn(f"Vamp/Melodia unavailable or failed ({e}); falling back to librosa.pyin")

        if times is None or len(times) == 0:
            # Default hop_length for librosa.pyin is 512, frame_length 2048
            f0_pyin, voiced_flag, voiced_probs = librosa.pyin(y, sr=sr, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
            times = librosa.times_like(f0_pyin, sr=sr) # Correctly get times for pyin output
            f0 = f0_pyin
            f0[~voiced_flag] = 0.0  # Set unvoiced frames to 0 Hz
            method_used = AnalysisMethod.LIBROSA
        
        if f0 is None: # Ensure f0 is not None
            f0 = np.array([])
            times = np.array([])

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        onsets_sec = librosa.frames_to_time(onset_frames, sr=sr)
        intervals = np.diff(onsets_sec)
        
        segments_internal: List[MelodySegment] = [] # Use the internal MelodySegment from tango_analyzer_gpt for classification
        if len(intervals) > 0:
            # Map to the internal MelodySegment for _classify_melody_segments_internal
            temp_segments_for_classification = self._classify_melody_segments_internal(onsets_sec[:-1], intervals)
            # Convert back to the main MelodySegment dataclass
            for seg_internal in temp_segments_for_classification:
                seg_type_enum = SegmentType.LEGATO if seg_internal.segment_type == "legato" else SegmentType.STACCATO
                segments_internal.append(MelodySegment(
                    start=seg_internal.start,
                    end=seg_internal.end,
                    segment_type=seg_type_enum,
                    confidence=seg_internal.confidence,
                    duration=seg_internal.duration
                ))
        return times, f0, segments_internal, method_used

    # Internal MelodySegment class as defined in tango_analyzer_gpt for _classify_melody_segments_internal
    @dataclass
    class _InternalMelodySegment:
        start: float
        end: float
        segment_type: str # "legato" | "staccato"
        confidence: float
        duration: float

    def _classify_melody_segments_internal(self, onsets: np.ndarray, intervals: np.ndarray) -> List[_InternalMelodySegment]:
        cfg = self.cfg
        segments_out: List[MusicAnalyzer._InternalMelodySegment] = [] # Explicitly use the nested class
        if len(intervals) == 0:
            return segments_out

        X = intervals.reshape(-1, 1)
        if cfg.articulation_log:
            # Ensure no zero or negative values before log
            X = X[X > 0]
            if len(X) == 0: return segments_out # Not enough valid intervals
            X = np.log(X)
        
        if len(X) < cfg.gmm_components: # Not enough samples for GMM
            warnings.warn(f"Not enough valid intervals ({len(X)}) for GMM with {cfg.gmm_components} components. Skipping segment classification.")
            return segments_out

        try:
            gmm = GaussianMixture(n_components=cfg.gmm_components, n_init=10, random_state=0).fit(X)
            labels = gmm.predict(X)
            means = np.exp(gmm.means_.ravel()) if cfg.articulation_log else gmm.means_.ravel()
            legato_lab = int(np.argmax(means))
            post = gmm.predict_proba(X).max(axis=1)

            # Adjust onsets to match the filtered X length if log was applied and some intervals were removed
            valid_indices = np.where(intervals > 0)[0] if cfg.articulation_log else np.arange(len(intervals))
            if len(valid_indices) != len(labels):
                 warnings.warn("Mismatch in length after GMM, skipping some segments.") # Should not happen if X filtering is correct
                 return segments_out

            for i in range(len(labels)):
                original_interval_index = valid_indices[i]
                seg_type_str = "legato" if labels[i] == legato_lab else "staccato"
                confidence = float(post[i])
                current_onset = float(onsets[original_interval_index])
                current_interval = float(intervals[original_interval_index])
                segments_out.append(MusicAnalyzer._InternalMelodySegment(
                    start=current_onset,
                    end=current_onset + current_interval,
                    segment_type=seg_type_str,
                    confidence=confidence,
                    duration=current_interval
                ))
        except ValueError as e:
            warnings.warn(f"GMM fitting failed for melody segments ({e}). Returning empty segments.")
            return []
        return segments_out

    def analyze_melody(self) -> MelodyAnalysisResult:
        try:
            times_arr, f0_arr, segments_list, method_used = self._detect_melody_internal(self.audio_data, self.sample_rate)

            # Calculate melody statistics from segments_list
            legato_count = sum(1 for s in segments_list if s.segment_type == SegmentType.LEGATO)
            staccato_count = sum(1 for s in segments_list if s.segment_type == SegmentType.STACCATO)
            total_segments = len(segments_list)
            
            legato_percentage = (legato_count / total_segments) * 100 if total_segments > 0 else 0.0
            staccato_percentage = (staccato_count / total_segments) * 100 if total_segments > 0 else 0.0
            
            segment_durations = [s.duration for s in segments_list]
            average_segment_duration = float(np.mean(segment_durations)) if segment_durations else 0.0
            
            # Median interval calculation needs raw intervals before GMM filtering
            onset_env = librosa.onset.onset_strength(y=self.audio_data, sr=self.sample_rate)
            onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=self.sample_rate)
            onsets_sec = librosa.frames_to_time(onset_frames, sr=self.sample_rate)
            raw_intervals = np.diff(onsets_sec)
            median_interval = float(np.median(raw_intervals)) if len(raw_intervals) > 0 else 0.0

            statistics = MelodyStatistics(
                legato_percentage=legato_percentage,
                staccato_percentage=staccato_percentage,
                total_segments=total_segments,
                average_segment_duration=average_segment_duration,
                median_interval=median_interval
            )

            return MelodyAnalysisResult(
                segments=segments_list,
                statistics=statistics,
                f0_times=[float(t) for t in times_arr],
                f0_values=[float(v) for v in f0_arr]
            )
        except Exception as e:
            warnings.warn(f"Melody analysis failed: {e}")
            return MelodyAnalysisResult(
                segments=[], 
                statistics=MelodyStatistics(0,0,0,0,0),
                f0_times=[], f0_values=[]
            )

    # ==================== PAUSE AND PHRASE ANALYSIS (from tango_analyzer_gpt) ====================
    def _detect_pauses_internal(self, y: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        cfg = self.cfg
        hop = 512
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
        threshold = np.percentile(rms, cfg.pause_rms_percentile)
        below = rms < threshold
        pauses_tuples: List[Tuple[float, float]] = []
        if np.any(below):
            # Find contiguous regions below threshold
            # pad `below` with False at both ends to correctly identify groups at edges
            padded_below = np.concatenate(([False], below, [False]))
            diffs = np.diff(padded_below.astype(int))
            starts = np.where(diffs == 1)[0] # frame indices where pause starts
            ends = np.where(diffs == -1)[0]   # frame indices where pause ends (exclusive)

            for s_frame, e_frame in zip(starts, ends):
                num_frames_in_pause = e_frame - s_frame
                dur = librosa.frames_to_time(num_frames_in_pause, sr=sr, hop_length=hop)
                if dur >= cfg.pause_min_duration:
                    start_time = librosa.frames_to_time(s_frame, sr=sr, hop_length=hop)
                    end_time = librosa.frames_to_time(e_frame, sr=sr, hop_length=hop) # e_frame is exclusive end
                    pauses_tuples.append((float(start_time), float(end_time)))
        return pauses_tuples

    def _detect_phrases_internal(self, y: np.ndarray, sr: int, pauses_tuples: Optional[List[Tuple[float, float]]] = None) -> np.ndarray:
        cfg = self.cfg
        boundaries_arr: np.ndarray
        if pauses_tuples and len(pauses_tuples) > 0:
            # Phrase ends at the END of a pause, or start of next phrase is start of pause
            # tango_analyzer_gpt uses pause END as boundary. Let's stick to that.
            boundaries_arr = np.array(sorted([p[1] for p in pauses_tuples]))
        else:
            tempo_val, _ = librosa.beat.beat_track(y=y, sr=sr, trim=False)
            if tempo_val == 0:
                return np.array([])
            bar_dur = 240.0 / tempo_val  # Assumes 4 beats per bar for tempo calculation
            total_dur = len(y) / sr
            if bar_dur * cfg.phrase_min_bars <= 0: # Avoid issues with zero or negative bar_dur
                 return np.array([])
            boundaries_arr = np.arange(bar_dur * cfg.phrase_min_bars, total_dur, bar_dur * cfg.phrase_min_bars)
        return boundaries_arr

    # ==================== SECTION ANALYSIS (from tango_analyzer_gpt) ====================
    def _detect_sections_internal(self, y: np.ndarray, sr: int) -> np.ndarray:
        cfg = self.cfg
        # chroma = librosa.feature.chroma_cqt(y=y, sr=sr) # CQT can be slow
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=512)
        
        # Ensure chroma and mfcc have same number of frames if hop_length is consistent
        # If not, one might need to be trimmed or padded, or use consistent hop_length
        min_frames = min(chroma.shape[1], mfcc.shape[1])
        chroma = chroma[:, :min_frames]
        mfcc = mfcc[:, :min_frames]

        X = np.vstack([chroma, librosa.util.normalize(mfcc, axis=1)])
        # S = librosa.segment.recurrence_matrix(X, mode="affinity", width=3) # width=3 is small, default is 1
        S = librosa.segment.recurrence_matrix(X, width=cfg.novelty_kernel//4, mode='affinity', sym=True)
        # novelty = librosa.segment.nn_filter(S, aggregate=None).mean(0) # This was in tango_analyzer_gpt
        # A common way to get novelty from recurrence matrix for segmentation:
        L = librosa.segment.path_enhance(S, cfg.novelty_kernel, R_smooth=2, L_smooth=2)
        novelty_curve = np.sum(np.diff(L, axis=1)**2, axis=0)
        novelty_curve = np.concatenate(([0.0], novelty_curve, [0.0])) # Pad for boundary conditions

        # peaks = librosa.util.peak_pick(novelty_curve, cfg.novelty_kernel, cfg.novelty_kernel, cfg.novelty_kernel, cfg.novelty_kernel, 0.2, 5)
        # Simpler peak picking, or use librosa.segment.agglomerative or find_peaks from scipy.signal
        # Using a simpler approach based on librosa.segment.agglomerative example logic for boundaries
        # Or, stick to peak_pick if it works well enough.
        # Let's try a slightly different peak picking for robustness, or stick to original if it was fine.
        # The original peak_pick in tango_analyzer_gpt was: 
        # peaks = librosa.util.peak_pick(novelty, cfg.novelty_kernel, cfg.novelty_kernel, cfg.novelty_kernel, cfg.novelty_kernel, 0.2, 5)
        # where novelty was `np.diff(np.concatenate([[0], librosa.segment.nn_filter(S, aggregate=None).mean(0)]))`
        # Let's try to replicate that more closely if the path_enhance method is too different.
        
        # Reverting to a structure closer to original tango_analyzer_gpt for novelty and peaks:
        df = librosa.segment.timelag_filter(scipy.signal.medfilt, size=(1, cfg.novelty_kernel // 2))(S)
        # Path enhancement can be good but let's use the simpler novelty from tango_analyzer_gpt first
        # novelty_from_S = librosa.segment.nn_filter(S, aggregate=None).mean(0)
        # novelty_from_S = np.diff(np.concatenate([[0], novelty_from_S]))
        # Using librosa's segmentation function for boundaries directly from features
        # This is often more robust than manual peak picking on a novelty curve
        # However, to match tango_analyzer_gpt, let's use its novelty approach.

        # Replicating tango_analyzer_gpt's novelty calculation more directly:
        # S is already calculated. Original was: S = librosa.segment.recurrence_matrix(X, mode="affinity", width=3)
        # Let's use a slightly larger width for S if novelty_kernel is large
        # S_for_novelty = librosa.segment.recurrence_matrix(X, width=max(3, cfg.novelty_kernel // 8), mode='affinity', sym=True)
        # novelty_original_style = librosa.segment.nn_filter(S_for_novelty, aggregate=None).mean(0)
        # novelty_original_style = np.diff(np.concatenate([[0.0], novelty_original_style]))
        # peaks = librosa.util.peak_pick(novelty_original_style, pre_max=cfg.novelty_kernel, post_max=cfg.novelty_kernel, pre_avg=cfg.novelty_kernel, post_avg=cfg.novelty_kernel, delta=0.2, wait=5)
        
        # Using librosa.segment.boundaries for a more direct approach from features X
        # This might be more robust than manual peak picking on a derived novelty curve.
        # It finds boundaries based on feature dissimilarity.
        # `librosa.segment.agglomerative` or `librosa.segment.subsegment` can be used.
        # Let's try `librosa.segment.agglomerative` which is common.
        # We need to estimate k (number of segments) or use a distance threshold.
        # For simplicity and to avoid estimating k, let's stick to novelty + peak picking from tango_analyzer_gpt.

        # Using the novelty curve from path_enhance as it's a common segmentation approach
        # novelty_curve is already calculated above.
        # Adjust peak picking parameters based on novelty_curve characteristics
        # The hop_length for X (and thus novelty_curve) is 512 samples.
        # Convert cfg.novelty_kernel (frames for checkerboard) to something meaningful for peak_pick on this novelty_curve
        # Let's assume cfg.novelty_kernel was originally for frames of a different hop_length or for the recurrence matrix itself.
        # For peak_pick on novelty_curve (length of audio frames), wait parameter is important.
        # A wait time of a few seconds in frames: e.g., 5s * sr / hop_length_X
        wait_frames = int(5 * sr / 512) # Wait 5 seconds
        delta_val = 0.1 * np.std(novelty_curve) # Dynamic delta based on novelty variance
        
        # Try scipy's find_peaks for more control
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(novelty_curve, height=delta_val, distance=wait_frames)

        # boundaries_frames = librosa.util.peak_pick(novelty_curve, مبلغ=wait_frames//2, post_max=wait_frames//2, pre_avg=wait_frames//2, post_avg=wait_frames//2, delta=delta_val, wait=wait_frames//4)
        boundaries_frames = peaks # Use peaks from find_peaks
        boundaries_times = librosa.frames_to_time(boundaries_frames, sr=sr, hop_length=512) # hop_length used for X
        
        # Add start and end of the audio as boundaries
        final_boundaries = np.concatenate(([0.0], boundaries_times, [self.duration]))
        final_boundaries = np.unique(final_boundaries) # Remove duplicates and sort
        final_boundaries = final_boundaries[final_boundaries <= self.duration] # Ensure within bounds
        return final_boundaries

    def analyze_sections(self) -> SectionAnalysisResult:
        try:
            section_boundaries_arr = self._detect_sections_internal(self.audio_data, self.sample_rate)
            pauses_tuples = self._detect_pauses_internal(self.audio_data, self.sample_rate)
            phrase_boundaries_arr = self._detect_phrases_internal(self.audio_data, self.sample_rate, pauses_tuples)

            sections_list: List[MusicalSection] = []
            for i in range(len(section_boundaries_arr) - 1):
                start_time = section_boundaries_arr[i]
                end_time = section_boundaries_arr[i+1]
                # Simple classification for now, can be enhanced
                sec_type = SectionType.UNKNOWN # Default
                # Could try to map to A/B/C based on some criteria later if needed
                sections_list.append(MusicalSection(
                    section_type=sec_type,
                    start=float(start_time),
                    end=float(end_time),
                    duration=float(end_time - start_time),
                    description=f"Section {i+1}"
                ))

            phrases_list: List[TangoPhrase] = []
            # Create TangoPhrase objects from phrase_boundaries_arr
            # Need to associate pauses with phrases if possible
            for i in range(len(phrase_boundaries_arr)):
                phrase_start = phrase_boundaries_arr[i-1] if i > 0 else 0.0
                phrase_end = phrase_boundaries_arr[i]
                if phrase_start >= phrase_end: continue # Skip zero or negative duration phrases

                # Find pause within this phrase (simplistic: first pause that starts within the phrase)
                current_pause_start, current_pause_end, current_pause_duration = 0.0, 0.0, 0.0
                for p_start, p_end in pauses_tuples:
                    if phrase_start <= p_start < phrase_end:
                        current_pause_start = p_start
                        current_pause_end = min(p_end, phrase_end) # Pause can't extend beyond phrase
                        current_pause_duration = current_pause_end - current_pause_start
                        break
                
                phrases_list.append(TangoPhrase(
                    start=float(phrase_start),
                    end=float(phrase_end),
                    duration=float(phrase_end - phrase_start),
                    pause_start=float(current_pause_start),
                    pause_end=float(current_pause_end),
                    pause_duration=float(current_pause_duration),
                    volume_drop_ratio=0.0, # Placeholder
                    frequency_drop_ratio=0.0, # Placeholder
                    confidence=1.0 # Placeholder
                ))
            
            # Calculate phrase statistics
            total_phrases = len(phrases_list)
            avg_phrase_dur = float(np.mean([p.duration for p in phrases_list])) if total_phrases > 0 else 0.0
            avg_pause_dur = float(np.mean([p.pause_duration for p in phrases_list if p.pause_duration > 0])) if total_phrases > 0 and any(p.pause_duration > 0 for p in phrases_list) else 0.0
            total_pause_time = sum(p.pause_duration for p in phrases_list)
            phrase_density = (total_phrases / self.duration) * 60 if self.duration and self.duration > 0 else 0.0

            phrase_stats = PhraseStatistics(
                total_phrases=total_phrases,
                average_phrase_duration=avg_phrase_dur,
                average_pause_duration=avg_pause_dur,
                phrase_density=phrase_density,
                total_pause_time=total_pause_time
            )
            
            # Time signature: tango_analyzer_gpt doesn't detect it. Defaulting.
            time_sig = "4/4"
            if self.cfg.enabled.get("beat", False) and hasattr(self, 
'beat_analysis_result') and self.beat_analysis_result:
                 # A more sophisticated time signature detection could be added here based on downbeats
                 # For now, keeping it simple
                 pass 

            return SectionAnalysisResult(
                time_signature=time_sig,
                sections=sections_list,
                phrases=phrases_list,
                phrase_statistics=phrase_stats,
                total_duration=float(self.duration or 0.0),
                section_count=len(sections_list),
                section_boundaries=[float(b) for b in section_boundaries_arr]
            )
        except Exception as e:
            warnings.warn(f"Section analysis failed: {e}")
            return SectionAnalysisResult(
                time_signature="unknown", sections=[], phrases=[], 
                phrase_statistics=PhraseStatistics(0,0,0,0,0),
                total_duration=float(self.duration or 0.0), section_count=0, section_boundaries=[]
            )

    def analyze_complete(self) -> CompleteAnalysisResult:
        """
        Performs a complete analysis of the audio file.

        Returns:
            CompleteAnalysisResult: All analysis results combined.
        """
        start_time_proc = time.time()

        # Initialize with fallback/empty results
        beat_res = BeatAnalysisResult(bpm=0.0, beats=[], method=AnalysisMethod.FALLBACK, confidence=0.0, total_beats=0, downbeats=[], marcato_type=BeatType.UNKNOWN)
        melody_res = MelodyAnalysisResult(segments=[], statistics=MelodyStatistics(0,0,0,0,0), f0_times=[], f0_values=[])
        section_res = SectionAnalysisResult(time_signature="unknown", sections=[], phrases=[], phrase_statistics=PhraseStatistics(0,0,0,0,0), total_duration=float(self.duration or 0.0), section_count=0, section_boundaries=[])

        if self.cfg.enabled.get("beat", False):
            beat_res = self.analyze_beats()
            # Store for potential use in section analysis (e.g. time signature)
            self.beat_analysis_result = beat_res 

        if self.cfg.enabled.get("melody", False):
            melody_res = self.analyze_melody()
        
        # Section analysis internally calls pause and phrase detection
        if self.cfg.enabled.get("section", False) or self.cfg.enabled.get("phrase", False) or self.cfg.enabled.get("pause", False):
            section_res = self.analyze_sections()

        end_time_proc = time.time()
        processing_duration = end_time_proc - start_time_proc

        file_info = {
            "audio_path": self.audio_path,
            "sample_rate": self.sample_rate,
            "duration": self.duration
        }

        return CompleteAnalysisResult(
            beat_analysis=beat_res,
            melody_analysis=melody_res,
            section_analysis=section_res,
            processing_duration=processing_duration,
            file_info=file_info
        )


if __name__ == "__main__":
    # Example Usage (for testing purposes)
    dummy_audio_path = "dummy_audio.wav"
    if not os.path.exists(dummy_audio_path):
        print(f"Creating a dummy audio file: {dummy_audio_path}")
        sr_dummy = 22050
        duration_dummy = 20 # seconds, longer for more sections/phrases
        y_dummy = librosa.tone(frequency=440, sr=sr_dummy, duration=duration_dummy/4) 
        y_dummy = np.concatenate([y_dummy, np.zeros(int(sr_dummy*0.5)), librosa.tone(frequency=220, sr=sr_dummy, duration=duration_dummy/4)])
        y_dummy = np.concatenate([y_dummy, np.zeros(int(sr_dummy*0.5)), librosa.tone(frequency=880, sr=sr_dummy, duration=duration_dummy/4)])
        y_dummy = np.concatenate([y_dummy, np.zeros(int(sr_dummy*1.0)), librosa.tone(frequency=330, sr=sr_dummy, duration=duration_dummy/4)])
        # Add some noise to make it a bit more realistic for RMS based pause detection
        y_dummy += 0.01 * np.random.randn(len(y_dummy))
        import soundfile as sf
        sf.write(dummy_audio_path, y_dummy, sr_dummy)
        print(f"Dummy audio created: {dummy_audio_path} with duration {librosa.get_duration(y=y_dummy, sr=sr_dummy)}s")

    print(f"Analyzing: {dummy_audio_path}")
    # Test with default config (madmom/vamp disabled by default in this script's TangoConfig)
    analyzer = MusicAnalyzer(audio_path=dummy_audio_path)
    
    # To test with madmom/vamp enabled if installed:
    # test_config = TangoConfig(use_madmom=True, use_vamp_melodia=True)
    # analyzer = MusicAnalyzer(audio_path=dummy_audio_path, config=test_config)
    
    complete_results = analyzer.analyze_all()

    def dataclass_to_dict(obj):
        if isinstance(obj, Enum):
            return obj.value
        if hasattr(obj, '__dict__'): # For dataclasses or regular classes
            return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
        elif isinstance(obj, dict):
            return {k: dataclass_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [dataclass_to_dict(i) for i in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist() # Convert numpy arrays to lists for JSON
        return obj

    results_dict = dataclass_to_dict(complete_results)
    print(json.dumps(results_dict, indent=2))

    # Clean up dummy file
    # os.remove(dummy_audio_path)
    # print(f"Cleaned up {dummy_audio_path}")



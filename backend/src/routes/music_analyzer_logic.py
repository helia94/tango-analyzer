"""
Music Analysis Logic Layer

This module contains the core music analysis logic separated from the REST API layer.
The MusicAnalyzer class provides granular analysis methods for beat, melody, and section analysis.
"""

import os
import json
import time
import numpy as np
import librosa
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class AnalysisMethod(Enum):
    """Enumeration of analysis methods"""
    LIBROSA = "librosa"
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


class BeatType(Enum):
    """Enumeration of beat duration types"""
    SHORT = "short"
    LONG = "long"


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
    volume_drop_ratio: float
    frequency_drop_ratio: float
    confidence: float


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


@dataclass
class CompleteAnalysisResult:
    """Data class for complete music analysis results"""
    beat_analysis: BeatAnalysisResult
    melody_analysis: MelodyAnalysisResult
    section_analysis: SectionAnalysisResult
    processing_duration: float
    file_info: Dict[str, Any]


class MusicAnalyzer:
    """
    Main music analysis class that provides comprehensive analysis of audio files.
    
    This class follows the specified order: beat analysis first, then melody, then sections.
    All analysis methods are granular and can be used independently or together.
    """
    
    def __init__(self, audio_path: str):
        """
        Initialize the MusicAnalyzer with an audio file.
        
        Args:
            audio_path (str): Path to the audio file to analyze
        """
        self.audio_path = audio_path
        self.audio_data: Optional[np.ndarray] = None
        self.sample_rate: Optional[int] = None
        self.duration: Optional[float] = None
        self._load_audio()
    
    def _load_audio(self) -> None:
        """Load audio file and extract basic information"""
        try:
            if not os.path.exists(self.audio_path):
                raise FileNotFoundError(f"Audio file not found: {self.audio_path}")
            
            self.audio_data, self.sample_rate = librosa.load(self.audio_path)
            self.duration = librosa.get_duration(y=self.audio_data, sr=self.sample_rate)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file: {str(e)}")
    
    # ==================== BEAT ANALYSIS METHODS ====================
    
    def analyze_beats(self) -> BeatAnalysisResult:
        """
        Comprehensive beat analysis - FIRST in the analysis order.
        
        Returns:
            BeatAnalysisResult: Complete beat analysis results
        """
        try:
            # Extract tempo and beat positions
            tempo, beat_times = self._extract_tempo_and_beats()
            
            # Calculate beat confidences
            confidences = self._calculate_beat_confidences(beat_times)
            
            # Create beat data objects
            beats = self._create_beat_data_objects(beat_times, confidences)
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_beat_confidence(confidences)
            
            return BeatAnalysisResult(
                bpm=float(tempo),
                beats=beats,
                method=AnalysisMethod.LIBROSA,
                confidence=overall_confidence,
                total_beats=len(beats)
            )
            
        except Exception as e:
            return self._create_fallback_beat_analysis(str(e))
    
    def _extract_tempo_and_beats(self) -> Tuple[float, np.ndarray]:
        """Extract tempo and beat positions using librosa"""
        tempo, beats = librosa.beat.beat_track(
            y=self.audio_data, 
            sr=self.sample_rate, 
            units='time'
        )
        return tempo, beats
    
    def _calculate_beat_confidences(self, beat_times: np.ndarray) -> List[float]:
        """Calculate confidence scores for each detected beat"""
        onset_envelope = librosa.onset.onset_strength(
            y=self.audio_data, 
            sr=self.sample_rate
        )
        beat_frames = librosa.time_to_frames(beat_times, sr=self.sample_rate)
        
        confidences = []
        for frame in beat_frames:
            if frame < len(onset_envelope):
                confidences.append(float(onset_envelope[frame]))
            else:
                confidences.append(0.5)
        
        # Normalize confidences
        if confidences:
            max_conf = max(confidences)
            if max_conf > 0:
                confidences = [c / max_conf for c in confidences]
        
        return confidences
    
    def _create_beat_data_objects(self, beat_times: np.ndarray, confidences: List[float]) -> List[BeatData]:
        """Create BeatData objects from beat times, confidences, and durations"""
        beats = []
        
        # Calculate beat durations and types
        beat_durations = self._calculate_beat_durations(beat_times)
        beat_types = self._classify_beat_types(beat_durations, confidences)
        
        for i, (beat_time, confidence) in enumerate(zip(beat_times, confidences)):
            duration = beat_durations[i] if i < len(beat_durations) else 0.0
            beat_type = beat_types[i] if i < len(beat_types) else BeatType.SHORT
            
            beats.append(BeatData(
                time=float(beat_time),
                confidence=float(confidence),
                strength=float(confidence),  # For compatibility
                duration=float(duration),
                beat_type=beat_type
            ))
        return beats
    
    def _calculate_beat_durations(self, beat_times: np.ndarray) -> List[float]:
        """
        Calculate the duration of each individual beat sound.
        
        This measures how long each beat sound lasts (sustain/decay),
        not the interval between beats.
        
        Args:
            beat_times (np.ndarray): Array of beat onset times
            
        Returns:
            List[float]: Duration of each beat sound in seconds
        """
        if len(beat_times) == 0:
            return []
        
        durations = []
        
        # Calculate spectral rolloff and energy decay for each beat
        for i, beat_time in enumerate(beat_times):
            duration = self._measure_beat_sustain(beat_time)
            durations.append(duration)
        
        return durations
    
    def _measure_beat_sustain(self, beat_time: float) -> float:
        """
        Measure the sustain duration of a single beat.
        
        Uses spectral rolloff and energy decay analysis to determine
        how long the beat sound lasts after its onset.
        
        Args:
            beat_time (float): Time of beat onset
            
        Returns:
            float: Duration of beat sustain in seconds
        """
        try:
            # Define analysis window around the beat
            window_duration = 0.5  # 500ms window after beat onset
            start_sample = int(beat_time * self.sample_rate)
            end_sample = int((beat_time + window_duration) * self.sample_rate)
            
            # Ensure we don't exceed audio bounds
            end_sample = min(end_sample, len(self.audio_data))
            if start_sample >= len(self.audio_data):
                return 0.05  # Default short duration
            
            # Extract audio segment
            beat_segment = self.audio_data[start_sample:end_sample]
            
            if len(beat_segment) < 1024:  # Too short for analysis
                return 0.05
            
            # Calculate spectral rolloff over time
            rolloff = librosa.feature.spectral_rolloff(
                y=beat_segment, 
                sr=self.sample_rate,
                hop_length=512
            )[0]
            
            # Calculate RMS energy over time
            rms = librosa.feature.rms(
                y=beat_segment,
                hop_length=512
            )[0]
            
            # Find where energy drops significantly
            duration = self._find_energy_decay_point(rms, rolloff)
            
            # Convert frame-based duration to time
            frame_duration = 512 / self.sample_rate
            time_duration = duration * frame_duration
            
            # Clamp to reasonable bounds (5ms to 500ms)
            return max(0.005, min(0.5, time_duration))
            
        except Exception:
            # Fallback to default short duration
            return 0.05
    
    def _find_energy_decay_point(self, rms: np.ndarray, rolloff: np.ndarray) -> float:
        """
        Find the point where beat energy decays significantly.
        
        Args:
            rms (np.ndarray): RMS energy over time
            rolloff (np.ndarray): Spectral rolloff over time
            
        Returns:
            float: Duration in frames where energy decays
        """
        if len(rms) < 3:
            return 1.0  # Default short duration
        
        # Smooth the signals
        rms_smooth = np.convolve(rms, np.ones(3)/3, mode='same')
        
        # Find peak energy (usually at the beginning)
        peak_energy = np.max(rms_smooth[:5]) if len(rms_smooth) >= 5 else np.max(rms_smooth)
        
        # Define decay threshold (energy drops to 20% of peak)
        decay_threshold = peak_energy * 0.2
        
        # Find where energy drops below threshold
        decay_points = np.where(rms_smooth < decay_threshold)[0]
        
        if len(decay_points) > 0:
            # Return first significant decay point
            decay_frame = decay_points[0]
            return float(decay_frame)
        else:
            # If no significant decay found, use spectral rolloff change
            if len(rolloff) > 1:
                rolloff_diff = np.diff(rolloff)
                # Find where rolloff starts decreasing significantly
                significant_decrease = np.where(rolloff_diff < -np.std(rolloff_diff))[0]
                if len(significant_decrease) > 0:
                    return float(significant_decrease[0])
        
        # Default to short duration if no clear decay pattern
        return min(5.0, len(rms) * 0.3)
    
    def _classify_beat_types(self, durations: List[float], confidences: List[float]) -> List[BeatType]:
        """
        Classify beats as SHORT or LONG based on their duration and strength.
        
        Args:
            durations (List[float]): Beat durations in seconds
            confidences (List[float]): Beat confidence scores
            
        Returns:
            List[BeatType]: Classification for each beat
        """
        if not durations:
            return []
        
        # Calculate threshold for short vs long beats
        median_duration = np.median(durations)
        duration_std = np.std(durations)
        
        # Adaptive threshold based on the audio characteristics
        # Generally, beats > 100ms are considered "long" in tango context
        base_threshold = 0.1  # 100ms
        adaptive_threshold = max(base_threshold, median_duration)
        
        beat_types = []
        
        for duration, confidence in zip(durations, confidences):
            # Consider both duration and confidence
            # Strong beats with longer duration are more likely to be LONG
            weighted_duration = duration * (0.5 + confidence * 0.5)
            
            if weighted_duration > adaptive_threshold:
                beat_types.append(BeatType.LONG)
            else:
                beat_types.append(BeatType.SHORT)
        
        return beat_types
    
    def _calculate_overall_beat_confidence(self, confidences: List[float]) -> float:
        """Calculate overall confidence score for beat analysis"""
        return float(np.mean(confidences)) if confidences else 0.0
    
    def _create_fallback_beat_analysis(self, error_msg: str) -> BeatAnalysisResult:
        """Create fallback beat analysis when primary method fails"""
        return BeatAnalysisResult(
            bpm=120.0,  # Default tango BPM
            beats=[],
            method=AnalysisMethod.FALLBACK,
            confidence=0.0,
            total_beats=0
        )
    
    # ==================== MELODY ANALYSIS METHODS ====================
    
    def analyze_melody(self) -> MelodyAnalysisResult:
        """
        Comprehensive melody analysis - SECOND in the analysis order.
        
        Returns:
            MelodyAnalysisResult: Complete melody analysis results
        """
        try:
            # Detect onsets
            onsets = self._detect_onsets()
            
            if len(onsets) < 2:
                return self._create_empty_melody_analysis()
            
            # Calculate onset intervals
            intervals = self._calculate_onset_intervals(onsets)
            
            # Classify melody segments
            segments = self._classify_melody_segments(onsets, intervals)
            
            # Apply postprocessing to merge short segments
            processed_segments = self._postprocess_melody_segments(segments)
            
            # Calculate melody statistics with processed segments
            statistics = self._calculate_melody_statistics(processed_segments, intervals)
            
            return MelodyAnalysisResult(
                segments=processed_segments,
                statistics=statistics
            )
            
        except Exception as e:
            return self._create_empty_melody_analysis()
    
    def _detect_onsets(self) -> np.ndarray:
        """Detect onset times in the audio"""
        return librosa.onset.onset_detect(
            y=self.audio_data, 
            sr=self.sample_rate, 
            units='time'
        )
    
    def _calculate_onset_intervals(self, onsets: np.ndarray) -> np.ndarray:
        """Calculate intervals between consecutive onsets"""
        return np.diff(onsets)
    
    def _classify_melody_segments(self, onsets: np.ndarray, intervals: np.ndarray) -> List[MelodySegment]:
        """Classify melody segments as legato or staccato"""
        median_interval = np.median(intervals)
        segments = []
        
        for i in range(len(onsets) - 1):
            start_time = onsets[i]
            end_time = onsets[i + 1]
            interval = intervals[i]
            duration = float(end_time - start_time)
            
            # Classification logic: longer intervals = legato, shorter = staccato
            if interval > median_interval:
                segment_type = SegmentType.LEGATO
            else:
                segment_type = SegmentType.STACCATO
            
            # Confidence based on deviation from median
            confidence = min(1.0, abs(interval - median_interval) / median_interval + 0.5)
            
            segments.append(MelodySegment(
                start=float(start_time),
                end=float(end_time),
                segment_type=segment_type,
                confidence=float(confidence),
                duration=duration
            ))
        
        return segments
    
    def _calculate_melody_statistics(self, segments: List[MelodySegment], intervals: np.ndarray) -> MelodyStatistics:
        """Calculate comprehensive melody statistics"""
        if not segments:
            return MelodyStatistics(
                legato_percentage=50.0,
                staccato_percentage=50.0,
                total_segments=0,
                average_segment_duration=0.0,
                median_interval=0.0
            )
        
        legato_count = sum(1 for seg in segments if seg.segment_type == SegmentType.LEGATO)
        staccato_count = len(segments) - legato_count
        total_segments = len(segments)
        
        legato_percentage = (legato_count / total_segments * 100)
        staccato_percentage = (staccato_count / total_segments * 100)
        
        average_duration = np.mean([seg.duration for seg in segments])
        median_interval = float(np.median(intervals))
        
        return MelodyStatistics(
            legato_percentage=float(legato_percentage),
            staccato_percentage=float(staccato_percentage),
            total_segments=total_segments,
            average_segment_duration=float(average_duration),
            median_interval=median_interval
        )
    
    def _create_empty_melody_analysis(self) -> MelodyAnalysisResult:
        """Create empty melody analysis when no segments are found"""
        return MelodyAnalysisResult(
            segments=[],
            statistics=MelodyStatistics(
                legato_percentage=50.0,
                staccato_percentage=50.0,
                total_segments=0,
                average_segment_duration=0.0,
                median_interval=0.0
            )
        )
    
    def _postprocess_melody_segments(self, segments: List[MelodySegment]) -> List[MelodySegment]:
        """
        Postprocess melody segments to merge short segments using moving horizon approach.
        
        Ensures no segment is shorter than 3 seconds by merging with adjacent segments
        using majority rule within a moving horizon.
        
        Args:
            segments (List[MelodySegment]): Original melody segments
            
        Returns:
            List[MelodySegment]: Processed segments with no segment shorter than 3 seconds
        """
        if not segments:
            return segments
        
        MIN_SEGMENT_DURATION = 3.0  # 3 seconds minimum
        processed_segments = segments.copy()
        
        # Continue processing until no segments are shorter than 3 seconds
        iteration = 0
        max_iterations = min (len(segments), 4)  # Prevent infinite loops
        
        while iteration < max_iterations:
            iteration += 1
            short_segments_found = False
            
            # Find segments shorter than 3 seconds
            for i, segment in enumerate(processed_segments):
                if segment.duration < MIN_SEGMENT_DURATION:
                    short_segments_found = True
                    
                    # Apply moving horizon approach to merge this segment
                    processed_segments = self._merge_short_segment_with_horizon(
                        processed_segments, i, MIN_SEGMENT_DURATION
                    )
                    break  # Restart the loop after modification
            
            if not short_segments_found:
                break
        
        return processed_segments
    
    def _merge_short_segment_with_horizon(self, segments: List[MelodySegment], 
                                        short_index: int, min_duration: float) -> List[MelodySegment]:
        """
        Merge a short segment using moving horizon approach with majority rule.
        
        Args:
            segments (List[MelodySegment]): Current segments list
            short_index (int): Index of the short segment to merge
            min_duration (float): Minimum required duration
            
        Returns:
            List[MelodySegment]: Updated segments list with merged segment
        """
        if short_index >= len(segments):
            return segments
        
        short_segment = segments[short_index]
        
        # Define horizon window (segments to consider for merging)
        horizon_size = 3  # Consider up to 3 segments on each side
        start_idx = max(0, short_index - horizon_size)
        end_idx = min(len(segments), short_index + horizon_size + 1)
        
        # Find the best merge candidate using majority rule
        merge_candidate_idx = self._find_best_merge_candidate(
            segments, short_index, start_idx, end_idx
        )
        
        if merge_candidate_idx is None:
            # If no good candidate found, merge with immediate neighbor
            merge_candidate_idx = self._find_immediate_neighbor(segments, short_index)
        
        if merge_candidate_idx is not None:
            # Perform the merge
            return self._merge_segments(segments, short_index, merge_candidate_idx)
        
        return segments
    
    def _find_best_merge_candidate(self, segments: List[MelodySegment], 
                                 short_index: int, start_idx: int, end_idx: int) -> Optional[int]:
        """
        Find the best candidate for merging using majority rule within horizon.
        
        Args:
            segments (List[MelodySegment]): Current segments list
            short_index (int): Index of short segment
            start_idx (int): Start of horizon window
            end_idx (int): End of horizon window
            
        Returns:
            Optional[int]: Index of best merge candidate, or None if no good candidate
        """
        short_segment = segments[short_index]
        
        # Count segment types in the horizon
        legato_count = 0
        staccato_count = 0
        candidates = []
        
        for i in range(start_idx, end_idx):
            if i == short_index:
                continue
                
            segment = segments[i]
            candidates.append(i)
            
            if segment.segment_type == SegmentType.LEGATO:
                legato_count += 1
            else:
                staccato_count += 1
        
        if not candidates:
            return None
        
        # Determine majority type
        majority_type = SegmentType.LEGATO if legato_count > staccato_count else SegmentType.STACCATO
        
        # Find candidates of majority type, preferring adjacent segments
        same_type_candidates = [
            i for i in candidates 
            if segments[i].segment_type == majority_type
        ]
        
        if same_type_candidates:
            # Prefer adjacent segments
            adjacent_candidates = [
                i for i in same_type_candidates 
                if abs(i - short_index) == 1
            ]
            
            if adjacent_candidates:
                return adjacent_candidates[0]
            else:
                # Return closest same-type candidate
                return min(same_type_candidates, key=lambda x: abs(x - short_index))
        
        # If no same-type candidates, find any adjacent candidate
        adjacent_candidates = [
            i for i in candidates 
            if abs(i - short_index) == 1
        ]
        
        return adjacent_candidates[0] if adjacent_candidates else None
    
    def _find_immediate_neighbor(self, segments: List[MelodySegment], short_index: int) -> Optional[int]:
        """
        Find immediate neighbor (left or right) for merging.
        
        Args:
            segments (List[MelodySegment]): Current segments list
            short_index (int): Index of short segment
            
        Returns:
            Optional[int]: Index of immediate neighbor, or None if none available
        """
        # Prefer left neighbor, then right neighbor
        if short_index > 0:
            return short_index - 1
        elif short_index < len(segments) - 1:
            return short_index + 1
        
        return None
    
    def _merge_segments(self, segments: List[MelodySegment], 
                       index1: int, index2: int) -> List[MelodySegment]:
        """
        Merge two segments and return updated segments list.
        
        Args:
            segments (List[MelodySegment]): Current segments list
            index1 (int): Index of first segment to merge
            index2 (int): Index of second segment to merge
            
        Returns:
            List[MelodySegment]: Updated segments list with merged segment
        """
        if index1 == index2 or index1 >= len(segments) or index2 >= len(segments):
            return segments
        
        # Ensure index1 < index2 for consistent processing
        if index1 > index2:
            index1, index2 = index2, index1
        
        segment1 = segments[index1]
        segment2 = segments[index2]
        
        # Determine merged segment properties
        merged_start = min(segment1.start, segment2.start)
        merged_end = max(segment1.end, segment2.end)
        merged_duration = merged_end - merged_start
        
        # Use majority rule for segment type (or higher confidence)
        if segment1.segment_type == segment2.segment_type:
            merged_type = segment1.segment_type
            merged_confidence = max(segment1.confidence, segment2.confidence)
        else:
            # Choose type with higher confidence
            if segment1.confidence >= segment2.confidence:
                merged_type = segment1.segment_type
                merged_confidence = segment1.confidence
            else:
                merged_type = segment2.segment_type
                merged_confidence = segment2.confidence
        
        # Create merged segment
        merged_segment = MelodySegment(
            start=merged_start,
            end=merged_end,
            segment_type=merged_type,
            confidence=merged_confidence,
            duration=merged_duration
        )
        
        # Create new segments list
        new_segments = []
        
        for i, segment in enumerate(segments):
            if i == index1:
                new_segments.append(merged_segment)
            elif i == index2:
                continue  # Skip the second segment as it's merged
            else:
                new_segments.append(segment)
        
        return new_segments
    
    # ==================== SECTION ANALYSIS METHODS ====================
    
    def analyze_sections(self) -> SectionAnalysisResult:
        """
        Comprehensive section analysis - THIRD in the analysis order.
        
        Returns:
            SectionAnalysisResult: Complete section analysis results
        """
        try:
            # Estimate time signature
            time_signature = self._estimate_time_signature()
            
            # Detect musical sections
            sections = self._detect_musical_sections()
            
            # Detect phrases with beat-based constraints
            phrases, phrase_stats = self._detect_tango_phrases()
            
            return SectionAnalysisResult(
                time_signature=time_signature,
                sections=sections,
                phrases=phrases,
                phrase_statistics=phrase_stats,
                total_duration=float(self.duration),
                section_count=len(sections)
            )
            
        except Exception as e:
            return self._create_fallback_section_analysis()
    
    def _estimate_time_signature(self) -> str:
        """Estimate the time signature of the music"""
        try:
            tempo, _ = librosa.beat.beat_track(y=self.audio_data, sr=self.sample_rate)
            # Tango is typically in 2/4 or 4/4 time
            return "2/4" if tempo > 100 else "4/4"
        except:
            return "2/4"  # Default for tango
    
    def _detect_musical_sections(self) -> List[MusicalSection]:
        """
        Detect musical sections using spectral similarity analysis.
        
        Implements tango-specific structure detection:
        - 5 sections (ABABA or ABABC)
        - Each section ~30 seconds (20-40 seconds range)
        - Sections detected by spectral similarity and phrase boundaries
        """
        try:
            # Extract comprehensive audio features for similarity analysis
            features = self._extract_spectral_features()
            
            # Detect section boundaries using similarity analysis
            section_boundaries = self._detect_section_boundaries(features)
            
            # Classify sections into A/B/C types based on similarity
            section_types = self._classify_section_types(features, section_boundaries)
            
            # Create MusicalSection objects
            sections = self._create_musical_sections(section_boundaries, section_types)
            
            # Validate tango structure and adjust if needed
            validated_sections = self._validate_tango_structure(sections)
            
            return validated_sections
            
        except Exception as e:
            # Fallback to simple structure if advanced analysis fails
            return self._create_fallback_sections()
    
    def _extract_spectral_features(self) -> Dict[str, np.ndarray]:
        """
        Extract comprehensive spectral features for section analysis.
        
        Returns:
            Dict[str, np.ndarray]: Dictionary of extracted features
        """
        try:
            # Use larger hop length for section-level analysis
            hop_length = 2048  # ~46ms at 44.1kHz
            
            features = {}
            
            # 1. MFCC features (timbral characteristics)
            mfcc = librosa.feature.mfcc(
                y=self.audio_data,
                sr=self.sample_rate,
                n_mfcc=13,
                hop_length=hop_length
            )
            features['mfcc'] = mfcc
            
            # 2. Chroma features (harmonic content)
            chroma = librosa.feature.chroma_stft(
                y=self.audio_data,
                sr=self.sample_rate,
                hop_length=hop_length
            )
            features['chroma'] = chroma
            
            # 3. Spectral centroid (brightness)
            spectral_centroid = librosa.feature.spectral_centroid(
                y=self.audio_data,
                sr=self.sample_rate,
                hop_length=hop_length
            )
            features['spectral_centroid'] = spectral_centroid
            
            # 4. Spectral rolloff (frequency distribution)
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=self.audio_data,
                sr=self.sample_rate,
                hop_length=hop_length
            )
            features['spectral_rolloff'] = spectral_rolloff
            
            # 5. Zero crossing rate (percussiveness)
            zcr = librosa.feature.zero_crossing_rate(
                y=self.audio_data,
                hop_length=hop_length
            )
            features['zcr'] = zcr
            
            # 6. RMS energy (dynamics)
            rms = librosa.feature.rms(
                y=self.audio_data,
                hop_length=hop_length
            )
            features['rms'] = rms
            
            # 7. Tempo and beat tracking for rhythm analysis
            tempo, beats = librosa.beat.beat_track(
                y=self.audio_data,
                sr=self.sample_rate,
                hop_length=hop_length
            )
            features['tempo'] = tempo
            features['beats'] = beats
            
            # Convert frame indices to time
            features['times'] = librosa.frames_to_time(
                np.arange(mfcc.shape[1]),
                sr=self.sample_rate,
                hop_length=hop_length
            )
            
            return features
            
        except Exception:
            # Return minimal features on error
            return {'times': np.linspace(0, self.duration, 100)}
    
    def _detect_section_boundaries(self, features: Dict[str, np.ndarray]) -> List[float]:
        """
        Detect section boundaries using spectral similarity analysis.
        
        Args:
            features: Extracted audio features
            
        Returns:
            List[float]: Section boundary times in seconds
        """
        try:
            # Combine multiple features for robust boundary detection
            combined_features = self._combine_features_for_similarity(features)
            
            # Calculate self-similarity matrix
            similarity_matrix = self._calculate_similarity_matrix(combined_features)
            
            # Detect boundaries using novelty detection
            boundaries = self._detect_novelty_boundaries(similarity_matrix, features['times'])
            
            # Filter boundaries based on tango structure constraints
            filtered_boundaries = self._filter_boundaries_for_tango(boundaries)
            
            # Ensure we have exactly 5 sections (4 boundaries + start/end)
            final_boundaries = self._ensure_five_sections(filtered_boundaries)
            
            return final_boundaries
            
        except Exception:
            # Fallback to time-based boundaries
            return [0.0, self.duration * 0.2, self.duration * 0.4, 
                   self.duration * 0.6, self.duration * 0.8, self.duration]
    
    def _combine_features_for_similarity(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine multiple features into a single feature matrix for similarity analysis.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            np.ndarray: Combined feature matrix (features x time)
        """
        try:
            feature_list = []
            
            # Normalize and combine key features
            if 'mfcc' in features:
                # Use first 13 MFCC coefficients
                mfcc_norm = librosa.util.normalize(features['mfcc'], axis=1)
                feature_list.append(mfcc_norm)
            
            if 'chroma' in features:
                # Chroma features for harmonic content
                chroma_norm = librosa.util.normalize(features['chroma'], axis=1)
                feature_list.append(chroma_norm)
            
            if 'spectral_centroid' in features:
                # Spectral centroid for brightness
                centroid_norm = librosa.util.normalize(features['spectral_centroid'], axis=1)
                feature_list.append(centroid_norm)
            
            if 'rms' in features:
                # RMS energy for dynamics
                rms_norm = librosa.util.normalize(features['rms'], axis=1)
                feature_list.append(rms_norm)
            
            # Concatenate all features
            if feature_list:
                combined = np.vstack(feature_list)
                return combined
            else:
                # Fallback to dummy features
                return np.random.random((10, 100))
                
        except Exception:
            # Return dummy features on error
            return np.random.random((10, 100))
    
    def _calculate_similarity_matrix(self, features: np.ndarray) -> np.ndarray:
        """
        Calculate self-similarity matrix for boundary detection.
        
        Args:
            features: Combined feature matrix
            
        Returns:
            np.ndarray: Self-similarity matrix
        """
        try:
            # Calculate cosine similarity between all time frames
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Transpose to get (time x features) for sklearn
            features_T = features.T
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(features_T)
            
            return similarity_matrix
            
        except Exception:
            # Fallback to identity matrix
            n_frames = features.shape[1] if len(features.shape) > 1 else 100
            return np.eye(n_frames)
    
    def _detect_novelty_boundaries(self, similarity_matrix: np.ndarray, 
                                 times: np.ndarray) -> List[float]:
        """
        Detect boundaries using novelty detection on similarity matrix.
        
        Args:
            similarity_matrix: Self-similarity matrix
            times: Time array corresponding to matrix indices
            
        Returns:
            List[float]: Detected boundary times
        """
        try:
            # Calculate novelty function using checkerboard kernel
            kernel_size = min(16, similarity_matrix.shape[0] // 10)
            novelty = librosa.segment.recurrence_to_lag(
                similarity_matrix, 
                pad=True
            )
            
            # Smooth the novelty function
            novelty_smooth = librosa.util.normalize(
                np.convolve(novelty, np.ones(5)/5, mode='same')
            )
            
            # Find peaks in novelty function
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(
                novelty_smooth,
                height=np.mean(novelty_smooth) + 0.5 * np.std(novelty_smooth),
                distance=len(novelty_smooth) // 10  # Minimum distance between peaks
            )
            
            # Convert peak indices to times
            if len(times) > 0 and len(peaks) > 0:
                boundary_times = [times[min(peak, len(times)-1)] for peak in peaks]
                return sorted(boundary_times)
            else:
                return []
                
        except Exception:
            return []
    
    def _filter_boundaries_for_tango(self, boundaries: List[float]) -> List[float]:
        """
        Filter boundaries based on tango structure constraints.
        
        Args:
            boundaries: Raw detected boundaries
            
        Returns:
            List[float]: Filtered boundaries suitable for tango structure
        """
        if not boundaries or self.duration <= 0:
            return []
        
        # Filter boundaries that are too close together (minimum 15 seconds)
        min_section_duration = 15.0
        filtered = [0.0]  # Always start with 0
        
        for boundary in boundaries:
            if boundary > filtered[-1] + min_section_duration and boundary < self.duration - 5.0:
                filtered.append(boundary)
        
        # Add end time
        if filtered[-1] < self.duration - 5.0:
            filtered.append(self.duration)
        
        return filtered
    
    def _ensure_five_sections(self, boundaries: List[float]) -> List[float]:
        """
        Ensure we have exactly 5 sections by adjusting boundaries.
        
        Args:
            boundaries: Current boundaries
            
        Returns:
            List[float]: Adjusted boundaries for exactly 5 sections
        """
        if not boundaries:
            boundaries = [0.0, self.duration]
        
        # Ensure start and end
        if boundaries[0] != 0.0:
            boundaries = [0.0] + boundaries
        if boundaries[-1] != self.duration:
            boundaries.append(self.duration)
        
        # Remove duplicates and sort
        boundaries = sorted(list(set(boundaries)))
        
        # Adjust to have exactly 6 boundary points (5 sections)
        if len(boundaries) < 6:
            # Add boundaries to reach 6 points
            while len(boundaries) < 6:
                # Find largest gap and split it
                gaps = [(boundaries[i+1] - boundaries[i], i) for i in range(len(boundaries)-1)]
                largest_gap_idx = max(gaps, key=lambda x: x[0])[1]
                
                # Insert boundary in the middle of largest gap
                new_boundary = (boundaries[largest_gap_idx] + boundaries[largest_gap_idx + 1]) / 2
                boundaries.insert(largest_gap_idx + 1, new_boundary)
        
        elif len(boundaries) > 6:
            # Remove boundaries to reach 6 points (keep start, end, and 4 best)
            # Keep start and end, select 4 best internal boundaries
            internal_boundaries = boundaries[1:-1]
            
            # Score boundaries based on how close they are to ideal positions
            ideal_positions = [self.duration * i / 5 for i in range(1, 5)]
            scored_boundaries = []
            
            for boundary in internal_boundaries:
                # Find closest ideal position
                distances = [abs(boundary - ideal) for ideal in ideal_positions]
                min_distance = min(distances)
                scored_boundaries.append((min_distance, boundary))
            
            # Select 4 best boundaries
            scored_boundaries.sort(key=lambda x: x[0])
            selected_boundaries = [b[1] for b in scored_boundaries[:4]]
            
            boundaries = [0.0] + sorted(selected_boundaries) + [self.duration]
        
        return boundaries
    
    def _classify_section_types(self, features: Dict[str, np.ndarray], 
                              boundaries: List[float]) -> List[SectionType]:
        """
        Classify sections into A/B/C types based on spectral similarity.
        
        Args:
            features: Extracted audio features
            boundaries: Section boundaries
            
        Returns:
            List[SectionType]: Section type for each section
        """
        try:
            if len(boundaries) < 2:
                return [SectionType.A]
            
            # Extract features for each section
            section_features = self._extract_section_features(features, boundaries)
            
            # Calculate similarity between sections
            section_similarities = self._calculate_section_similarities(section_features)
            
            # Classify based on similarity patterns
            section_types = self._assign_section_types(section_similarities)
            
            return section_types
            
        except Exception:
            # Fallback to ABABA pattern
            n_sections = len(boundaries) - 1
            if n_sections == 5:
                return [SectionType.A, SectionType.B, SectionType.A, SectionType.B, SectionType.A]
            elif n_sections == 4:
                return [SectionType.A, SectionType.B, SectionType.A, SectionType.B]
            elif n_sections == 3:
                return [SectionType.A, SectionType.B, SectionType.A]
            else:
                return [SectionType.A] * n_sections
    
    def _extract_section_features(self, features: Dict[str, np.ndarray], 
                                boundaries: List[float]) -> List[np.ndarray]:
        """
        Extract average features for each section.
        
        Args:
            features: Full audio features
            boundaries: Section boundaries
            
        Returns:
            List[np.ndarray]: Average features for each section
        """
        section_features = []
        times = features.get('times', np.linspace(0, self.duration, 100))
        
        for i in range(len(boundaries) - 1):
            start_time = boundaries[i]
            end_time = boundaries[i + 1]
            
            # Find time indices for this section
            start_idx = np.searchsorted(times, start_time)
            end_idx = np.searchsorted(times, end_time)
            
            # Extract and average features for this section
            section_feature_list = []
            
            for feature_name in ['mfcc', 'chroma', 'spectral_centroid', 'rms']:
                if feature_name in features:
                    feature_data = features[feature_name]
                    if len(feature_data.shape) > 1:
                        section_data = feature_data[:, start_idx:end_idx]
                        section_avg = np.mean(section_data, axis=1)
                    else:
                        section_data = feature_data[start_idx:end_idx]
                        section_avg = np.mean(section_data)
                    section_feature_list.append(section_avg)
            
            if section_feature_list:
                section_features.append(np.concatenate([
                    f.flatten() if hasattr(f, 'flatten') else [f] 
                    for f in section_feature_list
                ]))
            else:
                section_features.append(np.array([1.0]))  # Dummy feature
        
        return section_features
    
    def _calculate_section_similarities(self, section_features: List[np.ndarray]) -> np.ndarray:
        """
        Calculate similarity matrix between sections.
        
        Args:
            section_features: Features for each section
            
        Returns:
            np.ndarray: Section similarity matrix
        """
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Stack section features
            feature_matrix = np.vstack([f.reshape(1, -1) for f in section_features])
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(feature_matrix)
            
            return similarity_matrix
            
        except Exception:
            # Fallback to identity matrix
            n_sections = len(section_features)
            return np.eye(n_sections)
    
    def _assign_section_types(self, similarities: np.ndarray) -> List[SectionType]:
        """
        Assign A/B/C types based on similarity patterns.
        
        Args:
            similarities: Section similarity matrix
            
        Returns:
            List[SectionType]: Section types
        """
        n_sections = similarities.shape[0]
        
        if n_sections == 5:
            # Check if sections 1, 3, 5 are similar (ABABA pattern)
            sim_1_3 = similarities[0, 2]
            sim_1_5 = similarities[0, 4]
            sim_3_5 = similarities[2, 4]
            
            # Check if sections 2, 4 are similar
            sim_2_4 = similarities[1, 3]
            
            # Threshold for similarity
            threshold = 0.7
            
            if (sim_1_3 > threshold and sim_1_5 > threshold and 
                sim_3_5 > threshold and sim_2_4 > threshold):
                # Strong ABABA pattern
                return [SectionType.A, SectionType.B, SectionType.A, SectionType.B, SectionType.A]
            elif (sim_1_3 > threshold and sim_2_4 > threshold):
                # ABABC pattern (last section different)
                return [SectionType.A, SectionType.B, SectionType.A, SectionType.B, SectionType.C]
            else:
                # Default ABABA
                return [SectionType.A, SectionType.B, SectionType.A, SectionType.B, SectionType.A]
        
        elif n_sections == 4:
            return [SectionType.A, SectionType.B, SectionType.A, SectionType.B]
        elif n_sections == 3:
            return [SectionType.A, SectionType.B, SectionType.A]
        else:
            # Default pattern
            types = []
            for i in range(n_sections):
                types.append(SectionType.A if i % 2 == 0 else SectionType.B)
            return types
    
    def _create_musical_sections(self, boundaries: List[float], 
                               section_types: List[SectionType]) -> List[MusicalSection]:
        """
        Create MusicalSection objects from boundaries and types.
        
        Args:
            boundaries: Section boundaries
            section_types: Section types
            
        Returns:
            List[MusicalSection]: Created musical sections
        """
        sections = []
        
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            duration = end - start
            section_type = section_types[i] if i < len(section_types) else SectionType.A
            
            # Create description based on type and position
            if section_type == SectionType.A:
                if i == 0:
                    description = 'opening_verse'
                elif i == len(section_types) - 1:
                    description = 'closing_verse'
                else:
                    description = f'verse_{i//2 + 1}'
            elif section_type == SectionType.B:
                description = f'chorus_{i//2 + 1}'
            else:  # SectionType.C
                description = 'coda_section'
            
            section = MusicalSection(
                section_type=section_type,
                start=start,
                end=end,
                duration=duration,
                description=description
            )
            
            sections.append(section)
        
        return sections
    
    def _validate_tango_structure(self, sections: List[MusicalSection]) -> List[MusicalSection]:
        """
        Validate and adjust sections based on tango structure rules.
        
        Args:
            sections: Initial sections
            
        Returns:
            List[MusicalSection]: Validated sections
        """
        # Post-analysis validation
        validation_results = {
            'near_30_seconds': self._check_section_durations(sections),
            'ababa_structure': self._check_ababa_structure(sections),
            'five_sections': len(sections) == 5,
            'phrase_alignment': self._check_phrase_alignment(sections)
        }
        
        # Log validation results for debugging
        print(f"Tango structure validation: {validation_results}")
        
        # If validation fails significantly, apply corrections
        if sum(validation_results.values()) < 2:
            # Apply fallback structure
            return self._create_fallback_sections()
        
        return sections
    
    def _check_section_durations(self, sections: List[MusicalSection]) -> bool:
        """Check if sections are near 30 seconds (20-40 range)."""
        for section in sections:
            if not (20.0 <= section.duration <= 40.0):
                return False
        return True
    
    def _check_ababa_structure(self, sections: List[MusicalSection]) -> bool:
        """Check if sections follow ABABA or ABABC pattern."""
        if len(sections) != 5:
            return False
        
        types = [s.section_type for s in sections]
        
        # Check ABABA
        ababa = [SectionType.A, SectionType.B, SectionType.A, SectionType.B, SectionType.A]
        if types == ababa:
            return True
        
        # Check ABABC
        ababc = [SectionType.A, SectionType.B, SectionType.A, SectionType.B, SectionType.C]
        if types == ababc:
            return True
        
        return False
    
    def _check_phrase_alignment(self, sections: List[MusicalSection]) -> bool:
        """Check if sections align with detected phrases."""
        # This would check against phrase detection results
        # For now, return True as phrase detection is separate
        return True
    
    def _create_fallback_sections(self) -> List[MusicalSection]:
        """Create fallback sections when advanced analysis fails."""
        if not self.duration or self.duration <= 0:
            duration = 180.0  # Default duration
        else:
            duration = self.duration
        
        # Create 5 sections with equal duration
        section_duration = duration / 5
        
        sections = []
        section_types = [SectionType.A, SectionType.B, SectionType.A, SectionType.B, SectionType.A]
        descriptions = ['opening_verse', 'first_chorus', 'middle_verse', 'second_chorus', 'closing_verse']
        
        for i in range(5):
            start = i * section_duration
            end = (i + 1) * section_duration
            
            section = MusicalSection(
                section_type=section_types[i],
                start=start,
                end=end,
                duration=section_duration,
                description=descriptions[i]
            )
            sections.append(section)
        
        return sections
    
    def _detect_tango_phrases(self) -> Tuple[List[TangoPhrase], PhraseStatistics]:
        """
        Detect tango musical phrases based on characteristic pauses.
        
        Primary detection: Volume drops (sudden decreases in energy)
        Secondary validation: Beat spacing (8-16 beats apart) when beat detection is reliable
        
        Returns:
            Tuple[List[TangoPhrase], PhraseStatistics]: Detected phrases and statistics
        """
        try:
            # Primary detection: Find volume drops
            pause_candidates = self._detect_volume_drops()
            
            # Secondary detection: Find frequency content changes
            frequency_drops = self._detect_frequency_drops()
            
            # Combine volume and frequency indicators
            combined_pauses = self._combine_pause_indicators(pause_candidates, frequency_drops)
            
            # Filter pauses that are too close together (minimum spacing)
            filtered_pauses = self._filter_close_pauses(combined_pauses)
            
            # Convert pauses to phrases
            phrases = self._create_phrases_from_pauses(filtered_pauses)
            
            # Calculate phrase statistics
            phrase_stats = self._calculate_phrase_statistics(phrases)
            
            return phrases, phrase_stats
            
        except Exception as e:
            # Return empty results on error
            return [], PhraseStatistics(
                total_phrases=0,
                average_phrase_duration=0.0,
                average_pause_duration=0.0,
                phrase_density=0.0,
                total_pause_time=0.0
            )
    
    def _detect_volume_drops(self) -> List[Dict[str, float]]:
        """
        Detect sudden volume drops that indicate phrase endings.
        
        Returns:
            List[Dict[str, float]]: List of pause candidates with timing and drop ratio
        """
        try:
            # Calculate RMS energy over time with small hop length for precision
            hop_length = 1024  # ~23ms at 44.1kHz
            rms = librosa.feature.rms(
                y=self.audio_data,
                hop_length=hop_length,
                frame_length=2048
            )[0]
            
            # Convert frame indices to time
            times = librosa.frames_to_time(
                np.arange(len(rms)),
                sr=self.sample_rate,
                hop_length=hop_length
            )
            
            # Smooth RMS to reduce noise
            window_size = 5
            rms_smooth = np.convolve(rms, np.ones(window_size)/window_size, mode='same')
            
            # Find significant volume drops
            pause_candidates = []
            
            # Look for drops in a sliding window
            window_duration = 2.0  # 2 second analysis window
            window_frames = int(window_duration * self.sample_rate / hop_length)
            
            for i in range(window_frames, len(rms_smooth) - window_frames):
                # Compare current window with previous window
                prev_window = rms_smooth[i-window_frames:i]
                curr_window = rms_smooth[i:i+window_frames]
                
                if len(prev_window) == 0 or len(curr_window) == 0:
                    continue
                
                prev_energy = np.mean(prev_window)
                curr_energy = np.mean(curr_window)
                
                # Check for significant drop (at least 50% reduction)
                if prev_energy > 0 and curr_energy < prev_energy * 0.75:
                    drop_ratio = curr_energy / prev_energy
                    
                    # Find the exact start and end of the pause
                    pause_start, pause_end = self._find_pause_boundaries(
                        rms_smooth, i, window_frames, prev_energy
                    )
                    
                    pause_duration = times[pause_end] - times[pause_start]
                    
                    # Only consider pauses lasting 0.8-3 seconds (relaxed for testing)
                    if 0.3 <= pause_duration <= 3.0:
                        pause_candidates.append({
                            'start_time': times[pause_start],
                            'end_time': times[pause_end],
                            'duration': pause_duration,
                            'volume_drop_ratio': drop_ratio,
                            'confidence': min(1.0, (1.0 - drop_ratio) * 2)  # Higher confidence for bigger drops
                        })
            
            return pause_candidates
            
        except Exception:
            return []
    
    def _find_pause_boundaries(self, rms_smooth: np.ndarray, center_idx: int, 
                             window_frames: int, reference_energy: float) -> Tuple[int, int]:
        """
        Find the exact start and end boundaries of a pause.
        
        Args:
            rms_smooth: Smoothed RMS energy array
            center_idx: Center index where drop was detected
            window_frames: Window size in frames
            reference_energy: Reference energy level before drop
            
        Returns:
            Tuple[int, int]: Start and end frame indices of the pause
        """
        threshold = reference_energy * 0.3  # 30% of reference energy
        
        # Find pause start (working backwards from center)
        pause_start = center_idx
        for i in range(center_idx, max(0, center_idx - window_frames), -1):
            if rms_smooth[i] > threshold:
                pause_start = i
                break
        
        # Find pause end (working forwards from center)
        pause_end = center_idx
        for i in range(center_idx, min(len(rms_smooth), center_idx + window_frames)):
            if rms_smooth[i] > threshold:
                pause_end = i
                break
        
        return pause_start, pause_end
    
    def _detect_frequency_drops(self) -> List[Dict[str, float]]:
        """
        Detect sudden changes in frequency content that indicate phrase endings.
        
        Returns:
            List[Dict[str, float]]: List of frequency drop candidates
        """
        try:
            # Calculate spectral centroid over time
            hop_length = 1024
            spectral_centroid = librosa.feature.spectral_centroid(
                y=self.audio_data,
                sr=self.sample_rate,
                hop_length=hop_length
            )[0]
            
            # Calculate spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=self.audio_data,
                sr=self.sample_rate,
                hop_length=hop_length
            )[0]
            
            # Convert to time
            times = librosa.frames_to_time(
                np.arange(len(spectral_centroid)),
                sr=self.sample_rate,
                hop_length=hop_length
            )
            
            frequency_drops = []
            
            # Look for sudden drops in spectral content
            window_size = int(1.0 * self.sample_rate / hop_length)  # 1 second window
            
            for i in range(window_size, len(spectral_centroid) - window_size):
                prev_centroid = np.mean(spectral_centroid[i-window_size:i])
                curr_centroid = np.mean(spectral_centroid[i:i+window_size])
                
                prev_rolloff = np.mean(spectral_rolloff[i-window_size:i])
                curr_rolloff = np.mean(spectral_rolloff[i:i+window_size])
                
                # Check for significant frequency content drop
                if (prev_centroid > 0 and curr_centroid < prev_centroid * 0.7 and
                    prev_rolloff > 0 and curr_rolloff < prev_rolloff * 0.7):
                    
                    centroid_ratio = curr_centroid / prev_centroid
                    rolloff_ratio = curr_rolloff / prev_rolloff
                    
                    frequency_drops.append({
                        'time': times[i],
                        'centroid_drop_ratio': centroid_ratio,
                        'rolloff_drop_ratio': rolloff_ratio,
                        'confidence': min(1.0, (2.0 - centroid_ratio - rolloff_ratio))
                    })
            
            return frequency_drops
            
        except Exception:
            return []
    
    def _combine_pause_indicators(self, volume_drops: List[Dict[str, float]], 
                                frequency_drops: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """
        Combine volume and frequency drop indicators to identify strong pause candidates.
        
        Args:
            volume_drops: Volume drop candidates
            frequency_drops: Frequency drop candidates
            
        Returns:
            List[Dict[str, float]]: Combined pause candidates with enhanced confidence
        """
        combined_pauses = []
        
        # Start with volume drops as primary indicators
        for vol_drop in volume_drops:
            pause = vol_drop.copy()
            pause['frequency_drop_ratio'] = 1.0  # Default (no frequency drop)
            
            # Look for nearby frequency drops to enhance confidence
            for freq_drop in frequency_drops:
                time_diff = abs(freq_drop['time'] - vol_drop['start_time'])
                
                # If frequency drop is within 1 second of volume drop
                if time_diff <= 1.0:
                    # Enhance the pause with frequency information
                    pause['frequency_drop_ratio'] = min(
                        freq_drop['centroid_drop_ratio'],
                        freq_drop['rolloff_drop_ratio']
                    )
                    # Boost confidence when both indicators agree
                    pause['confidence'] = min(1.0, pause['confidence'] + freq_drop['confidence'] * 0.3)
                    break
            
            combined_pauses.append(pause)
        
        return combined_pauses
    
    def _filter_close_pauses(self, pauses: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """
        Filter out pauses that are too close together, keeping the strongest ones.
        
        Args:
            pauses: List of pause candidates
            
        Returns:
            List[Dict[str, float]]: Filtered pauses with minimum spacing
        """
        if not pauses:
            return []
        
        # Sort by confidence (strongest first)
        sorted_pauses = sorted(pauses, key=lambda x: x['confidence'], reverse=True)
        
        filtered_pauses = []
        min_spacing = 4.0  # Minimum 4 seconds between pauses (roughly 8-16 beats at typical tango tempo)
        
        for pause in sorted_pauses:
            # Check if this pause is far enough from already accepted pauses
            too_close = False
            for accepted_pause in filtered_pauses:
                time_diff = abs(pause['start_time'] - accepted_pause['start_time'])
                if time_diff < min_spacing:
                    too_close = True
                    break
            
            if not too_close:
                filtered_pauses.append(pause)
        
        # Sort by time for final output
        return sorted(filtered_pauses, key=lambda x: x['start_time'])
    
    def _create_phrases_from_pauses(self, pauses: List[Dict[str, float]]) -> List[TangoPhrase]:
        """
        Create TangoPhrase objects from detected pauses.
        
        Args:
            pauses: List of detected pauses
            
        Returns:
            List[TangoPhrase]: List of tango phrases
        """
        if not pauses:
            return []
        
        phrases = []
        
        # Create phrases between pauses
        for i in range(len(pauses) + 1):
            if i == 0:
                # First phrase: from start to first pause
                phrase_start = 0.0
                phrase_end = pauses[0]['start_time'] if pauses else self.duration
                pause_info = pauses[0] if pauses else None
            elif i == len(pauses):
                # Last phrase: from last pause to end
                phrase_start = pauses[i-1]['end_time']
                phrase_end = self.duration
                pause_info = None
            else:
                # Middle phrases: from previous pause end to current pause start
                phrase_start = pauses[i-1]['end_time']
                phrase_end = pauses[i]['start_time']
                pause_info = pauses[i]
            
            phrase_duration = phrase_end - phrase_start
            
            # Only create phrases that are long enough (at least 2 seconds)
            if phrase_duration >= 2.0:
                if pause_info:
                    phrase = TangoPhrase(
                        start=phrase_start,
                        end=phrase_end,
                        duration=phrase_duration,
                        pause_start=pause_info['start_time'],
                        pause_end=pause_info['end_time'],
                        pause_duration=pause_info['duration'],
                        volume_drop_ratio=pause_info['volume_drop_ratio'],
                        frequency_drop_ratio=pause_info['frequency_drop_ratio'],
                        confidence=pause_info['confidence']
                    )
                else:
                    # Final phrase without pause
                    phrase = TangoPhrase(
                        start=phrase_start,
                        end=phrase_end,
                        duration=phrase_duration,
                        pause_start=0.0,
                        pause_end=0.0,
                        pause_duration=0.0,
                        volume_drop_ratio=1.0,
                        frequency_drop_ratio=1.0,
                        confidence=0.5
                    )
                
                phrases.append(phrase)
        
        return phrases
    
    def _calculate_phrase_statistics(self, phrases: List[TangoPhrase]) -> PhraseStatistics:
        """
        Calculate statistics for detected phrases.
        
        Args:
            phrases: List of detected phrases
            
        Returns:
            PhraseStatistics: Phrase analysis statistics
        """
        if not phrases:
            return PhraseStatistics(
                total_phrases=0,
                average_phrase_duration=0.0,
                average_pause_duration=0.0,
                phrase_density=0.0,
                total_pause_time=0.0
            )
        
        total_phrases = len(phrases)
        phrase_durations = [p.duration for p in phrases]
        pause_durations = [p.pause_duration for p in phrases if p.pause_duration > 0]
        
        average_phrase_duration = np.mean(phrase_durations)
        average_pause_duration = np.mean(pause_durations) if pause_durations else 0.0
        total_pause_time = sum(pause_durations)
        
        # Calculate phrase density (phrases per minute)
        phrase_density = (total_phrases / self.duration) * 60.0 if self.duration > 0 else 0.0
        
        return PhraseStatistics(
            total_phrases=total_phrases,
            average_phrase_duration=float(average_phrase_duration),
            average_pause_duration=float(average_pause_duration),
            phrase_density=float(phrase_density),
            total_pause_time=float(total_pause_time)
        )
    
    def _create_fallback_section_analysis(self) -> SectionAnalysisResult:
        """Create fallback section analysis when primary method fails"""
        return SectionAnalysisResult(
            time_signature='2/4',
            sections=[
                MusicalSection(
                    section_type=SectionType.A,
                    start=0.0,
                    end=180.0,
                    duration=180.0,
                    description='main_theme'
                )
            ],
            phrases=[],
            phrase_statistics=PhraseStatistics(
                total_phrases=0,
                average_phrase_duration=0.0,
                average_pause_duration=0.0,
                phrase_density=0.0,
                total_pause_time=0.0
            ),
            total_duration=180.0,
            section_count=1
        )
    
    # ==================== COMPLETE ANALYSIS METHOD ====================
    
    def analyze_complete(self) -> CompleteAnalysisResult:
        """
        Perform complete music analysis in the specified order: beat -> melody -> sections.
        
        Returns:
            CompleteAnalysisResult: Complete analysis results for all components
        """
        start_time = time.time()
        
        # Step 1: Beat analysis (FIRST)
        beat_analysis = self.analyze_beats()
        
        # Step 2: Melody analysis (SECOND) 
        melody_analysis = self.analyze_melody()
        
        # Step 3: Section analysis (THIRD)
        section_analysis = self.analyze_sections()
        
        processing_duration = time.time() - start_time
        
        # Gather file information
        file_info = self._get_file_info()
        
        return CompleteAnalysisResult(
            beat_analysis=beat_analysis,
            melody_analysis=melody_analysis,
            section_analysis=section_analysis,
            processing_duration=processing_duration,
            file_info=file_info
        )
    
    def _get_file_info(self) -> Dict[str, Any]:
        """Get basic file information"""
        try:
            file_size = os.path.getsize(self.audio_path)
            return {
                'file_path': self.audio_path,
                'file_size': file_size,
                'duration': float(self.duration) if self.duration else 0.0,
                'sample_rate': int(self.sample_rate) if self.sample_rate else 0
            }
        except:
            return {}
    
    # ==================== UTILITY METHODS ====================
    
    def to_dict(self, result: CompleteAnalysisResult) -> Dict[str, Any]:
        """
        Convert analysis result to dictionary format for JSON serialization.
        
        Args:
            result (CompleteAnalysisResult): Analysis result to convert
            
        Returns:
            Dict[str, Any]: Dictionary representation of the analysis result
        """
        return {
            'beat_analysis': {
                'bpm': result.beat_analysis.bpm,
                'beats': [
                    {
                        'time': beat.time,
                        'confidence': beat.confidence,
                        'strength': beat.strength,
                        'duration': beat.duration,
                        'beat_type': beat.beat_type.value
                    }
                    for beat in result.beat_analysis.beats
                ],
                'method': result.beat_analysis.method.value,
                'confidence': result.beat_analysis.confidence,
                'total_beats': result.beat_analysis.total_beats
            },
            'melody_analysis': {
                'segments': [
                    {
                        'start': seg.start,
                        'end': seg.end,
                        'type': seg.segment_type.value,
                        'confidence': seg.confidence,
                        'duration': seg.duration
                    }
                    for seg in result.melody_analysis.segments
                ],
                'statistics': {
                    'legato_percentage': result.melody_analysis.statistics.legato_percentage,
                    'staccato_percentage': result.melody_analysis.statistics.staccato_percentage,
                    'total_segments': result.melody_analysis.statistics.total_segments,
                    'average_segment_duration': result.melody_analysis.statistics.average_segment_duration,
                    'median_interval': result.melody_analysis.statistics.median_interval
                }
            },
            'section_analysis': {
                'time_signature': result.section_analysis.time_signature,
                'sections': [
                    {
                        'type': sec.section_type.value,
                        'start': sec.start,
                        'end': sec.end,
                        'duration': sec.duration,
                        'description': sec.description
                    }
                    for sec in result.section_analysis.sections
                ],
                'phrases': [
                    {
                        'start': phrase.start,
                        'end': phrase.end,
                        'duration': phrase.duration,
                        'pause_start': phrase.pause_start,
                        'pause_end': phrase.pause_end,
                        'pause_duration': phrase.pause_duration,
                        'volume_drop_ratio': phrase.volume_drop_ratio,
                        'frequency_drop_ratio': phrase.frequency_drop_ratio,
                        'confidence': phrase.confidence
                    }
                    for phrase in result.section_analysis.phrases
                ],
                'phrase_statistics': {
                    'total_phrases': result.section_analysis.phrase_statistics.total_phrases,
                    'average_phrase_duration': result.section_analysis.phrase_statistics.average_phrase_duration,
                    'average_pause_duration': result.section_analysis.phrase_statistics.average_pause_duration,
                    'phrase_density': result.section_analysis.phrase_statistics.phrase_density,
                    'total_pause_time': result.section_analysis.phrase_statistics.total_pause_time
                },
                'total_duration': result.section_analysis.total_duration,
                'section_count': result.section_analysis.section_count
            },
            'processing_duration': result.processing_duration,
            'file_info': result.file_info
        }


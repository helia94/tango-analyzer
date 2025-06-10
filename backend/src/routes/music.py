import os
import uuid
import tempfile
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import librosa
import numpy as np
from datetime import datetime
import json

# Try to import essentia, fall back to librosa only if not available
try:
    import essentia.standard as es
    ESSENTIA_AVAILABLE = True
    print("Essentia loaded successfully")
except ImportError as e:
    ESSENTIA_AVAILABLE = False
    print(f"Warning: Essentia not available ({e}), using librosa only")
except Exception as e:
    ESSENTIA_AVAILABLE = False
    print(f"Warning: Essentia failed to load ({e}), using librosa only")

music_bp = Blueprint('music', __name__)

# Configuration
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'flac', 'm4a', 'aac', 'ogg'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_beats_librosa(audio_path):
    """Analyze beats using librosa"""
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # Beat tracking
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
        
        # Get onset strength for confidence estimation
        onset_envelope = librosa.onset.onset_strength(y=y, sr=sr)
        beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='frames')[1]
        
        # Calculate confidence scores (simplified)
        confidences = []
        for beat_frame in beat_frames:
            if beat_frame < len(onset_envelope):
                confidences.append(float(onset_envelope[beat_frame]))
            else:
                confidences.append(0.5)
        
        # Normalize confidences
        if confidences:
            max_conf = max(confidences)
            confidences = [c / max_conf for c in confidences]
        
        return {
            'bpm': float(tempo),
            'beats': [{'time': float(beat), 'confidence': conf, 'strength': conf} 
                     for beat, conf in zip(beats, confidences)],
            'method': 'librosa'
        }
    except Exception as e:
        raise Exception(f"Librosa beat analysis failed: {str(e)}")

def analyze_beats_essentia(audio_path):
    """Analyze beats using Essentia"""
    try:
        # Load audio
        loader = es.MonoLoader(filename=audio_path)
        audio = loader()
        
        # Beat tracking with RhythmExtractor2013
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)
        
        # Format results
        beat_list = []
        for i, beat_time in enumerate(beats):
            confidence = float(beats_confidence) if i == 0 else float(beats_confidence)
            beat_list.append({
                'time': float(beat_time),
                'confidence': confidence,
                'strength': confidence
            })
        
        return {
            'bpm': float(bpm),
            'beats': beat_list,
            'method': 'essentia'
        }
    except Exception as e:
        raise Exception(f"Essentia beat analysis failed: {str(e)}")

def analyze_melody_articulation(audio_path):
    """Analyze melody for legato/staccato segments"""
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # Onset detection
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        
        if len(onsets) < 2:
            return {
                'segments': [],
                'statistics': {
                    'legato_percentage': 0,
                    'staccato_percentage': 0,
                    'total_segments': 0
                }
            }
        
        # Calculate inter-onset intervals
        intervals = np.diff(onsets)
        
        # Simple rule-based classification
        # Shorter intervals with sharp attacks = staccato
        # Longer intervals with smooth connections = legato
        segments = []
        legato_count = 0
        staccato_count = 0
        
        for i in range(len(intervals)):
            start_time = onsets[i]
            end_time = onsets[i + 1]
            interval_duration = intervals[i]
            
            # Simple heuristic: shorter intervals tend to be staccato
            if interval_duration < 0.3:  # Less than 300ms
                segment_type = 'staccato'
                staccato_count += 1
                confidence = 0.7
            else:
                segment_type = 'legato'
                legato_count += 1
                confidence = 0.6
            
            segments.append({
                'start': float(start_time),
                'end': float(end_time),
                'type': segment_type,
                'confidence': confidence,
                'characteristics': [f'{segment_type}_style', f'duration_{interval_duration:.2f}s']
            })
        
        total_segments = len(segments)
        legato_percentage = (legato_count / total_segments * 100) if total_segments > 0 else 0
        staccato_percentage = (staccato_count / total_segments * 100) if total_segments > 0 else 0
        
        return {
            'segments': segments,
            'statistics': {
                'legato_percentage': legato_percentage,
                'staccato_percentage': staccato_percentage,
                'total_segments': total_segments
            }
        }
    except Exception as e:
        raise Exception(f"Melody analysis failed: {str(e)}")

def analyze_tango_structure(audio_path, beats_data):
    """Analyze tango-specific structure"""
    try:
        # Load audio for additional analysis
        y, sr = librosa.load(audio_path, sr=None)
        duration = len(y) / sr
        
        # Simple section detection based on beat patterns and audio features
        # This is a simplified implementation - could be enhanced with more sophisticated analysis
        
        # Estimate sections based on typical tango structure
        section_duration = duration / 5  # Assume 5 sections (ABABA or ABABC)
        
        sections = []
        section_types = ['A', 'B', 'A', 'B', 'C']
        
        for i, section_type in enumerate(section_types):
            start_time = i * section_duration
            end_time = min((i + 1) * section_duration, duration)
            
            if start_time < duration:
                description = 'verse_instrumental' if section_type == 'A' and i == 0 else \
                             'verse_vocal' if section_type == 'A' else \
                             'chorus' if section_type == 'B' else \
                             'instrumental_solo'
                
                sections.append({
                    'type': section_type,
                    'start': float(start_time),
                    'end': float(end_time),
                    'description': description
                })
        
        # Generate phrase boundaries (4 phrases per section, 4 measures per phrase)
        phrases = []
        beats = beats_data.get('beats', [])
        
        if beats:
            # Estimate measures based on beats (assuming 4/4 or 2/4 time)
            beats_per_measure = 4  # Can be adjusted based on time signature detection
            
            for section in sections:
                section_beats = [b for b in beats if section['start'] <= b['time'] <= section['end']]
                if len(section_beats) >= beats_per_measure:
                    beats_per_phrase = beats_per_measure * 4  # 4 measures per phrase
                    
                    for phrase_num in range(4):  # 4 phrases per section
                        start_beat_idx = phrase_num * beats_per_phrase
                        end_beat_idx = min((phrase_num + 1) * beats_per_phrase, len(section_beats))
                        
                        if start_beat_idx < len(section_beats):
                            phrase_start = section_beats[start_beat_idx]['time']
                            phrase_end = section_beats[end_beat_idx - 1]['time'] if end_beat_idx > start_beat_idx else phrase_start + 8
                            
                            phrases.append({
                                'section': section['type'],
                                'phrase': phrase_num + 1,
                                'start': float(phrase_start),
                                'end': float(phrase_end)
                            })
        
        return {
            'sections': sections,
            'phrases': phrases,
            'time_signature': '2/4',  # Typical for tango
            'key_signature': 'unknown'  # Would need more sophisticated analysis
        }
    except Exception as e:
        raise Exception(f"Tango structure analysis failed: {str(e)}")

@music_bp.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'File type not allowed. Supported: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        
        # Generate unique filename
        upload_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        stored_filename = f"{upload_id}.{file_extension}"
        file_path = os.path.join(UPLOAD_FOLDER, stored_filename)
        
        # Save file
        file.save(file_path)
        
        # Get file info
        file_size = os.path.getsize(file_path)
        if file_size > MAX_FILE_SIZE:
            os.remove(file_path)
            return jsonify({'error': f'File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB'}), 400
        
        return jsonify({
            'upload_id': upload_id,
            'filename': filename,
            'size': file_size,
            'message': 'File uploaded successfully'
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@music_bp.route('/analyze/<upload_id>', methods=['POST'])
def analyze_music(upload_id):
    """Analyze uploaded music file"""
    try:
        # Find the uploaded file
        file_path = None
        for ext in ALLOWED_EXTENSIONS:
            potential_path = os.path.join(UPLOAD_FOLDER, f"{upload_id}.{ext}")
            if os.path.exists(potential_path):
                file_path = potential_path
                break
        
        if not file_path:
            return jsonify({'error': 'File not found'}), 404
        
        # Get file info
        y, sr = librosa.load(file_path, sr=None)
        duration = len(y) / sr
        
        file_info = {
            'filename': f"{upload_id}.{file_path.split('.')[-1]}",
            'duration': float(duration),
            'sample_rate': int(sr),
            'channels': 1  # librosa loads as mono by default
        }
        
        # Perform beat analysis
        print("Starting beat analysis...")
        if ESSENTIA_AVAILABLE:
            try:
                beat_analysis = analyze_beats_essentia(file_path)
            except Exception as e:
                print(f"Essentia failed, falling back to librosa: {e}")
                beat_analysis = analyze_beats_librosa(file_path)
        else:
            beat_analysis = analyze_beats_librosa(file_path)
        
        # Perform melody analysis
        print("Starting melody analysis...")
        melody_analysis = analyze_melody_articulation(file_path)
        
        # Perform tango structure analysis
        print("Starting tango structure analysis...")
        tango_structure = analyze_tango_structure(file_path, beat_analysis)
        
        # Generate analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Compile results
        results = {
            'analysis_id': analysis_id,
            'timestamp': datetime.now().isoformat(),
            'file_info': file_info,
            'beat_analysis': beat_analysis,
            'melody_analysis': melody_analysis,
            'tango_structure': tango_structure
        }
        
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except:
            pass  # File cleanup is not critical
        
        return jsonify(results), 200
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@music_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'essentia_available': ESSENTIA_AVAILABLE,
        'supported_formats': list(ALLOWED_EXTENSIONS)
    }), 200


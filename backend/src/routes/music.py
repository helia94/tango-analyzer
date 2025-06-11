import os
import uuid
import time
import json
import tempfile
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from flask_cors import cross_origin
import librosa
import numpy as np
from src.models.user import db
from src.models.music import MusicUpload, AnalysisResult

music_bp = Blueprint('music', __name__)

# Configuration
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'flac', 'm4a', 'aac', 'ogg'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


def get_models():
    from src.models.user import db
    from src.models.music import MusicUpload, AnalysisResult
    return db, MusicUpload, AnalysisResult

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_beats(audio_path):
    """Analyze beats using librosa"""
    try:
        # Load audio file
        y, sr = librosa.load(audio_path)
        
        # Beat tracking
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
        
        # Calculate beat confidence (simplified)
        onset_envelope = librosa.onset.onset_strength(y=y, sr=sr)
        beat_frames = librosa.time_to_frames(beats, sr=sr)
        
        # Get confidence scores for each beat
        confidences = []
        for frame in beat_frames:
            if frame < len(onset_envelope):
                confidences.append(float(onset_envelope[frame]))
            else:
                confidences.append(0.5)
        
        # Normalize confidences
        if confidences:
            max_conf = max(confidences)
            confidences = [c / max_conf for c in confidences]
        
        # Create beat data
        beats_data = []
        for i, (beat_time, confidence) in enumerate(zip(beats, confidences)):
            beats_data.append({
                'time': float(beat_time),
                'confidence': float(confidence),
                'strength': float(confidence)  # For compatibility
            })
        
        return {
            'bpm': float(tempo),
            'beats': beats_data,
            'method': 'librosa',
            'confidence': float(np.mean(confidences)) if confidences else 0.0
        }
        
    except Exception as e:
        current_app.logger.error(f"Beat analysis error: {str(e)}")
        return {
            'bpm': 120.0,  # Default tango BPM
            'beats': [],
            'method': 'fallback',
            'confidence': 0.0
        }

def analyze_melody(audio_path):
    """Analyze melody for legato/staccato classification"""
    try:
        # Load audio file
        y, sr = librosa.load(audio_path)
        
        # Onset detection
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        
        # Calculate onset intervals
        if len(onsets) < 2:
            return {
                'segments': [],
                'statistics': {
                    'legato_percentage': 50.0,
                    'staccato_percentage': 50.0,
                    'total_segments': 0
                }
            }
        
        intervals = np.diff(onsets)
        
        # Classify segments based on onset intervals
        # Shorter intervals = staccato, longer intervals = legato
        median_interval = np.median(intervals)
        
        segments = []
        legato_count = 0
        staccato_count = 0
        
        for i in range(len(onsets) - 1):
            start_time = onsets[i]
            end_time = onsets[i + 1]
            interval = intervals[i]
            
            # Classification logic
            if interval > median_interval:
                segment_type = 'legato'
                legato_count += 1
            else:
                segment_type = 'staccato'
                staccato_count += 1
            
            # Confidence based on how far from median
            confidence = min(1.0, abs(interval - median_interval) / median_interval + 0.5)
            
            segments.append({
                'start': float(start_time),
                'end': float(end_time),
                'type': segment_type,
                'confidence': float(confidence)
            })
        
        total_segments = len(segments)
        legato_percentage = (legato_count / total_segments * 100) if total_segments > 0 else 50.0
        staccato_percentage = (staccato_count / total_segments * 100) if total_segments > 0 else 50.0
        
        return {
            'segments': segments,
            'statistics': {
                'legato_percentage': float(legato_percentage),
                'staccato_percentage': float(staccato_percentage),
                'total_segments': total_segments
            }
        }
        
    except Exception as e:
        current_app.logger.error(f"Melody analysis error: {str(e)}")
        return {
            'segments': [],
            'statistics': {
                'legato_percentage': 50.0,
                'staccato_percentage': 50.0,
                'total_segments': 0
            }
        }

def analyze_tango_structure(audio_path):
    """Analyze tango musical structure"""
    try:
        # Load audio file
        y, sr = librosa.load(audio_path)
        
        # Estimate tempo and time signature
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Tango is typically in 2/4 or 4/4 time
        time_signature = "2/4" if tempo > 100 else "4/4"
        
        # Simple section detection based on spectral changes
        # This is a simplified approach - real tango analysis would be more complex
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Create basic A-B-A structure (common in tango)
        sections = []
        if duration > 60:  # If song is longer than 1 minute
            sections = [
                {
                    'type': 'A',
                    'start': 0.0,
                    'end': duration * 0.3,
                    'description': 'opening_theme'
                },
                {
                    'type': 'B',
                    'start': duration * 0.3,
                    'end': duration * 0.7,
                    'description': 'contrasting_section'
                },
                {
                    'type': 'A',
                    'start': duration * 0.7,
                    'end': duration,
                    'description': 'return_theme'
                }
            ]
        else:
            sections = [
                {
                    'type': 'A',
                    'start': 0.0,
                    'end': duration,
                    'description': 'main_theme'
                }
            ]
        
        return {
            'time_signature': time_signature,
            'sections': sections,
            'phrases': []  # Could be extended with phrase detection
        }
        
    except Exception as e:
        current_app.logger.error(f"Structure analysis error: {str(e)}")
        return {
            'time_signature': '2/4',
            'sections': [
                {
                    'type': 'A',
                    'start': 0.0,
                    'end': 180.0,
                    'description': 'main_theme'
                }
            ],
            'phrases': []
        }

@music_bp.route('/health', methods=['GET'])
@cross_origin()
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'tango-music-analyzer',
        'version': '1.0.0',
        'librosa_available': True
    })

@music_bp.route('/upload', methods=['POST'])
@cross_origin()
def upload_file():
    """Upload music file for analysis"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({'error': 'File too large'}), 400
        
        # Generate unique upload ID
        upload_id = str(uuid.uuid4())
        
        # Secure filename
        filename = secure_filename(file.filename)
        if not filename:
            filename = f"upload_{upload_id}.mp3"
        
        # Create uploads directory
        upload_dir = os.path.join(tempfile.gettempdir(), 'tango_uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save file
        file_path = os.path.join(upload_dir, f"{upload_id}_{filename}")
        file.save(file_path)
        
        # Create database record
        upload_record = MusicUpload(
            upload_id=upload_id,
            filename=f"{upload_id}_{filename}",
            original_filename=file.filename,
            file_size=file_size,
            file_type=file.content_type or 'audio/mpeg'
        )
        
        db.session.add(upload_record)
        db.session.commit()
        
        return jsonify({
            'upload_id': upload_id,
            'filename': filename,
            'file_size': file_size,
            'status': 'uploaded'
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': 'Upload failed'}), 500

@music_bp.route('/analyze/<upload_id>', methods=['POST'])
@cross_origin()
def analyze_music(upload_id):
    """Analyze uploaded music file"""
    try:
        # Find upload record
        upload_record = MusicUpload.query.filter_by(upload_id=upload_id).first()
        if not upload_record:
            return jsonify({'error': 'Upload not found'}), 404
        
        # Update status
        upload_record.status = 'analyzing'
        db.session.commit()
        
        # Get file path
        upload_dir = os.path.join(tempfile.gettempdir(), 'tango_uploads')
        file_path = os.path.join(upload_dir, upload_record.filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        start_time = time.time()
        
        # Perform analysis
        beat_analysis = analyze_beats(file_path)
        melody_analysis = analyze_melody(file_path)
        structure_analysis = analyze_tango_structure(file_path)
        
        processing_duration = time.time() - start_time
        
        # Save analysis results
        analysis_result = AnalysisResult(
            upload_id=upload_id,
            bpm=beat_analysis['bpm'],
            beat_count=len(beat_analysis['beats']),
            beat_confidence=beat_analysis['confidence'],
            beat_method=beat_analysis['method'],
            beats_data=json.dumps(beat_analysis['beats']),
            legato_percentage=melody_analysis['statistics']['legato_percentage'],
            staccato_percentage=melody_analysis['statistics']['staccato_percentage'],
            melody_segments_count=melody_analysis['statistics']['total_segments'],
            melody_data=json.dumps(melody_analysis['segments']),
            time_signature=structure_analysis['time_signature'],
            sections_count=len(structure_analysis['sections']),
            structure_data=json.dumps(structure_analysis['sections']),
            processing_duration=processing_duration
        )
        
        db.session.add(analysis_result)
        
        # Update upload status
        upload_record.status = 'completed'
        db.session.commit()
        
        # Clean up file
        try:
            os.remove(file_path)
        except:
            pass
        
        return jsonify(analysis_result.to_dict()), 200
        
    except Exception as e:
        current_app.logger.error(f"Analysis error: {str(e)}")
        
        # Update status to error
        upload_record = MusicUpload.query.filter_by(upload_id=upload_id).first()
        if upload_record:
            upload_record.status = 'error'
            db.session.commit()
        
        return jsonify({'error': 'Analysis failed'}), 500

@music_bp.route('/results/<upload_id>', methods=['GET'])
@cross_origin()
def get_results(upload_id):
    """Get analysis results for an upload"""
    try:
        analysis_result = AnalysisResult.query.filter_by(upload_id=upload_id).first()
        if not analysis_result:
            return jsonify({'error': 'Results not found'}), 404
        
        return jsonify(analysis_result.to_dict()), 200
        
    except Exception as e:
        current_app.logger.error(f"Get results error: {str(e)}")
        return jsonify({'error': 'Failed to retrieve results'}), 500

@music_bp.route('/uploads', methods=['GET'])
@cross_origin()
def list_uploads():
    """List all uploads"""
    try:
        uploads = MusicUpload.query.order_by(MusicUpload.upload_time.desc()).limit(50).all()
        return jsonify([upload.to_dict() for upload in uploads]), 200
        
    except Exception as e:
        current_app.logger.error(f"List uploads error: {str(e)}")
        return jsonify({'error': 'Failed to list uploads'}), 500


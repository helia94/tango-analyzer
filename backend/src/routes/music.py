"""
Refactored Music Analysis REST API

This module provides REST API endpoints for music analysis using the MusicAnalyzer logic layer.
The API is separated from the business logic for better maintainability and testability.
"""

import os
import uuid
import json
import tempfile
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from flask_cors import cross_origin

# Import the logic layer
from src.routes.music_analyzer_logic import MusicAnalyzer, CompleteAnalysisResult

# Import models after Flask app is initialized
def get_models():
    from src.models.user import db
    from src.models.music import MusicUpload, AnalysisResult
    return db, MusicUpload, AnalysisResult

music_bp = Blueprint('music', __name__)

# Configuration
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'flac', 'm4a', 'aac', 'ogg'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


class MusicAnalysisAPI:
    """
    REST API handler class for music analysis endpoints.
    
    This class provides a clean interface between the REST API and the MusicAnalyzer logic layer.
    """
    
    @staticmethod
    def allowed_file(filename: str) -> bool:
        """Check if the uploaded file has an allowed extension"""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    @staticmethod
    def validate_file_upload(file) -> tuple:
        """
        Validate uploaded file for size and type.
        
        Returns:
            tuple: (is_valid, error_message, file_size)
        """
        if not file or file.filename == '':
            return False, 'No file selected', 0
        
        if not MusicAnalysisAPI.allowed_file(file.filename):
            return False, 'File type not allowed', 0
        
        # Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return False, 'File too large', file_size
        
        return True, '', file_size
    
    @staticmethod
    def save_uploaded_file(file, upload_id: str) -> tuple:
        """
        Save uploaded file to temporary directory.
        
        Returns:
            tuple: (success, file_path, filename)
        """
        try:
            # Secure filename
            filename = secure_filename(file.filename)
            if not filename:
                filename = f"upload_{upload_id}.mp3"
            
            # Create uploads directory
            upload_dir = os.path.join(tempfile.gettempdir(), 'tango_uploads')
            os.makedirs(upload_dir, exist_ok=True)
            
            # Save file
            final_filename = f"{upload_id}_{filename}"
            file_path = os.path.join(upload_dir, final_filename)
            file.save(file_path)
            
            return True, file_path, final_filename
            
        except Exception as e:
            current_app.logger.error(f"File save error: {str(e)}")
            return False, '', ''
    
    @staticmethod
    def create_upload_record(upload_id: str, filename: str, original_filename: str, 
                           file_size: int, content_type: str) -> bool:
        """Create database record for uploaded file"""
        try:
            db, MusicUpload, AnalysisResult = get_models()
            
            upload_record = MusicUpload(
                upload_id=upload_id,
                filename=filename,
                original_filename=original_filename,
                file_size=file_size,
                file_type=content_type or 'audio/mpeg'
            )
            
            db.session.add(upload_record)
            db.session.commit()
            return True
            
        except Exception as e:
            current_app.logger.error(f"Database error: {str(e)}")
            return False
    
    @staticmethod
    def perform_analysis(file_path: str) -> tuple:
        """
        Perform complete music analysis using the MusicAnalyzer logic layer.
        
        Returns:
            tuple: (success, analysis_result_dict, error_message)
        """
        try:
            # Initialize the analyzer
            analyzer = MusicAnalyzer(file_path)
            
            # Perform complete analysis (beat -> melody -> sections)
            analysis_result = analyzer.analyze_complete()
            
            # Convert to dictionary format
            result_dict = analyzer.to_dict(analysis_result)
            
            return True, result_dict, ''
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            current_app.logger.error(error_msg)
            return False, {}, error_msg
    
    @staticmethod
    def save_analysis_results(upload_id: str, analysis_dict: dict, processing_duration: float) -> bool:
        """Save analysis results to database"""
        try:
            db, MusicUpload, AnalysisResult = get_models()
            
            # Extract data from analysis dictionary
            beat_analysis = analysis_dict.get('beat_analysis', {})
            melody_analysis = analysis_dict.get('melody_analysis', {})
            section_analysis = analysis_dict.get('section_analysis', {})
            
            analysis_result = AnalysisResult(
                upload_id=upload_id,
                bpm=beat_analysis.get('bpm', 120.0),
                beat_count=beat_analysis.get('total_beats', 0),
                beat_confidence=beat_analysis.get('confidence', 0.0),
                beat_method=beat_analysis.get('method', 'fallback'),
                beats_data=json.dumps(beat_analysis.get('beats', [])),
                legato_percentage=melody_analysis.get('statistics', {}).get('legato_percentage', 50.0),
                staccato_percentage=melody_analysis.get('statistics', {}).get('staccato_percentage', 50.0),
                melody_segments_count=melody_analysis.get('statistics', {}).get('total_segments', 0),
                melody_data=json.dumps(melody_analysis.get('segments', [])),
                time_signature=section_analysis.get('time_signature', '2/4'),
                sections_count=section_analysis.get('section_count', 0),
                structure_data=json.dumps(section_analysis.get('sections', [])),
                processing_duration=processing_duration
            )
            
            db.session.add(analysis_result)
            db.session.commit()
            return True
            
        except Exception as e:
            current_app.logger.error(f"Save results error: {str(e)}")
            return False
    
    @staticmethod
    def update_upload_status(upload_id: str, status: str) -> bool:
        """Update upload status in database"""
        try:
            db, MusicUpload, AnalysisResult = get_models()
            
            upload_record = MusicUpload.query.filter_by(upload_id=upload_id).first()
            if upload_record:
                upload_record.status = status
                db.session.commit()
                return True
            return False
            
        except Exception as e:
            current_app.logger.error(f"Status update error: {str(e)}")
            return False
    
    @staticmethod
    def cleanup_file(file_path: str) -> None:
        """Clean up temporary file"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            current_app.logger.warning(f"File cleanup warning: {str(e)}")


# ==================== REST API ENDPOINTS ====================

@music_bp.route('/health', methods=['GET'])
@cross_origin()
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'tango-music-analyzer',
        'version': '2.0.0',
        'librosa_available': True,
        'logic_layer': 'MusicAnalyzer'
    })


@music_bp.route('/upload', methods=['POST'])
@cross_origin()
def upload_file():
    """Upload music file for analysis"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Validate file
        is_valid, error_msg, file_size = MusicAnalysisAPI.validate_file_upload(file)
        if not is_valid:
            return jsonify({'error': error_msg}), 400
        
        # Generate unique upload ID
        upload_id = str(uuid.uuid4())
        
        # Save file
        success, file_path, filename = MusicAnalysisAPI.save_uploaded_file(file, upload_id)
        if not success:
            return jsonify({'error': 'Failed to save file'}), 500
        
        # Create database record
        success = MusicAnalysisAPI.create_upload_record(
            upload_id, filename, file.filename, file_size, file.content_type
        )
        if not success:
            MusicAnalysisAPI.cleanup_file(file_path)
            return jsonify({'error': 'Failed to create upload record'}), 500
        
        return jsonify({
            'upload_id': upload_id,
            'filename': file.filename,
            'file_size': file_size,
            'status': 'uploaded'
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': 'Upload failed'}), 500


@music_bp.route('/analyze/<upload_id>', methods=['POST'])
@cross_origin()
def analyze_music(upload_id):
    """Analyze uploaded music file using the MusicAnalyzer logic layer"""
    try:
        # Find upload record
        db, MusicUpload, AnalysisResult = get_models()
        upload_record = MusicUpload.query.filter_by(upload_id=upload_id).first()
        if not upload_record:
            return jsonify({'error': 'Upload not found'}), 404
        
        # Update status to analyzing
        MusicAnalysisAPI.update_upload_status(upload_id, 'analyzing')
        
        # Get file path
        upload_dir = os.path.join(tempfile.gettempdir(), 'tango_uploads')
        file_path = os.path.join(upload_dir, upload_record.filename)
        
        if not os.path.exists(file_path):
            MusicAnalysisAPI.update_upload_status(upload_id, 'error')
            return jsonify({'error': 'File not found'}), 404
        
        # Perform analysis using logic layer
        success, analysis_dict, error_msg = MusicAnalysisAPI.perform_analysis(file_path)
        
        if not success:
            MusicAnalysisAPI.update_upload_status(upload_id, 'error')
            return jsonify({'error': error_msg}), 500
        
        # Save analysis results
        processing_duration = analysis_dict.get('processing_duration', 0.0)
        success = MusicAnalysisAPI.save_analysis_results(upload_id, analysis_dict, processing_duration)
        
        if not success:
            MusicAnalysisAPI.update_upload_status(upload_id, 'error')
            return jsonify({'error': 'Failed to save analysis results'}), 500
        
        # Update status to completed
        MusicAnalysisAPI.update_upload_status(upload_id, 'completed')
        
        # Clean up file
        MusicAnalysisAPI.cleanup_file(file_path)
        
        # Return analysis results
        return jsonify(analysis_dict), 200
        
    except Exception as e:
        current_app.logger.error(f"Analysis error: {str(e)}")
        MusicAnalysisAPI.update_upload_status(upload_id, 'error')
        return jsonify({'error': 'Analysis failed'}), 500


@music_bp.route('/analyze/<upload_id>/beat', methods=['POST'])
@cross_origin()
def analyze_beat_only(upload_id):
    """Analyze only the beat component of uploaded music file"""
    try:
        # Find upload record
        db, MusicUpload, AnalysisResult = get_models()
        upload_record = MusicUpload.query.filter_by(upload_id=upload_id).first()
        if not upload_record:
            return jsonify({'error': 'Upload not found'}), 404
        
        # Get file path
        upload_dir = os.path.join(tempfile.gettempdir(), 'tango_uploads')
        file_path = os.path.join(upload_dir, upload_record.filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Perform beat analysis only
        analyzer = MusicAnalyzer(file_path)
        beat_result = analyzer.analyze_beats()
        
        # Convert to dictionary
        beat_dict = {
            'bpm': beat_result.bpm,
            'beats': [
                {
                    'time': beat.time,
                    'confidence': beat.confidence,
                    'strength': beat.strength
                }
                for beat in beat_result.beats
            ],
            'method': beat_result.method.value,
            'confidence': beat_result.confidence,
            'total_beats': beat_result.total_beats
        }
        
        return jsonify({'beat_analysis': beat_dict}), 200
        
    except Exception as e:
        current_app.logger.error(f"Beat analysis error: {str(e)}")
        return jsonify({'error': 'Beat analysis failed'}), 500


@music_bp.route('/analyze/<upload_id>/melody', methods=['POST'])
@cross_origin()
def analyze_melody_only(upload_id):
    """Analyze only the melody component of uploaded music file"""
    try:
        # Find upload record
        db, MusicUpload, AnalysisResult = get_models()
        upload_record = MusicUpload.query.filter_by(upload_id=upload_id).first()
        if not upload_record:
            return jsonify({'error': 'Upload not found'}), 404
        
        # Get file path
        upload_dir = os.path.join(tempfile.gettempdir(), 'tango_uploads')
        file_path = os.path.join(upload_dir, upload_record.filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Perform melody analysis only
        analyzer = MusicAnalyzer(file_path)
        melody_result = analyzer.analyze_melody()
        
        # Convert to dictionary
        melody_dict = {
            'segments': [
                {
                    'start': seg.start,
                    'end': seg.end,
                    'type': seg.segment_type.value,
                    'confidence': seg.confidence,
                    'duration': seg.duration
                }
                for seg in melody_result.segments
            ],
            'statistics': {
                'legato_percentage': melody_result.statistics.legato_percentage,
                'staccato_percentage': melody_result.statistics.staccato_percentage,
                'total_segments': melody_result.statistics.total_segments,
                'average_segment_duration': melody_result.statistics.average_segment_duration,
                'median_interval': melody_result.statistics.median_interval
            }
        }
        
        return jsonify({'melody_analysis': melody_dict}), 200
        
    except Exception as e:
        current_app.logger.error(f"Melody analysis error: {str(e)}")
        return jsonify({'error': 'Melody analysis failed'}), 500


@music_bp.route('/analyze/<upload_id>/sections', methods=['POST'])
@cross_origin()
def analyze_sections_only(upload_id):
    """Analyze only the sections component of uploaded music file"""
    try:
        # Find upload record
        db, MusicUpload, AnalysisResult = get_models()
        upload_record = MusicUpload.query.filter_by(upload_id=upload_id).first()
        if not upload_record:
            return jsonify({'error': 'Upload not found'}), 404
        
        # Get file path
        upload_dir = os.path.join(tempfile.gettempdir(), 'tango_uploads')
        file_path = os.path.join(upload_dir, upload_record.filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Perform section analysis only
        analyzer = MusicAnalyzer(file_path)
        section_result = analyzer.analyze_sections()
        
        # Convert to dictionary
        section_dict = {
            'time_signature': section_result.time_signature,
            'sections': [
                {
                    'type': sec.section_type.value,
                    'start': sec.start,
                    'end': sec.end,
                    'duration': sec.duration,
                    'description': sec.description
                }
                for sec in section_result.sections
            ],
            'phrases': section_result.phrases,
            'total_duration': section_result.total_duration,
            'section_count': section_result.section_count
        }
        
        return jsonify({'section_analysis': section_dict}), 200
        
    except Exception as e:
        current_app.logger.error(f"Section analysis error: {str(e)}")
        return jsonify({'error': 'Section analysis failed'}), 500


@music_bp.route('/results/<upload_id>', methods=['GET'])
@cross_origin()
def get_results(upload_id):
    """Get analysis results for an upload"""
    try:
        db, MusicUpload, AnalysisResult = get_models()
        
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
        db, MusicUpload, AnalysisResult = get_models()
        
        uploads = MusicUpload.query.order_by(MusicUpload.upload_time.desc()).limit(50).all()
        return jsonify([upload.to_dict() for upload in uploads]), 200
        
    except Exception as e:
        current_app.logger.error(f"List uploads error: {str(e)}")
        return jsonify({'error': 'Failed to list uploads'}), 500


@music_bp.route('/analyzer/info', methods=['GET'])
@cross_origin()
def get_analyzer_info():
    """Get information about the MusicAnalyzer logic layer"""
    return jsonify({
        'analyzer_class': 'MusicAnalyzer',
        'analysis_order': ['beat', 'melody', 'sections'],
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size_mb': MAX_FILE_SIZE // (1024 * 1024),
        'features': {
            'beat_analysis': {
                'tempo_detection': True,
                'beat_tracking': True,
                'confidence_scoring': True
            },
            'melody_analysis': {
                'onset_detection': True,
                'legato_staccato_classification': True,
                'segment_statistics': True
            },
            'section_analysis': {
                'time_signature_estimation': True,
                'structural_segmentation': True,
                'phrase_detection': False  # Placeholder for future
            }
        },
        'version': '2.0.0'
    })


from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

# Import db from user.py to use the same instance
from src.models.user import db

class MusicUpload(db.Model):
    __tablename__ = 'music_uploads'
    
    id = db.Column(db.Integer, primary_key=True)
    upload_id = db.Column(db.String(36), unique=True, nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_size = db.Column(db.Integer, nullable=False)
    file_type = db.Column(db.String(50), nullable=False)
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default='uploaded')  # uploaded, analyzing, completed, error
    
    # Relationship to analysis results
    analysis_result = db.relationship('AnalysisResult', backref='upload', uselist=False, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'upload_id': self.upload_id,
            'filename': self.filename,
            'original_filename': self.original_filename,
            'file_size': self.file_size,
            'file_type': self.file_type,
            'upload_time': self.upload_time.isoformat() if self.upload_time else None,
            'status': self.status
        }

class AnalysisResult(db.Model):
    __tablename__ = 'analysis_results'
    
    id = db.Column(db.Integer, primary_key=True)
    upload_id = db.Column(db.String(36), db.ForeignKey('music_uploads.upload_id'), nullable=False)
    
    # Beat analysis results
    bpm = db.Column(db.Float)
    beat_count = db.Column(db.Integer)
    beat_confidence = db.Column(db.Float)
    beat_method = db.Column(db.String(50))
    beats_data = db.Column(db.Text)  # JSON string of beat timestamps and confidences
    
    # Melody analysis results
    legato_percentage = db.Column(db.Float)
    staccato_percentage = db.Column(db.Float)
    melody_segments_count = db.Column(db.Integer)
    melody_data = db.Column(db.Text)  # JSON string of melody segments
    
    # Tango structure analysis
    time_signature = db.Column(db.String(10))
    sections_count = db.Column(db.Integer)
    structure_data = db.Column(db.Text)  # JSON string of structure analysis
    
    # Analysis metadata
    analysis_time = db.Column(db.DateTime, default=datetime.utcnow)
    processing_duration = db.Column(db.Float)  # seconds
    
    def to_dict(self):
        return {
            'id': self.id,
            'upload_id': self.upload_id,
            'beat_analysis': {
                'bpm': self.bpm,
                'beat_count': self.beat_count,
                'confidence': self.beat_confidence,
                'method': self.beat_method,
                'beats': json.loads(self.beats_data) if self.beats_data else []
            },
            'melody_analysis': {
                'statistics': {
                    'legato_percentage': self.legato_percentage,
                    'staccato_percentage': self.staccato_percentage,
                    'total_segments': self.melody_segments_count
                },
                'segments': json.loads(self.melody_data) if self.melody_data else []
            },
            'tango_structure': {
                'time_signature': self.time_signature,
                'sections_count': self.sections_count,
                'sections': json.loads(self.structure_data) if self.structure_data else [],
                'phrases': []  # Can be extended later
            },
            'metadata': {
                'analysis_time': self.analysis_time.isoformat() if self.analysis_time else None,
                'processing_duration': self.processing_duration
            }
        }


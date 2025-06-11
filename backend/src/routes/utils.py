from src.routes.tango_analyzer_gpt import  CompleteAnalysisResult
from typing import Dict, List, Any, Optional, Tuple



def to_dict(result: CompleteAnalysisResult) -> Dict[str, Any]:
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
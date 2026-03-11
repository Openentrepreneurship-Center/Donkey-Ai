"""
Diarization 서비스 - pyannote.audio
화자 분리 수행
"""
from typing import List, Tuple

from models.loader import get_diarization_pipeline


def run_diarization(audio_path: str) -> List[Tuple[float, float, str]]:
    """
    pyannote로 화자 분리 수행
    
    Args:
        audio_path: 오디오 파일 경로
        
    Returns:
        [(start, end, speaker), ...] 리스트
        speaker: "SPEAKER_00", "SPEAKER_01" 등
    """
    pipeline = get_diarization_pipeline()
    if pipeline is None:
        return []
    
    diarization = pipeline(audio_path)
    
    result = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        result.append((turn.start, turn.end, speaker))
    
    return result

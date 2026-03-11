"""
STT 서비스 - Faster-Whisper
"""
from typing import List, Tuple, Any

from config import get_settings
from models.loader import get_stt_model


def transcribe(audio_path: str) -> Tuple[List[Any], Any]:
    """
    Faster-Whisper로 STT 수행
    
    Args:
        audio_path: 16kHz mono wav 경로
        
    Returns:
        (segments, info) - segments는 Segment 객체 리스트
    """
    model = get_stt_model()
    if model is None:
        raise RuntimeError("STT model not loaded")

    cfg = get_settings()
    segments, info = model.transcribe(
        audio_path,
        beam_size=cfg.stt_beam_size,
        vad_filter=cfg.stt_vad_filter,
        vad_parameters=dict(
            min_silence_duration_ms=cfg.stt_vad_min_silence_ms,
            speech_pad_ms=cfg.stt_vad_speech_pad_ms,
        ),
    )
    # generator -> list
    segments = list(segments)
    return segments, info


def segments_to_list(segments: List[Any]) -> List[dict]:
    """
    Whisper Segment 객체 리스트를 dict 리스트로 변환
    """
    return [
        {"start": s.start, "end": s.end, "text": s.text.strip()}
        for s in segments
    ]

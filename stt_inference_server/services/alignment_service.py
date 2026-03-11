"""
Alignment 서비스 - STT segments + Diarization segments 결합
각 STT segment에 대해 overlap이 가장 큰 speaker 할당
"""
from typing import List, Tuple


def compute_overlap(
    stt_start: float, stt_end: float,
    spk_start: float, spk_end: float
) -> float:
    """두 구간의 겹치는 길이 반환"""
    overlap_start = max(stt_start, spk_start)
    overlap_end = min(stt_end, spk_end)
    return max(0.0, overlap_end - overlap_start)


def align_speakers(
    stt_segments: List[dict],
    diarization: List[Tuple[float, float, str]]
) -> List[dict]:
    """
    STT segments에 speaker 라벨 부여
    각 segment에 대해 overlap이 가장 큰 speaker 선택
    
    Args:
        stt_segments: [{"start", "end", "text"}, ...]
        diarization: [(start, end, speaker), ...]
        
    Returns:
        [{"speaker", "start", "end", "text"}, ...]
    """
    result = []
    default_speaker = "SPEAKER_00"
    
    for seg in stt_segments:
        start = seg["start"]
        end = seg["end"]
        text = seg["text"]
        
        best_speaker = default_speaker
        best_overlap = 0.0
        
        for spk_start, spk_end, speaker in diarization:
            overlap = compute_overlap(start, end, spk_start, spk_end)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
        
        result.append({
            "speaker": best_speaker,
            "start": start,
            "end": end,
            "text": text,
        })
    
    return result

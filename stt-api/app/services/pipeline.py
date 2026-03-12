"""병렬 전사+화자분리 파이프라인."""

import asyncio
import logging
from pathlib import Path

from app.config import get_settings
from app.services.diarization import run_diarization
from app.services.postprocessing import deduplicate_segments, postprocess_text
from app.services.transcription import transcribe_with_segments

logger = logging.getLogger(__name__)


def _assign_speaker_to_segments(
    transcript_segments: list[dict],
    diar_segments: list[dict],
) -> list[dict]:
    """
    전사 세그먼트에 화자 라벨 할당.
    시간 겹침이 가장 큰 diarization 구간의 speaker를 할당.
    """
    if not diar_segments:
        return transcript_segments

    result = []
    for seg in transcript_segments:
        s_start, s_end = seg["start"], seg["end"]
        best_speaker = None
        best_overlap = 0.0

        for d in diar_segments:
            d_start, d_end = d["start"], d["end"]
            overlap = min(s_end, d_end) - max(s_start, d_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d["speaker"]

        out = {**seg, "speaker": best_speaker}
        result.append(out)
    return result


def _run_transcribe_sync(wav_path: Path, language: str) -> list[dict]:
    """동기 전사 실행 (스레드 풀용)."""
    segments = transcribe_with_segments(wav_path, language=language)
    settings = get_settings()
    if settings.enable_postprocessing:
        segments = deduplicate_segments(segments)
        for seg in segments:
            seg["text"] = postprocess_text(seg.get("text", ""))
    return segments


def _run_diarization_sync(wav_path: Path) -> list[dict]:
    """동기 화자분리 실행 (스레드 풀용)."""
    return run_diarization(wav_path)


async def transcribe_with_diarization(
    wav_path: Path,
    language: str = "ko",
) -> list[dict]:
    """
    전사 + 화자분리 병렬 실행 후 결과 병합.
    Returns list of {"start": float, "end": float, "text": str, "speaker": str | None}.
    """
    settings = get_settings()
    loop = asyncio.get_running_loop()

    if not settings.enable_diarization:
        # 화자분리 비활성화: 기존 전사만
        segments = await loop.run_in_executor(
            None,
            lambda: _run_transcribe_sync(wav_path, language),
        )
        return [{"speaker": None, **s} for s in segments]

    # 병렬 실행: 전사 + 화자분리
    trans_task = loop.run_in_executor(
        None,
        lambda: _run_transcribe_sync(wav_path, language),
    )
    diar_task = loop.run_in_executor(
        None,
        lambda: _run_diarization_sync(wav_path),
    )

    transcript_segments, diar_segments = await asyncio.gather(trans_task, diar_task)

    # 화자 라벨 할당
    return _assign_speaker_to_segments(transcript_segments, diar_segments)

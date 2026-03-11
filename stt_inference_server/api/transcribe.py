"""
Transcribe API - POST /transcribe
음성 업로드 → 변환 → Diarization → STT → Alignment → JSON 응답
"""
import logging
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException

from services.stt_service import transcribe, segments_to_list
from services.diarization_service import run_diarization
from services.alignment_service import align_speakers
from config import get_settings, get_max_file_size_bytes
from utils.audio import convert_to_wav, get_converted_path, AudioConversionError
from utils.temp_file import temp_directory, save_upload_to_disk

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".webm"}


def validate_file(filename: str) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Allowed: wav, mp3, m4a, webm",
        )


def validate_file_size(size: int) -> None:
    max_bytes = get_max_file_size_bytes()
    max_mb = get_settings().max_file_size_mb
    if size > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max {max_mb}MB",
        )


@router.post("/transcribe")
def transcribe_api(file: UploadFile = File(...)):
    """
    음성 파일 업로드 → STT + Diarization → JSON 반환
    
    Processing pipeline:
    1. temp dir 생성
    2. 업로드 파일 저장 (디스크, RAM 비사용)
    3. ffmpeg 16kHz mono wav 변환
    4. Diarization
    5. STT
    6. Alignment
    7. JSON 응답
    """
    validate_file(file.filename or "")
    
    with temp_directory() as tmpdir:
        # 1. 원본 파일 저장 (스트리밍, RAM 비사용)
        original_path = str(Path(tmpdir) / (file.filename or "audio"))
        try:
            save_upload_to_disk(file.file, original_path)
        except Exception as e:
            logger.exception("Upload save failed")
            raise HTTPException(status_code=500, detail=f"Upload failed: {e}")
        
        file_size = Path(original_path).stat().st_size
        validate_file_size(file_size)
        
        # 2. ffmpeg 16kHz mono wav 변환
        wav_path = get_converted_path(original_path, tmpdir)
        try:
            convert_to_wav(original_path, wav_path)
        except AudioConversionError as e:
            logger.exception("Audio conversion failed")
            raise HTTPException(status_code=500, detail=str(e))
        
        # 3. Diarization
        try:
            diarization = run_diarization(wav_path)
        except Exception as e:
            logger.warning(f"Diarization failed, using empty: {e}")
            diarization = []
        
        # 4. STT
        try:
            segments, info = transcribe(wav_path)
        except Exception as e:
            logger.exception("STT failed")
            raise HTTPException(status_code=500, detail=f"STT failed: {e}")
        
        stt_list = segments_to_list(segments)
        
        # 5. Alignment
        aligned = align_speakers(stt_list, diarization)
        
        # 6. 전체 텍스트
        full_text = " ".join(s["text"] for s in aligned).strip()
        
        return {
            "text": full_text,
            "segments": [
                {
                    "speaker": s["speaker"],
                    "start": round(s["start"], 2),
                    "end": round(s["end"], 2),
                    "text": s["text"],
                }
                for s in aligned
            ],
        }

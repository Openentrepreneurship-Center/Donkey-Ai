"""
오디오 변환 - ffmpeg로 16kHz mono wav 변환
"""
import subprocess
import logging
from pathlib import Path

from config import get_settings

logger = logging.getLogger(__name__)

SUPPORTED_INPUT_FORMATS = {".wav", ".mp3", ".m4a", ".webm"}


class AudioConversionError(Exception):
    """ffmpeg 변환 실패 시"""
    pass


def convert_to_wav(input_path: str, output_path: str) -> str:
    """
    ffmpeg로 16kHz mono wav로 변환
    
    Args:
        input_path: 입력 오디오 경로
        output_path: 출력 wav 경로
        
    Returns:
        output_path
        
    Raises:
        AudioConversionError: ffmpeg 실패 시
    """
    cfg = get_settings()
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ar", str(cfg.audio_sample_rate),
        "-ac", str(cfg.audio_channels),
        "-hide_banner", "-loglevel", "error",
        output_path,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=cfg.ffmpeg_timeout,
        )
        if result.returncode != 0:
            raise AudioConversionError(
                f"ffmpeg failed: {result.stderr or result.stdout}"
            )
        if not Path(output_path).exists():
            raise AudioConversionError("ffmpeg did not produce output file")
        return output_path
    except subprocess.TimeoutExpired:
        raise AudioConversionError(f"ffmpeg timeout ({cfg.ffmpeg_timeout}s)")
    except FileNotFoundError:
        raise AudioConversionError("ffmpeg not found. Install ffmpeg.")


def get_converted_path(original_path: str, temp_dir: str) -> str:
    """원본 경로에서 변환된 wav 경로 생성"""
    stem = Path(original_path).stem
    return str(Path(temp_dir) / f"{stem}_16k.wav")

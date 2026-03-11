"""
설정 - pydantic-settings + .env
환경변수 또는 .env 파일에서 로드
"""
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # HuggingFace
    hf_token: str = ""

    # STT - Faster-Whisper
    stt_model: str = "large-v3"
    stt_device: Literal["auto", "cuda", "cpu"] = "auto"
    stt_compute_type: Literal["float16", "int8"] = "float16"
    stt_beam_size: int = 3
    stt_vad_filter: bool = True
    stt_vad_min_silence_ms: int = 500
    stt_vad_speech_pad_ms: int = 200

    # Diarization
    diarization_model: str = "pyannote/speaker-diarization-3.1"

    # Audio
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    ffmpeg_timeout: int = 300

    # API
    max_file_size_mb: int = 500


@lru_cache
def get_settings() -> Settings:
    """싱글톤 설정 인스턴스"""
    return Settings()


def get_max_file_size_bytes() -> int:
    return get_settings().max_file_size_mb * 1024 * 1024

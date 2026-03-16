"""STT 설정. Faster-Whisper 오픈소스 전용."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Faster-Whisper 로컬 STT
    faster_whisper_model: str = Field(
        default="large-v3", validation_alias="FASTER_WHISPER_MODEL",
    )
    faster_whisper_beam_size: int = Field(
        default=5, validation_alias="FASTER_WHISPER_BEAM_SIZE",
    )
    faster_whisper_compute_type: str = Field(
        default="float16", validation_alias="FASTER_WHISPER_COMPUTE_TYPE",
    )

    # 공통
    default_language: str = Field(default="ko", validation_alias="DEFAULT_LANGUAGE")

    # 후처리 (의료 용어 교정, 환각 제거) 적용 여부
    enable_postprocessing: bool = Field(default=True, validation_alias="ENABLE_POSTPROCESSING")

    # 화자 분리 (pyannote) - HUGGINGFACE_TOKEN 필요 (또는 huggingface-cli login)
    enable_diarization: bool = Field(default=True, validation_alias="ENABLE_DIARIZATION")
    huggingface_token: str = Field(default="", validation_alias="HUGGINGFACE_TOKEN")
    # 화자 수. None이면 min/max만 적용(자동). 2 등 지정 시 해당 인원으로 고정
    diarization_num_speakers: int | None = Field(default=None, validation_alias="DIARIZATION_NUM_SPEAKERS")
    # 자동 탐지 시 최소 화자 수. 1=기본(완전 자동). 2로 두면 2명 대화에서 1명으로 나오는 것 방지
    diarization_min_speakers: int = Field(default=1, validation_alias="DIARIZATION_MIN_SPEAKERS")
    diarization_max_speakers: int | None = Field(default=None, validation_alias="DIARIZATION_MAX_SPEAKERS")

    @field_validator("diarization_num_speakers", "diarization_max_speakers", mode="before")
    @classmethod
    def empty_str_to_none(cls, v):
        if v == "" or v is None:
            return None
        return v

    @field_validator("diarization_min_speakers", mode="before")
    @classmethod
    def min_speakers_default(cls, v):
        if v == "" or v is None:
            return 1
        return int(v)

    model_config = {
        "env_file": Path(__file__).resolve().parent.parent / ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()

"""
모델 로딩 - 서버 시작 시 한 번만 수행
STT (Faster-Whisper), Diarization (pyannote.audio)
"""
import logging

from config import get_settings

logger = logging.getLogger(__name__)

# 전역 모델 인스턴스
_stt_model = None
_diarization_pipeline = None


def load_models():
    """서버 시작 시 STT, Diarization 모델 로딩"""
    global _stt_model, _diarization_pipeline

    import torch
    cfg = get_settings()

    # device: auto -> cuda if available else cpu
    if cfg.stt_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg.stt_device
    compute_type = cfg.stt_compute_type
    if device == "cpu" and compute_type == "float16":
        compute_type = "int8"

    # STT - Faster-Whisper
    logger.info(f"Loading Faster-Whisper {cfg.stt_model}...")
    from faster_whisper import WhisperModel
    _stt_model = WhisperModel(
        cfg.stt_model,
        device=device,
        compute_type=compute_type,
    )
    logger.info("STT model loaded.")

    # Diarization - pyannote.audio
    try:
        from pyannote.audio import Pipeline
        if not cfg.hf_token:
            logger.warning(
                "HF_TOKEN not set. Diarization will fail. "
                "Set HF_TOKEN env var and accept model terms at "
                "https://huggingface.co/pyannote/speaker-diarization"
            )
        _diarization_pipeline = Pipeline.from_pretrained(
            cfg.diarization_model,
            use_auth_token=cfg.hf_token or None,
        )
        _diarization_pipeline.to(torch.device(device))
        logger.info("Diarization pipeline loaded.")
    except Exception as e:
        logger.warning(f"Diarization pipeline load failed: {e}. Diarization disabled.")
        _diarization_pipeline = None


def get_stt_model():
    return _stt_model


def get_diarization_pipeline():
    return _diarization_pipeline


def models_loaded():
    return _stt_model is not None

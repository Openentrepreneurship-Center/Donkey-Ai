"""STT API Server - Standalone Speech-to-Text API."""

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.services.audio import download_audio, ensure_wav_16k_mono
from app.services.pipeline import transcribe_with_diarization

app = FastAPI(
    title="STT API",
    description="Standalone Speech-to-Text API (Faster-Whisper 오픈소스)",
    version="0.1.0",
)


class TranscribeUrlRequest(BaseModel):
    """URL로 음성 전사 요청."""

    url: str = Field(..., description="오디오 파일 URL")
    language: str = Field(default="ko", description="언어 코드 (ko, en 등)")


class SegmentResponse(BaseModel):
    """전사 세그먼트."""

    start: float
    end: float
    text: str
    speaker: str | None = None


class TranscribeResponse(BaseModel):
    """전사 결과."""

    segments: list[SegmentResponse]
    full_text: str


@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok"}


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_url(req: TranscribeUrlRequest):
    """
    오디오 URL을 받아 전사 결과 반환.
    지원 형식: wav, mp3, m4a 등 (ffmpeg 지원 형식)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        src = tmp / "audio"
        try:
            await download_audio(req.url, src)
        except Exception as e:
            raise HTTPException(400, f"Failed to download audio: {e}") from e

        try:
            wav_path = ensure_wav_16k_mono(src)
        except Exception as e:
            raise HTTPException(400, f"Failed to convert audio: {e}") from e

        try:
            segments = await transcribe_with_diarization(wav_path, language=req.language)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        except Exception as e:
            raise HTTPException(500, f"Transcription failed: {e}") from e

    out = [
        SegmentResponse(
            start=s["start"],
            end=s["end"],
            text=s["text"],
            speaker=s.get("speaker"),
        )
        for s in segments
    ]
    full_text = " ".join(s["text"] for s in segments).strip()
    return TranscribeResponse(segments=out, full_text=full_text)


@app.post("/transcribe/file", response_model=TranscribeResponse)
async def transcribe_file(
    file: UploadFile = File(...),
    language: str = Form(default="ko"),
):
    """
    오디오 파일 업로드로 전사.
    """
    suffix = Path(file.filename or "audio").suffix or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        try:
            content = await file.read()
            tmp.write(content)
            tmp.flush()
            tmp_path = Path(tmp.name)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    try:
        wav_path = ensure_wav_16k_mono(tmp_path)
        segments = await transcribe_with_diarization(wav_path, language=language)
    except ValueError as e:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Transcription failed: {e}") from e
    finally:
        tmp_path.unlink(missing_ok=True)

    out = [
        SegmentResponse(
            start=s["start"],
            end=s["end"],
            text=s["text"],
            speaker=s.get("speaker"),
        )
        for s in segments
    ]
    full_text = " ".join(s["text"] for s in segments).strip()
    return TranscribeResponse(segments=out, full_text=full_text)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

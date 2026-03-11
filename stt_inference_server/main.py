"""
AI Inference Server - STT + Diarization
서버 시작 시 모델 로딩, API 라우터 등록
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.transcribe import router as transcribe_router
from models.loader import load_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 모델 로딩, 종료 시 정리"""
    load_models()
    yield
    # 정리 작업 (필요 시)


app = FastAPI(
    title="STT Inference Server",
    description="AI Inference API - Speech-to-Text + Speaker Diarization",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(transcribe_router, tags=["transcribe"])


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
    )

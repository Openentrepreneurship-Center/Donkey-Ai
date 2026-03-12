# STT API - Faster-Whisper + pyannote 화자분리
FROM python:3.12-slim

# ffmpeg 설치 (오디오 변환용)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 설치
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

# 앱 코드
COPY app/ ./app/

EXPOSE 8000

# GPU 사용 시: docker run --gpus all ...
# CUDA 있으면 자동 사용, 없으면 CPU fallback
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

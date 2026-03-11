# AI Inference Server - STT + Diarization
# GPU: nvidia/cuda base image

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps: ffmpeg, python
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    python3.10 \
    python3-pip \
    python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps - PyTorch CUDA 먼저 설치
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt

# App
COPY . .

EXPOSE 8000

# Run server (1 worker for inference)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

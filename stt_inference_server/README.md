# STT Inference Server

FastAPI 기반 AI Inference API - Speech-to-Text + Speaker Diarization

## 기능

- 음성 파일 업로드 (wav, mp3, m4a, webm)
- STT: Faster-Whisper large-v3
- 화자 분리: pyannote.audio
- speaker-labeled transcript JSON 반환

## 실행

### 로컬

```bash
pip install -r requirements.txt
# HF_TOKEN=xxx python -m uvicorn main:app --host 0.0.0.0 --port 8000
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### Docker (GPU)

```bash
docker build -t stt-inference .
docker run --gpus all -p 8000:8000 -e HF_TOKEN=your_token stt-inference
```

## API

- `GET /health` - Health check
- `POST /transcribe` - `multipart/form-data`, body: `file=<audio>`

## 환경변수

- `HF_TOKEN`: HuggingFace token (pyannote 모델 접근용, 필수 for diarization)

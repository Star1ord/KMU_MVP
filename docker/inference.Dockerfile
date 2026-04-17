FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    AUTO_YES=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.runtime.txt ./requirements.runtime.txt

RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.runtime.txt

COPY api ./api
COPY config ./config
COPY inference ./inference
COPY src ./src
COPY video_integration ./video_integration
COPY run_all.py ./run_all.py
COPY README.md ./README.md

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# KMU MVP

Multimodal MVP for video-based risk assessment.

The repository contains:

- `api/`: FastAPI backend
- `web_ui/`: static frontend
- `src/`: preprocessing, inference, and pipeline code
- `models/`: committed runtime models required for local inference

## Quick Start

Recommended local setup:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.runtime.txt
python run_all.py --auto-yes
```

Open:

- API docs: `http://127.0.0.1:8000/docs`
- UI: `http://127.0.0.1:5500`

## Docker

Start backend + frontend with Docker Compose:

```bash
docker compose up --build
# or, on older Docker installs:
docker-compose up --build
```

Services:

- Backend: `http://127.0.0.1:8000`
- Frontend: `http://127.0.0.1:5500`

Notes:

- `models/` is mounted into the backend container as read-only.
- `data/` is mounted as a writable volume so generated artifacts stay on the host.
- The first build can take a while because of ML dependencies.

## Runtime Notes

- Target Python version: `3.10`
- Generated data is intentionally not tracked in git:
  - `data/raw/`
  - `data/processed/`
  - `data/ml/`
  - `data/cache/`
  - `data/processed_csv/`
- If you want to test with your own video, pass any local file path to the helper scripts.

Examples:

```bash
python examples/run_sample_inference.py path\to\video.mp4
python test_endpoints.py path\to\video.mp4
```

## Repository Hygiene

The repo is prepared for sharing with teammates:

- caches and notebook checkpoints are ignored
- generated datasets are ignored
- large local sample media is not supposed to live in git
- Docker entrypoints are provided for a reproducible start

## Structure

```text
api/                    FastAPI application
config/                 runtime config and example secrets
docker/                 Dockerfiles
examples/               helper scripts for local checks
inference/              root-level inference import bridge
models/                 committed model artifacts
src/                    core source code
tests/                  placeholder test package
video_integration/      legacy CV runtime modules
web_ui/                 static frontend
run_all.py              local backend + frontend launcher
requirements.runtime.txt runtime dependency set
docker-compose.yml      Docker Compose entrypoint
```

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import List, Optional

# КРИТИЧНО: Устанавливаем переменные окружения ДО импорта torch
# Это предотвращает проблемы с mutex lock на macOS
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
if platform.system() == "Darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    # Отключаем MPS для предотвращения проблем с mutex
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch

# Устанавливаем количество потоков для macOS
if platform.system() == "Darwin":
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

def patch_torch_load():
    # patch for pytorch 2.6+ compatibility
    try:
        from omegaconf import ListConfig
        from omegaconf.base import ContainerMetadata
        import typing
        
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([
                ListConfig,
                ContainerMetadata,
                typing.Any
            ])
    except ImportError:
        pass
    
    original_load = torch.load
    
    def patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)
    
    torch.load = patched_load

patch_torch_load()

import psutil
from tqdm import tqdm

WHISPERX_IMPORT_ERROR: Exception | None = None
try:
    import whisperx
except Exception as exc:  # pragma: no cover - import-time dependency guard
    whisperx = None  # type: ignore[assignment]
    WHISPERX_IMPORT_ERROR = exc

from pipeline.media_utils import (
    MediaSample,
    discover_media,
    filter_samples,
    migrate_transcript,
)
from src.utils.paths import AUDIO_WAV_DIR, TRANSCRIPTS_DIR


def _describe_whisperx_import_error(exc: Exception | None) -> str:
    if isinstance(exc, ModuleNotFoundError) and exc.name == "pkg_resources":
        return (
            "WhisperX import failed because `pkg_resources` is missing. "
            "Install a setuptools build that still provides it, for example "
            "`pip install \"setuptools<81\"`."
        )

    if exc is None:
        return "WhisperX import failed for an unknown reason."

    return f"WhisperX import failed: {type(exc).__name__}: {exc}"


VAD_MODEL_SHA256 = "0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea"
VAD_MIRROR_REPO = "Synthetai/whisperx-vad-segmentation"
VAD_MIRROR_FILE = "pytorch_model.bin"


def _resolve_vad_model_fp() -> str | None:
    """Локальный путь к весам VAD.

    whisperx 3.1.1 качает их с S3-бакета, который больше не отдаёт файл (403),
    поэтому берём тот же checkpoint с зеркала и сверяем sha256 с ожидаемым.
    """
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        print(f"warning: huggingface_hub unavailable, VAD fallback skipped: {exc}", flush=True)
        return None

    try:
        path = hf_hub_download(VAD_MIRROR_REPO, VAD_MIRROR_FILE)
    except Exception as exc:
        print(f"warning: could not fetch VAD weights from mirror: {exc}", flush=True)
        return None

    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != VAD_MODEL_SHA256:
        print(f"warning: VAD weights checksum mismatch ({digest}), falling back to whisperx default", flush=True)
        return None

    return path


def _patch_wav2vec2_processor_sampling_rate() -> None:
    """Добавляет Wav2Vec2Processor.sampling_rate.

    whisperx 3.1.1 читает его при выравнивании, но в актуальных transformers
    значение доступно только через feature_extractor.
    """
    try:
        from transformers import Wav2Vec2Processor
    except Exception as exc:
        print(f"warning: could not patch Wav2Vec2Processor: {exc}", flush=True)
        return

    if hasattr(Wav2Vec2Processor, "sampling_rate"):
        return

    Wav2Vec2Processor.sampling_rate = property(
        lambda self: self.feature_extractor.sampling_rate
    )


def _faster_whisper_compat_options() -> dict:
    """Поля TranscriptionOptions, которых whisperx не передаёт.

    faster-whisper >= 1.0 добавил обязательные поля (multilingual, hotwords и др.),
    из-за чего whisperx.load_model падает с TypeError. Подставляем их значения
    по умолчанию, но только те, которые реально есть в установленной версии.
    """
    try:
        import dataclasses
        from faster_whisper.transcribe import TranscriptionOptions
    except Exception:
        return {}

    defaults = {
        "multilingual": False,
        "max_new_tokens": None,
        "clip_timestamps": "0",
        "hallucination_silence_threshold": None,
        "hotwords": None,
    }
    available = {f.name for f in dataclasses.fields(TranscriptionOptions)}
    return {key: value for key, value in defaults.items() if key in available}


def detect_environment() -> dict:
    is_mac = platform.system() == "Darwin"
    total_memory_gb = psutil.virtual_memory().total / (1024**3)

    if is_mac or total_memory_gb <= 16:
        return {
            "mode": "lightweight",
            "description": "Mac or low-memory environment",
            "default_model": "medium",
            "batch_size": 4,
        }

    return {
        "mode": "server",
        "description": "Server or higher-memory environment",
        "default_model": "medium",
        "batch_size": 16,
    }


def select_samples(
    samples: List[MediaSample],
    transcripts_root: Path,
    file_ids: Optional[List[str]],
    limit: Optional[int],
    force: bool,
) -> List[MediaSample]:
    selected = filter_samples(samples, file_ids)
    tasks: List[MediaSample] = []

    for sample in selected:
        if not sample.wav_path.exists():
            print(f"warning: skipping {sample.file_id}: wav not found ({sample.wav_path})")
            continue

        migrated = migrate_transcript(sample, transcripts_root)

        if not force and not migrated and not sample.needs_transcript(transcripts_root):
            continue

        tasks.append(sample)
        if limit is not None and len(tasks) >= limit:
            break

    return tasks


def transcribe_audio(
    sample: MediaSample,
    transcripts_root: Path,
    model,
    align_model,
    align_metadata,
    language: str,
    device: str,
    batch_size: int,
) -> bool:
    output_path = sample.transcript_path(transcripts_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"WhisperX started: {sample.file_id}", flush=True)
        print(f"processing {sample.file_id}...", flush=True)
        
        # Загрузка аудио с обработкой ошибок
        try:
            audio = whisperx.load_audio(str(sample.wav_path))
            print(f"loading audio: {len(audio) / 16000:.1f} sec", flush=True)
        except Exception as exc:
            print(f"error loading audio for {sample.file_id}: {exc}", flush=True)
            return False
        
        # Транскрибация с обработкой ошибок
        try:
            print(f"transcribing...", flush=True)
            result = model.transcribe(audio, language=language, batch_size=batch_size)
            
            if not result or "segments" not in result or len(result["segments"]) == 0:
                print(f"warning: no segments found for {sample.file_id}", flush=True)
                return False
        except Exception as exc:
            print(f"error transcribing {sample.file_id}: {exc}", flush=True)
            import traceback
            traceback.print_exc()
            return False
        
        # Выравнивание временных меток с обработкой ошибок
        try:
            print(f"aligning timestamps...", flush=True)
            aligned = whisperx.align(
                result["segments"],
                align_model,
                align_metadata,
                audio,
                device=device,
                return_char_alignments=False,
            )
            
            if not aligned or "segments" not in aligned:
                print(f"warning: alignment failed for {sample.file_id}", flush=True)
                return False
        except Exception as exc:
            print(f"error aligning {sample.file_id}: {exc}", flush=True)
            import traceback
            traceback.print_exc()
            return False

        # Сохранение результата
        try:
            with open(output_path, "w", encoding="utf-8") as fp:
                json.dump(aligned, fp, ensure_ascii=False, indent=2)
            print(f"WhisperX finished: {sample.file_id}", flush=True)
            print(f"done: {sample.file_id}", flush=True)
            return True
        except Exception as exc:
            print(f"error saving transcript for {sample.file_id}: {exc}", flush=True)
            return False
            
    except Exception as exc:
        print(f"unexpected error transcribing {sample.file_id}: {exc}", flush=True)
        import traceback
        traceback.print_exc()
        return False


def batch_transcribe(
    input_dir: Path,
    output_dir: Path,
    model_name: Optional[str],
    language: str,
    device: str,
    batch_size: Optional[int],
    compute_type: str,
    file_ids: Optional[List[str]],
    limit: Optional[int],
    force: bool,
) -> None:
    if whisperx is None:
        raise RuntimeError(_describe_whisperx_import_error(WHISPERX_IMPORT_ERROR))

    env = detect_environment()
    model_name = model_name or env["default_model"]
    batch_size = batch_size or env["batch_size"]

    print(f"\n=== whisperx ===")
    print(f"environment: {env['description']}")
    print(f"model: {model_name}")
    print(f"batch size: {batch_size}")
    print(f"device: {device}, compute_type: {compute_type}")
    print("================\n")
    print(f"WhisperX input dir: {input_dir}")
    print(f"WhisperX output dir: {output_dir}")

    samples = discover_media(input_dir)
    if not samples:
        print(f"warning: no files found in {input_dir}")
        return

    tasks = select_samples(samples, output_dir, file_ids, limit, force)
    if not tasks:
        print("all selected files already have transcripts")
        return

    print(f"loading whisper model '{model_name}'...")
    print("(this may take 30-60 seconds on first run)", flush=True)
    
    # Загружаем модель с обработкой ошибок mutex lock
    model = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            model = whisperx.load_model(
                model_name,
                device=device,
                compute_type=compute_type,
                asr_options=_faster_whisper_compat_options(),
                vad_model_fp=_resolve_vad_model_fp(),
            )
            print("model loaded\n", flush=True)
            break
        except Exception as exc:
            error_msg = str(exc)
            if "mutex" in error_msg.lower() or "lock" in error_msg.lower():
                if attempt < max_retries - 1:
                    import time
                    wait_time = (attempt + 1) * 2
                    print(f"mutex lock error, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})", flush=True)
                    time.sleep(wait_time)
                    continue
            print(f"error loading whisper model: {exc}", flush=True)
            import traceback
            traceback.print_exc()
            raise
    
    if model is None:
        raise RuntimeError("Failed to load whisper model after multiple attempts")

    print(f"loading alignment model for language '{language}'...", flush=True)
    _patch_wav2vec2_processor_sampling_rate()
    try:
        align_model, align_metadata = whisperx.load_align_model(
            language_code=language,
            device=device,
        )
        print("alignment model loaded\n", flush=True)
    except Exception as exc:
        print(f"error loading alignment model: {exc}", flush=True)
        import traceback
        traceback.print_exc()
        raise

    print(f"starting transcription of {len(tasks)} files...\n")
    success = 0
    for idx, sample in enumerate(tasks, 1):
        print(f"[{idx}/{len(tasks)}] ", end="", flush=True)
        if transcribe_audio(
            sample,
            transcripts_root=output_dir,
            model=model,
            align_model=align_model,
            align_metadata=align_metadata,
            language=language,
            device=device,
            batch_size=batch_size,
        ):
            success += 1
        print()

    print(f"success: {success}/{len(tasks)} files transcribed")


def main():
    parser = argparse.ArgumentParser(description="transcribe audio with whisperx")
    parser.add_argument("--input-dir", type=str, default=str(AUDIO_WAV_DIR))
    parser.add_argument("--output-dir", type=str, default=str(TRANSCRIPTS_DIR))
    parser.add_argument("--model", type=str, default=None, choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--language", type=str, default="ru")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--compute-type", type=str, default="int8")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--file-ids", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    batch_transcribe(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        model_name=args.model,
        language=args.language,
        device=args.device,
        batch_size=args.batch_size,
        compute_type=args.compute_type,
        file_ids=args.file_ids,
        limit=args.limit,
        force=args.force,
    )


if __name__ == "__main__":
    main()

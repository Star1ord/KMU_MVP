from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable

import joblib

from .paths import resolve_repo_path


class ModelLoadError(RuntimeError):
    def __init__(self, model_name: str, model_path: Path, reason: str, hint: str | None = None):
        self.model_name = model_name
        self.model_path = model_path
        self.reason = reason
        self.hint = hint

        message = f"{model_name} load failed at {model_path}: {reason}"
        if hint:
            message = f"{message}. {hint}"
        super().__init__(message)


def first_existing_path(candidates: Iterable[str | Path]) -> Path | None:
    for candidate in candidates:
        resolved = resolve_repo_path(candidate)
        if resolved.exists():
            return resolved
    return None


def _model_hint(exc: BaseException) -> str | None:
    if isinstance(exc, ModuleNotFoundError):
        missing = exc.name or "unknown dependency"
        if missing == "numpy._core":
            return (
                "The artifact was likely serialized with a different NumPy/scikit-learn stack. "
                "Re-save the model in the current Python 3.10 environment or align numpy, "
                "scikit-learn, and joblib with the training environment."
            )
        if "catboost" in missing.lower():
            return "Install catboost or re-save the model without CatBoost-specific objects."
        return f"Install the missing dependency '{missing}' or re-save the model in this environment."

    if isinstance(exc, (pickle.UnpicklingError, EOFError)):
        return "The serialized model looks corrupted or was saved in an unsupported format. Re-download or re-export it."

    if "invalid load key" in str(exc):
        return "This usually means the file is not a plain pickle. Use joblib-compatible loading or re-export the model."

    return None


def _raise_model_error(
    model_name: str,
    model_path: Path,
    joblib_error: BaseException | None,
    pickle_error: BaseException | None,
) -> None:
    primary_error = joblib_error or pickle_error or RuntimeError("unknown model loading error")
    reason_parts = [f"joblib: {type(primary_error).__name__}: {primary_error}"]

    if pickle_error is not None and pickle_error is not primary_error:
        reason_parts.append(f"pickle: {type(pickle_error).__name__}: {pickle_error}")

    raise ModelLoadError(
        model_name=model_name,
        model_path=model_path,
        reason="; ".join(reason_parts),
        hint=_model_hint(primary_error) or (_model_hint(pickle_error) if pickle_error else None),
    )


def safe_load_serialized_model(path: str | Path, model_name: str = "model") -> Any:
    model_path = resolve_repo_path(path)

    if not model_path.exists():
        raise ModelLoadError(model_name, model_path, "file not found")

    if model_path.stat().st_size <= 0:
        raise ModelLoadError(model_name, model_path, "file is empty")

    joblib_error: BaseException | None = None
    try:
        return joblib.load(model_path)
    except Exception as exc:  # pragma: no cover - broad by design for model IO
        joblib_error = exc

    pickle_error: BaseException | None = None
    try:
        with model_path.open("rb") as handle:
            return pickle.load(handle)
    except Exception as exc:  # pragma: no cover - broad by design for model IO
        pickle_error = exc

    _raise_model_error(model_name, model_path, joblib_error, pickle_error)

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from brad.audio.ffmpeg import get_ffmpeg_version, project_ffmpeg_candidates, resolve_ffmpeg_command
from brad.config import ASR_MODEL_ALIASES, Settings


@dataclass(slots=True)
class DoctorCheck:
    name: str
    status: str
    detail: str


def _detect_compute_mode() -> tuple[str, str]:
    try:
        import ctranslate2  # type: ignore

        count = int(getattr(ctranslate2, "get_cuda_device_count", lambda: 0)())
        if count > 0:
            return "ok", f"GPU available (CUDA devices: {count}); auto mode can use CUDA."
        return "warn", "No CUDA device detected; CPU mode will be used."
    except Exception as exc:  # pragma: no cover - hardware and package dependent
        return "warn", f"CTranslate2 GPU check unavailable ({exc!s}); CPU fallback expected."


def _check_db(settings: Settings) -> DoctorCheck:
    try:
        settings.ensure_dirs()
        with sqlite3.connect(settings.db_path) as conn:
            conn.execute("SELECT 1")
        return DoctorCheck("Database", "ok", f"SQLite writable at {settings.db_path}")
    except Exception as exc:  # pragma: no cover - environment dependent
        return DoctorCheck("Database", "fail", f"Cannot initialize SQLite at {settings.db_path}: {exc}")


def _check_model_paths(settings: Settings) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    for alias in ASR_MODEL_ALIASES:
        model_dir = settings.resolve_asr_model_path(alias)
        if model_dir.exists():
            checks.append(DoctorCheck(f"ASR model ({alias})", "ok", f"Found: {model_dir}"))
        else:
            checks.append(
                DoctorCheck(
                    f"ASR model ({alias})",
                    "warn",
                    f"Missing: {model_dir} (manual download required; no auto-download).",
                )
            )
    return checks


def _check_llm_path(settings: Settings) -> DoctorCheck:
    model_path = settings.llm_default_model
    if model_path is None:
        return DoctorCheck(
            "LLM model",
            "warn",
            "No BRAD_LLM_DEFAULT_MODEL set. Summarization will use extractive fallback.",
        )
    if Path(model_path).exists():
        return DoctorCheck("LLM model", "ok", f"Found: {model_path}")
    return DoctorCheck("LLM model", "warn", f"Configured path missing: {model_path}")


def run_doctor(settings: Settings) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []

    if settings.ffmpeg_path is not None and not settings.ffmpeg_path.exists():
        checks.append(
            DoctorCheck(
                "ffmpeg",
                "fail",
                f"Configured BRAD_FFMPEG_PATH does not exist: {settings.ffmpeg_path}",
            )
        )
        ffmpeg_version = None
    else:
        ffmpeg_version = get_ffmpeg_version(settings.ffmpeg_path)

    if ffmpeg_version:
        checks.append(DoctorCheck("ffmpeg", "ok", ffmpeg_version))
    else:
        local_candidates = ", ".join(str(path) for path in project_ffmpeg_candidates())
        command = resolve_ffmpeg_command(settings.ffmpeg_path)
        checks.append(
            DoctorCheck(
                "ffmpeg",
                "fail",
                "ffmpeg not found. "
                f"Tried command '{command}'. "
                f"You can install ffmpeg on PATH, place it in project (candidates: {local_candidates}), "
                "or set BRAD_FFMPEG_PATH.",
            )
        )

    compute_status, compute_detail = _detect_compute_mode()
    checks.append(DoctorCheck("Compute mode", compute_status, compute_detail))
    checks.append(_check_db(settings))
    checks.extend(_check_model_paths(settings))
    checks.append(_check_llm_path(settings))
    return checks

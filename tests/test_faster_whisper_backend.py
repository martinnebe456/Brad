from __future__ import annotations

from pathlib import Path

import pytest

from brad.asr.faster_whisper_backend import FasterWhisperBackend


def test_model_init_falls_back_from_cuda_to_cpu(monkeypatch, tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    calls: list[tuple[str, str]] = []

    class FakeWhisperModel:
        def __init__(self, model_path: str, *, device: str, compute_type: str, local_files_only: bool):
            calls.append((device, compute_type))
            if device == "cuda":
                raise RuntimeError("cublas missing")

    monkeypatch.setattr(
        FasterWhisperBackend,
        "_whisper_model_class",
        staticmethod(lambda: FakeWhisperModel),
    )

    backend = FasterWhisperBackend(model_dir, device="cuda", compute_type="float16")
    backend._load_model()

    assert calls == [("cuda", "float16"), ("cpu", "int8")]
    assert backend.active_device == "cpu"
    assert backend.active_compute_type == "int8"


def test_model_init_raises_if_all_candidates_fail(monkeypatch, tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    class FakeWhisperModel:
        def __init__(self, model_path: str, *, device: str, compute_type: str, local_files_only: bool):
            raise RuntimeError("init failed")

    monkeypatch.setattr(
        FasterWhisperBackend,
        "_whisper_model_class",
        staticmethod(lambda: FakeWhisperModel),
    )

    backend = FasterWhisperBackend(model_dir, device="cpu", compute_type="int8")

    with pytest.raises(RuntimeError, match="Failed to initialize faster-whisper model"):
        backend._load_model()

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from brad.asr.onnx_backend_stub import ONNXWhisperBackend


def _write_silent_wav(path: Path, sample_rate: int = 16000, seconds: float = 0.1) -> None:
    samples = int(sample_rate * seconds)
    data = np.zeros(samples, dtype=np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(data.tobytes())


def _prepare_dummy_onnx_model_dir(path: Path) -> None:
    path.mkdir()
    (path / "encoder_model.onnx").write_bytes(b"dummy")
    (path / "decoder_model.onnx").write_bytes(b"dummy")


def test_onnx_backend_maps_chunks(monkeypatch, tmp_path: Path) -> None:
    model_dir = tmp_path / "onnx-model"
    _prepare_dummy_onnx_model_dir(model_dir)
    wav_path = tmp_path / "audio.wav"
    _write_silent_wav(wav_path)

    backend = ONNXWhisperBackend(model_dir, chunk_length_s=30.0)
    calls: dict[str, object] = {}

    def fake_pipeline(waveform, **kwargs):
        calls["waveform"] = waveform
        calls["kwargs"] = kwargs
        return {
            "language": "en",
            "chunks": [
                {"timestamp": (0.0, 1.2), "text": " Hello "},
                {"timestamp": (1.2, 2.5), "text": "world"},
            ],
        }

    monkeypatch.setattr(backend, "_load_pipeline", lambda: fake_pipeline)
    result = backend.transcribe(wav_path, language="en")

    assert result.backend == "onnx-whisper"
    assert result.language == "en"
    assert len(result.segments) == 2
    assert result.segments[0].text == "Hello"
    assert result.segments[1].start == 1.2
    assert isinstance(calls["waveform"], np.ndarray)
    kwargs = calls["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["return_timestamps"] is True
    assert kwargs["chunk_length_s"] == 30.0
    assert kwargs["generate_kwargs"] == {"language": "en", "task": "transcribe"}


def test_onnx_backend_falls_back_to_full_text(monkeypatch, tmp_path: Path) -> None:
    model_dir = tmp_path / "onnx-model"
    _prepare_dummy_onnx_model_dir(model_dir)
    wav_path = tmp_path / "audio.wav"
    _write_silent_wav(wav_path)

    backend = ONNXWhisperBackend(model_dir)
    monkeypatch.setattr(
        backend,
        "_load_pipeline",
        lambda: (lambda waveform, **kwargs: {"text": "Only fallback text"}),
    )
    result = backend.transcribe(wav_path, language="auto")

    assert result.language is None
    assert len(result.segments) == 1
    assert result.segments[0].start == 0.0
    assert result.segments[0].end == 0.0
    assert result.segments[0].text == "Only fallback text"


def test_provider_candidates_auto_prefers_cuda_then_cpu(monkeypatch, tmp_path: Path) -> None:
    model_dir = tmp_path / "onnx-model"
    _prepare_dummy_onnx_model_dir(model_dir)
    backend = ONNXWhisperBackend(model_dir, provider="auto")

    monkeypatch.setattr(
        ONNXWhisperBackend,
        "_available_runtime_providers",
        staticmethod(lambda: {"CUDAExecutionProvider", "CPUExecutionProvider"}),
    )
    assert backend._provider_candidates() == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_provider_candidates_fallback_to_cpu(monkeypatch, tmp_path: Path) -> None:
    model_dir = tmp_path / "onnx-model"
    _prepare_dummy_onnx_model_dir(model_dir)
    backend = ONNXWhisperBackend(model_dir, provider="CUDAExecutionProvider")

    monkeypatch.setattr(
        ONNXWhisperBackend,
        "_available_runtime_providers",
        staticmethod(lambda: {"CPUExecutionProvider"}),
    )
    assert backend._provider_candidates() == ["CPUExecutionProvider"]


def test_dependency_install_hint_mentions_missing_module() -> None:
    error = ModuleNotFoundError("No module named 'onnx'", name="onnx")
    hint = ONNXWhisperBackend._dependency_install_hint(error)

    assert "pip install -e '.[onnx]'" in hint
    assert "Missing module: onnx." in hint


def test_dependency_install_hint_without_module_name() -> None:
    hint = ONNXWhisperBackend._dependency_install_hint(RuntimeError("boom"))

    assert "pip install -e '.[onnx]'" in hint
    assert "Missing module:" not in hint

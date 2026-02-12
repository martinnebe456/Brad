from __future__ import annotations

from pathlib import Path

from brad.asr.base import TranscriptionResult


class ONNXWhisperBackendStub:
    """Guided placeholder for a future ONNX Runtime backend."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path

    def transcribe(self, audio_path: Path, language: str | None = None) -> TranscriptionResult:
        # TODO(LP-01): Implement ONNX Runtime + Optimum pipeline and map outputs
        # to TranscriptionResult for interface parity with FasterWhisperBackend.
        raise NotImplementedError(
            "ONNX backend is a guided exercise. See docs/LEARNING_PATH.md -> LP-01."
        )

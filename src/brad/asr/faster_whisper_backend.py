from __future__ import annotations

from pathlib import Path

from brad.asr.base import TranscriptSegment, TranscriptionResult


def _auto_device() -> str:
    try:
        import ctranslate2  # type: ignore

        count = int(getattr(ctranslate2, "get_cuda_device_count", lambda: 0)())
        return "cuda" if count > 0 else "cpu"
    except Exception:
        return "cpu"


class FasterWhisperBackend:
    """Local faster-whisper backend with explicit local model path."""

    def __init__(
        self,
        model_path: Path,
        *,
        device: str = "auto",
        compute_type: str = "int8",
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model path not found: {model_path}. "
                "Download manually and place it under ~/.brad/models/faster-whisper/."
            )
        self.model_path = model_path
        self.device = _auto_device() if device == "auto" else device
        self.compute_type = compute_type
        self.active_device: str | None = None
        self.active_compute_type: str | None = None
        self._model = None

    @staticmethod
    def _whisper_model_class():
        from faster_whisper import WhisperModel  # type: ignore

        return WhisperModel

    def _load_model(self):
        if self._model is None:
            try:
                whisper_model_class = self._whisper_model_class()
            except Exception as exc:
                raise RuntimeError(
                    "faster-whisper is not installed. Install project dependencies first."
                ) from exc

            candidates: list[tuple[str, str]] = [(self.device, self.compute_type)]
            if self.device == "cuda":
                # Keep the app usable when CUDA libraries are unavailable or misconfigured.
                candidates.append(("cpu", "int8"))

            errors: list[str] = []
            for candidate_device, candidate_compute in candidates:
                try:
                    self._model = whisper_model_class(
                        str(self.model_path),
                        device=candidate_device,
                        compute_type=candidate_compute,
                        local_files_only=True,
                    )
                    self.active_device = candidate_device
                    self.active_compute_type = candidate_compute
                    break
                except Exception as exc:  # pragma: no cover - hardware and package dependent
                    errors.append(f"{candidate_device}/{candidate_compute}: {exc}")

            if self._model is None:
                details = " | ".join(errors)
                raise RuntimeError(
                    "Failed to initialize faster-whisper model. "
                    f"Tried: {', '.join(f'{dev}/{comp}' for dev, comp in candidates)}. "
                    f"Errors: {details}"
                )
        return self._model

    def transcribe(self, audio_path: Path, language: str | None = None) -> TranscriptionResult:
        model = self._load_model()
        normalized_language = None if language in (None, "auto") else language
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=normalized_language,
            beam_size=5,
            condition_on_previous_text=False,
            vad_filter=False,
        )

        segments: list[TranscriptSegment] = []
        for item in segments_iter:
            text = item.text.strip()
            if not text:
                continue
            segments.append(
                TranscriptSegment(
                    start=float(item.start),
                    end=float(item.end),
                    text=text,
                )
            )

        return TranscriptionResult(
            segments=segments,
            language=getattr(info, "language", normalized_language),
            backend="faster-whisper",
        )

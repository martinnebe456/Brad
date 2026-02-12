from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np

from brad.asr.base import TranscriptSegment, TranscriptionResult


class ONNXWhisperBackend:
    """ONNX Runtime backend for local Whisper transcription via Optimum."""

    def __init__(
        self,
        model_path: Path,
        *,
        provider: str = "auto",
        use_cache: bool = False,
        chunk_length_s: float | None = None,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX model path not found: {model_path}. "
                "Export/download the ONNX model manually under ./models/onnx-whisper/."
            )
        self._ensure_onnx_files_present(model_path)
        self.model_path = model_path
        self.provider_preference = provider
        self.active_provider: str | None = None
        self.use_cache = use_cache
        self.chunk_length_s = chunk_length_s
        self._pipeline = None

    @staticmethod
    def _ensure_onnx_files_present(model_path: Path) -> None:
        required = ("encoder_model.onnx", "decoder_model.onnx")
        missing = [name for name in required if not (model_path / name).exists()]
        if not missing:
            return

        model_alias = model_path.name
        suggested_model_id = f"openai/whisper-{model_alias}"
        missing_str = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing ONNX files in {model_path}: {missing_str}. "
            "This folder must contain an exported ONNX Whisper model. "
            f"Example export command:\n"
            f"python -m optimum.commands.optimum_cli export onnx -m {suggested_model_id} "
            f"--task automatic-speech-recognition --device cpu {model_path}"
        )

    @staticmethod
    def _normalize_provider_name(provider: str) -> str:
        normalized = provider.strip()
        lowered = normalized.lower()
        if lowered in {"cuda", "cudaexecutionprovider"}:
            return "CUDAExecutionProvider"
        if lowered in {"cpu", "cpuexecutionprovider"}:
            return "CPUExecutionProvider"
        return normalized

    @staticmethod
    def _available_runtime_providers() -> set[str] | None:
        try:
            import onnxruntime  # type: ignore

            providers = onnxruntime.get_available_providers()
        except Exception:
            return None
        return {str(provider) for provider in providers}

    def _provider_candidates(self) -> list[str]:
        requested: list[str]
        preference = self.provider_preference.strip().lower()
        if preference in {"", "auto"}:
            requested = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            requested = [
                self._normalize_provider_name(item)
                for item in self.provider_preference.split(",")
                if item.strip()
            ]
            if "CPUExecutionProvider" not in requested:
                requested.append("CPUExecutionProvider")

        available = self._available_runtime_providers()
        if available is None:
            # If runtime providers are unknown, attempt requested order and let loading decide.
            return list(dict.fromkeys(requested))

        filtered = [provider for provider in requested if provider in available]
        if not filtered and "CPUExecutionProvider" in available:
            filtered = ["CPUExecutionProvider"]
        return list(dict.fromkeys(filtered or requested))

    @staticmethod
    def _dependency_install_hint(exc: Exception) -> str:
        message = (
            "ONNX dependencies are missing or incompatible. "
            "Install with: pip install -e '.[onnx]'"
        )
        if isinstance(exc, ImportError) and "_attention_scale" in str(exc):
            return (
                f"{message} Detected torch/optimum mismatch. "
                "Run: pip uninstall -y optimum && "
                "pip install -U \"optimum-onnx[onnxruntime]\"."
            )
        missing_name = getattr(exc, "name", None)
        if isinstance(missing_name, str) and missing_name:
            message = f"{message} Missing module: {missing_name}."
        return message

    def _load_pipeline(self):
        if self._pipeline is None:
            try:
                from optimum.onnxruntime import ORTModelForSpeechSeq2Seq  # type: ignore
                from transformers import AutoProcessor, pipeline  # type: ignore
            except Exception as exc:
                raise RuntimeError(self._dependency_install_hint(exc)) from exc

            candidates = self._provider_candidates()
            errors: list[str] = []
            for provider in candidates:
                try:
                    model = ORTModelForSpeechSeq2Seq.from_pretrained(
                        str(self.model_path),
                        provider=provider,
                        local_files_only=True,
                        use_cache=self.use_cache,
                    )
                    processor = AutoProcessor.from_pretrained(
                        str(self.model_path),
                        local_files_only=True,
                    )
                    self._pipeline = pipeline(
                        "automatic-speech-recognition",
                        model=model,
                        tokenizer=processor.tokenizer,
                        feature_extractor=processor.feature_extractor,
                    )
                    self.active_provider = provider
                    break
                except Exception as exc:
                    errors.append(f"{provider}: {exc}")

            if self._pipeline is None:
                details = " | ".join(errors) if errors else "no provider attempts executed"
                raise RuntimeError(
                    "Failed to initialize ONNX Whisper backend. "
                    f"Tried providers: {', '.join(candidates)}. Errors: {details}"
                )
        return self._pipeline

    @staticmethod
    def _read_wav_mono_16k(audio_path: Path) -> np.ndarray:
        """Read WAV audio into float32 waveform expected by HF ASR pipeline."""

        with wave.open(str(audio_path), "rb") as handle:
            channels = handle.getnchannels()
            sample_rate = handle.getframerate()
            sample_width = handle.getsampwidth()
            frames = handle.readframes(handle.getnframes())

        if channels != 1:
            raise ValueError(f"ONNX backend expects mono WAV input, got channels={channels}")
        if sample_rate != 16_000:
            raise ValueError(f"ONNX backend expects 16kHz WAV input, got {sample_rate}Hz")

        if sample_width == 1:
            waveform = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
            return (waveform - 128.0) / 128.0
        if sample_width == 2:
            waveform = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            return waveform / 32768.0
        if sample_width == 4:
            waveform = np.frombuffer(frames, dtype=np.int32).astype(np.float32)
            return waveform / 2147483648.0

        raise ValueError(f"Unsupported WAV sample width: {sample_width * 8} bits")

    @staticmethod
    def _normalize_language(language: str | None) -> str | None:
        if language is None:
            return None
        normalized = language.strip().lower()
        return None if normalized in {"", "auto"} else normalized

    @staticmethod
    def _parse_timestamp(raw_timestamp: Any) -> tuple[float, float] | None:
        if not isinstance(raw_timestamp, (list, tuple)) or len(raw_timestamp) != 2:
            return None

        start_raw, end_raw = raw_timestamp
        if start_raw is None:
            start_raw = 0.0
        if end_raw is None:
            end_raw = start_raw

        start = float(start_raw)
        end = float(end_raw)
        if end < start:
            end = start
        return start, end

    def transcribe(self, audio_path: Path, language: str | None = None) -> TranscriptionResult:
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        asr_pipeline = self._load_pipeline()
        waveform = self._read_wav_mono_16k(audio_path)
        normalized_language = self._normalize_language(language)

        inference_kwargs: dict[str, Any] = {"return_timestamps": True}
        if self.chunk_length_s is not None:
            inference_kwargs["chunk_length_s"] = self.chunk_length_s
        if normalized_language is not None:
            inference_kwargs["generate_kwargs"] = {
                "language": normalized_language,
                "task": "transcribe",
            }

        raw_output = asr_pipeline(waveform, **inference_kwargs)

        segments: list[TranscriptSegment] = []
        if isinstance(raw_output, dict):
            raw_chunks = raw_output.get("chunks")
            if isinstance(raw_chunks, list):
                for chunk in raw_chunks:
                    if not isinstance(chunk, dict):
                        continue
                    text = str(chunk.get("text", "")).strip()
                    if not text:
                        continue
                    parsed = self._parse_timestamp(chunk.get("timestamp"))
                    if parsed is None:
                        continue
                    start, end = parsed
                    segments.append(TranscriptSegment(start=start, end=end, text=text))

        if not segments and isinstance(raw_output, dict):
            full_text = str(raw_output.get("text", "")).strip()
            if full_text:
                segments.append(TranscriptSegment(start=0.0, end=0.0, text=full_text))

        detected_language: str | None = None
        if isinstance(raw_output, dict) and isinstance(raw_output.get("language"), str):
            detected_language = str(raw_output["language"]).strip() or None

        return TranscriptionResult(
            segments=segments,
            language=detected_language or normalized_language,
            backend="onnx-whisper",
        )


class ONNXWhisperBackendStub(ONNXWhisperBackend):
    """Backward compatible alias for previous stub class name."""

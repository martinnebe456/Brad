from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from brad.asr.base import TranscriptSegment
from brad.asr.faster_whisper_backend import FasterWhisperBackend
from brad.asr.onnx_backend_stub import ONNXWhisperBackend
from brad.audio.chunking import build_chunks_from_vad
from brad.audio.ffmpeg import convert_to_mono_16k_wav, extract_wav_segment
from brad.audio.vad import detect_speech_spans
from brad.config import ASR_BACKENDS, Settings, get_settings
from brad.export import json as json_export
from brad.export.md import render_markdown
from brad.export.srt import to_srt
from brad.nlp.summarizer import MeetingSummarizer, SummaryResult, segments_to_text
from brad.storage.db import BradDB
from brad.storage.models import SearchHit


@dataclass(slots=True)
class TranscriptionOutcome:
    meeting_id: int
    language: str
    segment_count: int
    export_paths: dict[str, Path]


@dataclass(slots=True)
class SummaryOutcome:
    summary: SummaryResult
    meeting_id: int | None


class BradService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        self.db = BradDB(self.settings.db_path)
        self.db.initialize()
        self.summarizer = MeetingSummarizer()

    def _temp_run_dir(self) -> Path:
        run_dir = self.settings.temp_dir / f"run_{uuid.uuid4().hex[:8]}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _resolve_llm_model(self, llm_model: Path | None) -> Path | None:
        if llm_model is not None:
            return llm_model
        return self.settings.llm_default_model

    def transcribe_file(
        self,
        audio_file: Path,
        *,
        backend_name: str,
        model_name: str,
        language: str,
        use_vad: bool,
    ) -> TranscriptionOutcome:
        if not audio_file.exists():
            raise FileNotFoundError(f"Input audio file does not exist: {audio_file}")

        normalized_backend = backend_name.strip().lower()
        if normalized_backend not in ASR_BACKENDS:
            raise ValueError(
                f"Unsupported backend '{backend_name}'. Allowed: {', '.join(ASR_BACKENDS)}"
            )

        if normalized_backend == "faster-whisper":
            model_path = self.settings.resolve_asr_model_path(model_name)
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model path not found: {model_path}\n"
                    "Brad will not auto-download models. Download manually and rerun."
                )
            backend = FasterWhisperBackend(
                model_path=model_path,
                compute_type=self.settings.default_compute_type,
            )
        else:
            model_path = self.settings.resolve_onnx_model_path(model_name)
            if not model_path.exists():
                raise FileNotFoundError(
                    f"ONNX model path not found: {model_path}\n"
                    "Export/download the ONNX model manually and rerun."
                )
            backend = ONNXWhisperBackend(
                model_path=model_path,
                provider=self.settings.onnx_provider,
                use_cache=False,
            )

        run_dir = self._temp_run_dir()
        prepared_wav = run_dir / "input_16k.wav"
        detected_language: str | None = None
        collected_segments: list[TranscriptSegment] = []

        try:
            convert_to_mono_16k_wav(
                audio_file,
                prepared_wav,
                ffmpeg_path=self.settings.ffmpeg_path,
            )
            if use_vad:
                spans = detect_speech_spans(prepared_wav)
                chunks = build_chunks_from_vad(spans)
                if not chunks:
                    result = backend.transcribe(prepared_wav, language=language)
                    detected_language = result.language
                    collected_segments = result.segments
                else:
                    for index, chunk in enumerate(chunks):
                        chunk_wav = run_dir / f"chunk_{index:04d}.wav"
                        extract_wav_segment(
                            prepared_wav,
                            chunk_wav,
                            chunk.start,
                            chunk.end,
                            ffmpeg_path=self.settings.ffmpeg_path,
                        )
                        result = backend.transcribe(chunk_wav, language=language)
                        if detected_language is None and result.language:
                            detected_language = result.language
                        for segment in result.segments:
                            collected_segments.append(
                                TranscriptSegment(
                                    start=segment.start + chunk.start,
                                    end=segment.end + chunk.start,
                                    text=segment.text,
                                )
                            )
            else:
                result = backend.transcribe(prepared_wav, language=language)
                detected_language = result.language
                collected_segments = result.segments
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

        collected_segments.sort(key=lambda item: item.start)
        duration_seconds = max((item.end for item in collected_segments), default=0.0)

        meeting_id = self.db.create_meeting(
            source_path=str(audio_file.resolve()),
            language=detected_language or (language if language != "auto" else "unknown"),
            model_name=f"{normalized_backend}:{model_name}",
            duration_seconds=duration_seconds,
        )
        self.db.add_segments(meeting_id, collected_segments)
        export_paths = self.export_all_formats(meeting_id)

        return TranscriptionOutcome(
            meeting_id=meeting_id,
            language=detected_language or "unknown",
            segment_count=len(collected_segments),
            export_paths=export_paths,
        )

    def summarize_target(
        self,
        meeting_or_path: str,
        *,
        template_name: str,
        llm_model: Path | None,
    ) -> SummaryOutcome:
        model_path = self._resolve_llm_model(llm_model)
        meeting_id: int | None = None

        if meeting_or_path.isdigit():
            possible_id = int(meeting_or_path)
            meeting = self.db.get_meeting(possible_id)
            if meeting is not None:
                meeting_id = possible_id
                segments = self.db.get_segments(meeting_id)
                transcript_text = segments_to_text(segments)
                summary = self.summarizer.summarize_text(
                    transcript_text,
                    template_name=template_name,
                    llm_model=model_path,
                )
                self.db.add_summary(
                    meeting_id=meeting_id,
                    template_name=summary.template_name,
                    method=summary.method,
                    text=summary.text,
                    llm_model=summary.llm_model,
                )
                return SummaryOutcome(summary=summary, meeting_id=meeting_id)

        transcript_path = Path(meeting_or_path)
        if not transcript_path.exists():
            raise FileNotFoundError(
                f"Target is neither a known meeting_id nor a transcript path: {meeting_or_path}"
            )

        summary = self.summarizer.summarize_path(
            transcript_path,
            template_name=template_name,
            llm_model=model_path,
        )
        return SummaryOutcome(summary=summary, meeting_id=None)

    def export_meeting(self, meeting_id: int, export_format: str) -> Path:
        fmt = export_format.lower().strip()
        if fmt not in {"md", "srt", "json"}:
            raise ValueError("Unsupported export format. Use: md, srt, json")

        meeting = self.db.get_meeting(meeting_id)
        if meeting is None:
            raise ValueError(f"Meeting not found: {meeting_id}")

        segments = self.db.get_segments(meeting_id)
        summary = self.db.get_latest_summary(meeting_id)

        export_dir = self.settings.exports_dir / f"meeting_{meeting_id}"
        export_dir.mkdir(parents=True, exist_ok=True)
        output_path = export_dir / f"meeting_{meeting_id}.{fmt}"

        if fmt == "md":
            content = render_markdown(meeting, segments, summary)
        elif fmt == "srt":
            content = to_srt(segments)
        else:
            payload = json_export.build_payload(meeting, segments, summary)
            content = json_export.dumps_payload(payload)

        output_path.write_text(content, encoding="utf-8")
        self.db.add_export(meeting_id=meeting_id, export_format=fmt, path=str(output_path))
        return output_path

    def export_all_formats(self, meeting_id: int) -> dict[str, Path]:
        return {fmt: self.export_meeting(meeting_id, fmt) for fmt in ("md", "srt", "json")}

    def search(
        self,
        query: str,
        *,
        meeting_id: int | None = None,
        limit: int = 25,
    ) -> list[SearchHit]:
        return self.db.search_segments(query, meeting_id=meeting_id, limit=limit)

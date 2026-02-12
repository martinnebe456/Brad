from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(slots=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass(slots=True)
class TranscriptionResult:
    segments: list[TranscriptSegment]
    language: str | None
    backend: str


class ASRBackend(Protocol):
    def transcribe(self, audio_path: Path, language: str | None = None) -> TranscriptionResult:
        """Transcribe an audio file into timestamped segments."""

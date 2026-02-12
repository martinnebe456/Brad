from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class MeetingRecord:
    id: int
    created_at: str
    source_path: str
    language: str
    model_name: str
    duration_seconds: float


@dataclass(slots=True)
class SegmentRecord:
    id: int
    meeting_id: int
    start: float
    end: float
    text: str


@dataclass(slots=True)
class SummaryRecord:
    id: int
    meeting_id: int
    created_at: str
    template_name: str
    method: str
    llm_model: str | None
    text: str


@dataclass(slots=True)
class ExportRecord:
    id: int
    meeting_id: int
    created_at: str
    export_format: str
    path: str


@dataclass(slots=True)
class SearchHit:
    segment_id: int
    meeting_id: int
    start: float
    end: float
    text: str
    snippet: str

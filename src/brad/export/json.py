from __future__ import annotations

import json
from typing import Any

from brad.storage.models import MeetingRecord, SegmentRecord, SummaryRecord


def build_payload(
    meeting: MeetingRecord,
    segments: list[SegmentRecord],
    summary: SummaryRecord | None,
) -> dict[str, Any]:
    return {
        "meeting": {
            "id": meeting.id,
            "created_at": meeting.created_at,
            "source_path": meeting.source_path,
            "language": meeting.language,
            "model_name": meeting.model_name,
            "duration_seconds": meeting.duration_seconds,
        },
        "segments": [
            {"id": item.id, "start": item.start, "end": item.end, "text": item.text}
            for item in segments
        ],
        "summary": (
            {
                "id": summary.id,
                "template_name": summary.template_name,
                "method": summary.method,
                "llm_model": summary.llm_model,
                "text": summary.text,
            }
            if summary is not None
            else None
        ),
    }


def dumps_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)

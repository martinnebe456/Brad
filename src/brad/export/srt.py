from __future__ import annotations

from typing import Protocol, Sequence


class SRTSegment(Protocol):
    start: float
    end: float
    text: str


def format_timestamp(seconds: float) -> str:
    total_ms = int(round(max(seconds, 0.0) * 1000.0))
    hours = total_ms // 3_600_000
    total_ms %= 3_600_000
    minutes = total_ms // 60_000
    total_ms %= 60_000
    secs = total_ms // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def to_srt(segments: Sequence[SRTSegment]) -> str:
    """Build an SRT string from timestamped segments.

    TODO(LP-06): add configurable line wrapping to improve subtitle readability.
    """

    lines: list[str] = []
    for index, segment in enumerate(segments, start=1):
        lines.append(str(index))
        lines.append(f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}")
        lines.append(segment.text.strip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"

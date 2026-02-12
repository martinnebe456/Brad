from __future__ import annotations

from brad.storage.models import MeetingRecord, SegmentRecord, SummaryRecord


def render_markdown(
    meeting: MeetingRecord,
    segments: list[SegmentRecord],
    summary: SummaryRecord | None,
) -> str:
    lines: list[str] = []
    lines.append(f"# Brad transcript export: meeting {meeting.id}")
    lines.append("")
    lines.append(f"- Created at: {meeting.created_at}")
    lines.append(f"- Source file: `{meeting.source_path}`")
    lines.append(f"- Language: `{meeting.language}`")
    lines.append(f"- ASR model: `{meeting.model_name}`")
    lines.append(f"- Duration: `{meeting.duration_seconds:.2f}s`")
    lines.append("")

    if summary:
        lines.append("## Summary")
        lines.append("")
        lines.append(summary.text.strip())
        lines.append("")

    lines.append("## Transcript")
    lines.append("")
    for segment in segments:
        lines.append(f"- [{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
    lines.append("")
    return "\n".join(lines)

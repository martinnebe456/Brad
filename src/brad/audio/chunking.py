from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class TimeSpan:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def merge_speech_spans(
    spans: Iterable[TimeSpan],
    max_gap_s: float = 0.75,
    min_duration_s: float = 0.4,
) -> list[TimeSpan]:
    """Merge VAD spans that are close in time and remove tiny artifacts.

    TODO(LP-05): experiment with adaptive gap thresholds based on speaking rate.
    """

    ordered = sorted(spans, key=lambda item: item.start)
    if not ordered:
        return []

    merged: list[TimeSpan] = []
    current = ordered[0]
    for nxt in ordered[1:]:
        if nxt.start <= current.end + max_gap_s:
            current = TimeSpan(start=current.start, end=max(current.end, nxt.end))
            continue
        if current.duration >= min_duration_s:
            merged.append(current)
        current = nxt

    if current.duration >= min_duration_s:
        merged.append(current)
    return merged


def split_long_spans(spans: Iterable[TimeSpan], max_duration_s: float = 30.0) -> list[TimeSpan]:
    """Split long spans into fixed-size chunks for stable ASR latency."""

    output: list[TimeSpan] = []
    for span in spans:
        start = span.start
        while start < span.end:
            end = min(start + max_duration_s, span.end)
            output.append(TimeSpan(start=start, end=end))
            start = end
    return output


def build_chunks_from_vad(
    spans: Iterable[TimeSpan],
    max_gap_s: float = 0.75,
    min_duration_s: float = 0.4,
    max_duration_s: float = 30.0,
) -> list[TimeSpan]:
    """Turn raw VAD spans into ASR-friendly chunks."""

    merged = merge_speech_spans(spans, max_gap_s=max_gap_s, min_duration_s=min_duration_s)
    return split_long_spans(merged, max_duration_s=max_duration_s)

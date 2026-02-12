from brad.asr.base import TranscriptSegment
from brad.export.srt import format_timestamp, to_srt


def test_format_timestamp() -> None:
    assert format_timestamp(0.0) == "00:00:00,000"
    assert format_timestamp(1.234) == "00:00:01,234"
    assert format_timestamp(3723.045) == "01:02:03,045"


def test_to_srt() -> None:
    segments = [
        TranscriptSegment(start=0.0, end=1.0, text="Hello world."),
        TranscriptSegment(start=1.2, end=2.5, text="Second line."),
    ]
    srt = to_srt(segments)
    assert "1" in srt
    assert "00:00:00,000 --> 00:00:01,000" in srt
    assert "Hello world." in srt
    assert "00:00:01,200 --> 00:00:02,500" in srt

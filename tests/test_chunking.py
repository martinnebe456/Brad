from brad.audio.chunking import TimeSpan, build_chunks_from_vad, merge_speech_spans


def test_merge_speech_spans() -> None:
    spans = [
        TimeSpan(0.0, 1.0),
        TimeSpan(1.2, 2.0),
        TimeSpan(5.0, 5.1),  # removed by min_duration_s
        TimeSpan(6.0, 8.0),
    ]
    merged = merge_speech_spans(spans, max_gap_s=0.3, min_duration_s=0.4)
    assert merged == [TimeSpan(0.0, 2.0), TimeSpan(6.0, 8.0)]


def test_build_chunks_from_vad_split_long() -> None:
    spans = [TimeSpan(0.0, 65.0)]
    chunks = build_chunks_from_vad(spans, max_duration_s=30.0)
    assert chunks == [TimeSpan(0.0, 30.0), TimeSpan(30.0, 60.0), TimeSpan(60.0, 65.0)]

from __future__ import annotations

from pathlib import Path

from brad.audio.chunking import TimeSpan


def silero_vad_available() -> bool:
    try:
        import silero_vad  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def detect_speech_spans(
    wav_path: Path,
    sampling_rate: int = 16000,
    threshold: float = 0.5,
) -> list[TimeSpan]:
    """Run Silero VAD and return detected speech spans in seconds."""

    try:
        from silero_vad import get_speech_timestamps, load_silero_vad, read_audio  # type: ignore
    except Exception as exc:
        raise RuntimeError("Silero VAD is not installed. Install with: pip install -e '.[vad]'") from exc

    model = load_silero_vad()
    waveform = read_audio(str(wav_path), sampling_rate=sampling_rate)
    timestamps = get_speech_timestamps(
        waveform,
        model,
        sampling_rate=sampling_rate,
        threshold=threshold,
    )

    spans: list[TimeSpan] = []
    for stamp in timestamps:
        start = float(stamp["start"]) / sampling_rate
        end = float(stamp["end"]) / sampling_rate
        spans.append(TimeSpan(start=start, end=end))
    return spans

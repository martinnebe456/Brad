from __future__ import annotations

import subprocess
from pathlib import Path


class FfmpegError(RuntimeError):
    """Raised when ffmpeg or ffprobe commands fail."""


def get_ffmpeg_version() -> str | None:
    try:
        completed = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if completed.returncode != 0:
        return None
    first_line = completed.stdout.splitlines()[0] if completed.stdout else "ffmpeg detected"
    return first_line.strip()


def _run_ffmpeg(args: list[str]) -> None:
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise FfmpegError("ffmpeg is not installed or not available on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "Unknown ffmpeg error."
        raise FfmpegError(stderr) from exc


def convert_to_mono_16k_wav(input_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            "-y",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-vn",
            str(output_path),
        ]
    )
    return output_path


def extract_wav_segment(input_wav: Path, output_wav: Path, start_s: float, end_s: float) -> Path:
    if end_s <= start_s:
        raise ValueError("Chunk end time must be greater than start time.")
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            "-y",
            "-ss",
            f"{start_s:.3f}",
            "-to",
            f"{end_s:.3f}",
            "-i",
            str(input_wav),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output_wav),
        ]
    )
    return output_wav

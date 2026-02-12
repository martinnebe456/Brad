from __future__ import annotations

import os
import subprocess
from pathlib import Path


class FfmpegError(RuntimeError):
    """Raised when ffmpeg or ffprobe commands fail."""


def _ffmpeg_executable_name() -> str:
    return "ffmpeg.exe" if os.name == "nt" else "ffmpeg"


def project_ffmpeg_candidates() -> list[Path]:
    exe_name = _ffmpeg_executable_name()
    cwd = Path.cwd()
    return [
        cwd / "tools" / "ffmpeg" / "bin" / exe_name,
        cwd / "ffmpeg" / "bin" / exe_name,
        cwd / "bin" / exe_name,
    ]


def resolve_ffmpeg_command(ffmpeg_path: Path | None = None) -> str:
    if ffmpeg_path is not None:
        return str(Path(ffmpeg_path).expanduser())

    for candidate in project_ffmpeg_candidates():
        if candidate.exists():
            return str(candidate)

    return _ffmpeg_executable_name()


def get_ffmpeg_version(ffmpeg_path: Path | None = None) -> str | None:
    command = resolve_ffmpeg_command(ffmpeg_path)
    try:
        completed = subprocess.run(
            [command, "-version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if completed.returncode != 0:
        return None
    first_line = completed.stdout.splitlines()[0] if completed.stdout else "ffmpeg detected"
    return f"{first_line.strip()} (command: {command})"


def _run_ffmpeg(args: list[str], ffmpeg_path: Path | None = None) -> None:
    command = resolve_ffmpeg_command(ffmpeg_path)
    try:
        subprocess.run(
            [command, "-hide_banner", "-loglevel", "error", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise FfmpegError(
            f"ffmpeg not found (command: {command}). "
            "Install ffmpeg, place it under ./tools/ffmpeg/bin/, or set BRAD_FFMPEG_PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "Unknown ffmpeg error."
        raise FfmpegError(stderr) from exc


def convert_to_mono_16k_wav(
    input_path: Path,
    output_path: Path,
    ffmpeg_path: Path | None = None,
) -> Path:
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
        ],
        ffmpeg_path=ffmpeg_path,
    )
    return output_path


def extract_wav_segment(
    input_wav: Path,
    output_wav: Path,
    start_s: float,
    end_s: float,
    ffmpeg_path: Path | None = None,
) -> Path:
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
        ],
        ffmpeg_path=ffmpeg_path,
    )
    return output_wav

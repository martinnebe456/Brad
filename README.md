# Brad - Your local AI meeting assistant

Brad is a learning-first, local-only meeting assistant implemented from scratch in Python.

It follows a practical workflow:
transcribe -> summarize -> search -> export

## Local-only promise

- No cloud inference.
- No telemetry, no analytics, no tracking.
- No automatic uploads of audio or transcripts.
- Runtime is offline-safe by default: Brad never downloads models automatically.

## Features (MVP)

- Local ASR with faster-whisper (CTranslate2), CPU/GPU capable.
- ffmpeg preprocessing to mono 16kHz WAV.
- Optional Silero VAD chunking.
- Local summarization with llama-cpp-python (GGUF), with extractive fallback when no LLM is configured.
- SQLite storage with FTS5 transcript search.
- Exports: Markdown, JSON, SRT.
- Typer CLI, native desktop UI, and optional local web UI.

## Project status

This repo is intentionally learning-first:

- Core flow works end to end.
- Advanced features are left as guided TODOs in code and `docs/LEARNING_PATH.md`.

## Prerequisites

1. Python 3.11+
2. ffmpeg available either:
   - on PATH, or
   - as a project-local binary in one of:
     - `./tools/ffmpeg/bin/ffmpeg` (or `ffmpeg.exe` on Windows)
     - `./ffmpeg/bin/ffmpeg`
     - `./bin/ffmpeg`

Useful references:
- https://ffmpeg.org/download.html
- https://github.com/SYSTRAN/faster-whisper
- https://github.com/snakers4/silero-vad
- https://github.com/abetlen/llama-cpp-python

## Installation

### Option A: uv

```bash
uv venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
uv pip install -e ".[dev,vad]"
# Optional local LLM support:
uv pip install -e ".[llm]"
```

### Option B: pip

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev,vad]"
# Optional local LLM support:
pip install -e ".[llm]"
```

## Model setup (manual, explicit)

Brad will not auto-download any model. You must place models yourself.

Default model root:

- `./models` (relative to current working directory, typically repo root)

You can override with env vars:

- `BRAD_DATA_DIR`
- `BRAD_MODELS_DIR`
- `BRAD_FFMPEG_PATH`

### Alternate model locations (optional)

If you prefer a different model cache location, set `BRAD_MODELS_DIR`.

macOS/Linux:

```bash
export BRAD_MODELS_DIR="$HOME/.brad/models"
# optional: keep DB/exports local too
export BRAD_DATA_DIR="$(pwd)/data"
```

Windows PowerShell:

```powershell
$env:BRAD_MODELS_DIR="$HOME\.brad\models"
# optional: keep DB/exports local too
$env:BRAD_DATA_DIR="$PWD\data"
# optional: explicit ffmpeg binary
# $env:BRAD_FFMPEG_PATH="$PWD\tools\ffmpeg\bin\ffmpeg.exe"
```

### ffmpeg setup (system or project-local)

Brad checks ffmpeg in this order:

1. `BRAD_FFMPEG_PATH` (if set)
2. `./tools/ffmpeg/bin/ffmpeg(.exe)`
3. `./ffmpeg/bin/ffmpeg(.exe)`
4. `./bin/ffmpeg(.exe)`
5. `ffmpeg` on PATH

If you want ffmpeg inside the repo, put the binary under `./tools/ffmpeg/bin/`.

### ASR model folders

Expected local paths for faster-whisper (CTranslate2):

- `./models/faster-whisper/small`
- `./models/faster-whisper/medium`
- `./models/faster-whisper/large-v3`

Suggested model sources:

- https://huggingface.co/Systran/faster-whisper-small
- https://huggingface.co/Systran/faster-whisper-medium
- https://huggingface.co/Systran/faster-whisper-large-v3

If you use `huggingface-cli`, that is still explicit manual action:

```bash
hf download Systran/faster-whisper-small --local-dir ./models/faster-whisper/small
hf download Systran/faster-whisper-medium --local-dir ./models/faster-whisper/medium
hf download Systran/faster-whisper-large-v3 --local-dir ./models/faster-whisper/large-v3
```

### Local GGUF model for summarization (optional)

Example location:

- `./models/llm/your-model.gguf`

Then either pass `--llm-model` at runtime or set:

```bash
export BRAD_LLM_DEFAULT_MODEL=./models/llm/your-model.gguf
# PowerShell:
# $env:BRAD_LLM_DEFAULT_MODEL="$PWD\models\llm\your-model.gguf"
```

## Quickstart

Run health checks:

```bash
brad doctor
```

Transcribe:

```bash
brad transcribe ./samples/sample-3.mp3 --model small --language auto --vad off
```

Summarize:

```bash
brad summarize 1 --template general
brad summarize 1 --template engineering --llm-model ./models/llm/your-model.gguf
```

Export:

```bash
brad export 1 --format md
brad export 1 --format srt
brad export 1 --format json
```

Search:

```bash
brad search "deadline risk"
brad search "budget" --meeting 1
```

Run desktop UI:

```bash
brad ui
```

Optional web UI (Gradio):

```bash
brad ui --mode web
```

## CPU and GPU examples

CPU-focused run:

```bash
export BRAD_DEFAULT_COMPUTE_TYPE=int8
brad transcribe ./meeting.mp3 --model small --language en --vad on
```

GPU-focused run (CUDA machine):

```bash
export BRAD_DEFAULT_COMPUTE_TYPE=float16
export CUDA_VISIBLE_DEVICES=0
brad transcribe ./meeting.mp3 --model medium --language auto --vad on
```

Windows PowerShell equivalent:

```powershell
$env:BRAD_DEFAULT_COMPUTE_TYPE="float16"
$env:CUDA_VISIBLE_DEVICES="0"
brad transcribe .\meeting.mp3 --model medium --language auto --vad on
```

## Data layout

By default Brad stores runtime data under `~/.brad`:

- `~/.brad/brad.db` (SQLite database)
- `~/.brad/exports/` (generated markdown/json/srt exports)
- `~/.brad/tmp/` (temporary audio files)

By default Brad stores models under `./models/`:

- `./models/` (manually downloaded ASR/LLM model files)

Optional project-local ffmpeg location:

- `./tools/ffmpeg/bin/ffmpeg(.exe)`

## Architecture and learning docs

- `docs/ARCHITECTURE.md`
- `docs/LEARNING_PATH.md`
- `docs/prompts/*.md`

Prompt templates are plain markdown files in `docs/prompts/`. Edit them directly to change summary behavior for `general`, `sales`, and `engineering`.

## Contributing

Contributors should read `AGENTS.md` first for the project playbook and active next steps.

Then see `CONTRIBUTING.md` for setup, lint, tests, and PR workflow.

## Security

See `SECURITY.md` for threat model and reporting guidance.

## Roadmap

1. Harden transcription pipeline and chunking defaults.
2. Add better SRT formatting and post-processing.
3. Add diarization exercise.
4. Add optional semantic search with local embeddings.

## Changelog

See `CHANGELOG.md`.

## License

GPL-3.0 license. See `LICENSE`.

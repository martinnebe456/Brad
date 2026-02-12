# Changelog

## 2026-02-12

- Fixed ONNX optional dependency set:
  - Added `onnx` to `.[onnx]` extras in `pyproject.toml` (required by Optimum ONNX runtime imports).
- Improved ONNX backend startup error message:
  - Reports missing/incompatible dependency state.
  - Includes missing module name when available (for example `onnx`).
- Initial repository scaffold.
- Implemented local-only MVP pipeline:
  - ffmpeg preprocessing
  - faster-whisper transcription
  - optional Silero VAD chunking
  - SQLite + FTS5 storage
  - local summarization (llama-cpp optional, extractive fallback)
  - CLI + Gradio UI
- Added docs, prompts, tests, CI, and AGENTS playbook.
- Added LP-01 baseline implementation:
  - ONNX Whisper backend via Optimum + ONNX Runtime
  - `--backend` selection in CLI transcribe flow (`faster-whisper|onnx`)
  - ONNX backend wiring in shared service layer and Gradio UI
  - ONNX provider default set to `auto` (prefer CUDA, fallback CPU)
  - Unit tests with mocked ONNX pipeline output

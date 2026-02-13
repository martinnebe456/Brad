# Changelog

## 2026-02-12

- Added a native desktop UI (Tkinter) with tabs for transcribe, summarize, search, export, and health checks.
- Updated `brad ui` to launch desktop mode by default, with optional web mode via `--mode web`.
- Added CLI mode parsing tests for desktop/web UI selection.
- Added safer faster-whisper model initialization:
  - Tries CUDA first (when selected), then automatically falls back to CPU (`int8`) on init failure.
  - Adds explicit error details if both attempts fail.
- Added regression tests for faster-whisper init fallback behavior.
- Simplified ASR stack to a single backend (`faster-whisper`) for stability.
- Removed ONNX backend implementation and ONNX-specific tests.
- Removed ONNX runtime/dependency extras from packaging and install flow.
- Updated CLI/UI/doctor/docs to match faster-whisper-only behavior.
- Initial repository scaffold.
- Implemented local-only MVP pipeline:
  - ffmpeg preprocessing
  - faster-whisper transcription
  - optional Silero VAD chunking
  - SQLite + FTS5 storage
  - local summarization (llama-cpp optional, extractive fallback)
  - CLI + Gradio UI
- Added docs, prompts, tests, CI, and AGENTS playbook.

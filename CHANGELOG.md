# Changelog

## 2026-02-12

- Initial repository scaffold.
- Implemented local-only MVP pipeline:
  - ffmpeg preprocessing
  - faster-whisper transcription
  - optional Silero VAD chunking
  - SQLite + FTS5 storage
  - local summarization (llama-cpp optional, extractive fallback)
  - CLI + Gradio UI
- Added docs, prompts, tests, CI, and AGENTS playbook.

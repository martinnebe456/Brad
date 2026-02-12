# Brad architecture (MVP)

## Pipeline overview

```text
[Audio/File Input]
       |
       v
[Preprocess]
  - ffmpeg: convert to mono, 16kHz WAV
  - optional Silero VAD speech spans
       |
       v
[ASR]
  - primary: faster-whisper (CTranslate2)
  - optional future: ONNX Runtime backend
       |
       v
[Postprocess]
  - normalize segments
  - offset timestamps when chunked
       |
       v
[Storage]
  - SQLite tables: meetings, segments, summaries, exports
  - SQLite FTS5 index: segments_fts
       |
       +--------------------------+
       |                          |
       v                          v
[Summarize]                 [Search]
  - llama-cpp (GGUF)          - FTS5 match query
  - fallback extractive
       |
       v
[Export]
  - Markdown
  - JSON
  - SRT
       |
       v
[Interfaces]
  - Typer CLI
  - Gradio local UI
```

## Runtime design notes

- Offline-safe by default:
  - No runtime model downloads.
  - Model paths must already exist locally.
- Orchestration lives in `src/brad/services.py` so CLI and UI share behavior.
- Advanced features remain as TODOs mapped to `docs/LEARNING_PATH.md`.

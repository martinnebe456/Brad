# Learning path

This project is intentionally structured so you can extend it step-by-step.
Created by GPT-5.3-Codex

## LP-01: Implement ONNX backend (core extension) - DONE

- File: `src/brad/asr/onnx_backend_stub.py`
- Goal: replace `NotImplementedError` with a working ONNX Runtime pipeline via Optimum.
- Why: understand backend abstraction and model portability.
- Deliverables:
  - ONNX backend class implementing `ASRBackend`
  - CLI option to select backend
  - Unit test with mocked ONNX pipeline output

## LP-02: Add diarization (advanced)

- Goal: speaker labels (`SPEAKER_1`, `SPEAKER_2`) in stored segments/exports.
- Why: makes summaries and transcripts far more useful for meetings.
- Suggested approach: optional pyannote or lightweight diarization alternatives.

## LP-03: Add live transcription mode (advanced)

- Goal: stream microphone input in rolling windows.
- Why: introduces latency vs quality tradeoffs and buffering strategy.
- Suggested approach: chunked capture + incremental transcription + append-only storage.

## LP-04: Add semantic search (advanced)

- Goal: embeddings-based retrieval beyond literal FTS matching.
- Why: enables concept-level search ("timeline risk") instead of exact words.
- Suggested approach: local embedding model + vector index (FAISS or sqlite extension).

## LP-05: Improve chunking logic

- File: `src/brad/audio/chunking.py`
- Goal: smarter merge/split policies around pauses and max chunk durations.
- Why: chunk quality strongly affects ASR quality and throughput.
- Deliverables:
  - better heuristics
  - regression tests in `tests/test_chunking.py`

## LP-06: Improve SRT formatting

- File: `src/brad/export/srt.py`
- Goal: better line breaking and punctuation cleanup.
- Why: SRT readability impacts downstream usage in media players.
- Deliverables:
  - configurable line limits
  - stable tests in `tests/test_srt_export.py`

## LP-07: Improve prompts and summary QA

- Files: `docs/prompts/*.md`, `src/brad/nlp/summarizer.py`
- Goal: more reliable structured summaries per meeting type.
- Why: prompt clarity reduces hallucination and improves consistency.

## LP-08: Add migration story

- Goal: basic schema versioning and migrations for SQLite.
- Why: future changes should be safe for existing users.
- Suggested approach: metadata table + migration scripts.

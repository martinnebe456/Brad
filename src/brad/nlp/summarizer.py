from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from brad.nlp.prompts import load_template
from brad.storage.models import SegmentRecord

_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "for",
    "of",
    "in",
    "on",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "it",
    "that",
    "this",
    "we",
    "they",
    "you",
    "i",
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_SPLIT = re.compile(r"[A-Za-z0-9_]+")


@dataclass(slots=True)
class SummaryResult:
    text: str
    method: str
    template_name: str
    llm_model: str | None


def _normalize_sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT.split(text)
    clean = [item.strip() for item in parts if item.strip()]
    return clean


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _WORD_SPLIT.finditer(text)]


def extractive_summary(transcript_text: str, max_sentences: int = 6) -> str:
    sentences = _normalize_sentences(transcript_text)
    if not sentences:
        return "No transcript content was available."

    freq: dict[str, int] = {}
    for token in _tokenize(transcript_text):
        if token in _STOPWORDS or len(token) < 3:
            continue
        freq[token] = freq.get(token, 0) + 1

    scored: list[tuple[float, str, int]] = []
    for idx, sentence in enumerate(sentences):
        tokens = _tokenize(sentence)
        if not tokens:
            continue
        score = sum(freq.get(token, 0) for token in tokens) / len(tokens)
        scored.append((score, sentence, idx))

    if not scored:
        top = sentences[:max_sentences]
    else:
        top = [item[1] for item in sorted(scored, key=lambda row: row[0], reverse=True)[:max_sentences]]
        top = sorted(top, key=lambda sentence: sentences.index(sentence))

    action_lines = [
        sentence
        for sentence in sentences
        if any(marker in sentence.lower() for marker in ("action", "todo", "next step", "will "))
    ][:4]

    lines = ["Summary (extractive):", ""]
    lines.extend(f"- {sentence}" for sentence in top)
    if action_lines:
        lines.append("")
        lines.append("Likely action items:")
        lines.extend(f"- {sentence}" for sentence in action_lines)
    return "\n".join(lines)


def _llama_summary(prompt: str, model_path: Path) -> str:
    try:
        from llama_cpp import Llama  # type: ignore
    except Exception as exc:
        raise RuntimeError("llama-cpp-python not installed. Install with: pip install -e '.[llm]'") from exc

    if not model_path.exists():
        raise FileNotFoundError(f"LLM model not found: {model_path}")

    n_threads = max((os.cpu_count() or 4) - 1, 1)
    llm = Llama(
        model_path=str(model_path),
        n_ctx=4096,
        n_threads=n_threads,
        verbose=False,
    )
    response = llm(
        prompt,
        max_tokens=700,
        temperature=0.2,
        top_p=0.9,
        stop=["</s>"],
    )
    text = response["choices"][0]["text"].strip()
    return text or "LLM returned an empty summary."


def _load_transcript_text(transcript_path: Path) -> str:
    raw = transcript_path.read_text(encoding="utf-8")
    if transcript_path.suffix.lower() == ".json":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        segments = data.get("segments")
        if isinstance(segments, list):
            texts = [str(item.get("text", "")).strip() for item in segments if isinstance(item, dict)]
            return "\n".join(filter(None, texts))
    return raw


def segments_to_text(segments: list[SegmentRecord]) -> str:
    return "\n".join(f"[{item.start:.2f}-{item.end:.2f}] {item.text}" for item in segments)


class MeetingSummarizer:
    def summarize_text(
        self,
        transcript_text: str,
        *,
        template_name: str,
        llm_model: Path | None,
    ) -> SummaryResult:
        prompt_template = load_template(template_name)

        # TODO(LP-07): improve prompt packing strategy for long transcripts.
        clipped_text = transcript_text[-18_000:] if len(transcript_text) > 18_000 else transcript_text
        prompt = (
            f"{prompt_template}\n\n"
            "Transcript:\n"
            f"{clipped_text}\n\n"
            "Write the requested summary now.\n"
        )

        if llm_model is not None:
            summary_text = _llama_summary(prompt, llm_model)
            return SummaryResult(
                text=summary_text,
                method="llama-cpp",
                template_name=template_name,
                llm_model=str(llm_model),
            )

        return SummaryResult(
            text=extractive_summary(transcript_text),
            method="extractive",
            template_name=template_name,
            llm_model=None,
        )

    def summarize_path(
        self,
        transcript_path: Path,
        *,
        template_name: str,
        llm_model: Path | None,
    ) -> SummaryResult:
        transcript_text = _load_transcript_text(transcript_path)
        return self.summarize_text(
            transcript_text,
            template_name=template_name,
            llm_model=llm_model,
        )

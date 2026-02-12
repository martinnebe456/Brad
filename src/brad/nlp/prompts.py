from __future__ import annotations

from pathlib import Path

from brad.config import project_root

PROMPT_FILES: dict[str, str] = {
    "general": "summary_general.md",
    "sales": "summary_sales.md",
    "engineering": "summary_engineering.md",
}

_FALLBACK_PROMPTS: dict[str, str] = {
    "general": (
        "Summarize this meeting with objective, key points, decisions, action items, and open questions."
    ),
    "sales": (
        "Summarize this sales call with context, pain points, objections, commitments, and next steps."
    ),
    "engineering": (
        "Summarize this engineering meeting with technical decisions, blockers, risks, and owner TODOs."
    ),
}


def prompts_dir() -> Path:
    return project_root() / "docs" / "prompts"


def list_templates() -> list[str]:
    return sorted(PROMPT_FILES)


def load_template(template_name: str) -> str:
    key = template_name.lower().strip()
    if key not in PROMPT_FILES:
        allowed = ", ".join(list_templates())
        raise ValueError(f"Unknown template '{template_name}'. Allowed: {allowed}")

    prompt_path = prompts_dir() / PROMPT_FILES[key]
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return _FALLBACK_PROMPTS[key]

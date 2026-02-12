from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ASR_MODEL_ALIASES: dict[str, str] = {
    "small": "small",
    "medium": "medium",
    "large": "large-v3",
}


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    data_dir: Path = Field(default_factory=lambda: Path.home() / ".brad")
    models_dir: Path | None = None
    exports_dir: Path | None = None
    temp_dir: Path | None = None
    db_filename: str = "brad.db"

    default_asr_model: str = "small"
    default_language: str = "auto"
    default_compute_type: str = "int8"
    llm_default_model: Path | None = None

    model_config = SettingsConfigDict(env_prefix="BRAD_", extra="ignore")

    @model_validator(mode="after")
    def _derive_paths(self) -> "Settings":
        if "models_dir" not in self.model_fields_set or self.models_dir is None:
            # Default to a project-local model cache so users can keep models in ./models.
            self.models_dir = Path.cwd() / "models"
        if "exports_dir" not in self.model_fields_set or self.exports_dir is None:
            self.exports_dir = self.data_dir / "exports"
        if "temp_dir" not in self.model_fields_set or self.temp_dir is None:
            self.temp_dir = self.data_dir / "tmp"
        return self

    @property
    def db_path(self) -> Path:
        return self.data_dir / self.db_filename

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.models_dir, self.exports_dir, self.temp_dir):
            path.mkdir(parents=True, exist_ok=True)

    def resolve_asr_model_path(self, model_name: str) -> Path:
        key = model_name.lower().strip()
        if key not in ASR_MODEL_ALIASES:
            allowed = ", ".join(sorted(ASR_MODEL_ALIASES))
            raise ValueError(f"Unsupported ASR model '{model_name}'. Allowed: {allowed}")
        return self.models_dir / "faster-whisper" / ASR_MODEL_ALIASES[key]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

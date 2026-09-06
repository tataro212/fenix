"""Typed settings loaded from environment variables and optional .env files."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FENIX_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: SecretStr | None = Field(default=None, alias="GEMINI_API_KEY")
    translation_model: str = "gemini-2.5-flash"
    translation_concurrency: int = Field(default=5, ge=1, le=32)
    translation_batch_characters: int = Field(default=12_000, ge=1_000)
    translation_batch_segments: int = Field(default=32, ge=1)
    translation_retries: int = Field(default=2, ge=0, le=8)
    layout_model_path: Path | None = None
    enable_layout_detection: bool = True
    enable_ocr: bool = False
    output_directory: Path = Path("output")


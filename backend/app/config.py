"""Environment-driven settings. No secret has a baked-in default."""

from __future__ import annotations

import logging
import secrets
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent.parent
_WEAK_FRAGMENTS = (
    "changeme",
    "change-me",
    "secret",
    "password",
    "example",
    "placeholder",
    "replace",
    "default",
    "test",
    "dev",
    "1234",
)
MIN_SECRET_LEN = 32


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    env: Literal["dev", "prod"] = Field("dev", validation_alias="ENV")
    secret_key: SecretStr = Field(SecretStr(""), validation_alias="SECRET_KEY")
    database_url: str = Field(
        f"sqlite:///{BACKEND_DIR / 'scout.db'}", validation_alias="DATABASE_URL"
    )
    cookie_secure: bool = Field(True, validation_alias="COOKIE_SECURE")
    cookie_name: str = "scout_session"
    token_ttl_seconds: int = 12 * 3600

    # New names first, legacy names accepted.
    tba_api_key: SecretStr = Field(
        SecretStr(""), validation_alias=AliasChoices("TBA_API_KEY", "TBA_AUTH_KEY")
    )
    ollama_url: str = Field(
        "http://localhost:11434", validation_alias=AliasChoices("OLLAMA_URL", "OLLAMA_HOST")
    )
    tba_base_url: str = "https://www.thebluealliance.com/api/v3"

    cors_origins: str = Field("", validation_alias="CORS_ORIGINS")

    # Vision pipeline output (vision/match_pipeline/scout_table.csv) and the DB backup folder.
    vision_scout_table: str = Field(
        str(BACKEND_DIR.parent / "vision" / "match_pipeline" / "scout_table.csv"),
        validation_alias="VISION_SCOUT_TABLE",
    )
    backup_dir: str = Field(str(BACKEND_DIR / "backups"), validation_alias="BACKUP_DIR")

    login_max_failures: int = 5
    login_window_seconds: int = 900

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @model_validator(mode="after")
    def _check(self) -> Settings:
        if self.cookie_secure is False and self.env != "dev":
            raise ValueError("COOKIE_SECURE=false is only allowed when ENV=dev")
        key = self.secret_key.get_secret_value()
        if self.env == "prod":
            low = key.lower()
            if (
                len(key) < MIN_SECRET_LEN
                or len(set(key)) < 10
                or any(w in low for w in _WEAK_FRAGMENTS)
            ):
                raise ValueError(
                    f"SECRET_KEY missing or weak: need >= {MIN_SECRET_LEN} random chars "
                    "(e.g. `python -c 'import secrets;print(secrets.token_urlsafe(48))'`)"
                )
        elif not key:
            # Dev only: ephemeral random key, sessions die on restart.
            self.secret_key = SecretStr(secrets.token_urlsafe(48))
            log.warning("SECRET_KEY unset; using an ephemeral random key (dev only)")
        if "*" in self.cors_origin_list:
            raise ValueError("CORS_ORIGINS must list explicit origins, never '*'")
        if self.env == "dev" and not self.cors_origins:
            self.cors_origins = "http://localhost:5173"
        return self


def get_settings() -> Settings:
    return Settings()

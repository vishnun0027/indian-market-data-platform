"""Configuration management using Pydantic Settings.

Loads settings from environment variables and .env files.
Auto-creates required data directories on initialization.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- Paths ---
    data_dir: Path = Path("./data")

    # --- Logging ---
    log_level: str = "INFO"

    # --- Download ---
    max_workers: int = 4
    rate_limit_per_second: float = 2.0
    retry_attempts: int = 3
    retry_wait_seconds: float = 5.0
    yfinance_threads: int = 2

    # --- Anti-blocking ---
    batch_cooldown_size: int = 50       # Pause after this many downloads
    batch_cooldown_secs: float = 30.0   # Cooldown duration in seconds

    # --- HuggingFace Hub ---
    hf_repo_id: str = ""                # e.g. "username/indian-market-data"
    hf_token: str = ""                  # Write-access token (HF_TOKEN env var)
    hf_private: bool = False            # Whether the HF dataset repo is private

    @field_validator("data_dir", mode="before")
    @classmethod
    def resolve_data_dir(cls, v: str | Path) -> Path:
        """Resolve the data directory to an absolute path."""
        return Path(v).resolve()

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is a recognized Python logging level."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            msg = f"Invalid log level '{v}'. Must be one of: {', '.join(sorted(allowed))}"
            raise ValueError(msg)
        return upper

    # --- Derived paths ---
    @property
    def stocks_dir(self) -> Path:
        return self.data_dir / "stocks"

    @property
    def indices_dir(self) -> Path:
        return self.data_dir / "indices"

    @property
    def etfs_dir(self) -> Path:
        return self.data_dir / "etfs"

    @property
    def commodities_dir(self) -> Path:
        return self.data_dir / "commodities"

    @property
    def forex_dir(self) -> Path:
        return self.data_dir / "forex"

    @property
    def metadata_dir(self) -> Path:
        return self.data_dir / "metadata"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    def ensure_directories(self) -> None:
        """Create all required data directories if they don't exist."""
        dirs = [
            self.stocks_dir,
            self.indices_dir,
            self.etfs_dir,
            self.commodities_dir,
            self.forex_dir,
            self.metadata_dir,
            self.logs_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def get_asset_dir(self, asset_type: str) -> Path:
        """Return the storage directory for a given asset type string.

        Args:
            asset_type: One of 'EQUITY', 'INDEX', 'ETF', 'COMMODITY', 'FOREX'.

        Returns:
            Path to the directory for that asset type.
        """
        mapping = {
            "EQUITY": self.stocks_dir,
            "INDEX": self.indices_dir,
            "ETF": self.etfs_dir,
            "COMMODITY": self.commodities_dir,
            "FOREX": self.forex_dir,
        }
        result = mapping.get(asset_type.upper())
        if result is None:
            msg = f"Unknown asset type: {asset_type}"
            raise ValueError(msg)
        return result


def get_settings() -> Settings:
    """Create and return a Settings instance.

    Uses a .env file in the current working directory if present.
    """
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        return Settings(_env_file=str(env_path))
    return Settings()

"""Tests for configuration module."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from market_data.config import Settings, get_settings


class TestSettings:
    """Tests for the Settings class."""

    def test_default_settings(self) -> None:
        """Settings should have sensible defaults."""
        settings = Settings(data_dir="./test_data")
        assert settings.log_level == "INFO"
        assert settings.max_workers == 4
        assert settings.rate_limit_per_second == 2.0
        assert settings.retry_attempts == 3

    def test_data_dir_resolved(self) -> None:
        """data_dir should be resolved to an absolute path."""
        settings = Settings(data_dir="./relative/path")
        assert settings.data_dir.is_absolute()

    def test_log_level_validation(self) -> None:
        """Invalid log level should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid log level"):
            Settings(data_dir="./test_data", log_level="INVALID")

    def test_log_level_case_insensitive(self) -> None:
        """Log level should accept any case."""
        settings = Settings(data_dir="./test_data", log_level="debug")
        assert settings.log_level == "DEBUG"

    def test_derived_paths(self) -> None:
        """Derived directory paths should be under data_dir."""
        settings = Settings(data_dir="/tmp/market_test")
        assert settings.stocks_dir == Path("/tmp/market_test/stocks")
        assert settings.indices_dir == Path("/tmp/market_test/indices")
        assert settings.commodities_dir == Path("/tmp/market_test/commodities")
        assert settings.forex_dir == Path("/tmp/market_test/forex")
        assert settings.metadata_dir == Path("/tmp/market_test/metadata")
        assert settings.logs_dir == Path("/tmp/market_test/logs")

    def test_ensure_directories(self, tmp_path: Path) -> None:
        """ensure_directories should create all required directories."""
        settings = Settings(data_dir=str(tmp_path / "data"))
        settings.ensure_directories()

        assert settings.stocks_dir.exists()
        assert settings.indices_dir.exists()
        assert settings.commodities_dir.exists()
        assert settings.forex_dir.exists()
        assert settings.metadata_dir.exists()
        assert settings.logs_dir.exists()

    def test_get_asset_dir(self) -> None:
        """get_asset_dir should return the correct directory for each type."""
        settings = Settings(data_dir="/tmp/market_test")
        assert settings.get_asset_dir("EQUITY") == Path("/tmp/market_test/stocks")
        assert settings.get_asset_dir("INDEX") == Path("/tmp/market_test/indices")
        assert settings.get_asset_dir("COMMODITY") == Path("/tmp/market_test/commodities")

    def test_get_asset_dir_invalid(self) -> None:
        """get_asset_dir should raise ValueError for unknown types."""
        settings = Settings(data_dir="/tmp/market_test")
        with pytest.raises(ValueError, match="Unknown asset type"):
            settings.get_asset_dir("INVALID")

    def test_env_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Settings should be overridable via environment variables."""
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("MAX_WORKERS", "8")

        settings = Settings()
        assert settings.data_dir == tmp_path
        assert settings.log_level == "DEBUG"
        assert settings.max_workers == 8

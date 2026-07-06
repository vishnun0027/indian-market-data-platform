"""Tests for Parquet storage layer."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from market_data.assets.models import AssetInfo, AssetType, Exchange
from market_data.storage.parquet_store import ParquetStore


class TestParquetStore:
    """Tests for the ParquetStore class."""

    def test_save_and_load(
        self, tmp_data_dir: Path, sample_equity: AssetInfo, sample_ohlcv_df: pd.DataFrame
    ) -> None:
        """Save and load should roundtrip correctly."""
        store = ParquetStore(data_dir=tmp_data_dir)

        store.save(sample_equity, sample_ohlcv_df)
        loaded = store.load(sample_equity)

        assert len(loaded) == len(sample_ohlcv_df)
        assert list(loaded.columns) == list(sample_ohlcv_df.columns)
        assert loaded["close"].tolist() == sample_ohlcv_df["close"].tolist()

    def test_exists(
        self, tmp_data_dir: Path, sample_equity: AssetInfo, sample_ohlcv_df: pd.DataFrame
    ) -> None:
        """exists() should return True only after saving."""
        store = ParquetStore(data_dir=tmp_data_dir)

        assert not store.exists(sample_equity)
        store.save(sample_equity, sample_ohlcv_df)
        assert store.exists(sample_equity)

    def test_get_last_date(
        self, tmp_data_dir: Path, sample_equity: AssetInfo, sample_ohlcv_df: pd.DataFrame
    ) -> None:
        """get_last_date should return the most recent date."""
        store = ParquetStore(data_dir=tmp_data_dir)

        assert store.get_last_date(sample_equity) is None
        store.save(sample_equity, sample_ohlcv_df)
        assert store.get_last_date(sample_equity) == date(2024, 1, 8)

    def test_get_row_count(
        self, tmp_data_dir: Path, sample_equity: AssetInfo, sample_ohlcv_df: pd.DataFrame
    ) -> None:
        """get_row_count should return the correct count."""
        store = ParquetStore(data_dir=tmp_data_dir)

        assert store.get_row_count(sample_equity) == 0
        store.save(sample_equity, sample_ohlcv_df)
        assert store.get_row_count(sample_equity) == 5

    def test_append_deduplicates(
        self, tmp_data_dir: Path, sample_equity: AssetInfo, sample_ohlcv_df: pd.DataFrame
    ) -> None:
        """Appending overlapping data should deduplicate by date."""
        store = ParquetStore(data_dir=tmp_data_dir)
        store.save(sample_equity, sample_ohlcv_df)

        # Append with overlapping date (Jan 8) and a new date (Jan 9)
        new_df = pd.DataFrame({
            "date": [date(2024, 1, 8), date(2024, 1, 9)],
            "open": [105.0, 110.0],
            "high": [110.0, 115.0],
            "low": [104.0, 108.0],
            "close": [108.0, 112.0],
            "adj_close": [108.0, 112.0],
            "volume": [1500000, 1600000],
            "dividends": [0.0, 0.0],
            "stock_splits": [0.0, 0.0],
            "symbol": ["RELIANCE", "RELIANCE"],
        })
        store.save(sample_equity, new_df)

        loaded = store.load(sample_equity)
        assert len(loaded) == 6  # 5 original + 1 new (Jan 8 deduplicated)

    def test_load_with_date_filter(
        self, tmp_data_dir: Path, sample_equity: AssetInfo, sample_ohlcv_df: pd.DataFrame
    ) -> None:
        """load() should support date range filtering."""
        store = ParquetStore(data_dir=tmp_data_dir)
        store.save(sample_equity, sample_ohlcv_df)

        filtered = store.load(
            sample_equity,
            start_date=date(2024, 1, 3),
            end_date=date(2024, 1, 5),
        )
        assert len(filtered) == 3

    def test_delete(
        self, tmp_data_dir: Path, sample_equity: AssetInfo, sample_ohlcv_df: pd.DataFrame
    ) -> None:
        """delete() should remove the Parquet file."""
        store = ParquetStore(data_dir=tmp_data_dir)
        store.save(sample_equity, sample_ohlcv_df)

        assert store.exists(sample_equity)
        assert store.delete(sample_equity)
        assert not store.exists(sample_equity)

    def test_delete_nonexistent(self, tmp_data_dir: Path, sample_equity: AssetInfo) -> None:
        """delete() on nonexistent file should return False."""
        store = ParquetStore(data_dir=tmp_data_dir)
        assert not store.delete(sample_equity)

    def test_empty_df_save(self, tmp_data_dir: Path, sample_equity: AssetInfo) -> None:
        """Saving an empty DataFrame should not create a file."""
        store = ParquetStore(data_dir=tmp_data_dir)
        store.save(sample_equity, pd.DataFrame())
        assert not store.exists(sample_equity)

    def test_list_stored_assets(
        self, tmp_data_dir: Path, sample_equity: AssetInfo, sample_ohlcv_df: pd.DataFrame
    ) -> None:
        """list_stored_assets should return symbols of stored files."""
        store = ParquetStore(data_dir=tmp_data_dir)
        assert store.list_stored_assets() == []

        store.save(sample_equity, sample_ohlcv_df)
        stored = store.list_stored_assets()
        assert "RELIANCE" in stored

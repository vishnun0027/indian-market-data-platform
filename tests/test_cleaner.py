"""Tests for the DataCleaner class."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from market_data.assets.models import AssetInfo, AssetType, Exchange
from market_data.storage.parquet_store import ParquetStore
from market_data.validation.cleaner import DataCleaner


class TestDataCleaner:
    """Tests for DataCleaner functions."""

    def test_clean_dataframe_no_violations(self) -> None:
        """Dataframe without violations should not be modified."""
        df = pd.DataFrame({
            "open": [100.0, 102.0],
            "high": [105.0, 106.0],
            "low": [99.0, 100.0],
            "close": [103.0, 104.0],
        })
        cleaned, count = DataCleaner.clean_dataframe(df)
        assert count == 0
        assert cleaned["open"].tolist() == df["open"].tolist()
        assert cleaned["high"].tolist() == df["high"].tolist()

    def test_clean_dataframe_high_lt_low(self) -> None:
        """If high < low, high should become max and low should become min."""
        df = pd.DataFrame({
            "open": [100.0],
            "high": [98.0],  # high < open and high < low
            "low": [99.0],
            "close": [103.0],
        })
        cleaned, count = DataCleaner.clean_dataframe(df)
        assert count == 1
        # high should be max(100, 98, 99, 103) = 103
        assert cleaned.loc[0, "high"] == 103.0
        # low should be min(100, 98, 99, 103) = 98
        assert cleaned.loc[0, "low"] == 98.0

    def test_clean_dataframe_open_out_of_bounds(self) -> None:
        """If open is outside high/low, boundaries should be adjusted."""
        df = pd.DataFrame({
            "open": [110.0],  # open > high
            "high": [105.0],
            "low": [99.0],
            "close": [103.0],
        })
        cleaned, count = DataCleaner.clean_dataframe(df)
        assert count == 1
        assert cleaned.loc[0, "high"] == 110.0
        assert cleaned.loc[0, "low"] == 99.0

    def test_clean_dataframe_close_out_of_bounds(self) -> None:
        """If close is outside high/low, boundaries should be adjusted."""
        df = pd.DataFrame({
            "open": [100.0],
            "high": [105.0],
            "low": [99.0],
            "close": [95.0],  # close < low
        })
        cleaned, count = DataCleaner.clean_dataframe(df)
        assert count == 1
        assert cleaned.loc[0, "high"] == 105.0
        assert cleaned.loc[0, "low"] == 95.0

    def test_clean_asset(
        self, tmp_data_dir: Path, sample_equity: AssetInfo, bad_ohlcv_df: pd.DataFrame
    ) -> None:
        """clean_asset should repair, save, and update the catalog."""
        store = ParquetStore(data_dir=tmp_data_dir)
        store.save(sample_equity, bad_ohlcv_df)

        cleaner = DataCleaner(store=store, catalog=None)
        count = cleaner.clean_asset(sample_equity)

        assert count > 0
        assert store.exists(sample_equity)

        # Loaded data should now have no violations
        cleaned = store.load(sample_equity)
        # Verify high >= open/close/low
        assert (cleaned["high"] >= cleaned["low"]).all()
        assert (cleaned["high"] >= cleaned["open"]).all()
        assert (cleaned["high"] >= cleaned["close"]).all()
        assert (cleaned["low"] <= cleaned["open"]).all()
        assert (cleaned["low"] <= cleaned["close"]).all()

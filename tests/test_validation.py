"""Tests for data validation checks."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from market_data.validation.checks import (
    check_data_range,
    check_duplicates,
    check_missing_trading_days,
    check_negative_volume,
    check_null_values,
    check_ohlc_consistency,
)
from market_data.validation.models import CheckStatus


class TestCheckDuplicates:
    """Tests for the duplicate check."""

    def test_no_duplicates(self, sample_ohlcv_df: pd.DataFrame) -> None:
        result = check_duplicates(sample_ohlcv_df)
        assert result.status == CheckStatus.PASSED

    def test_with_duplicates(self, bad_ohlcv_df: pd.DataFrame) -> None:
        result = check_duplicates(bad_ohlcv_df)
        assert result.status == CheckStatus.FAILED
        assert result.rows_affected > 0

    def test_empty_df(self) -> None:
        result = check_duplicates(pd.DataFrame())
        assert result.status == CheckStatus.SKIPPED


class TestCheckOHLCConsistency:
    """Tests for OHLC consistency checks."""

    def test_valid_ohlc(self, sample_ohlcv_df: pd.DataFrame) -> None:
        result = check_ohlc_consistency(sample_ohlcv_df)
        assert result.status == CheckStatus.PASSED

    def test_invalid_ohlc(self, bad_ohlcv_df: pd.DataFrame) -> None:
        result = check_ohlc_consistency(bad_ohlcv_df)
        assert result.status == CheckStatus.FAILED
        assert result.rows_affected > 0

    def test_empty_df(self) -> None:
        result = check_ohlc_consistency(pd.DataFrame())
        assert result.status == CheckStatus.SKIPPED


class TestCheckNegativeVolume:
    """Tests for negative volume check."""

    def test_valid_volume(self, sample_ohlcv_df: pd.DataFrame) -> None:
        result = check_negative_volume(sample_ohlcv_df)
        assert result.status == CheckStatus.PASSED

    def test_negative_volume(self, bad_ohlcv_df: pd.DataFrame) -> None:
        result = check_negative_volume(bad_ohlcv_df)
        assert result.status == CheckStatus.FAILED
        assert result.rows_affected == 1


class TestCheckNullValues:
    """Tests for null value check."""

    def test_no_nulls(self, sample_ohlcv_df: pd.DataFrame) -> None:
        result = check_null_values(sample_ohlcv_df)
        assert result.status == CheckStatus.PASSED

    def test_with_nulls(self) -> None:
        df = pd.DataFrame({
            "date": [date(2024, 1, 2), date(2024, 1, 3)],
            "open": [100.0, None],
            "high": [105.0, 106.0],
            "low": [99.0, 100.0],
            "close": [103.0, 104.0],
            "volume": [1000000, 1200000],
        })
        result = check_null_values(df)
        assert result.status == CheckStatus.WARNING
        assert result.rows_affected > 0


class TestCheckDataRange:
    """Tests for date range check."""

    def test_valid_range(self, sample_ohlcv_df: pd.DataFrame) -> None:
        result = check_data_range(sample_ohlcv_df)
        assert result.status == CheckStatus.PASSED

    def test_future_dates(self) -> None:
        future = date.today() + timedelta(days=30)
        df = pd.DataFrame({
            "date": [date(2024, 1, 2), future],
            "open": [100.0, 102.0],
            "high": [105.0, 106.0],
            "low": [99.0, 100.0],
            "close": [103.0, 104.0],
            "volume": [1000000, 1200000],
        })
        result = check_data_range(df)
        assert result.status == CheckStatus.FAILED


class TestCheckMissingTradingDays:
    """Tests for missing trading days check."""

    def test_no_large_gaps(self, sample_ohlcv_df: pd.DataFrame) -> None:
        result = check_missing_trading_days(sample_ohlcv_df)
        assert result.status == CheckStatus.PASSED

    def test_large_gap(self) -> None:
        df = pd.DataFrame({
            "date": [date(2024, 1, 2), date(2024, 2, 1)],  # 30-day gap
            "open": [100.0, 102.0],
            "high": [105.0, 106.0],
            "low": [99.0, 100.0],
            "close": [103.0, 104.0],
            "volume": [1000000, 1200000],
        })
        result = check_missing_trading_days(df)
        assert result.status == CheckStatus.WARNING

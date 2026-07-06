"""Shared test fixtures."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from market_data.assets.models import AssetInfo, AssetType, Exchange


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory with all subdirectories."""
    for subdir in ("stocks", "indices", "etfs", "commodities", "forex", "metadata", "logs"):
        (tmp_path / subdir).mkdir()
    return tmp_path


@pytest.fixture
def sample_equity() -> AssetInfo:
    """A sample NSE equity asset."""
    return AssetInfo(
        symbol="RELIANCE",
        name="Reliance Industries Limited",
        isin="INE002A01018",
        exchange=Exchange.NSE,
        asset_type=AssetType.EQUITY,
    )


@pytest.fixture
def sample_index() -> AssetInfo:
    """A sample index asset."""
    return AssetInfo(
        symbol="NIFTY 50",
        name="NSE Nifty 50 Index",
        exchange=Exchange.NSE,
        asset_type=AssetType.INDEX,
        yfinance_ticker="^NSEI",
    )


@pytest.fixture
def sample_commodity() -> AssetInfo:
    """A sample commodity asset."""
    return AssetInfo(
        symbol="GOLD",
        name="Gold Futures",
        exchange=Exchange.MULTI,
        asset_type=AssetType.COMMODITY,
        yfinance_ticker="GC=F",
    )


@pytest.fixture
def sample_forex() -> AssetInfo:
    """A sample forex asset."""
    return AssetInfo(
        symbol="USDINR",
        name="US Dollar / Indian Rupee",
        exchange=Exchange.MULTI,
        asset_type=AssetType.FOREX,
        yfinance_ticker="USDINR=X",
    )


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """A small valid OHLCV DataFrame for testing."""
    return pd.DataFrame({
        "date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4),
                 date(2024, 1, 5), date(2024, 1, 8)],
        "open": [100.0, 102.0, 101.0, 103.0, 104.0],
        "high": [105.0, 106.0, 104.0, 107.0, 108.0],
        "low": [99.0, 100.0, 99.5, 101.0, 102.0],
        "close": [103.0, 104.0, 102.0, 106.0, 107.0],
        "adj_close": [103.0, 104.0, 102.0, 106.0, 107.0],
        "volume": [1000000, 1200000, 900000, 1100000, 1300000],
        "dividends": [0.0, 0.0, 0.0, 0.0, 0.0],
        "stock_splits": [0.0, 0.0, 0.0, 0.0, 0.0],
        "symbol": ["RELIANCE"] * 5,
    })


@pytest.fixture
def bad_ohlcv_df() -> pd.DataFrame:
    """A DataFrame with known data quality issues for testing validation."""
    return pd.DataFrame({
        "date": [date(2024, 1, 2), date(2024, 1, 2), date(2024, 1, 4),  # duplicate date
                 date(2024, 1, 5), date(2024, 1, 8)],
        "open": [100.0, 102.0, 101.0, 110.0, 104.0],  # row 4: open > high
        "high": [105.0, 106.0, 104.0, 107.0, 108.0],
        "low": [99.0, 100.0, 105.0, 101.0, 102.0],    # row 3: low > high
        "close": [103.0, 104.0, 102.0, 106.0, 107.0],
        "adj_close": [103.0, 104.0, 102.0, 106.0, 107.0],
        "volume": [1000000, -500, 900000, 1100000, 1300000],  # negative volume
        "dividends": [0.0, 0.0, 0.0, 0.0, 0.0],
        "stock_splits": [0.0, 0.0, 0.0, 0.0, 0.0],
        "symbol": ["RELIANCE"] * 5,
    })

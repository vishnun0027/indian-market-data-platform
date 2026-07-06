"""Tests for asset models."""

from __future__ import annotations

from datetime import date

import pytest

from market_data.assets.models import AssetInfo, AssetType, Exchange


class TestAssetInfo:
    """Tests for the AssetInfo model."""

    def test_equity_yfinance_ticker(self) -> None:
        """NSE equity should get .NS suffix."""
        asset = AssetInfo(symbol="RELIANCE", exchange=Exchange.NSE, asset_type=AssetType.EQUITY)
        assert asset.yfinance_ticker == "RELIANCE.NS"

    def test_bse_equity_yfinance_ticker(self) -> None:
        """BSE equity should get .BO suffix."""
        asset = AssetInfo(symbol="RELIANCE", exchange=Exchange.BSE, asset_type=AssetType.EQUITY)
        assert asset.yfinance_ticker == "RELIANCE.BO"

    def test_index_yfinance_ticker(self) -> None:
        """Known indices should map to yfinance tickers."""
        asset = AssetInfo(
            symbol="NIFTY 50",
            exchange=Exchange.NSE,
            asset_type=AssetType.INDEX,
        )
        assert asset.yfinance_ticker == "^NSEI"

    def test_index_explicit_ticker(self) -> None:
        """Explicitly provided ticker should not be overridden."""
        asset = AssetInfo(
            symbol="CUSTOM INDEX",
            asset_type=AssetType.INDEX,
            yfinance_ticker="^CUSTOM",
        )
        assert asset.yfinance_ticker == "^CUSTOM"

    def test_commodity_yfinance_ticker(self) -> None:
        """Known commodities should map to futures tickers."""
        asset = AssetInfo(symbol="GOLD", asset_type=AssetType.COMMODITY)
        assert asset.yfinance_ticker == "GC=F"

    def test_forex_yfinance_ticker(self) -> None:
        """Known forex pairs should map to yfinance tickers."""
        asset = AssetInfo(symbol="USDINR", asset_type=AssetType.FOREX)
        assert asset.yfinance_ticker == "USDINR=X"

    def test_safe_filename(self) -> None:
        """safe_filename should replace special characters."""
        asset = AssetInfo(symbol="NIFTY 50", asset_type=AssetType.INDEX, yfinance_ticker="^NSEI")
        assert asset.safe_filename == "NIFTY_50"

    def test_safe_filename_slash(self) -> None:
        """Slashes should be replaced in safe_filename."""
        asset = AssetInfo(symbol="USD/INR", asset_type=AssetType.FOREX, yfinance_ticker="USDINR=X")
        assert asset.safe_filename == "USD_INR"

    def test_full_model(self) -> None:
        """Full model creation with all fields."""
        asset = AssetInfo(
            symbol="TCS",
            name="Tata Consultancy Services",
            isin="INE467B01029",
            exchange=Exchange.NSE,
            sector="Information Technology",
            industry="IT Services",
            listing_date=date(2004, 8, 25),
            asset_type=AssetType.EQUITY,
        )
        assert asset.symbol == "TCS"
        assert asset.yfinance_ticker == "TCS.NS"
        assert asset.listing_date == date(2004, 8, 25)

    def test_model_serialization(self, sample_equity: AssetInfo) -> None:
        """Model should serialize to dict and back."""
        data = sample_equity.model_dump()
        assert data["symbol"] == "RELIANCE"
        assert data["exchange"] == "NSE"
        assert data["asset_type"] == "EQUITY"

        restored = AssetInfo(**data)
        assert restored.symbol == sample_equity.symbol

"""Pydantic models for asset metadata."""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class AssetType(str, Enum):
    """Supported asset types."""

    EQUITY = "EQUITY"
    INDEX = "INDEX"
    ETF = "ETF"
    COMMODITY = "COMMODITY"
    FOREX = "FOREX"


class Exchange(str, Enum):
    """Supported exchanges."""

    NSE = "NSE"
    BSE = "BSE"
    MULTI = "MULTI"  # For assets listed on multiple exchanges or non-exchange assets


class AssetInfo(BaseModel):
    """Metadata for a single tradeable asset.

    Attributes:
        symbol: The exchange-local symbol (e.g. 'RELIANCE', 'NIFTY 50').
        name: Human-readable name (e.g. 'Reliance Industries Limited').
        isin: ISIN code if available.
        exchange: Primary exchange.
        sector: Sector classification if available.
        industry: Industry classification if available.
        listing_date: Date the asset was listed, if known.
        asset_type: Type of asset (EQUITY, INDEX, etc.).
        yfinance_ticker: Ticker string for yfinance (e.g. 'RELIANCE.NS', '^NSEI').
    """

    symbol: str
    name: str = ""
    isin: str = ""
    exchange: Exchange = Exchange.NSE
    sector: str = ""
    industry: str = ""
    listing_date: date | None = None
    asset_type: AssetType = AssetType.EQUITY
    yfinance_ticker: str = Field(
        default="",
        description="Ticker symbol used by yfinance (e.g. 'RELIANCE.NS', '^NSEI')",
    )

    def model_post_init(self, __context: object) -> None:
        """Auto-generate yfinance ticker if not explicitly provided."""
        if not self.yfinance_ticker:
            self.yfinance_ticker = self._derive_yfinance_ticker()

    def _derive_yfinance_ticker(self) -> str:
        """Derive the yfinance ticker from symbol, exchange, and asset type."""
        if self.asset_type == AssetType.INDEX:
            # yfinance indices use ^ prefix
            index_map = {
                "NIFTY 50": "^NSEI",
                "NIFTY NEXT 50": "^NSMIDCP50",
                "NIFTY BANK": "^NSEBANK",
                "NIFTY IT": "^CNXIT",
                "NIFTY MIDCAP 50": "^NSMIDCP50",
                "NIFTY MIDCAP 100": "^NSEMDCP100",
                "INDIA VIX": "^INDIAVIX",
                "SENSEX": "^BSESN",
            }
            return index_map.get(self.symbol.upper(), f"^{self.symbol}")

        if self.asset_type == AssetType.COMMODITY:
            commodity_map = {
                "GOLD": "GC=F",
                "SILVER": "SI=F",
                "CRUDE OIL": "CL=F",
                "NATURAL GAS": "NG=F",
            }
            return commodity_map.get(self.symbol.upper(), self.symbol)

        if self.asset_type == AssetType.FOREX:
            forex_map = {
                "USDINR": "USDINR=X",
                "EURINR": "EURINR=X",
                "GBPINR": "GBPINR=X",
                "JPYINR": "JPYINR=X",
            }
            return forex_map.get(self.symbol.upper(), f"{self.symbol}=X")

        # Equities and ETFs — append exchange suffix
        suffix = ".NS" if self.exchange == Exchange.NSE else ".BO"
        return f"{self.symbol}{suffix}"

    @property
    def safe_filename(self) -> str:
        """Return a filesystem-safe version of the symbol for use as a filename."""
        # Replace characters that are problematic in filenames
        return self.symbol.replace(" ", "_").replace("/", "_").replace("^", "").replace("=", "_")

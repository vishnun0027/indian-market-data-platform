"""Asset discovery — discovers the full universe of Indian market assets.

Fetches NSE/BSE equity listings, and provides curated lists for indices,
ETFs, commodities, and forex pairs.
"""

from __future__ import annotations

import csv
import io
import logging
from pathlib import Path

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from market_data.assets.models import AssetInfo, AssetType, Exchange

logger = logging.getLogger("market_data.assets.discovery")

# --- NSE / BSE listing URLs ---
# NSE provides a CSV of all listed equities.  This URL may change or be blocked.
NSE_EQUITY_CSV_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
BSE_EQUITY_CSV_URL = "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w"

# Browser-like headers to avoid being blocked
_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


class AssetDiscovery:
    """Discovers and maintains the universe of supported assets.

    Usage:
        discovery = AssetDiscovery()
        assets = discovery.discover_all()
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout

    # ------------------------------------------------------------------
    # NSE equities
    # ------------------------------------------------------------------
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _fetch_nse_csv(self) -> str:
        """Fetch the NSE equity listing CSV."""
        with httpx.Client(
            headers=_HTTP_HEADERS,
            timeout=self._timeout,
            follow_redirects=True,
        ) as client:
            resp = client.get(NSE_EQUITY_CSV_URL)
            resp.raise_for_status()
            return resp.text

    def discover_nse_equities(self) -> list[AssetInfo]:
        """Discover all NSE-listed equities.

        Tries the public CSV endpoint first.  On failure, falls back to a
        curated seed list of major NIFTY constituents.
        """
        try:
            csv_text = self._fetch_nse_csv()
            return self._parse_nse_csv(csv_text)
        except Exception:
            logger.warning(
                "Failed to fetch NSE equity CSV from %s — falling back to seed list",
                NSE_EQUITY_CSV_URL,
            )
            return self._nse_seed_list()

    @staticmethod
    def _parse_nse_csv(csv_text: str) -> list[AssetInfo]:
        """Parse the NSE EQUITY_L.csv format."""
        assets: list[AssetInfo] = []
        reader = csv.DictReader(io.StringIO(csv_text))

        for row in reader:
            symbol = row.get("SYMBOL", "").strip()
            if not symbol:
                continue
            name = row.get("NAME OF COMPANY", "").strip()
            isin = row.get(" ISIN NUMBER", row.get("ISIN NUMBER", "")).strip()
            listing_raw = row.get(" DATE OF LISTING", row.get("DATE OF LISTING", "")).strip()

            listing_date = None
            if listing_raw:
                try:
                    from datetime import datetime

                    listing_date = datetime.strptime(listing_raw, "%d-%b-%Y").date()
                except ValueError:
                    pass

            assets.append(
                AssetInfo(
                    symbol=symbol,
                    name=name,
                    isin=isin,
                    exchange=Exchange.NSE,
                    asset_type=AssetType.EQUITY,
                    listing_date=listing_date,
                )
            )

        logger.info("Discovered %d NSE equities from CSV", len(assets))
        return assets

    @staticmethod
    def _nse_seed_list() -> list[AssetInfo]:
        """Curated seed list of major NSE equities (NIFTY 50 + key large-caps)."""
        symbols = [
            # NIFTY 50 constituents (representative sample — the full list is much larger)
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
            "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
            "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "TITAN",
            "SUNPHARMA", "BAJFINANCE", "BAJAJFINSV", "WIPRO", "HCLTECH",
            "ULTRACEMCO", "NESTLEIND", "ONGC", "NTPC", "POWERGRID",
            "M&M", "TATAMOTORS", "TATASTEEL", "JSWSTEEL", "ADANIENT",
            "ADANIPORTS", "TECHM", "INDUSINDBK", "COALINDIA", "GRASIM",
            "CIPLA", "DRREDDY", "EICHERMOT", "BPCL", "BRITANNIA",
            "APOLLOHOSP", "DIVISLAB", "HEROMOTOCO", "SBILIFE", "TATACONSUM",
            "BAJAJ-AUTO", "HDFCLIFE", "HINDALCO", "UPL", "LTIM",
            # Additional large-caps
            "VEDL", "BANKBARODA", "IOC", "GAIL", "PNB",
            "IRCTC", "ZOMATO", "PAYTM", "NYKAA", "DELHIVERY",
            "PIIND", "HAVELLS", "GODREJCP", "DABUR", "MARICO",
            "TRENT", "PERSISTENT", "COFORGE", "MPHASIS", "LTTS",
            "JUBLFOOD", "PAGEIND", "ASTRAL", "ATUL", "NAVINFLUOR",
            "CHOLAFIN", "MUTHOOTFIN", "MANAPPURAM", "AUBANK", "IDFCFIRSTB",
            "FEDERALBNK", "RBLBANK", "BANDHANBNK", "CANBK", "UNIONBANK",
            "INDIANB", "CENTRALBK", "IDBI", "YESBANK", "RECLTD",
            "PFC", "IRFC", "NHPC", "SJVN", "TATAPOWER",
            "ADANIGREEN", "ADANIPOWER", "TORNTPOWER", "CESC", "JSL",
            "NATIONALUM", "NMDC", "SAIL", "APLAPOLLO", "RATNAMANI",
            "HAL", "BEL", "BDL", "COCHINSHIP", "MAZAGON",
            "LICI", "SBICARD", "ICICIPRULI", "ICICIGI", "HDFCAMC",
            "MOTHERSON", "BOSCHLTD", "MRF", "CUMMINSIND", "THERMAX",
            "ABB", "SIEMENS", "HONAUT", "3MINDIA", "SCHAEFFLER",
        ]
        assets = [
            AssetInfo(
                symbol=s,
                name=s,
                exchange=Exchange.NSE,
                asset_type=AssetType.EQUITY,
            )
            for s in symbols
        ]
        logger.info("Using seed list of %d NSE equities", len(assets))
        return assets

    # ------------------------------------------------------------------
    # BSE equities
    # ------------------------------------------------------------------
    def discover_bse_equities(self) -> list[AssetInfo]:
        """Discover BSE-listed equities.

        BSE API is less stable, so we use a curated seed list of major
        BSE-only stocks (most large-caps are dual-listed and covered by NSE).
        """
        symbols = [
            "BERGEPAINT", "RAJESHEXPO", "RELAXO", "VGUARD", "CENTURYTEX",
            "GRSE", "GARDENREACH", "TITAGARH", "IRCON", "RVNL",
            "RAILTEL", "RITES", "HUDCO", "NCC", "KEC",
            "KALPATPOWR", "GPPL", "GSPL", "GUJGASLTD", "MGL",
        ]
        assets = [
            AssetInfo(
                symbol=s,
                name=s,
                exchange=Exchange.BSE,
                asset_type=AssetType.EQUITY,
            )
            for s in symbols
        ]
        logger.info("Discovered %d BSE equities (seed list)", len(assets))
        return assets

    # ------------------------------------------------------------------
    # Indices
    # ------------------------------------------------------------------
    def discover_indices(self) -> list[AssetInfo]:
        """Return curated list of major Indian market indices."""
        indices = [
            ("NIFTY 50", "^NSEI", "NSE Nifty 50 Index"),
            ("NIFTY NEXT 50", "^NSMIDCP50", "NSE Nifty Next 50 Index"),
            ("NIFTY BANK", "^NSEBANK", "NSE Bank Nifty Index"),
            ("NIFTY IT", "^CNXIT", "NSE Nifty IT Index"),
            ("NIFTY MIDCAP 100", "^NSEMDCP100", "NSE Nifty Midcap 100 Index"),
            ("INDIA VIX", "^INDIAVIX", "India Volatility Index"),
            ("SENSEX", "^BSESN", "BSE Sensex 30 Index"),
            ("NIFTY PHARMA", "^CNXPHARMA", "NSE Nifty Pharma Index"),
            ("NIFTY AUTO", "^CNXAUTO", "NSE Nifty Auto Index"),
            ("NIFTY METAL", "^CNXMETAL", "NSE Nifty Metal Index"),
            ("NIFTY REALTY", "^CNXREALTY", "NSE Nifty Realty Index"),
            ("NIFTY ENERGY", "^CNXENERGY", "NSE Nifty Energy Index"),
            ("NIFTY FMCG", "^CNXFMCG", "NSE Nifty FMCG Index"),
            ("NIFTY INFRA", "^CNXINFRA", "NSE Nifty Infrastructure Index"),
            ("NIFTY PSE", "^CNXPSE", "NSE Nifty PSE Index"),
            ("NIFTY MEDIA", "^CNXMEDIA", "NSE Nifty Media Index"),
            ("NIFTY COMMODITIES", "^CNXCMDT", "NSE Nifty Commodities Index"),
            ("NIFTY CONSUMPTION", "^CNXCONSUMP", "NSE Nifty Consumption Index"),
            ("NIFTY CPSE", "^CNXCPSE", "NSE Nifty CPSE Index"),
            ("NIFTY FIN SERVICE", "^CNXFINANCE", "NSE Nifty Financial Services Index"),
            ("NIFTY GROWSECT 15", "^CNXGROWTH", "NSE Nifty Growth Sectors 15 Index"),
            ("NIFTY MNC", "^CNXMNC", "NSE Nifty MNC Index"),
            ("NIFTY PSU BANK", "^CNXPSUBANK", "NSE Nifty PSU Bank Index"),
            ("NIFTY PVT BANK", "^CNXPVTBANK", "NSE Nifty Private Bank Index"),
            ("NIFTY SMLCAP 50", "^NSESL50", "NSE Nifty Smallcap 50 Index"),
            ("NIFTY SMLCAP 100", "^NSESL100", "NSE Nifty Smallcap 100 Index"),
            ("NIFTY SMLCAP 250", "^NSESL250", "NSE Nifty Smallcap 250 Index"),
            ("NIFTY MIDCAP 50", "^NSMIDCP50", "NSE Nifty Midcap 50 Index"),
            ("NIFTY MIDCAP 150", "^NSEMIDCP150", "NSE Nifty Midcap 150 Index"),
        ]
        assets = [
            AssetInfo(
                symbol=symbol,
                name=name,
                exchange=Exchange.NSE if "NSE" in name else Exchange.BSE,
                asset_type=AssetType.INDEX,
                yfinance_ticker=ticker,
            )
            for symbol, ticker, name in indices
        ]
        logger.info("Discovered %d indices", len(assets))
        return assets

    # ------------------------------------------------------------------
    # ETFs
    # ------------------------------------------------------------------
    def discover_etfs(self) -> list[AssetInfo]:
        """Return curated list of major Indian ETFs."""
        etfs = [
            ("NIFTYBEES", "Nippon India ETF Nifty BeES"),
            ("BANKBEES", "Nippon India ETF Bank BeES"),
            ("JUNIORBEES", "Nippon India ETF Junior BeES"),
            ("SETFNIF50", "SBI ETF Nifty 50"),
            ("SETFNIFBK", "SBI ETF Nifty Bank"),
            ("GOLDBEES", "Nippon India ETF Gold BeES"),
            ("SILVERBEES", "Nippon India ETF Silver"),
            ("CPSEETF", "Nippon India CPSE ETF"),
            ("ITETF", "Nippon India ETF Nifty IT"),
            ("PHARMABEES", "Nippon India ETF Pharma"),
            ("MOM50", "Motilal Oswal Momentum ETF"),
            ("MIDCAPETF", "Nippon India ETF Midcap 150"),
            ("INFRAETF", "Nippon India ETF Infra BeES"),
            ("PSUBNKBEES", "Nippon India ETF PSU Bank"),
            ("HABORETF", "Groww Nifty EV & New Age Auto ETF"),
            ("MON100", "Motilal Oswal NASDAQ 100 ETF"),
            ("MOM100", "Motilal Oswal Midcap 100 ETF"),
            ("LIQUIDBEES", "Nippon India ETF Liquid BeES"),
            ("HNGSNGBEES", "Nippon India ETF Hang Seng"),
            ("COMMOETF", "Nippon India ETF Commodities"),
        ]
        assets = [
            AssetInfo(
                symbol=symbol,
                name=name,
                exchange=Exchange.NSE,
                asset_type=AssetType.ETF,
            )
            for symbol, name in etfs
        ]
        logger.info("Discovered %d ETFs", len(assets))
        return assets

    # ------------------------------------------------------------------
    # Commodities
    # ------------------------------------------------------------------
    def discover_commodities(self) -> list[AssetInfo]:
        """Return commodity futures tracked via yfinance."""
        commodities = [
            ("GOLD", "GC=F", "Gold Futures"),
            ("SILVER", "SI=F", "Silver Futures"),
            ("CRUDE OIL", "CL=F", "Crude Oil WTI Futures"),
            ("NATURAL GAS", "NG=F", "Natural Gas Futures"),
            ("COPPER", "HG=F", "Copper Futures"),
            ("PLATINUM", "PL=F", "Platinum Futures"),
            ("PALLADIUM", "PA=F", "Palladium Futures"),
            ("ALUMINIUM", "ALI=F", "Aluminium Futures"),
        ]
        assets = [
            AssetInfo(
                symbol=symbol,
                name=name,
                exchange=Exchange.MULTI,
                asset_type=AssetType.COMMODITY,
                yfinance_ticker=ticker,
            )
            for symbol, ticker, name in commodities
        ]
        logger.info("Discovered %d commodities", len(assets))
        return assets

    # ------------------------------------------------------------------
    # Forex
    # ------------------------------------------------------------------
    def discover_forex(self) -> list[AssetInfo]:
        """Return major INR forex pairs."""
        pairs = [
            ("USDINR", "USDINR=X", "US Dollar / Indian Rupee"),
            ("EURINR", "EURINR=X", "Euro / Indian Rupee"),
            ("GBPINR", "GBPINR=X", "British Pound / Indian Rupee"),
            ("JPYINR", "JPYINR=X", "Japanese Yen / Indian Rupee"),
            ("AUDINR", "AUDINR=X", "Australian Dollar / Indian Rupee"),
            ("CADINR", "CADINR=X", "Canadian Dollar / Indian Rupee"),
            ("SGDINR", "SGDINR=X", "Singapore Dollar / Indian Rupee"),
            ("CHFINR", "CHFINR=X", "Swiss Franc / Indian Rupee"),
        ]
        assets = [
            AssetInfo(
                symbol=symbol,
                name=name,
                exchange=Exchange.MULTI,
                asset_type=AssetType.FOREX,
                yfinance_ticker=ticker,
            )
            for symbol, ticker, name in pairs
        ]
        logger.info("Discovered %d forex pairs", len(assets))
        return assets

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------
    def discover_all(self) -> list[AssetInfo]:
        """Discover the full universe of supported assets.

        Returns:
            Combined list of all discovered assets.
        """
        all_assets: list[AssetInfo] = []

        all_assets.extend(self.discover_nse_equities())
        all_assets.extend(self.discover_bse_equities())
        all_assets.extend(self.discover_indices())
        all_assets.extend(self.discover_etfs())
        all_assets.extend(self.discover_commodities())
        all_assets.extend(self.discover_forex())

        logger.info("Total assets discovered: %d", len(all_assets))
        return all_assets

    def save_universe(self, assets: list[AssetInfo], metadata_dir: Path) -> Path:
        """Persist the asset universe to a Parquet file.

        Args:
            assets: List of discovered assets.
            metadata_dir: Directory to write the Parquet file to.

        Returns:
            Path to the written file.
        """
        metadata_dir.mkdir(parents=True, exist_ok=True)
        out_path = metadata_dir / "asset_universe.parquet"

        records = [a.model_dump() for a in assets]
        df = pd.DataFrame(records)

        # Convert enums to their string values for storage
        for col in ("asset_type", "exchange"):
            if col in df.columns:
                df[col] = df[col].astype(str)

        # Convert listing_date to proper date type
        if "listing_date" in df.columns:
            df["listing_date"] = pd.to_datetime(df["listing_date"])

        df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
        logger.info("Saved asset universe (%d assets) to %s", len(assets), out_path)
        return out_path

"""Parquet-based storage for historical market data.

Each asset is stored in a separate Parquet file under a directory
corresponding to its asset type.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from market_data.assets.models import AssetInfo

logger = logging.getLogger("market_data.storage.parquet_store")

# Canonical schema for OHLCV data
OHLCV_SCHEMA = pa.schema([
    ("date", pa.date32()),
    ("open", pa.float64()),
    ("high", pa.float64()),
    ("low", pa.float64()),
    ("close", pa.float64()),
    ("adj_close", pa.float64()),
    ("volume", pa.int64()),
    ("dividends", pa.float64()),
    ("stock_splits", pa.float64()),
    ("symbol", pa.string()),
])


class ParquetStore:
    """Manages Parquet file storage for historical market data.

    Files are stored at:
        {data_dir}/{asset_type_dir}/{symbol}.parquet

    Args:
        data_dir: Root data directory (must exist or will be created).
    """

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir

    def _asset_path(self, asset: AssetInfo) -> Path:
        """Return the Parquet file path for an asset."""
        # Map asset_type to directory
        type_dir_map = {
            "EQUITY": "stocks",
            "INDEX": "indices",
            "ETF": "etfs",
            "COMMODITY": "commodities",
            "FOREX": "forex",
        }
        subdir = type_dir_map.get(asset.asset_type.value, "other")
        directory = self._data_dir / subdir
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{asset.safe_filename}.parquet"

    def save(self, asset: AssetInfo, df: pd.DataFrame) -> Path:
        """Save or append OHLCV data for an asset.

        If the file already exists, the new data is appended and
        duplicates (by date) are removed, keeping the latest.

        Args:
            asset: Asset metadata.
            df: DataFrame with OHLCV data.

        Returns:
            Path to the written Parquet file.
        """
        path = self._asset_path(asset)

        if df.empty:
            logger.debug("Empty DataFrame for %s — nothing to save", asset.symbol)
            return path

        # Ensure proper types before merging
        df = self._coerce_types(df)

        if path.exists():
            existing = self.load(asset)
            df = pd.concat([existing, df], ignore_index=True)
            # Remove duplicate dates, keeping the latest entry
            df = df.drop_duplicates(subset=["date"], keep="last")

        # Sort by date
        df = df.sort_values("date").reset_index(drop=True)

        # Write with pyarrow
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, path, compression="snappy")

        logger.debug("Saved %d rows for %s → %s", len(df), asset.symbol, path)
        return path

    def load(
        self,
        asset: AssetInfo,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """Load OHLCV data for an asset, optionally filtering by date range.

        Args:
            asset: Asset metadata.
            start_date: Inclusive start date filter.
            end_date: Inclusive end date filter.

        Returns:
            DataFrame with OHLCV data, or empty DataFrame if file doesn't exist.
        """
        path = self._asset_path(asset)
        if not path.exists():
            logger.debug("No data file for %s at %s", asset.symbol, path)
            return pd.DataFrame()

        df = pd.read_parquet(path, engine="pyarrow")

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date

            if start_date is not None:
                df = df[df["date"] >= start_date]
            if end_date is not None:
                df = df[df["date"] <= end_date]

        return df.reset_index(drop=True)

    def exists(self, asset: AssetInfo) -> bool:
        """Check whether a Parquet file exists for the asset."""
        return self._asset_path(asset).exists()

    def get_last_date(self, asset: AssetInfo) -> date | None:
        """Return the last (most recent) date in the stored data.

        Returns None if no data exists for this asset.
        """
        path = self._asset_path(asset)
        if not path.exists():
            return None

        try:
            table = pq.read_table(path, columns=["date"])
            if table.num_rows == 0:
                return None
            dates = table.column("date").to_pylist()
            return max(dates)
        except Exception:
            logger.warning("Could not read last date from %s", path, exc_info=True)
            return None

    def get_row_count(self, asset: AssetInfo) -> int:
        """Return the number of rows stored for an asset."""
        path = self._asset_path(asset)
        if not path.exists():
            return 0
        try:
            metadata = pq.read_metadata(path)
            return metadata.num_rows
        except Exception:
            return 0

    def get_file_size_bytes(self, asset: AssetInfo) -> int:
        """Return the file size in bytes for the asset's Parquet file."""
        path = self._asset_path(asset)
        if not path.exists():
            return 0
        return path.stat().st_size

    def delete(self, asset: AssetInfo) -> bool:
        """Delete the Parquet file for an asset.

        Returns True if the file was deleted, False if it didn't exist.
        """
        path = self._asset_path(asset)
        if path.exists():
            path.unlink()
            logger.info("Deleted data file for %s: %s", asset.symbol, path)
            return True
        return False

    def list_stored_assets(self) -> list[str]:
        """Return a list of all symbols that have stored Parquet files."""
        symbols: list[str] = []
        for subdir in ("stocks", "indices", "etfs", "commodities", "forex"):
            directory = self._data_dir / subdir
            if directory.exists():
                for f in directory.glob("*.parquet"):
                    symbols.append(f.stem)
        return sorted(symbols)

    @staticmethod
    def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
        """Coerce DataFrame columns to consistent types."""
        df = df.copy()

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date

        float_cols = ["open", "high", "low", "close", "adj_close", "dividends", "stock_splits"]
        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")

        return df

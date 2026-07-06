"""Data cleaning utilities to repair data quality anomalies in Parquet files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from market_data.assets.models import AssetInfo
    from market_data.storage.catalog import DuckDBCatalog
    from market_data.storage.parquet_store import ParquetStore

logger = logging.getLogger("market_data.validation.cleaner")


class DataCleaner:
    """Repairs data quality anomalies in stored Parquet data files.

    Args:
        store: ParquetStore instance to read/write data.
        catalog: Optional DuckDBCatalog to update validation status logs.
    """

    def __init__(
        self,
        store: ParquetStore,
        catalog: DuckDBCatalog | None = None,
    ) -> None:
        self._store = store
        self._catalog = catalog

    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Repair OHLC anomalies in-place.

        Ensures:
            - high >= low
            - open is within [low, high]
            - close is within [low, high]

        Args:
            df: DataFrame to clean.

        Returns:
            A tuple of (cleaned DataFrame, count of repaired rows).
        """
        if df.empty:
            return df, 0

        # Check for required columns
        required = {"open", "high", "low", "close"}
        if not required.issubset(df.columns):
            return df, 0

        # Find rows with violations before modifying them
        violation_mask = (
            (df["high"] < df["low"]) |
            (df["open"] > df["high"]) |
            (df["open"] < df["low"]) |
            (df["close"] > df["high"]) |
            (df["close"] < df["low"])
        )
        repaired_count = int(violation_mask.sum())

        if repaired_count == 0:
            return df, 0

        df = df.copy()

        # Compute new high and low from original values
        new_high = df[["open", "high", "low", "close"]].max(axis=1)
        new_low = df[["open", "high", "low", "close"]].min(axis=1)

        # Enforce bounds
        df["high"] = new_high
        df["low"] = new_low

        return df, repaired_count

    def clean_asset(self, asset: AssetInfo) -> int:
        """Load, clean, and save Parquet file for a single asset.

        Updates validation status to PASSED in catalog if repaired.

        Args:
            asset: Asset metadata.

        Returns:
            Number of repaired rows.
        """
        if not self._store.exists(asset):
            return 0

        df = self._store.load(asset)
        cleaned_df, repaired_count = self.clean_dataframe(df)

        if repaired_count > 0:
            self._store.save(asset, cleaned_df)
            logger.info("Repaired %d rows for %s", repaired_count, asset.symbol)

            # Update validation catalog log
            if self._catalog is not None:
                self._catalog.log_validation(
                    symbol=asset.symbol,
                    check_name="ohlc_consistency",
                    status="PASSED",
                    details=f"Repaired {repaired_count} OHLC consistency violations",
                )
        
        return repaired_count

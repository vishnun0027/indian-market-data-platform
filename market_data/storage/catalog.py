"""DuckDB-based metadata catalog.

Tracks asset metadata, download history, and validation results in a
lightweight embedded database alongside the Parquet data files.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from market_data.assets.models import AssetInfo

logger = logging.getLogger("market_data.storage.catalog")


class DuckDBCatalog:
    """Metadata catalog backed by DuckDB.

    Stores asset metadata, download logs, and validation results.

    Args:
        db_path: Path to the DuckDB database file.
    """

    def __init__(self, db_path: Path, read_only: bool = False) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._conn = duckdb.connect(str(db_path), read_only=read_only)
        if not read_only:
            self._init_tables()

    def _init_tables(self) -> None:
        """Create tables if they don't exist."""
        # Create sequences first — tables reference them via nextval()
        self._conn.execute(
            "CREATE SEQUENCE IF NOT EXISTS download_log_seq START 1"
        )
        self._conn.execute(
            "CREATE SEQUENCE IF NOT EXISTS validation_log_seq START 1"
        )

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS assets (
                symbol VARCHAR PRIMARY KEY,
                name VARCHAR,
                isin VARCHAR,
                exchange VARCHAR,
                sector VARCHAR,
                industry VARCHAR,
                listing_date DATE,
                asset_type VARCHAR,
                yfinance_ticker VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS download_log (
                id INTEGER PRIMARY KEY DEFAULT nextval('download_log_seq'),
                symbol VARCHAR NOT NULL,
                download_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                start_date DATE,
                end_date DATE,
                rows_downloaded INTEGER DEFAULT 0,
                file_path VARCHAR,
                file_size_bytes BIGINT DEFAULT 0,
                status VARCHAR DEFAULT 'success',
                error_message VARCHAR,
                FOREIGN KEY (symbol) REFERENCES assets(symbol)
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS validation_log (
                id INTEGER PRIMARY KEY DEFAULT nextval('validation_log_seq'),
                symbol VARCHAR NOT NULL,
                check_name VARCHAR NOT NULL,
                status VARCHAR NOT NULL,
                details VARCHAR,
                validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (symbol) REFERENCES assets(symbol)
            )
        """)

    # ------------------------------------------------------------------
    # Asset operations
    # ------------------------------------------------------------------
    def upsert_asset(self, asset: AssetInfo) -> None:
        """Insert or update an asset record."""
        from datetime import datetime

        now = datetime.now()
        self._conn.execute("""
            INSERT INTO assets (symbol, name, isin, exchange, sector, industry,
                                listing_date, asset_type, yfinance_ticker, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (symbol) DO UPDATE SET
                name = EXCLUDED.name,
                isin = EXCLUDED.isin,
                exchange = EXCLUDED.exchange,
                sector = EXCLUDED.sector,
                industry = EXCLUDED.industry,
                listing_date = EXCLUDED.listing_date,
                asset_type = EXCLUDED.asset_type,
                yfinance_ticker = EXCLUDED.yfinance_ticker,
                updated_at = EXCLUDED.updated_at
        """, [
            asset.symbol,
            asset.name,
            asset.isin,
            asset.exchange.value,
            asset.sector,
            asset.industry,
            asset.listing_date,
            asset.asset_type.value,
            asset.yfinance_ticker,
            now,
        ])

    def upsert_assets(self, assets: list[AssetInfo]) -> None:
        """Insert or update multiple asset records."""
        for asset in assets:
            self.upsert_asset(asset)
        logger.info("Upserted %d assets into catalog", len(assets))

    def get_asset(self, symbol: str) -> dict[str, Any] | None:
        """Retrieve an asset record by symbol."""
        result = self._conn.execute(
            "SELECT * FROM assets WHERE symbol = ?", [symbol]
        ).fetchone()
        if result is None:
            return None
        columns = [desc[0] for desc in self._conn.description]
        return dict(zip(columns, result))

    def get_all_assets(self) -> list[dict[str, Any]]:
        """Retrieve all asset records."""
        result = self._conn.execute("SELECT * FROM assets ORDER BY symbol").fetchall()
        columns = [desc[0] for desc in self._conn.description]
        return [dict(zip(columns, row)) for row in result]

    def count_assets(self) -> int:
        """Return the total number of assets in the catalog."""
        return self._conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0]

    # ------------------------------------------------------------------
    # Download log operations
    # ------------------------------------------------------------------
    def log_download(
        self,
        symbol: str,
        rows_downloaded: int,
        file_path: str = "",
        file_size_bytes: int = 0,
        status: str = "success",
        error_message: str = "",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> None:
        """Log a download event."""
        self._conn.execute("""
            INSERT INTO download_log (symbol, start_date, end_date, rows_downloaded,
                                       file_path, file_size_bytes, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            symbol,
            start_date,
            end_date,
            rows_downloaded,
            file_path,
            file_size_bytes,
            status,
            error_message,
        ])

    def get_download_history(self, symbol: str) -> list[dict[str, Any]]:
        """Get download history for a symbol."""
        result = self._conn.execute(
            "SELECT * FROM download_log WHERE symbol = ? ORDER BY download_time DESC",
            [symbol],
        ).fetchall()
        columns = [desc[0] for desc in self._conn.description]
        return [dict(zip(columns, row)) for row in result]

    # ------------------------------------------------------------------
    # Validation log operations
    # ------------------------------------------------------------------
    def log_validation(
        self,
        symbol: str,
        check_name: str,
        status: str,
        details: str = "",
    ) -> None:
        """Log a validation result."""
        self._conn.execute("""
            INSERT INTO validation_log (symbol, check_name, status, details)
            VALUES (?, ?, ?, ?)
        """, [symbol, check_name, status, details])

    def get_validation_results(self, symbol: str) -> list[dict[str, Any]]:
        """Get validation results for a symbol."""
        result = self._conn.execute(
            "SELECT * FROM validation_log WHERE symbol = ? ORDER BY validated_at DESC",
            [symbol],
        ).fetchall()
        columns = [desc[0] for desc in self._conn.description]
        return [dict(zip(columns, row)) for row in result]

    # ------------------------------------------------------------------
    # Status / summary
    # ------------------------------------------------------------------
    def get_status_summary(self) -> dict[str, Any]:
        """Return a summary of the catalog status."""
        total_assets = self.count_assets()

        asset_type_counts = self._conn.execute(
            "SELECT asset_type, COUNT(*) as cnt FROM assets GROUP BY asset_type ORDER BY asset_type"
        ).fetchall()

        recent_downloads = self._conn.execute(
            "SELECT COUNT(*) FROM download_log WHERE download_time > CURRENT_TIMESTAMP - INTERVAL '24 hours'"
        ).fetchone()[0]

        failed_downloads = self._conn.execute(
            "SELECT COUNT(*) FROM download_log WHERE status = 'error'"
        ).fetchone()[0]

        return {
            "total_assets": total_assets,
            "asset_type_counts": {row[0]: row[1] for row in asset_type_counts},
            "recent_downloads_24h": recent_downloads,
            "total_failed_downloads": failed_downloads,
        }

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def query(self, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
        """Execute an arbitrary read-only query.

        Uses parameterized queries only — no string interpolation.

        Args:
            sql: SQL query string with ? placeholders.
            params: Parameter values for the placeholders.
        """
        if params:
            result = self._conn.execute(sql, params).fetchall()
        else:
            result = self._conn.execute(sql).fetchall()
        columns = [desc[0] for desc in self._conn.description]
        return [dict(zip(columns, row)) for row in result]

    def close(self) -> None:
        """Close the DuckDB connection."""
        self._conn.close()

    def __enter__(self) -> DuckDBCatalog:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

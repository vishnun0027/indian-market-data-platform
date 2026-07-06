"""Download engine — fetches historical OHLCV data via yfinance.

Supports parallel downloads, resume from last downloaded date,
automatic retries with exponential backoff, rate limiting,
adaptive throttle detection, batch cooldowns, and randomized
request ordering to minimize IP blocking risk.
"""

from __future__ import annotations

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd
import yfinance as yf
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from market_data.downloader.rate_limiter import RateLimiter

if TYPE_CHECKING:
    from market_data.assets.models import AssetInfo
    from market_data.storage.parquet_store import ParquetStore

logger = logging.getLogger("market_data.downloader.engine")

# HTTP status / error strings that indicate throttling
_THROTTLE_INDICATORS = (
    "429",
    "too many requests",
    "rate limit",
    "connection reset",
    "connection aborted",
    "read timed out",
    "no data found",
)

# Default batch cooldown settings
_BATCH_COOLDOWN_SIZE = 50       # Pause every N successful downloads
_BATCH_COOLDOWN_SECS = 30.0    # Pause duration in seconds


class DownloadResult:
    """Result of a single asset download attempt."""

    __slots__ = ("asset", "df", "error", "skipped", "rows_added")

    def __init__(
        self,
        asset: AssetInfo,
        df: pd.DataFrame | None = None,
        error: str | None = None,
        skipped: bool = False,
        rows_added: int = 0,
    ) -> None:
        self.asset = asset
        self.df = df
        self.error = error
        self.skipped = skipped
        self.rows_added = rows_added

    @property
    def success(self) -> bool:
        return self.df is not None and not self.df.empty and self.error is None


class DownloadEngine:
    """Orchestrates parallel historical data downloads with IP-block protection.

    Protection layers:
        1. Rate limiting with random jitter (±30%)
        2. Adaptive backoff — slows down automatically on throttle signals
        3. Batch cooldown — pauses every N downloads to let the server breathe
        4. Randomized order — shuffles asset list to avoid predictable patterns
        5. Exponential backoff retries via tenacity

    Args:
        store: ParquetStore instance for resume detection and data persistence.
        max_workers: Maximum number of parallel download threads.
        rate_limit: Max requests per second.
        retry_attempts: Number of retry attempts per asset.
        retry_wait: Base wait time in seconds for exponential backoff.
        batch_cooldown_size: Pause after this many successful downloads.
        batch_cooldown_secs: How long to pause during batch cooldown.
    """

    def __init__(
        self,
        store: ParquetStore,
        max_workers: int = 4,
        rate_limit: float = 2.0,
        retry_attempts: int = 3,
        retry_wait: float = 5.0,
        batch_cooldown_size: int = _BATCH_COOLDOWN_SIZE,
        batch_cooldown_secs: float = _BATCH_COOLDOWN_SECS,
    ) -> None:
        self._store = store
        self._max_workers = max_workers
        self._rate_limiter = RateLimiter(max_per_second=rate_limit)
        self._retry_attempts = retry_attempts
        self._retry_wait = retry_wait
        self._batch_cooldown_size = batch_cooldown_size
        self._batch_cooldown_secs = batch_cooldown_secs
        self._consecutive_errors = 0

    # ------------------------------------------------------------------
    # Single asset download
    # ------------------------------------------------------------------
    def download_single(
        self,
        asset: AssetInfo,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> DownloadResult:
        """Download historical data for a single asset.

        Automatically determines the start date based on existing data
        (resume support).  If the data is already up-to-date, returns
        a skipped result.

        Args:
            asset: The asset to download.
            start_date: Override start date.  If None, resumes from last date.
            end_date: Override end date.  If None, uses today.

        Returns:
            DownloadResult with the fetched DataFrame or error details.
        """
        if end_date is None:
            end_date = date.today()

        # Resume: check last available date
        if start_date is None:
            last_date = self._store.get_last_date(asset)
            if last_date is not None:
                # Start from the day after the last available date
                start_date = last_date + timedelta(days=1)
                if start_date >= end_date:
                    logger.debug("Asset %s is up-to-date (last: %s)", asset.symbol, last_date)
                    return DownloadResult(asset=asset, skipped=True)
            else:
                # No existing data — download from the beginning
                start_date = date(1990, 1, 1)

        try:
            df = self._fetch_with_retry(
                ticker=asset.yfinance_ticker,
                start=start_date.isoformat(),
                end=end_date.isoformat(),
            )
            # Reset consecutive error counter on success
            self._consecutive_errors = 0
        except Exception as exc:
            error_str = str(exc).lower()
            self._consecutive_errors += 1

            # Detect throttling and signal the rate limiter
            if self._is_throttle_error(error_str):
                self._rate_limiter.report_throttle()
                logger.warning(
                    "Throttle detected for %s — adaptive backoff engaged "
                    "(consecutive errors: %d, current rate: %.2f req/s)",
                    asset.symbol,
                    self._consecutive_errors,
                    self._rate_limiter.current_rate,
                )
                # Extra sleep on throttle to let things cool down
                cooldown = min(10.0 * self._consecutive_errors, 120.0)
                logger.info("Cooling down for %.0fs before next request...", cooldown)
                time.sleep(cooldown)
            else:
                logger.error("Failed to download %s: %s", asset.symbol, exc)

            return DownloadResult(asset=asset, error=str(exc))

        if df is None or df.empty:
            logger.warning("No data returned for %s (%s → %s)", asset.symbol, start_date, end_date)
            return DownloadResult(
                asset=asset,
                df=pd.DataFrame(),
                error=None,
                rows_added=0,
            )

        # Normalize columns
        df = self._normalize_dataframe(df, asset)

        # Persist
        self._store.save(asset, df)
        rows = len(df)
        logger.info("Downloaded %d rows for %s", rows, asset.symbol)
        return DownloadResult(asset=asset, df=df, rows_added=rows)

    @staticmethod
    def _is_throttle_error(error_str: str) -> bool:
        """Check if an error message indicates rate-limiting / IP blocking."""
        return any(indicator in error_str for indicator in _THROTTLE_INDICATORS)

    def _fetch_with_retry(self, ticker: str, start: str, end: str) -> pd.DataFrame | None:
        """Fetch data from yfinance with rate limiting and retries."""

        @retry(
            stop=stop_after_attempt(self._retry_attempts),
            wait=wait_exponential(multiplier=self._retry_wait, min=2, max=60),
            retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
            reraise=True,
        )
        def _do_fetch() -> pd.DataFrame | None:
            self._rate_limiter.acquire()
            logger.debug("Fetching %s from %s to %s", ticker, start, end)
            df = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=False,
                actions=True,
                threads=False,
            )
            return df

        return _do_fetch()

    @staticmethod
    def _normalize_dataframe(df: pd.DataFrame, asset: AssetInfo) -> pd.DataFrame:
        """Normalize yfinance DataFrame to a consistent schema.

        Ensures columns: Date, Open, High, Low, Close, Adj Close, Volume,
        Dividends, Stock Splits, Symbol.
        """
        df = df.copy()

        # yfinance may return MultiIndex columns for single tickers
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure index is a proper DatetimeIndex named 'Date'
        if df.index.name != "Date":
            df.index.name = "Date"
        df = df.reset_index()

        # Standardize column names
        column_renames = {
            "Adj Close": "adj_close",
            "Stock Splits": "stock_splits",
        }
        df.columns = [column_renames.get(str(c), str(c).lower().replace(" ", "_")) for c in df.columns]

        # Ensure date column is proper datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date

        # Add symbol column
        df["symbol"] = asset.symbol

        # Fill missing action columns
        for col in ("dividends", "stock_splits"):
            if col not in df.columns:
                df[col] = 0.0

        # Desired column order
        desired = ["date", "open", "high", "low", "close", "adj_close", "volume",
                    "dividends", "stock_splits", "symbol"]
        existing = [c for c in desired if c in df.columns]
        df = df[existing]

        return df

    # ------------------------------------------------------------------
    # Batch download
    # ------------------------------------------------------------------
    def download_batch(
        self,
        assets: list[AssetInfo],
        start_date: date | None = None,
        end_date: date | None = None,
        show_progress: bool = True,
        shuffle: bool = True,
    ) -> list[DownloadResult]:
        """Download historical data for multiple assets in parallel.

        Args:
            assets: List of assets to download.
            start_date: Override start date for all assets.
            end_date: Override end date for all assets.
            show_progress: Whether to display a Rich progress bar.
            shuffle: Randomize download order (reduces pattern detection).

        Returns:
            List of DownloadResult for each asset.
        """
        results: list[DownloadResult] = []

        # Shuffle to avoid predictable sequential patterns that
        # look like bot traffic to the server
        work_list = list(assets)
        if shuffle:
            random.shuffle(work_list)
            logger.info("Shuffled download order to reduce bot-detection risk")

        if show_progress:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
            )
        else:
            progress = None

        success_counter = 0

        def _download_one(asset: AssetInfo) -> DownloadResult:
            return self.download_single(asset, start_date=start_date, end_date=end_date)

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            if progress:
                with progress:
                    task_id = progress.add_task("Downloading...", total=len(work_list))
                    futures = {
                        executor.submit(_download_one, asset): asset for asset in work_list
                    }
                    for future in as_completed(futures):
                        result = future.result()
                        results.append(result)
                        progress.advance(task_id)

                        # Batch cooldown: pause periodically to avoid sustained load
                        if result.success:
                            success_counter += 1
                            if (
                                success_counter > 0
                                and success_counter % self._batch_cooldown_size == 0
                            ):
                                cooldown = self._batch_cooldown_secs + random.uniform(0, 10)
                                progress.update(
                                    task_id,
                                    description=(
                                        f"[yellow]Cooling down ({cooldown:.0f}s) "
                                        f"after {success_counter} downloads..."
                                    ),
                                )
                                logger.info(
                                    "Batch cooldown: pausing %.0fs after %d downloads "
                                    "(rate: %.2f req/s, throttles: %d)",
                                    cooldown,
                                    success_counter,
                                    self._rate_limiter.current_rate,
                                    self._rate_limiter.throttle_count,
                                )
                                time.sleep(cooldown)
                                progress.update(task_id, description="Downloading...")
            else:
                futures = {
                    executor.submit(_download_one, asset): asset for asset in work_list
                }
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)

                    if result.success:
                        success_counter += 1
                        if (
                            success_counter > 0
                            and success_counter % self._batch_cooldown_size == 0
                        ):
                            cooldown = self._batch_cooldown_secs + random.uniform(0, 10)
                            logger.info("Batch cooldown: pausing %.0fs", cooldown)
                            time.sleep(cooldown)

        # Summary
        success_count = sum(1 for r in results if r.success)
        skip_count = sum(1 for r in results if r.skipped)
        fail_count = sum(1 for r in results if r.error)
        total_rows = sum(r.rows_added for r in results)

        logger.info(
            "Batch download complete: %d success, %d skipped, %d failed, %d total rows "
            "(throttle events: %d, final rate: %.2f req/s)",
            success_count,
            skip_count,
            fail_count,
            total_rows,
            self._rate_limiter.throttle_count,
            self._rate_limiter.current_rate,
        )

        return results

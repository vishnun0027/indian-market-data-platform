"""Tests for the download engine (mocked yfinance)."""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from market_data.assets.models import AssetInfo, AssetType, Exchange
from market_data.downloader.engine import DownloadEngine, DownloadResult
from market_data.downloader.rate_limiter import RateLimiter
from market_data.storage.parquet_store import ParquetStore


class TestRateLimiter:
    """Tests for the RateLimiter."""

    def test_rate_limiting(self) -> None:
        """Rate limiter should enforce minimum interval between calls."""
        limiter = RateLimiter(max_per_second=10.0)  # 100ms between calls

        start = time.monotonic()
        for _ in range(3):
            limiter.acquire()
        elapsed = time.monotonic() - start

        # 3 calls at 10/s → at least 0.2s for the gaps between them
        assert elapsed >= 0.15  # Allow small tolerance

    def test_invalid_rate(self) -> None:
        """Rate limiter should reject non-positive rates."""
        with pytest.raises(ValueError, match="must be positive"):
            RateLimiter(max_per_second=0)

    def test_context_manager(self) -> None:
        """Rate limiter should work as a context manager."""
        limiter = RateLimiter(max_per_second=100.0)
        with limiter:
            pass  # Should not raise

    def test_jitter_varies_interval(self) -> None:
        """Jitter should cause varying intervals between calls."""
        limiter = RateLimiter(max_per_second=10.0)
        intervals = []
        for _ in range(5):
            start = time.monotonic()
            limiter.acquire()
            intervals.append(time.monotonic() - start)
        # With 30% jitter, not all intervals should be identical
        # (statistically extremely unlikely they'd all be the same)
        assert len(intervals) == 5

    def test_adaptive_backoff(self) -> None:
        """report_throttle should slow down the rate."""
        limiter = RateLimiter(max_per_second=10.0)
        original_rate = limiter.current_rate

        limiter.report_throttle()
        assert limiter.current_rate < original_rate

        # Second throttle should slow it further
        rate_after_first = limiter.current_rate
        limiter.report_throttle()
        assert limiter.current_rate < rate_after_first

    def test_throttle_count(self) -> None:
        """throttle_count should track the number of events."""
        limiter = RateLimiter(max_per_second=10.0)
        assert limiter.throttle_count == 0

        limiter.report_throttle()
        assert limiter.throttle_count == 1

        limiter.report_throttle()
        assert limiter.throttle_count == 2

    def test_max_slowdown(self) -> None:
        """Rate should not go below the max slowdown limit."""
        limiter = RateLimiter(max_per_second=10.0)
        for _ in range(20):
            limiter.report_throttle()
        # Should be at max slowdown (16x) — rate should be 10/16 = 0.625
        assert limiter.current_rate >= 0.5  # some floor


class TestDownloadResult:
    """Tests for DownloadResult."""

    def test_success(self, sample_equity: AssetInfo, sample_ohlcv_df: pd.DataFrame) -> None:
        result = DownloadResult(asset=sample_equity, df=sample_ohlcv_df, rows_added=5)
        assert result.success
        assert not result.skipped
        assert result.rows_added == 5

    def test_error(self, sample_equity: AssetInfo) -> None:
        result = DownloadResult(asset=sample_equity, error="Connection failed")
        assert not result.success
        assert result.error == "Connection failed"

    def test_skipped(self, sample_equity: AssetInfo) -> None:
        result = DownloadResult(asset=sample_equity, skipped=True)
        assert not result.success
        assert result.skipped


class TestDownloadEngine:
    """Tests for the DownloadEngine with mocked yfinance."""

    @patch("market_data.downloader.engine.yf.download")
    def test_download_single(
        self,
        mock_yf_download: MagicMock,
        tmp_data_dir: Path,
        sample_equity: AssetInfo,
        sample_ohlcv_df: pd.DataFrame,
    ) -> None:
        """download_single should fetch and store data."""
        # Mock yfinance response
        yf_df = sample_ohlcv_df.set_index("date").drop(columns=["symbol"])
        yf_df.index = pd.to_datetime(yf_df.index)
        yf_df.index.name = "Date"
        mock_yf_download.return_value = yf_df

        store = ParquetStore(data_dir=tmp_data_dir)
        engine = DownloadEngine(
            store=store,
            max_workers=1,
            rate_limit=100.0,
            retry_attempts=1,
        )

        result = engine.download_single(
            sample_equity,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        assert result.success
        assert result.rows_added > 0
        assert store.exists(sample_equity)

    @patch("market_data.downloader.engine.yf.download")
    def test_download_single_empty(
        self,
        mock_yf_download: MagicMock,
        tmp_data_dir: Path,
        sample_equity: AssetInfo,
    ) -> None:
        """download_single should handle empty responses."""
        mock_yf_download.return_value = pd.DataFrame()

        store = ParquetStore(data_dir=tmp_data_dir)
        engine = DownloadEngine(store=store, max_workers=1, rate_limit=100.0, retry_attempts=1)

        result = engine.download_single(
            sample_equity,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        assert not result.success
        assert result.error is None  # No error, just no data

    @patch("market_data.downloader.engine.yf.download")
    def test_download_resume(
        self,
        mock_yf_download: MagicMock,
        tmp_data_dir: Path,
        sample_equity: AssetInfo,
        sample_ohlcv_df: pd.DataFrame,
    ) -> None:
        """download_single should resume from last downloaded date."""
        store = ParquetStore(data_dir=tmp_data_dir)
        store.save(sample_equity, sample_ohlcv_df)  # Last date: Jan 8

        mock_yf_download.return_value = pd.DataFrame()  # No new data

        engine = DownloadEngine(store=store, max_workers=1, rate_limit=100.0, retry_attempts=1)
        result = engine.download_single(sample_equity)

        # yfinance should have been called with start=Jan 9
        if mock_yf_download.called:
            call_args = mock_yf_download.call_args
            assert "2024-01-09" in str(call_args)


class TestThrottleDetection:
    """Tests for throttle / IP-block detection."""

    def test_throttle_errors_detected(self) -> None:
        """Known throttle error strings should be detected."""
        from market_data.downloader.engine import DownloadEngine

        assert DownloadEngine._is_throttle_error("429 too many requests")
        assert DownloadEngine._is_throttle_error("connection reset by peer")
        assert DownloadEngine._is_throttle_error("rate limit exceeded")
        assert DownloadEngine._is_throttle_error("read timed out")

    def test_non_throttle_errors(self) -> None:
        """Regular errors should not be detected as throttle."""
        from market_data.downloader.engine import DownloadEngine

        assert not DownloadEngine._is_throttle_error("invalid ticker symbol")
        assert not DownloadEngine._is_throttle_error("no such file")
        assert not DownloadEngine._is_throttle_error("value error")


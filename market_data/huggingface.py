"""HuggingFace Hub integration for dataset sync.

Handles pulling existing datasets from HuggingFace, pushing updated
Parquet files, and generating dataset cards with schema documentation.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError

logger = logging.getLogger("market_data.huggingface")

# Subdirectories that contain Parquet data files
_DATA_SUBDIRS = ("stocks", "indices", "etfs", "commodities", "forex")

# Files/dirs within data/ that should be synced to HuggingFace
_SYNC_SUBDIRS = _DATA_SUBDIRS + ("metadata",)


class HuggingFaceSync:
    """Sync local Parquet data with a HuggingFace dataset repository.

    Args:
        repo_id: HuggingFace dataset repository (e.g. "username/indian-market-data").
        token: HuggingFace write-access token.
        private: Whether the repo should be private.
    """

    def __init__(self, repo_id: str, token: str, private: bool = False) -> None:
        if not repo_id:
            msg = "HuggingFace repo_id is required. Set HF_REPO_ID in your .env or pass --repo-id."
            raise ValueError(msg)
        if not token:
            msg = "HuggingFace token is required. Set HF_TOKEN in your .env or as a GitHub secret."
            raise ValueError(msg)

        self._repo_id = repo_id
        self._token = token
        self._private = private
        self._api = HfApi(token=token)

    @property
    def repo_id(self) -> str:
        return self._repo_id

    # ------------------------------------------------------------------
    # Repository management
    # ------------------------------------------------------------------
    def ensure_repo(self) -> str:
        """Create the HuggingFace dataset repo if it doesn't exist.

        Returns:
            The repo URL.
        """
        try:
            info = self._api.repo_info(repo_id=self._repo_id, repo_type="dataset")
            logger.info("HuggingFace repo exists: %s", info.id)
            return f"https://huggingface.co/datasets/{self._repo_id}"
        except RepositoryNotFoundError:
            logger.info("Creating HuggingFace dataset repo: %s (private=%s)", self._repo_id, self._private)
            url = self._api.create_repo(
                repo_id=self._repo_id,
                repo_type="dataset",
                private=self._private,
                exist_ok=True,
            )
            logger.info("Created repo: %s", url)
            return str(url)

    # ------------------------------------------------------------------
    # Pull
    # ------------------------------------------------------------------
    def pull_dataset(self, local_dir: Path) -> Path:
        """Download the dataset from HuggingFace to a local directory.

        Only downloads Parquet files and metadata. If the repo doesn't
        exist yet, creates it and returns the empty local_dir.

        Args:
            local_dir: Local directory to download into.

        Returns:
            Path to the local directory with downloaded data.
        """
        self.ensure_repo()

        try:
            # Check if repo has any files
            files = self._api.list_repo_files(repo_id=self._repo_id, repo_type="dataset")
            if not files:
                logger.info("HuggingFace repo is empty — nothing to pull")
                return local_dir
        except RepositoryNotFoundError:
            logger.info("Repo not found during pull — will be created on push")
            return local_dir

        logger.info("Pulling dataset from %s → %s", self._repo_id, local_dir)

        # Download only relevant files (Parquet + metadata)
        allow_patterns = [
            "*.parquet",
            "metadata/**",
            "README.md",
            "sync_status.json",
        ]

        snapshot_path = snapshot_download(
            repo_id=self._repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
            allow_patterns=allow_patterns,
            token=self._token,
        )

        logger.info("Pulled dataset to %s", snapshot_path)
        return Path(snapshot_path)

    # ------------------------------------------------------------------
    # Push
    # ------------------------------------------------------------------
    def push_dataset(
        self,
        local_dir: Path,
        commit_message: str | None = None,
    ) -> str:
        """Push local data directory to HuggingFace.

        Uploads all Parquet files, metadata, and the dataset card.

        Args:
            local_dir: Local data directory to upload.
            commit_message: Git commit message for the upload.

        Returns:
            The commit URL.
        """
        self.ensure_repo()

        if commit_message is None:
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            commit_message = f"Update market data — {now}"

        # Write sync status
        self._write_sync_status(local_dir)

        # Generate dataset card
        card_path = self._generate_dataset_card(local_dir)

        logger.info("Pushing dataset to %s ...", self._repo_id)

        # Upload the data directory
        # We upload specific subdirectories to keep the repo clean
        commit_info = self._api.upload_folder(
            repo_id=self._repo_id,
            repo_type="dataset",
            folder_path=str(local_dir),
            commit_message=commit_message,
            ignore_patterns=[
                "logs/**",
                "*.duckdb",
                "*.duckdb.wal",
                ".huggingface/**",
            ],
        )

        logger.info("Pushed dataset: %s", commit_info)
        return str(commit_info)

    # ------------------------------------------------------------------
    # Sync status tracking
    # ------------------------------------------------------------------
    def _write_sync_status(self, local_dir: Path) -> Path:
        """Write a sync status JSON file with current statistics."""
        status = {
            "last_sync_utc": datetime.now(timezone.utc).isoformat(),
            "repo_id": self._repo_id,
            "stats": self._collect_stats(local_dir),
        }

        status_path = local_dir / "sync_status.json"
        status_path.write_text(json.dumps(status, indent=2, default=str))
        logger.debug("Wrote sync status to %s", status_path)
        return status_path

    @staticmethod
    def _collect_stats(local_dir: Path) -> dict:
        """Collect statistics about the local data."""
        stats: dict = {
            "total_files": 0,
            "total_size_bytes": 0,
            "asset_types": {},
        }

        for subdir_name in _DATA_SUBDIRS:
            subdir = local_dir / subdir_name
            if not subdir.exists():
                continue
            files = list(subdir.glob("*.parquet"))
            size = sum(f.stat().st_size for f in files)
            stats["asset_types"][subdir_name] = {
                "files": len(files),
                "size_bytes": size,
            }
            stats["total_files"] += len(files)
            stats["total_size_bytes"] += size

        return stats

    # ------------------------------------------------------------------
    # Dataset card generation
    # ------------------------------------------------------------------
    def _generate_dataset_card(self, local_dir: Path) -> Path:
        """Generate a HuggingFace dataset card (README.md) from template."""
        stats = self._collect_stats(local_dir)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # Build the asset types table
        asset_rows = ""
        for atype, info in stats.get("asset_types", {}).items():
            size_mb = info["size_bytes"] / (1024 * 1024)
            asset_rows += f"| {atype.title()} | {info['files']} | {size_mb:.1f} MB |\n"

        total_mb = stats["total_size_bytes"] / (1024 * 1024)

        card_content = f"""---
language:
- en
license: mit
tags:
- finance
- stock-market
- india
- nse
- bse
- time-series
- ohlcv
task_categories:
- time-series-forecasting
pretty_name: Indian Market Data (NSE/BSE)
size_categories:
- 1M<n<10M
---

# Indian Market Data (NSE/BSE)

Production-grade historical OHLCV dataset for Indian financial markets.
Auto-updated daily via GitHub Actions.

## Dataset Summary

| Metric | Value |
|--------|-------|
| Total Files | {stats['total_files']} |
| Total Size | {total_mb:.1f} MB |
| Last Updated | {now} |
| Update Frequency | Daily (weekdays) |
| Source | Yahoo Finance via yfinance |

### Asset Coverage

| Asset Type | Symbols | Size |
|-----------|---------|------|
{asset_rows}
## Schema

Each Parquet file contains daily OHLCV data with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `date` | `date32` | Trading date |
| `open` | `float64` | Opening price |
| `high` | `float64` | Day high |
| `low` | `float64` | Day low |
| `close` | `float64` | Closing price |
| `adj_close` | `float64` | Adjusted close (splits & dividends) |
| `volume` | `int64` | Trading volume |
| `dividends` | `float64` | Dividend amount |
| `stock_splits` | `float64` | Stock split ratio |
| `symbol` | `string` | Ticker symbol |

## Usage

```python
import pandas as pd

# Load a single stock
df = pd.read_parquet("hf://datasets/{self._repo_id}/stocks/RELIANCE.parquet")

# Load all stocks
from datasets import load_dataset
ds = load_dataset("{self._repo_id}", data_dir="stocks")
```

## File Structure

```
├── stocks/          # NSE/BSE equities (*.parquet)
├── indices/         # Market indices (*.parquet)
├── etfs/            # Exchange-traded funds (*.parquet)
├── commodities/     # Commodity futures (*.parquet)
├── forex/           # Currency pairs (*.parquet)
├── metadata/        # Asset universe & catalog
└── sync_status.json # Last sync timestamp & stats
```

## Data Quality

Each sync run validates data for:
- Duplicate dates
- OHLC consistency (High ≥ Low, bounds checks)
- Missing values
- Volume anomalies
- Trading day gaps

## License

MIT
"""
        card_path = local_dir / "README.md"
        card_path.write_text(card_content)
        logger.debug("Generated dataset card at %s", card_path)
        return card_path

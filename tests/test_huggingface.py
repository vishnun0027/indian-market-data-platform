"""Tests for the HuggingFace integration module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from market_data.huggingface import HuggingFaceSync


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory with sample Parquet files."""
    # Create subdirectories
    for subdir in ("stocks", "indices", "etfs", "commodities", "forex", "metadata"):
        (tmp_path / subdir).mkdir()

    # Create dummy parquet files
    import pandas as pd

    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"]).date,
        "open": [100.0, 101.0],
        "high": [105.0, 106.0],
        "low": [99.0, 100.0],
        "close": [103.0, 104.0],
        "adj_close": [103.0, 104.0],
        "volume": [1000000, 1100000],
        "dividends": [0.0, 0.0],
        "stock_splits": [0.0, 0.0],
        "symbol": ["RELIANCE", "RELIANCE"],
    })
    df.to_parquet(tmp_path / "stocks" / "RELIANCE.parquet", index=False)
    df["symbol"] = "TCS"
    df.to_parquet(tmp_path / "stocks" / "TCS.parquet", index=False)

    # Index
    idx_df = df.copy()
    idx_df["symbol"] = "NIFTY50"
    idx_df.to_parquet(tmp_path / "indices" / "NIFTY50.parquet", index=False)

    return tmp_path


@pytest.fixture
def mock_hf_api():
    """Provide a mock HfApi."""
    with patch("market_data.huggingface.HfApi") as mock_cls:
        api_instance = MagicMock()
        mock_cls.return_value = api_instance
        yield api_instance


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
class TestHuggingFaceSyncInit:
    """Tests for HuggingFaceSync construction and validation."""

    def test_requires_repo_id(self):
        with pytest.raises(ValueError, match="repo_id is required"):
            HuggingFaceSync(repo_id="", token="hf_test_token")

    def test_requires_token(self):
        with pytest.raises(ValueError, match="token is required"):
            HuggingFaceSync(repo_id="user/dataset", token="")

    def test_valid_construction(self, mock_hf_api):
        hf = HuggingFaceSync(repo_id="user/dataset", token="hf_test_token")
        assert hf.repo_id == "user/dataset"


# ---------------------------------------------------------------------------
# ensure_repo
# ---------------------------------------------------------------------------
class TestEnsureRepo:
    """Tests for repository creation / existence check."""

    def test_repo_exists(self, mock_hf_api):
        mock_hf_api.repo_info.return_value = MagicMock(id="user/dataset")

        hf = HuggingFaceSync(repo_id="user/dataset", token="hf_test_token")
        url = hf.ensure_repo()

        assert "user/dataset" in url
        mock_hf_api.repo_info.assert_called_once()

    def test_repo_created_when_missing(self, mock_hf_api):
        from huggingface_hub.utils import RepositoryNotFoundError

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.headers = {}
        mock_hf_api.repo_info.side_effect = RepositoryNotFoundError(
            "Not found", response=mock_response
        )
        mock_hf_api.create_repo.return_value = "https://huggingface.co/datasets/user/dataset"

        hf = HuggingFaceSync(repo_id="user/dataset", token="hf_test_token")
        url = hf.ensure_repo()

        assert "user/dataset" in url
        mock_hf_api.create_repo.assert_called_once_with(
            repo_id="user/dataset",
            repo_type="dataset",
            private=False,
            exist_ok=True,
        )


# ---------------------------------------------------------------------------
# pull_dataset
# ---------------------------------------------------------------------------
class TestPullDataset:
    """Tests for pulling datasets from HuggingFace."""

    def test_pull_empty_repo(self, mock_hf_api, tmp_path):
        mock_hf_api.repo_info.return_value = MagicMock(id="user/dataset")
        mock_hf_api.list_repo_files.return_value = []

        hf = HuggingFaceSync(repo_id="user/dataset", token="hf_test_token")
        result = hf.pull_dataset(tmp_path)

        assert result == tmp_path

    @patch("market_data.huggingface.snapshot_download")
    def test_pull_with_files(self, mock_download, mock_hf_api, tmp_path):
        mock_hf_api.repo_info.return_value = MagicMock(id="user/dataset")
        mock_hf_api.list_repo_files.return_value = ["stocks/RELIANCE.parquet"]
        mock_download.return_value = str(tmp_path)

        hf = HuggingFaceSync(repo_id="user/dataset", token="hf_test_token")
        result = hf.pull_dataset(tmp_path)

        assert result == tmp_path
        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args
        assert call_kwargs.kwargs["repo_id"] == "user/dataset"
        assert call_kwargs.kwargs["repo_type"] == "dataset"


# ---------------------------------------------------------------------------
# push_dataset
# ---------------------------------------------------------------------------
class TestPushDataset:
    """Tests for pushing datasets to HuggingFace."""

    def test_push_creates_sync_status(self, mock_hf_api, tmp_data_dir):
        mock_hf_api.repo_info.return_value = MagicMock(id="user/dataset")
        mock_hf_api.upload_folder.return_value = "https://huggingface.co/datasets/user/dataset/commit/abc123"

        hf = HuggingFaceSync(repo_id="user/dataset", token="hf_test_token")
        hf.push_dataset(local_dir=tmp_data_dir)

        # Check sync_status.json was created
        status_path = tmp_data_dir / "sync_status.json"
        assert status_path.exists()

        status = json.loads(status_path.read_text())
        assert "last_sync_utc" in status
        assert status["repo_id"] == "user/dataset"
        assert status["stats"]["total_files"] == 3  # 2 stocks + 1 index

    def test_push_generates_readme(self, mock_hf_api, tmp_data_dir):
        mock_hf_api.repo_info.return_value = MagicMock(id="user/dataset")
        mock_hf_api.upload_folder.return_value = "https://huggingface.co/datasets/user/dataset/commit/abc123"

        hf = HuggingFaceSync(repo_id="user/dataset", token="hf_test_token")
        hf.push_dataset(local_dir=tmp_data_dir)

        readme_path = tmp_data_dir / "README.md"
        assert readme_path.exists()

        content = readme_path.read_text()
        assert "Indian Market Data" in content
        assert "user/dataset" in content
        assert "Stocks" in content or "stocks" in content.lower()

    def test_push_calls_upload_folder(self, mock_hf_api, tmp_data_dir):
        mock_hf_api.repo_info.return_value = MagicMock(id="user/dataset")
        mock_hf_api.upload_folder.return_value = "https://huggingface.co/commit/abc"

        hf = HuggingFaceSync(repo_id="user/dataset", token="hf_test_token")
        hf.push_dataset(local_dir=tmp_data_dir, commit_message="test commit")

        mock_hf_api.upload_folder.assert_called_once()
        call_kwargs = mock_hf_api.upload_folder.call_args
        assert call_kwargs.kwargs["repo_id"] == "user/dataset"
        assert call_kwargs.kwargs["commit_message"] == "test commit"

    def test_push_with_custom_message(self, mock_hf_api, tmp_data_dir):
        mock_hf_api.repo_info.return_value = MagicMock(id="user/dataset")
        mock_hf_api.upload_folder.return_value = "https://huggingface.co/commit/abc"

        hf = HuggingFaceSync(repo_id="user/dataset", token="hf_test_token")
        result = hf.push_dataset(local_dir=tmp_data_dir, commit_message="Custom sync message")

        call_kwargs = mock_hf_api.upload_folder.call_args
        assert call_kwargs.kwargs["commit_message"] == "Custom sync message"


# ---------------------------------------------------------------------------
# Stats collection
# ---------------------------------------------------------------------------
class TestCollectStats:
    """Tests for data statistics collection."""

    def test_collect_stats(self, tmp_data_dir):
        stats = HuggingFaceSync._collect_stats(tmp_data_dir)

        assert stats["total_files"] == 3  # 2 stocks + 1 index
        assert stats["total_size_bytes"] > 0
        assert "stocks" in stats["asset_types"]
        assert stats["asset_types"]["stocks"]["files"] == 2
        assert "indices" in stats["asset_types"]
        assert stats["asset_types"]["indices"]["files"] == 1

    def test_collect_stats_empty_dir(self, tmp_path):
        stats = HuggingFaceSync._collect_stats(tmp_path)

        assert stats["total_files"] == 0
        assert stats["total_size_bytes"] == 0
        assert stats["asset_types"] == {}


# ---------------------------------------------------------------------------
# Dataset card generation
# ---------------------------------------------------------------------------
class TestDatasetCard:
    """Tests for HuggingFace dataset card (README) generation."""

    def test_card_contains_schema(self, mock_hf_api, tmp_data_dir):
        hf = HuggingFaceSync(repo_id="user/dataset", token="hf_test_token")
        card_path = hf._generate_dataset_card(tmp_data_dir)

        content = card_path.read_text()
        # Check YAML frontmatter
        assert content.startswith("---")
        assert "finance" in content
        # Check schema table
        assert "date" in content
        assert "open" in content
        assert "float64" in content
        # Check usage example
        assert "user/dataset" in content

# Indian Market Data Platform

Production-grade market data ingestion platform for Indian financial markets. Designed for backtesting, quantitative research, and AI/ML model development.

## Features

- **Asset Discovery** — Automatically discovers NSE/BSE equities, indices, ETFs, commodities, and forex pairs
- **Historical Data Download** — Complete OHLCV data with resume support, parallel downloads, and rate limiting
- **Parquet Storage** — Efficient columnar storage with DuckDB metadata catalog
- **Data Validation** — Quality checks for duplicates, OHLC consistency, gaps, and more
- **CLI Interface** — Rich-formatted command-line interface with progress bars
- **HuggingFace Hub Sync** — Publish and auto-update datasets on HuggingFace
- **GitHub Actions Automation** — Zero-infra daily sync pipeline

## Quick Start

```bash
# Install
uv pip install -e ".[dev]"

# Discover assets
market-data discover

# Download historical data
market-data download

# Validate data quality
market-data validate

# Check status
market-data status
```

## Daily Automation (GitHub Actions + HuggingFace)

The platform can run fully automated on GitHub Actions, syncing data to HuggingFace Hub daily.

### Setup

1. **Create a HuggingFace account** and generate a write-access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

2. **Add GitHub Secrets** in your repo → Settings → Secrets and variables → Actions:
   - `HF_TOKEN` — your HuggingFace write-access token

3. **Add GitHub Variables** (optional):
   - `HF_REPO_ID` — dataset repo name (default: `<your-username>/indian-market-data`)
   - `HF_PRIVATE` — set to `true` for a private dataset (default: `false`)

4. **Push to GitHub** — the workflow runs automatically on weekdays at IST 5:30 PM

### Manual Sync

```bash
# Full pipeline: pull → discover → download → validate → push
market-data sync

# Push local data to HuggingFace (one-shot)
market-data push-hf

# Sync with options
market-data sync --type EQUITY --skip-validate
market-data sync --skip-discover --workers 8
```

### Workflow Dispatch

You can also trigger the sync manually from GitHub → Actions → Daily Market Data Sync → Run workflow.

## Configuration

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `./data` | Local data directory |
| `MAX_WORKERS` | `4` | Parallel download threads |
| `RATE_LIMIT_PER_SECOND` | `2.0` | yfinance rate limit |
| `HF_REPO_ID` | — | HuggingFace dataset repo ID |
| `HF_TOKEN` | — | HuggingFace write-access token |
| `HF_PRIVATE` | `false` | Private dataset repo |

## CLI Commands

| Command | Description |
|---------|-------------|
| `market-data discover` | Discover all supported assets |
| `market-data download` | Download historical OHLCV data |
| `market-data validate` | Run data quality checks |
| `market-data clean` | Repair OHLC anomalies |
| `market-data status` | Show catalog summary |
| `market-data sync` | Full pipeline (pull → discover → download → validate → push) |
| `market-data push-hf` | One-shot push to HuggingFace |

## Technology Stack

Python 3.12, yfinance, pandas, pyarrow, DuckDB, Pydantic, httpx, tenacity, Typer, Rich, huggingface-hub

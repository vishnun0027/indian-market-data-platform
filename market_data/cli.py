"""CLI entrypoint for the Indian Market Data Platform.

Usage:
    market-data discover       — Discover all supported assets
    market-data download       — Download historical data
    market-data validate       — Run data validation checks
    market-data status         — Show catalog status
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from market_data import __version__
from market_data.config import get_settings
from market_data.logging_config import setup_logging

app = typer.Typer(
    name="market-data",
    help="Indian Market Data Platform — ingest, store, and validate market data for backtesting.",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


def _init(read_only: bool = False) -> tuple:
    """Initialize settings, logging, and core services."""
    settings = get_settings()
    settings.ensure_directories()
    logger = setup_logging(log_level=settings.log_level, logs_dir=settings.logs_dir)

    from market_data.storage.catalog import DuckDBCatalog
    from market_data.storage.parquet_store import ParquetStore

    store = ParquetStore(data_dir=settings.data_dir)
    catalog = DuckDBCatalog(db_path=settings.metadata_dir / "catalog.duckdb", read_only=read_only)

    return settings, logger, store, catalog


@app.command()
def discover() -> None:
    """Discover and catalog all supported Indian market assets."""
    settings, logger, store, catalog = _init()

    console.print(
        Panel(
            "[bold cyan]Asset Discovery[/bold cyan]\n"
            "Discovering NSE equities, BSE equities, indices, ETFs, commodities, and forex pairs...",
            title="Market Data Platform",
        )
    )

    from market_data.assets.discovery import AssetDiscovery

    discovery = AssetDiscovery()
    assets = discovery.discover_all()

    # Save to Parquet
    out_path = discovery.save_universe(assets, settings.metadata_dir)

    # Update catalog
    catalog.upsert_assets(assets)

    # Summary table
    from collections import Counter

    type_counts = Counter(a.asset_type.value for a in assets)

    table = Table(title="Discovery Summary", show_header=True, header_style="bold cyan")
    table.add_column("Asset Type", style="bold")
    table.add_column("Count", justify="right")

    for asset_type, count in sorted(type_counts.items()):
        table.add_row(asset_type, str(count))
    table.add_row("[bold]TOTAL[/bold]", f"[bold]{len(assets)}[/bold]")

    console.print(table)
    console.print(f"\n[green]✓[/green] Asset universe saved to [bold]{out_path}[/bold]")
    catalog.close()


@app.command()
def download(
    symbols: Optional[str] = typer.Option(
        None,
        "--symbols", "-s",
        help="Comma-separated symbols to download (e.g. RELIANCE,TCS,INFY). Default: all.",
    ),
    start: Optional[str] = typer.Option(
        None,
        "--start",
        help="Start date (YYYY-MM-DD). Default: earliest available / resume.",
    ),
    end: Optional[str] = typer.Option(
        None,
        "--end",
        help="End date (YYYY-MM-DD). Default: today.",
    ),
    asset_type: Optional[str] = typer.Option(
        None,
        "--type", "-t",
        help="Filter by asset type: EQUITY, INDEX, ETF, COMMODITY, FOREX",
    ),
    workers: Optional[int] = typer.Option(
        None,
        "--workers", "-w",
        help="Number of parallel download workers (overrides config)",
    ),
) -> None:
    """Download historical OHLCV data for all or selected assets."""
    settings, logger, store, catalog = _init()

    console.print(
        Panel(
            "[bold cyan]Historical Data Download[/bold cyan]\n"
            "Downloading OHLCV data with resume support, retries, and rate limiting...",
            title="Market Data Platform",
        )
    )

    # Parse dates
    start_date = date.fromisoformat(start) if start else None
    end_date = date.fromisoformat(end) if end else None

    # Load asset universe
    import pandas as pd

    from market_data.assets.models import AssetInfo, AssetType, Exchange

    universe_path = settings.metadata_dir / "asset_universe.parquet"
    if not universe_path.exists():
        console.print(
            "[red]✗[/red] Asset universe not found. Run [bold]market-data discover[/bold] first."
        )
        raise typer.Exit(code=1)

    universe_df = pd.read_parquet(universe_path)
    assets: list[AssetInfo] = []

    for _, row in universe_df.iterrows():
        asset = AssetInfo(
            symbol=row["symbol"],
            name=row.get("name", ""),
            isin=row.get("isin", ""),
            exchange=Exchange(row.get("exchange", "NSE")),
            sector=row.get("sector", ""),
            industry=row.get("industry", ""),
            asset_type=AssetType(row.get("asset_type", "EQUITY")),
            yfinance_ticker=row.get("yfinance_ticker", ""),
        )
        assets.append(asset)

    # Filter by symbols if specified
    if symbols:
        symbol_set = {s.strip().upper() for s in symbols.split(",")}
        assets = [a for a in assets if a.symbol.upper() in symbol_set]

    # Filter by asset type if specified
    if asset_type:
        assets = [a for a in assets if a.asset_type.value == asset_type.upper()]

    if not assets:
        console.print("[yellow]No assets matched the filter criteria.[/yellow]")
        raise typer.Exit()

    console.print(f"Downloading data for [bold]{len(assets)}[/bold] assets...")

    # Create download engine
    from market_data.downloader.engine import DownloadEngine

    max_w = workers if workers else settings.max_workers
    engine = DownloadEngine(
        store=store,
        max_workers=max_w,
        rate_limit=settings.rate_limit_per_second,
        retry_attempts=settings.retry_attempts,
        retry_wait=settings.retry_wait_seconds,
        batch_cooldown_size=settings.batch_cooldown_size,
        batch_cooldown_secs=settings.batch_cooldown_secs,
    )

    # Close catalog to release database lock during long-running download batch
    catalog.close()

    results = engine.download_batch(
        assets=assets,
        start_date=start_date,
        end_date=end_date,
        show_progress=True,
    )

    # Reopen catalog to log results
    from market_data.storage.catalog import DuckDBCatalog
    catalog = DuckDBCatalog(db_path=settings.metadata_dir / "catalog.duckdb")

    # Log to catalog
    for r in results:
        status = "success" if r.success else ("skipped" if r.skipped else "error")
        catalog.log_download(
            symbol=r.asset.symbol,
            rows_downloaded=r.rows_added,
            file_path=str(store._asset_path(r.asset)) if r.success else "",
            file_size_bytes=store.get_file_size_bytes(r.asset) if r.success else 0,
            status=status,
            error_message=r.error or "",
        )

    # Summary
    success = sum(1 for r in results if r.success)
    skipped = sum(1 for r in results if r.skipped)
    failed = sum(1 for r in results if r.error)
    total_rows = sum(r.rows_added for r in results)

    console.print(f"\n[green]✓ {success}[/green] downloaded, "
                  f"[yellow]⊘ {skipped}[/yellow] up-to-date, "
                  f"[red]✗ {failed}[/red] failed "
                  f"({total_rows:,} total rows)")

    if failed > 0:
        console.print("\n[red]Failed assets:[/red]")
        for r in results:
            if r.error:
                console.print(f"  • {r.asset.symbol}: {r.error}")

    catalog.close()


@app.command()
def validate(
    symbols: Optional[str] = typer.Option(
        None,
        "--symbols", "-s",
        help="Comma-separated symbols to validate (e.g. RELIANCE,TCS,INFY). Default: all with data.",
    ),
) -> None:
    """Run data quality validation on stored data."""
    settings, logger, store, catalog = _init()

    console.print(
        Panel(
            "[bold cyan]Data Validation[/bold cyan]\n"
            "Running quality checks: duplicates, OHLC consistency, gaps, volume, nulls...",
            title="Market Data Platform",
        )
    )

    import pandas as pd

    from market_data.assets.models import AssetInfo, AssetType, Exchange

    # Load asset universe
    universe_path = settings.metadata_dir / "asset_universe.parquet"
    if not universe_path.exists():
        console.print(
            "[red]✗[/red] Asset universe not found. Run [bold]market-data discover[/bold] first."
        )
        raise typer.Exit(code=1)

    universe_df = pd.read_parquet(universe_path)
    assets: list[AssetInfo] = []

    for _, row in universe_df.iterrows():
        asset = AssetInfo(
            symbol=row["symbol"],
            name=row.get("name", ""),
            isin=row.get("isin", ""),
            exchange=Exchange(row.get("exchange", "NSE")),
            asset_type=AssetType(row.get("asset_type", "EQUITY")),
            yfinance_ticker=row.get("yfinance_ticker", ""),
        )
        assets.append(asset)

    # Filter
    if symbols:
        symbol_set = {s.strip().upper() for s in symbols.split(",")}
        assets = [a for a in assets if a.symbol.upper() in symbol_set]

    # Only validate assets that have data
    assets = [a for a in assets if store.exists(a)]

    if not assets:
        console.print("[yellow]No stored data found to validate.[/yellow]")
        raise typer.Exit()

    console.print(f"Validating [bold]{len(assets)}[/bold] assets...")

    from market_data.validation.engine import ValidationEngine

    engine = ValidationEngine(store=store, catalog=catalog)
    reports = engine.validate_all(assets)

    # Print report
    engine.print_report(reports)

    # Save report
    report_path = engine.generate_report(reports, settings.metadata_dir)
    console.print(f"\n[green]✓[/green] Report saved to [bold]{report_path}[/bold]")

    catalog.close()


@app.command()
def clean(
    symbols: Optional[str] = typer.Option(
        None,
        "--symbols", "-s",
        help="Comma-separated symbols to clean (default: all stored assets)",
    ),
) -> None:
    """Repair OHLC data anomalies in stored files."""
    settings, logger, store, catalog = _init()

    console.print(
        Panel(
            "[bold cyan]Data Cleaning[/bold cyan]\n"
            "Enforcing OHLC bounds and high >= low...",
            title="Market Data Platform",
        )
    )

    import pandas as pd

    from market_data.assets.models import AssetInfo, AssetType, Exchange

    # Load asset universe
    universe_path = settings.metadata_dir / "asset_universe.parquet"
    if not universe_path.exists():
        console.print(
            "[red]✗[/red] Asset universe not found. Run [bold]market-data discover[/bold] first."
        )
        raise typer.Exit(code=1)

    universe_df = pd.read_parquet(universe_path)
    assets: list[AssetInfo] = []

    for _, row in universe_df.iterrows():
        asset = AssetInfo(
            symbol=row["symbol"],
            name=row.get("name", ""),
            isin=row.get("isin", ""),
            exchange=Exchange(row.get("exchange", "NSE")),
            asset_type=AssetType(row.get("asset_type", "EQUITY")),
            yfinance_ticker=row.get("yfinance_ticker", ""),
        )
        assets.append(asset)

    # Filter
    if symbols:
        symbol_set = {s.strip().upper() for s in symbols.split(",")}
        assets = [a for a in assets if a.symbol.upper() in symbol_set]

    # Only clean assets that have data
    assets = [a for a in assets if store.exists(a)]

    if not assets:
        console.print("[yellow]No stored data found to clean.[/yellow]")
        raise typer.Exit()

    console.print(f"Cleaning [bold]{len(assets)}[/bold] assets...")

    from market_data.validation.cleaner import DataCleaner

    cleaner = DataCleaner(store=store, catalog=catalog)
    total_repaired = 0
    repaired_assets = 0

    for asset in assets:
        repaired = cleaner.clean_asset(asset)
        if repaired > 0:
            total_repaired += repaired
            repaired_assets += 1

    console.print(
        f"\n[green]✓[/green] Completed cleaning. "
        f"Repaired [bold]{total_repaired}[/bold] rows across [bold]{repaired_assets}[/bold] assets."
    )

    catalog.close()


@app.command()
def status() -> None:
    """Show current platform status and catalog summary."""
    settings, logger, store, catalog = _init(read_only=True)

    console.print(
        Panel(
            f"[bold cyan]Market Data Platform v{__version__}[/bold cyan]",
            title="Status",
        )
    )

    # Settings
    table = Table(title="Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Setting", style="bold")
    table.add_column("Value")

    table.add_row("Data Directory", str(settings.data_dir))
    table.add_row("Log Level", settings.log_level)
    table.add_row("Max Workers", str(settings.max_workers))
    table.add_row("Rate Limit", f"{settings.rate_limit_per_second} req/s")
    table.add_row("Retry Attempts", str(settings.retry_attempts))
    console.print(table)

    # Catalog summary
    try:
        summary = catalog.get_status_summary()

        cat_table = Table(title="Catalog Summary", show_header=True, header_style="bold cyan")
        cat_table.add_column("Metric", style="bold")
        cat_table.add_column("Value", justify="right")

        cat_table.add_row("Total Assets", str(summary["total_assets"]))
        for atype, count in summary.get("asset_type_counts", {}).items():
            cat_table.add_row(f"  {atype}", str(count))
        cat_table.add_row("Downloads (24h)", str(summary["recent_downloads_24h"]))
        cat_table.add_row("Failed Downloads", str(summary["total_failed_downloads"]))

        console.print(cat_table)
    except Exception:
        console.print("[yellow]Catalog is empty — run 'market-data discover' first.[/yellow]")

    # Stored data summary
    stored = store.list_stored_assets()
    console.print(f"\n[bold]Stored assets:[/bold] {len(stored)} Parquet files")

    catalog.close()


@app.command(name="push-hf")
def push_hf(
    repo_id: Optional[str] = typer.Option(
        None,
        "--repo-id", "-r",
        help="HuggingFace dataset repo ID (e.g. username/indian-market-data). Overrides config.",
    ),
    message: Optional[str] = typer.Option(
        None,
        "--message", "-m",
        help="Commit message for the HuggingFace push.",
    ),
) -> None:
    """Push current local data to HuggingFace Hub."""
    settings, logger, store, catalog = _init(read_only=True)

    hf_repo = repo_id or settings.hf_repo_id
    hf_token = settings.hf_token

    if not hf_repo or not hf_token:
        console.print(
            "[red]✗[/red] HuggingFace not configured. Set [bold]HF_REPO_ID[/bold] "
            "and [bold]HF_TOKEN[/bold] in your .env or pass --repo-id."
        )
        raise typer.Exit(code=1)

    from market_data.huggingface import HuggingFaceSync

    console.print(
        Panel(
            f"[bold cyan]Push to HuggingFace[/bold cyan]\n"
            f"Uploading local data to [bold]{hf_repo}[/bold]...",
            title="Market Data Platform",
        )
    )

    hf = HuggingFaceSync(repo_id=hf_repo, token=hf_token, private=settings.hf_private)
    commit_url = hf.push_dataset(local_dir=settings.data_dir, commit_message=message)

    console.print(f"\n[green]✓[/green] Pushed to HuggingFace: [bold]{commit_url}[/bold]")
    catalog.close()


@app.command()
def sync(
    repo_id: Optional[str] = typer.Option(
        None,
        "--repo-id", "-r",
        help="HuggingFace dataset repo ID. Overrides config.",
    ),
    asset_type: Optional[str] = typer.Option(
        None,
        "--type", "-t",
        help="Filter by asset type: EQUITY, INDEX, ETF, COMMODITY, FOREX",
    ),
    workers: Optional[int] = typer.Option(
        None,
        "--workers", "-w",
        help="Number of parallel download workers (overrides config)",
    ),
    skip_discover: bool = typer.Option(
        False,
        "--skip-discover",
        help="Skip the asset discovery step (use existing universe).",
    ),
    skip_validate: bool = typer.Option(
        False,
        "--skip-validate",
        help="Skip the data validation step.",
    ),
    skip_push: bool = typer.Option(
        False,
        "--skip-push",
        help="Skip pushing to HuggingFace (download only).",
    ),
) -> None:
    """Full sync pipeline: pull from HF → discover → download → validate → push.

    This is the primary command used by GitHub Actions for daily automation.
    """
    settings, logger_inst, store, catalog = _init()

    hf_repo = repo_id or settings.hf_repo_id
    hf_token = settings.hf_token

    if not skip_push and (not hf_repo or not hf_token):
        console.print(
            "[red]✗[/red] HuggingFace not configured. Set [bold]HF_REPO_ID[/bold] "
            "and [bold]HF_TOKEN[/bold] in your .env, or pass --skip-push."
        )
        raise typer.Exit(code=1)

    console.print(
        Panel(
            "[bold cyan]Full Sync Pipeline[/bold cyan]\n"
            "Pull → Discover → Download → Validate → Push",
            title="Market Data Platform",
        )
    )

    # --- Step 1: Pull existing data from HuggingFace ---
    if not skip_push:
        console.print("\n[bold]Step 1/5:[/bold] Pulling existing data from HuggingFace...")
        from market_data.huggingface import HuggingFaceSync

        # Map --type filter to HF subdirectory names so we only pull
        # the relevant files instead of the full ~280 MB dataset.
        _ASSET_TYPE_TO_SUBDIR = {
            "EQUITY": "stocks",
            "INDEX": "indices",
            "ETF": "etfs",
            "COMMODITY": "commodities",
            "FOREX": "forex",
        }
        pull_subdirs = None
        if asset_type:
            subdir = _ASSET_TYPE_TO_SUBDIR.get(asset_type.upper())
            if subdir:
                pull_subdirs = [subdir]

        hf = HuggingFaceSync(repo_id=hf_repo, token=hf_token, private=settings.hf_private)
        hf.pull_dataset(local_dir=settings.data_dir, subdirs=pull_subdirs)
        console.print("[green]  ✓[/green] Pull complete")
    else:
        console.print("\n[bold]Step 1/5:[/bold] [dim]Skipped (--skip-push)[/dim]")

    # --- Step 2: Discover assets ---
    if not skip_discover:
        console.print("\n[bold]Step 2/5:[/bold] Discovering assets...")
        from market_data.assets.discovery import AssetDiscovery

        discovery = AssetDiscovery()
        assets = discovery.discover_all()
        discovery.save_universe(assets, settings.metadata_dir)
        catalog.upsert_assets(assets)
        console.print(f"[green]  ✓[/green] Discovered {len(assets)} assets")
    else:
        console.print("\n[bold]Step 2/5:[/bold] [dim]Skipped (--skip-discover)[/dim]")

    # --- Step 3: Download incremental data ---
    console.print("\n[bold]Step 3/5:[/bold] Downloading incremental data...")

    import pandas as pd

    from market_data.assets.models import AssetInfo, AssetType, Exchange

    universe_path = settings.metadata_dir / "asset_universe.parquet"
    if not universe_path.exists():
        console.print(
            "[red]✗[/red] Asset universe not found. Run without --skip-discover."
        )
        raise typer.Exit(code=1)

    universe_df = pd.read_parquet(universe_path)
    dl_assets: list[AssetInfo] = []

    for _, row in universe_df.iterrows():
        asset = AssetInfo(
            symbol=row["symbol"],
            name=row.get("name", ""),
            isin=row.get("isin", ""),
            exchange=Exchange(row.get("exchange", "NSE")),
            sector=row.get("sector", ""),
            industry=row.get("industry", ""),
            asset_type=AssetType(row.get("asset_type", "EQUITY")),
            yfinance_ticker=row.get("yfinance_ticker", ""),
        )
        dl_assets.append(asset)

    if asset_type:
        dl_assets = [a for a in dl_assets if a.asset_type.value == asset_type.upper()]

    if not dl_assets:
        console.print("[yellow]No assets matched the filter.[/yellow]")
        raise typer.Exit()

    from market_data.downloader.engine import DownloadEngine

    max_w = workers if workers else settings.max_workers
    engine = DownloadEngine(
        store=store,
        max_workers=max_w,
        rate_limit=settings.rate_limit_per_second,
        retry_attempts=settings.retry_attempts,
        retry_wait=settings.retry_wait_seconds,
        batch_cooldown_size=settings.batch_cooldown_size,
        batch_cooldown_secs=settings.batch_cooldown_secs,
    )

    # Close catalog during long download
    catalog.close()

    results = engine.download_batch(
        assets=dl_assets,
        show_progress=True,
    )

    # Reopen catalog
    from market_data.storage.catalog import DuckDBCatalog

    catalog = DuckDBCatalog(db_path=settings.metadata_dir / "catalog.duckdb")

    for r in results:
        status = "success" if r.success else ("skipped" if r.skipped else "error")
        catalog.log_download(
            symbol=r.asset.symbol,
            rows_downloaded=r.rows_added,
            file_path=str(store._asset_path(r.asset)) if r.success else "",
            file_size_bytes=store.get_file_size_bytes(r.asset) if r.success else 0,
            status=status,
            error_message=r.error or "",
        )

    success = sum(1 for r in results if r.success)
    skipped = sum(1 for r in results if r.skipped)
    failed = sum(1 for r in results if r.error)
    total_rows = sum(r.rows_added for r in results)

    console.print(
        f"[green]  ✓ {success}[/green] downloaded, "
        f"[yellow]⊘ {skipped}[/yellow] up-to-date, "
        f"[red]✗ {failed}[/red] failed "
        f"({total_rows:,} rows)"
    )

    # --- Step 4: Validate ---
    if not skip_validate:
        console.print("\n[bold]Step 4/5:[/bold] Validating data quality...")
        validate_assets = [a for a in dl_assets if store.exists(a)]

        if validate_assets:
            from market_data.validation.engine import ValidationEngine

            val_engine = ValidationEngine(store=store, catalog=catalog)
            reports = val_engine.validate_all(validate_assets)

            passed = sum(1 for r in reports if r.passed)
            console.print(
                f"[green]  ✓[/green] {passed}/{len(reports)} assets passed validation"
            )
        else:
            console.print("[dim]  No data to validate[/dim]")
    else:
        console.print("\n[bold]Step 4/5:[/bold] [dim]Skipped (--skip-validate)[/dim]")

    # --- Step 5: Push to HuggingFace ---
    if not skip_push:
        console.print("\n[bold]Step 5/5:[/bold] Pushing to HuggingFace...")
        from market_data.huggingface import HuggingFaceSync

        hf = HuggingFaceSync(repo_id=hf_repo, token=hf_token, private=settings.hf_private)
        commit_msg = (
            f"Daily sync — {success} updated, {skipped} up-to-date, "
            f"{failed} failed ({total_rows:,} new rows)"
        )
        commit_url = hf.push_dataset(local_dir=settings.data_dir, commit_message=commit_msg)
        console.print(f"[green]  ✓[/green] Pushed: [bold]{commit_url}[/bold]")
    else:
        console.print("\n[bold]Step 5/5:[/bold] [dim]Skipped (--skip-push)[/dim]")

    # --- Summary ---
    console.print(
        Panel(
            f"[bold green]Sync complete![/bold green]\n"
            f"Downloaded: {success} | Up-to-date: {skipped} | Failed: {failed} | New rows: {total_rows:,}",
            title="Summary",
        )
    )

    catalog.close()


@app.callback()
def main() -> None:
    """Indian Market Data Platform — data ingestion for backtesting and quantitative research."""


if __name__ == "__main__":
    app()


"""Validation engine — orchestrates data quality checks across assets."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from market_data.assets.models import AssetInfo
from market_data.storage.catalog import DuckDBCatalog
from market_data.storage.parquet_store import ParquetStore
from market_data.validation.checks import ALL_CHECKS
from market_data.validation.models import (
    AssetValidationReport,
    CheckStatus,
    ValidationResult,
)

logger = logging.getLogger("market_data.validation.engine")
console = Console()


class ValidationEngine:
    """Runs data quality checks on stored Parquet data.

    Args:
        store: ParquetStore instance to read data from.
        catalog: Optional DuckDBCatalog to log validation results.
    """

    def __init__(
        self,
        store: ParquetStore,
        catalog: DuckDBCatalog | None = None,
    ) -> None:
        self._store = store
        self._catalog = catalog

    def validate_asset(self, asset: AssetInfo) -> AssetValidationReport:
        """Run all validation checks on a single asset.

        Args:
            asset: The asset to validate.

        Returns:
            AssetValidationReport with results from all checks.
        """
        df = self._store.load(asset)
        report = AssetValidationReport(
            symbol=asset.symbol,
            total_rows=len(df),
        )

        if df.empty:
            report.checks.append(
                ValidationResult(
                    check_name="data_exists",
                    status=CheckStatus.SKIPPED,
                    message="No data available for validation",
                )
            )
            return report

        for check_fn in ALL_CHECKS:
            result = check_fn(df)
            report.checks.append(result)

            # Log to catalog if available
            if self._catalog is not None:
                self._catalog.log_validation(
                    symbol=asset.symbol,
                    check_name=result.check_name,
                    status=result.status.value,
                    details=result.message,
                )

        return report

    def validate_all(self, assets: list[AssetInfo]) -> list[AssetValidationReport]:
        """Validate all provided assets.

        Args:
            assets: List of assets to validate.

        Returns:
            List of validation reports, one per asset.
        """
        reports: list[AssetValidationReport] = []

        for asset in assets:
            report = self.validate_asset(asset)
            reports.append(report)

        # Summary
        total = len(reports)
        passed = sum(1 for r in reports if r.passed)
        logger.info(
            "Validation complete: %d/%d assets passed all checks",
            passed,
            total,
        )

        return reports

    def generate_report(
        self,
        reports: list[AssetValidationReport],
        output_dir: Path,
    ) -> Path:
        """Generate a validation summary as a Parquet file.

        Args:
            reports: List of validation reports.
            output_dir: Directory to write the report to.

        Returns:
            Path to the generated Parquet file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "validation_report.parquet"

        rows = []
        for report in reports:
            for check in report.checks:
                rows.append({
                    "symbol": report.symbol,
                    "total_rows": report.total_rows,
                    "check_name": check.check_name,
                    "status": check.status.value,
                    "message": check.message,
                    "rows_affected": check.rows_affected,
                    "validated_at": report.validated_at.isoformat(),
                })

        df = pd.DataFrame(rows)
        if not df.empty:
            df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)

        logger.info("Validation report saved to %s", out_path)
        return out_path

    def print_report(self, reports: list[AssetValidationReport]) -> None:
        """Print a formatted validation summary to the console using Rich."""
        table = Table(
            title="Validation Report",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Symbol", style="bold")
        table.add_column("Rows", justify="right")
        table.add_column("Passed", justify="center")
        table.add_column("Warnings", justify="center")
        table.add_column("Failed", justify="center")
        table.add_column("Status", justify="center")

        for report in reports:
            passed = sum(1 for c in report.checks if c.status == CheckStatus.PASSED)
            warned = sum(1 for c in report.checks if c.status == CheckStatus.WARNING)
            failed = sum(1 for c in report.checks if c.status == CheckStatus.FAILED)

            status = "[green]✓ PASS[/green]" if report.passed else "[red]✗ FAIL[/red]"

            table.add_row(
                report.symbol,
                str(report.total_rows),
                f"[green]{passed}[/green]",
                f"[yellow]{warned}[/yellow]" if warned else str(warned),
                f"[red]{failed}[/red]" if failed else str(failed),
                status,
            )

        console.print(table)

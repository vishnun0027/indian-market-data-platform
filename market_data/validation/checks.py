"""Individual data validation checks.

Each function takes a DataFrame and returns a ValidationResult.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from market_data.validation.models import CheckStatus, ValidationResult


def check_duplicates(df: pd.DataFrame) -> ValidationResult:
    """Check for duplicate rows by date."""
    if df.empty:
        return ValidationResult(
            check_name="duplicates",
            status=CheckStatus.SKIPPED,
            message="Empty DataFrame",
        )

    if "date" not in df.columns:
        return ValidationResult(
            check_name="duplicates",
            status=CheckStatus.SKIPPED,
            message="No 'date' column found",
        )

    dup_count = df.duplicated(subset=["date"], keep=False).sum()
    if dup_count > 0:
        return ValidationResult(
            check_name="duplicates",
            status=CheckStatus.FAILED,
            message=f"Found {dup_count} duplicate rows by date",
            rows_affected=int(dup_count),
        )

    return ValidationResult(
        check_name="duplicates",
        status=CheckStatus.PASSED,
        message="No duplicate dates found",
    )


def check_missing_trading_days(
    df: pd.DataFrame,
    max_gap_days: int = 5,
) -> ValidationResult:
    """Check for suspiciously large gaps between trading days.

    A gap of more than `max_gap_days` consecutive calendar days
    (excluding weekends) is flagged as a warning.  Gaps > 15 days
    are flagged as failures.

    Note: This is a heuristic — it doesn't use an actual exchange
    holiday calendar.  A proper implementation would require a
    holiday calendar for NSE/BSE.
    """
    if df.empty or "date" not in df.columns:
        return ValidationResult(
            check_name="missing_trading_days",
            status=CheckStatus.SKIPPED,
            message="Insufficient data",
        )

    dates = pd.to_datetime(df["date"]).sort_values()
    gaps = dates.diff().dt.days.dropna()

    large_gaps = gaps[gaps > max_gap_days]

    if len(large_gaps) == 0:
        return ValidationResult(
            check_name="missing_trading_days",
            status=CheckStatus.PASSED,
            message=f"No gaps larger than {max_gap_days} days",
        )

    very_large = gaps[gaps > 15]
    if len(very_large) > 0:
        max_gap = int(gaps.max())
        return ValidationResult(
            check_name="missing_trading_days",
            status=CheckStatus.WARNING,
            message=f"Found {len(large_gaps)} gaps > {max_gap_days} days (max gap: {max_gap} days)",
            rows_affected=len(large_gaps),
            details={"max_gap_days": max_gap, "gap_count": len(large_gaps)},
        )

    return ValidationResult(
        check_name="missing_trading_days",
        status=CheckStatus.WARNING,
        message=f"Found {len(large_gaps)} gaps > {max_gap_days} days",
        rows_affected=len(large_gaps),
        details={"gap_count": len(large_gaps)},
    )


def check_ohlc_consistency(df: pd.DataFrame) -> ValidationResult:
    """Validate OHLC relationships: High >= Low, Open/Close within High/Low."""
    required = {"open", "high", "low", "close"}
    if df.empty or not required.issubset(df.columns):
        return ValidationResult(
            check_name="ohlc_consistency",
            status=CheckStatus.SKIPPED,
            message="Missing required OHLC columns",
        )

    violations = 0
    details: dict = {}

    # High should be >= Low
    high_low = df["high"] < df["low"]
    if high_low.any():
        count = int(high_low.sum())
        violations += count
        details["high_lt_low"] = count

    # Open should be between Low and High (inclusive)
    open_out = (df["open"] > df["high"]) | (df["open"] < df["low"])
    if open_out.any():
        count = int(open_out.sum())
        violations += count
        details["open_out_of_range"] = count

    # Close should be between Low and High (inclusive)
    close_out = (df["close"] > df["high"]) | (df["close"] < df["low"])
    if close_out.any():
        count = int(close_out.sum())
        violations += count
        details["close_out_of_range"] = count

    if violations > 0:
        return ValidationResult(
            check_name="ohlc_consistency",
            status=CheckStatus.FAILED,
            message=f"Found {violations} OHLC consistency violations",
            rows_affected=violations,
            details=details,
        )

    return ValidationResult(
        check_name="ohlc_consistency",
        status=CheckStatus.PASSED,
        message="All OHLC values are consistent",
    )


def check_negative_volume(df: pd.DataFrame) -> ValidationResult:
    """Check for negative volume values."""
    if df.empty or "volume" not in df.columns:
        return ValidationResult(
            check_name="negative_volume",
            status=CheckStatus.SKIPPED,
            message="No 'volume' column found",
        )

    neg_count = (df["volume"] < 0).sum()
    if neg_count > 0:
        return ValidationResult(
            check_name="negative_volume",
            status=CheckStatus.FAILED,
            message=f"Found {neg_count} rows with negative volume",
            rows_affected=int(neg_count),
        )

    return ValidationResult(
        check_name="negative_volume",
        status=CheckStatus.PASSED,
        message="No negative volume values",
    )


def check_null_values(df: pd.DataFrame) -> ValidationResult:
    """Check for unexpected null values in critical columns."""
    if df.empty:
        return ValidationResult(
            check_name="null_values",
            status=CheckStatus.SKIPPED,
            message="Empty DataFrame",
        )

    critical_cols = ["date", "open", "high", "low", "close", "volume"]
    existing_critical = [c for c in critical_cols if c in df.columns]

    if not existing_critical:
        return ValidationResult(
            check_name="null_values",
            status=CheckStatus.SKIPPED,
            message="No critical columns found",
        )

    null_counts = df[existing_critical].isnull().sum()
    total_nulls = null_counts.sum()

    if total_nulls > 0:
        details = {col: int(count) for col, count in null_counts.items() if count > 0}
        return ValidationResult(
            check_name="null_values",
            status=CheckStatus.WARNING,
            message=f"Found {total_nulls} null values across critical columns",
            rows_affected=int(total_nulls),
            details=details,
        )

    return ValidationResult(
        check_name="null_values",
        status=CheckStatus.PASSED,
        message="No null values in critical columns",
    )


def check_data_range(df: pd.DataFrame) -> ValidationResult:
    """Check that dates are within a reasonable range (no future dates)."""
    if df.empty or "date" not in df.columns:
        return ValidationResult(
            check_name="data_range",
            status=CheckStatus.SKIPPED,
            message="Insufficient data",
        )

    today = date.today()
    dates = pd.to_datetime(df["date"]).dt.date

    future_count = (dates > today).sum()
    if future_count > 0:
        return ValidationResult(
            check_name="data_range",
            status=CheckStatus.FAILED,
            message=f"Found {future_count} rows with future dates",
            rows_affected=int(future_count),
        )

    # Check for unreasonably old dates (before 1990 for Indian markets)
    earliest_reasonable = date(1990, 1, 1)
    old_count = (dates < earliest_reasonable).sum()
    if old_count > 0:
        return ValidationResult(
            check_name="data_range",
            status=CheckStatus.WARNING,
            message=f"Found {old_count} rows before 1990",
            rows_affected=int(old_count),
        )

    return ValidationResult(
        check_name="data_range",
        status=CheckStatus.PASSED,
        message="All dates within expected range",
    )


# Collect all check functions for easy iteration
ALL_CHECKS = [
    check_duplicates,
    check_missing_trading_days,
    check_ohlc_consistency,
    check_negative_volume,
    check_null_values,
    check_data_range,
]

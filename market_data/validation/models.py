"""Pydantic models for validation results."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class CheckStatus(str, Enum):
    """Outcome of a single validation check."""

    PASSED = "PASSED"
    WARNING = "WARNING"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class ValidationResult(BaseModel):
    """Result of a single validation check on an asset's data."""

    check_name: str
    status: CheckStatus
    message: str = ""
    details: dict | None = None
    rows_affected: int = 0


class AssetValidationReport(BaseModel):
    """Aggregated validation report for a single asset."""

    symbol: str
    total_rows: int = 0
    checks: list[ValidationResult] = Field(default_factory=list)
    validated_at: datetime = Field(default_factory=datetime.now)

    @property
    def passed(self) -> bool:
        """True if no checks failed."""
        return all(c.status != CheckStatus.FAILED for c in self.checks)

    @property
    def summary(self) -> str:
        """Human-readable summary line."""
        passed = sum(1 for c in self.checks if c.status == CheckStatus.PASSED)
        warned = sum(1 for c in self.checks if c.status == CheckStatus.WARNING)
        failed = sum(1 for c in self.checks if c.status == CheckStatus.FAILED)
        return f"{self.symbol}: {passed} passed, {warned} warnings, {failed} failed"

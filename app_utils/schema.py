"""Schema validation utilities for the Streamlit app."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


BINARY_INDICATORS: List[str] = [
    "contact_telephone",
    "poutcome_success",
    "day_of_week_mon",
    "education_basic.6y",
    "job_management",
    "marital_single",
]

DERIVED_COLUMNS = ["campaign_log", "pdays_log", "previous_log", "duration_log"]

NUMERIC_LIMITS: Dict[str, Tuple[float | None, float | None]] = {
    "age": (17, 100),
    "duration": (0, None),
    "campaign": (0, None),
    "pdays": (0, None),
    "previous": (0, None),
    "emp.var.rate": (-10, 10),
    "cons.price.idx": (0, None),
    "cons.conf.idx": (-100, 100),
    "euribor3m": (0, None),
    "nr.employed": (0, None),
}


@dataclass
class ValidationReport:
    """Container describing validation issues."""

    missing_columns: List[str]
    invalid_indicators: Dict[str, List]
    row_errors: Dict[int, str]

    @property
    def is_valid(self) -> bool:
        return not (self.missing_columns or self.invalid_indicators or self.row_errors)


def validate_schema(df: pd.DataFrame, required_columns: Iterable[str]) -> ValidationReport:
    """Validate the dataframe against required columns and indicator constraints."""

    missing = [col for col in required_columns if col not in df.columns]
    indicator_errors: Dict[str, List] = {}
    for col in BINARY_INDICATORS:
        if col in df.columns:
            values = df[col].dropna().unique().tolist()
            if not set(values).issubset({0, 1}):
                indicator_errors[col] = values

    row_errors: Dict[int, str] = {}
    if not missing:
        for idx, row in df.iterrows():
            messages: List[str] = []
            for col, (lower, upper) in NUMERIC_LIMITS.items():
                value = row.get(col)
                if pd.isna(value):
                    messages.append(f"{col} is missing")
                    continue
                if lower is not None and value < lower:
                    messages.append(f"{col}={value} below minimum {lower}")
                if upper is not None and value > upper:
                    messages.append(f"{col}={value} above maximum {upper}")
            for indicator in BINARY_INDICATORS:
                if indicator in df.columns:
                    value = row.get(indicator)
                    if pd.notna(value) and value not in (0, 1):
                        messages.append(f"{indicator} must be 0/1, got {value}")
            if messages:
                row_errors[idx] = "; ".join(messages)

    return ValidationReport(missing, indicator_errors, row_errors)


def enforce_schema(df: pd.DataFrame, required_columns: Iterable[str]) -> pd.DataFrame:
    """Raise a ValueError when validation fails."""

    report = validate_schema(df, required_columns)
    if not report.is_valid:
        problems = []
        if report.missing_columns:
            problems.append(f"Missing columns: {', '.join(report.missing_columns)}")
        if report.invalid_indicators:
            invalid = ", ".join(
                f"{col}={values}" for col, values in report.invalid_indicators.items()
            )
            problems.append(f"Indicator columns must be binary: {invalid}")
        if report.row_errors:
            examples = ", ".join(
                f"row {idx}: {msg}" for idx, msg in list(report.row_errors.items())[:5]
            )
            problems.append(f"Row-level validation errors: {examples}")
        raise ValueError("; ".join(problems))
    return df


def ensure_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the engineered log features expected by the models."""

    for source in ["campaign", "pdays", "previous", "duration"]:
        if source in df.columns:
            log_col = f"{source}_log"
            df[log_col] = np.log(df[source].astype(float).clip(lower=0) + 1e-6)
    return df

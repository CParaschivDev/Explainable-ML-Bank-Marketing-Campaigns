"""Preprocessing helpers shared across the app and tests."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .schema import ensure_derived_features, enforce_schema, validate_schema


BOOLEAN_REPLACEMENTS = {"True": 1, "False": 0, True: 1, False: 0, "yes": 1, "no": 0}


def coerce_indicators(df: pd.DataFrame, indicator_columns: Iterable[str]) -> pd.DataFrame:
    """Coerce indicator columns to numeric 0/1 values."""

    for col in indicator_columns:
        if col in df.columns:
            series = df[col].replace(BOOLEAN_REPLACEMENTS)
            if hasattr(series, "infer_objects"):
                series = series.infer_objects(copy=False)
            df[col] = series.apply(_normalize_indicator_value)
    return df


def _normalize_indicator_value(value):
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"1", "true", "yes"}:
            return 1
        if lowered in {"0", "false", "no"}:
            return 0
        return value
    if isinstance(value, bool):
        return int(value)
    return value


def prepare_features(
    df: pd.DataFrame, required_columns: Iterable[str], strict: bool = True
) -> pd.DataFrame:
    """Validate and cast features for inference.

    Parameters
    ----------
    df:
        The input dataframe containing raw features.
    required_columns:
        Ordered collection of columns expected by the model pipeline.
    strict:
        When True (default) the function raises on missing columns. When False the
        function will add any missing columns filled with zeros, which is useful for
        single-record inference where not all one-hot encoded columns are captured by
        user inputs.
    """

    df = df.copy()
    df = coerce_indicators(
        df,
        [
            "contact_telephone",
            "poutcome_success",
            "day_of_week_mon",
            "education_basic.6y",
            "job_management",
            "marital_single",
        ],
    )
    df = ensure_derived_features(df)
    missing_report = validate_schema(df, required_columns)
    if missing_report.missing_columns and strict:
        raise ValueError(
            f"Missing columns: {', '.join(sorted(missing_report.missing_columns))}"
        )
    df = df.reindex(columns=required_columns, fill_value=0)
    df = df.apply(pd.to_numeric, errors="coerce").astype(np.float64)
    enforce_schema(df, required_columns)
    return df

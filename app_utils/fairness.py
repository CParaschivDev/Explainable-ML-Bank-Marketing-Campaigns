"""Fairness metrics utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class FairnessResult:
    protected_rate: float
    reference_rate: float
    disparate_impact: float
    equal_opportunity: float


def compute_group_metrics(
    df: pd.DataFrame,
    protected_attribute: str,
    predictions: np.ndarray,
    labels: np.ndarray | None = None,
) -> FairnessResult:
    """Compute disparate impact and equal opportunity metrics."""

    if protected_attribute not in df.columns:
        raise ValueError(f"Column '{protected_attribute}' not found in dataframe")

    binary_mask = df[protected_attribute].astype(int) == 1
    protected_pred = predictions[binary_mask]
    reference_pred = predictions[~binary_mask]

    protected_rate = float(protected_pred.mean()) if len(protected_pred) else float("nan")
    reference_rate = float(reference_pred.mean()) if len(reference_pred) else float("nan")

    disparate_impact = (
        protected_rate / reference_rate if reference_rate not in (0, np.nan) else float("nan")
    )

    if labels is None:
        return FairnessResult(protected_rate, reference_rate, disparate_impact, float("nan"))

    if labels.shape[0] != df.shape[0]:
        raise ValueError("Labels length must match dataframe")

    true_positive_mask = labels == 1
    protected_tp = predictions[binary_mask & true_positive_mask]
    reference_tp = predictions[~binary_mask & true_positive_mask]

    protected_recall = (
        float(protected_tp.mean()) if len(protected_tp) else float("nan")
    )
    reference_recall = (
        float(reference_tp.mean()) if len(reference_tp) else float("nan")
    )
    equal_opportunity = (
        protected_recall - reference_recall
        if not any(np.isnan([protected_recall, reference_recall]))
        else float("nan")
    )
    return FairnessResult(protected_rate, reference_rate, disparate_impact, equal_opportunity)


def to_readable(result: FairnessResult) -> Dict[str, float]:
    return {
        "Protected Rate": result.protected_rate,
        "Reference Rate": result.reference_rate,
        "Disparate Impact": result.disparate_impact,
        "Equal Opportunity Gap": result.equal_opportunity,
    }

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import warnings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app_utils.fairness import compute_group_metrics
from app_utils.preprocessing import prepare_features
from app_utils.schema import validate_schema
from streamlit_app import load_model_registry, load_models, load_sample_data


def test_load_models_returns_all_models():
    models = load_models()
    assert isinstance(models, dict)
    expected = {"Decision Tree", "Random Forest", "Naive Bayes", "K-Nearest Neighbors"}
    assert set(models.keys()) == expected


def test_load_sample_data_matches_model_features():
    models = load_models()
    df = load_sample_data()
    features = list(df.columns)
    for name, model in models.items():
        if hasattr(model, "feature_names_in_"):
            assert list(model.feature_names_in_) == features
        elif hasattr(model, "n_features_in_"):
            assert model.n_features_in_ == len(features)
        else:
            pytest.fail(f"Model {name} does not expose feature names")


def test_model_registry_metadata_aligns_with_models():
    registry = load_model_registry()
    models = load_models()
    assert set(registry.keys()) == set(models.keys())


def test_prepare_features_engineers_logs_and_coerces_indicators():
    df = load_sample_data().head(1).copy()
    features = df.columns.tolist()
    df.loc[:, "contact_telephone"] = "True"
    df.loc[:, "poutcome_success"] = "False"
    df.loc[:, "day_of_week_mon"] = "yes"
    df.loc[:, "education_basic.6y"] = False
    df.loc[:, "job_management"] = True
    df.loc[:, "marital_single"] = False
    prepared = prepare_features(df, features)
    assert set(features) == set(prepared.columns)
    assert prepared.loc[0, "contact_telephone"] == 1.0
    for col in ["campaign_log", "pdays_log", "previous_log", "duration_log"]:
        assert col in prepared.columns


def test_validate_schema_flags_invalid_indicator():
    reference = load_sample_data().copy()
    reference.loc[0, "contact_telephone"] = 2
    report = validate_schema(reference, reference.columns)
    assert not report.is_valid
    assert "contact_telephone" in report.invalid_indicators


def test_prepare_features_reports_missing_columns_before_fill():
    reference = load_sample_data()
    features = reference.columns.tolist()
    without_age = reference.drop(columns=["age"])
    with pytest.raises(ValueError, match="age"):
        prepare_features(without_age, features)


def test_prepare_features_can_fill_missing_when_not_strict():
    reference = load_sample_data()
    features = reference.columns.tolist()

    minimal_input = pd.DataFrame(
        {
            "age": [40],
            "duration": [300],
            "campaign": [2],
            "pdays": [999],
            "previous": [0],
            "emp.var.rate": [1.1],
            "cons.price.idx": [93.9],
            "cons.conf.idx": [-42.7],
            "euribor3m": [4.8],
            "nr.employed": [5191.0],
        }
    )

    prepared = prepare_features(minimal_input, features, strict=False)

    assert set(prepared.columns) == set(features)
    assert prepared.isna().sum().sum() == 0


def test_validate_schema_captures_numeric_limits_and_missing_values():
    df = load_sample_data().head(2).copy()
    df.loc[0, "age"] = 10
    df.loc[0, "emp.var.rate"] = 20
    df.loc[1, "campaign"] = np.nan

    report = validate_schema(df, df.columns)

    assert not report.is_valid
    assert "age=10.0 below minimum 17" in report.row_errors[0]
    assert "emp.var.rate=20.0 above maximum 10" in report.row_errors[0]
    assert "campaign is missing" in report.row_errors[1]


def test_compute_group_metrics_handles_zero_reference_rate():
    df = pd.DataFrame({"marital_single": [1, 0, 0]})
    preds = np.array([1, 0, 0])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error")
        result = compute_group_metrics(df, "marital_single", preds)

    assert result.reference_rate == 0
    assert np.isnan(result.disparate_impact)
    assert not caught


def test_compute_group_metrics_outputs_expected_ratios():
    df = pd.DataFrame({"marital_single": [1, 0, 1, 0]})
    preds = np.array([1, 0, 0, 1])
    labels = np.array([1, 1, 1, 1])
    result = compute_group_metrics(df, "marital_single", preds, labels=labels)
    assert pytest.approx(result.protected_rate, 0.01) == 0.5
    assert pytest.approx(result.reference_rate, 0.01) == 0.5


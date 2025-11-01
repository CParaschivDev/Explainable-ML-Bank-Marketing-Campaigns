import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
    reference = load_sample_data()
    features = reference.columns.tolist()
    row = {
        "age": 30,
        "duration": 100,
        "campaign": 2,
        "pdays": 10,
        "previous": 1,
        "emp.var.rate": 1.0,
        "cons.price.idx": 93.0,
        "cons.conf.idx": -42.0,
        "euribor3m": 4.5,
        "nr.employed": 5191.0,
        "contact_telephone": "True",
        "poutcome_success": "False",
        "day_of_week_mon": "yes",
        "education_basic.6y": False,
        "job_management": True,
        "marital_single": False,
    }
    df = pd.DataFrame([row])
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


def test_compute_group_metrics_outputs_expected_ratios():
    df = pd.DataFrame({"marital_single": [1, 0, 1, 0]})
    preds = np.array([1, 0, 0, 1])
    labels = np.array([1, 1, 1, 1])
    result = compute_group_metrics(df, "marital_single", preds, labels=labels)
    assert pytest.approx(result.protected_rate, 0.01) == 0.5
    assert pytest.approx(result.reference_rate, 0.01) == 0.5


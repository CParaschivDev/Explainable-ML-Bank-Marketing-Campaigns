import pytest
from streamlit_app import load_models, load_sample_data


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


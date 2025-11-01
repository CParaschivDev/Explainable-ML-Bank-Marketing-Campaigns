import json
import os
from pathlib import Path
from typing import Dict, List

import io

import joblib
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import streamlit as st

from app_utils.fairness import compute_group_metrics, to_readable
from app_utils.preprocessing import prepare_features
from app_utils.schema import ValidationReport, validate_schema
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

MODEL_REGISTRY_PATH = Path("models/registry.json")
_EXPLAINER_CACHE: Dict[tuple, shap.Explainer] = {}


# --- Helper Function for SHAP Force Plots ---
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)


def get_explainer(model_name: str, model, background: pd.DataFrame) -> shap.Explainer:
    signature = (
        model_name,
        background.shape[0],
        tuple(background.columns),
    )
    cached = _EXPLAINER_CACHE.get(signature)
    if cached is not None:
        return cached

    if "Tree" in model_name or hasattr(model, "estimators_"):
        explainer = shap.TreeExplainer(model, background)
    else:
        def predict_proba_wrapper(X):
            return model.predict_proba(X)[:, 1]

        explainer = shap.KernelExplainer(predict_proba_wrapper, background)

    _EXPLAINER_CACHE[signature] = explainer
    return explainer


@st.cache_data
def load_model_registry() -> Dict[str, Dict]:
    """Return the structured model registry."""

    if not MODEL_REGISTRY_PATH.exists():
        raise FileNotFoundError(
            "Model registry missing. Add models/registry.json describing each artifact."
        )
    with MODEL_REGISTRY_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    registry = {entry["name"]: entry for entry in payload.get("models", [])}
    if not registry:
        raise ValueError("Model registry is empty.")
    return registry


# --- 1. Load Pretrained Models ---
@st.cache_resource
def load_models():
    models_dir = MODEL_REGISTRY_PATH.parent
    if not os.path.isdir(models_dir):
        st.error(
            f"Error: The '{models_dir}' directory was not found. Please make sure it's in your GitHub repository."
        )
        st.stop()

    try:
        registry = load_model_registry()
    except Exception as exc:  # pragma: no cover - surfaced in UI
        st.error(str(exc))
        st.stop()

    loaded_models = {}
    for name, entry in registry.items():
        path = models_dir / entry["path"]
        try:
            loaded_models[name] = joblib.load(path)
        except FileNotFoundError:
            st.error(f"Model file not found at '{path}'. Ensure it exists.")
            st.stop()
        except Exception as e:
            st.error(f"Error loading {name}: {e}")
            st.stop()
    return loaded_models

# --- 2. Load Background Data for SHAP ---
@st.cache_data
def load_sample_data(path="sample_data.csv"):
    try:
        data = pd.read_csv(path)
        if 'y' in data.columns:
            data = data.drop(columns=['y'])
        required_cols = data.columns.tolist()
        data = prepare_features(data, required_cols)
        return data
    except FileNotFoundError:
        st.error(f"Sample data not found at '{path}'.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        st.stop()

# --- Page Setup ---
st.set_page_config(page_title="Bank Marketing Predictor", layout="wide")

models = load_models()
model_registry = load_model_registry()
sample_data = load_sample_data()

model_features = sample_data.columns.tolist()
X_background = sample_data[model_features].head(100)
X_background = X_background.apply(pd.to_numeric, errors='coerce').astype('float64')

st.title("Bank Marketing Campaign Predictor")
st.markdown("""
Predict whether a bank customer will subscribe to a term deposit
based on various input features and explore the explanations behind the predictions.
""")

# --- Sidebar Controls ---
st.sidebar.header("App Controls")
selected_models = st.sidebar.multiselect(
    "Choose Models:", list(models.keys()), default=list(models.keys())
)
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold:", 0.0, 1.0, 0.5, 0.01,
    help="Minimum probability required to label a customer as subscribed"
)

show_animations = st.sidebar.checkbox(
    "Show Animations", value=True,
    help="Display balloons or snow after predictions",
)

st.sidebar.header("Explanation Controls")
explanation_type = st.sidebar.selectbox("Explanation Type:", ("SHAP", "LIME"))

if explanation_type == "SHAP":
    shap_plot_type = st.sidebar.selectbox(
        "SHAP Plot Type:", ("Bar Plot", "Waterfall", "Force Plot", "Summary Plot")
    )
    selected_shap_model_name = st.sidebar.selectbox("Model for SHAP:", list(models.keys()))
else:  # LIME
    selected_lime_model_name = st.sidebar.selectbox("Model for LIME:", list(models.keys()))
    lime_max = max(3, min(20, len(model_features)))
    lime_feature_limit = st.sidebar.slider(
        "LIME: number of features",
        3,
        lime_max,
        min(10, lime_max),
    )
    normalize_lime = st.sidebar.checkbox("Normalize LIME weights", value=False)

st.sidebar.header("Appearance")
theme_choice = st.sidebar.selectbox("Theme", ["Light", "Dark", "Colorblind-Friendly"])
accent_color = st.sidebar.color_picker("Accent Color", "#FF4B4B")
if theme_choice == "Dark":
    st.markdown(
        """
        <style>
        .stApp {background-color: #0e1117; color: #FAFAFA;}
        label {color: #FAFAFA;}
        div[data-testid=\"stDataFrame\"] .dataframe {background-color: #1e1e1e; color: #FAFAFA;}
        div[data-testid=\"stDataFrame\"] .dataframe th, div[data-testid=\"stDataFrame\"] .dataframe td {background-color: #1e1e1e; color: #FAFAFA;}
        input, select, textarea {background-color: #262730; color: #FAFAFA;}
        </style>
        """,
        unsafe_allow_html=True,
    )
elif theme_choice == "Colorblind-Friendly":
    st.info(
        "High-contrast theme optimized for users with color vision deficiencies."
    )
    st.markdown(
        """
        <style>
        .stApp {background-color: #000000; color: #FFFFFF;}
        label {color: #FFFFFF;}
        div[data-testid=\"stDataFrame\"] .dataframe {background-color: #000000; color: #FFFFFF;}
        div[data-testid=\"stDataFrame\"] .dataframe th, div[data-testid=\"stDataFrame\"] .dataframe td {background-color: #000000; color: #FFFFFF;}
        input, select, textarea {background-color: #000000; color: #FFFFFF; border: 2px solid #FFFFFF;}
        a {color: #FFFF00;}
        .stButton>button {background-color:#000000;color:#FFFF00;border:2px solid #FFFF00;}
        </style>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
    <style>
    .stButton>button {{background-color:{accent_color}; color:white;}}
    div[data-testid=\"stMetricValue\"] {{color:{accent_color};}}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar.expander("About"):
    st.write(
        "Interactive dashboard for predicting term deposit subscriptions "
        "with explainable machine learning models."
    )

with st.sidebar.expander("Model catalog"):
    catalog_rows = []
    for name, entry in model_registry.items():
        metrics = entry.get("metrics", {})
        catalog_rows.append(
            {
                "Model": name,
                "Version": entry.get("version", "?"),
                "Accuracy": metrics.get("accuracy"),
                "ROC AUC": metrics.get("roc_auc"),
            }
        )
    st.dataframe(pd.DataFrame(catalog_rows))

if "prediction_ready" not in st.session_state:
    st.session_state["prediction_ready"] = False

default_values = {
    "age_input": 40,
    "duration_input": 300,
    "campaign_input": 2,
    "pdays_input": 999,
    "previous_input": 0,
    "emp_var_rate_input": 1.1,
    "cons_price_idx_input": 93.9,
    "cons_conf_idx_input": -42.7,
    "euribor3m_input": 4.8,
    "nr_employed_input": 5191.0,
    "contact_telephone_input": True,
    "poutcome_success_input": False,
    "day_of_week_mon_input": False,
    "education_basic_6y_input": False,
    "job_management_input": False,
    "marital_single_input": False,
}

if st.sidebar.button("Load Example Profile"):
    # Populate the form with values from the first row of the sample dataset
    example_row = sample_data.iloc[0]
    example_values = {
        "age_input": int(example_row["age"]),
        "duration_input": int(example_row["duration"]),
        "campaign_input": int(example_row["campaign"]),
        "pdays_input": int(example_row["pdays"]),
        "previous_input": int(example_row["previous"]),
        "emp_var_rate_input": float(example_row["emp.var.rate"]),
        "cons_price_idx_input": float(example_row["cons.price.idx"]),
        "cons_conf_idx_input": float(example_row["cons.conf.idx"]),
        "euribor3m_input": float(example_row["euribor3m"]),
        "nr_employed_input": float(example_row["nr.employed"]),
        "contact_telephone_input": bool(example_row["contact_telephone"]),
        "poutcome_success_input": bool(example_row["poutcome_success"]),
        "day_of_week_mon_input": bool(example_row["day_of_week_mon"]),
        "education_basic_6y_input": bool(example_row["education_basic.6y"]),
        "job_management_input": bool(example_row["job_management"]),
        "marital_single_input": bool(example_row["marital_single"]),
    }
    for k, v in example_values.items():
        st.session_state[k] = v
    st.session_state["prediction_ready"] = False

if st.sidebar.button("Reset Inputs"):
    for k, v in default_values.items():
        st.session_state[k] = v
    st.session_state["prediction_ready"] = False

# --- Input Form ---
st.header("Customer Input")
with st.form("customer_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider(
            "Age", 18, 100, 40, key="age_input",
            help="Customer age in years"
        )
        duration = st.slider(
            "Duration (s)", 0, 5000, 300, key="duration_input",
            help="Last contact duration in seconds"
        )
        campaign = st.number_input(
            "Campaign Contacts", 1, 60, 2, key="campaign_input",
            help="Number of contacts performed during this campaign"
        )
        pdays = st.number_input(
            "Days Since Last Contact", 0, 999, 999, key="pdays_input",
            help="Days since client was last contacted (-1 means never)"
        )
        previous = st.number_input(
            "Previous Contacts", 0, 10, 0, key="previous_input",
            help="Number of contacts before this campaign"
        )

    with col2:
        emp_var_rate = st.number_input(
            "Employment Var. Rate", -4.0, 2.0, 1.1, step=0.1, key="emp_var_rate_input",
            help="Quarterly employment variation rate"
        )
        cons_price_idx = st.number_input(
            "Consumer Price Index", 92.0, 95.0, 93.9, step=0.1, key="cons_price_idx_input",
            help="Consumer price index"
        )
        cons_conf_idx = st.number_input(
            "Consumer Confidence Index", -51.0, -26.0, -42.7, step=0.1, key="cons_conf_idx_input",
            help="Consumer confidence index"
        )
        euribor3m = st.number_input(
            "Euribor 3m", 0.5, 5.5, 4.8, step=0.1, key="euribor3m_input",
            help="Euribor 3-month rate"
        )
        nr_employed = st.number_input(
            "Number Employed", 4900.0, 5300.0, 5191.0, step=0.1, key="nr_employed_input",
            help="Number of employees"
        )

    with col3:
        contact_telephone = st.checkbox(
            "Contacted via Telephone?", True, key="contact_telephone_input",
            help="Was the client contacted via telephone?"
        )
        poutcome_success = st.checkbox(
            "Previous Outcome: Success?", False, key="poutcome_success_input",
            help="Was the outcome of the previous campaign successful?"
        )
        day_of_week_mon = st.checkbox(
            "Day of Week: Monday?", False, key="day_of_week_mon_input",
            help="Was the last contact on a Monday?"
        )
        education_basic_6y = st.checkbox(
            "Education: Basic 6y?", False, key="education_basic_6y_input",
            help="Does the client have basic 6 years education?"
        )
        job_management = st.checkbox(
            "Job: Management?", False, key="job_management_input",
            help="Is the client's job management?"
        )
        marital_single = st.checkbox(
            "Marital Status: Single?", False, key="marital_single_input",
            help="Is the client single?"
        )

    submitted = st.form_submit_button("Predict")
    if submitted:
        st.session_state["prediction_ready"] = True

# --- Preprocess Input ---
user_input_dict = {
    'age': age,
    'duration': duration,
    'campaign': campaign,
    'pdays': pdays,
    'previous': previous,
    'emp.var.rate': emp_var_rate,
    'cons.price.idx': cons_price_idx,
    'cons.conf.idx': cons_conf_idx,
    'euribor3m': euribor3m,
    'nr.employed': nr_employed,
    'contact_telephone': contact_telephone,
    'poutcome_success': poutcome_success,
    'day_of_week_mon': day_of_week_mon,
    'education_basic.6y': education_basic_6y,
    'job_management': job_management,
    'marital_single': marital_single,
}

user_data = pd.DataFrame([user_input_dict])
try:
    user_data_aligned = prepare_features(user_data, model_features)
    user_validation_report = validate_schema(
        user_data_aligned, model_features
    )
    st.session_state["user_validation_error"] = ""
except ValueError as exc:
    user_data_aligned = None
    user_validation_report = None
    st.session_state["prediction_ready"] = False
    st.session_state["user_validation_error"] = str(exc)

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Predictions",
    "🔍 Explanation",
    "📁 Batch Predictions",
    "📈 Data Exploration",
    "🛠️ Train & Evaluate",
])

with tab1:
    st.header("Prediction Results")
    if not st.session_state.get("prediction_ready"):
        st.info("Fill in the customer form and hit Predict to view results.")
    elif st.session_state.get("user_validation_error"):
        st.error(st.session_state["user_validation_error"])
    elif user_data_aligned is None:
        st.error("Input validation failed. Please review the customer form.")
    elif not selected_models:
        st.warning("Select at least one model.")
    else:
        st.dataframe(user_data_aligned)
        if user_validation_report and user_validation_report.is_valid:
            st.caption("✅ Input validated against the expected schema.")
        elif user_validation_report:
            st.warning("Input contains schema warnings; predictions may be unreliable.")
        predictions_df = pd.DataFrame(columns=["Model", "Prediction", "Confidence"])
        confidences, predictions = [], []

        with st.spinner("Generating predictions..."):
            for model_name in selected_models:
                model = models[model_name]
                try:
                    proba = model.predict_proba(user_data_aligned)[:, 1][0]
                    label = "Subscribed" if proba >= confidence_threshold else "Not Subscribed"
                    predictions_df.loc[len(predictions_df)] = [model_name, label, f"{proba:.2f}"]
                    confidences.append(proba)
                    predictions.append(1 if label == "Subscribed" else 0)
                except Exception as e:
                    st.error(f"Error with {model_name}: {e}")

        st.dataframe(predictions_df, use_container_width=True)
        if confidences:
            metric_cols = st.columns(len(selected_models))
            for idx, model_name in enumerate(selected_models):
                status = "Subscribed" if predictions[idx] else "Not Subscribed"
                metric_cols[idx].metric(model_name, f"{confidences[idx]:.2f}", status)

            vote = "Subscribed" if sum(predictions) >= len(predictions)/2 else "Not Subscribed"
            st.markdown(f"**Ensemble Vote:** `{vote}`")
            if show_animations:
                if vote == "Subscribed":
                    st.balloons()
                else:
                    st.snow()
            st.markdown(f"**Avg Confidence:** `{np.mean(confidences):.2f}`")
            fig, ax = plt.subplots()
            sns.barplot(x=selected_models, y=confidences, ax=ax)
            ax.set_ylim(0, 1)
            ax.set_ylabel("P=Subscribed")
            ax.set_title("Model Confidences")
            st.pyplot(fig)
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            st.download_button(
                "Download confidence chart",
                buf.getvalue(),
                file_name="confidences.png",
                mime="image/png",
            )
        csv_pred = predictions_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download predictions", csv_pred, "predictions.csv", "text/csv"
        )

with tab2:
    if not st.session_state.get("prediction_ready"):
        st.info("Generate a prediction first to view explanations.")
    elif explanation_type == "SHAP":
        st.header(f"SHAP Explanation: {selected_shap_model_name}")
        try:
            with st.spinner("Computing SHAP values..."):
                model = models[selected_shap_model_name]
                explainer = get_explainer(selected_shap_model_name, model, X_background)
                # Get raw SHAP values
                shap_vals_raw = explainer.shap_values(user_data_aligned)

                # Determine the correct SHAP values and base value for the plot
                if isinstance(shap_vals_raw, list):
                    # This case is for multi-output models where shap_vals_raw is a list of arrays,
                    # typically [shap_values_class_0, shap_values_class_1, ...]
                    # Each element in the list is (num_instances, num_features)
                    # We want the positive class (index 1) and the first instance
                    shap_values_for_plot = shap_vals_raw[1][0]
                    base_value_for_plot = explainer.expected_value[1]
                elif shap_vals_raw.ndim == 3:
                    # This case is for multi-output models where shap_vals_raw is (num_instances, num_features, num_classes)
                    # We want the first instance, and the SHAP values for the positive class (index 1)
                    shap_values_for_plot = shap_vals_raw[0][:, 1]
                    base_value_for_plot = explainer.expected_value[1]
                else: # shap_vals_raw.ndim == 2
                    # This case is for single-output models where shap_vals_raw is (num_instances, num_features)
                    # We want the first instance
                    shap_values_for_plot = shap_vals_raw[0]
                    base_value_for_plot = explainer.expected_value

                # Create the Explanation object
                explanation = shap.Explanation(
                    values=shap_values_for_plot,
                    base_values=base_value_for_plot,
                    data=np.array(user_data_aligned.iloc[0]),
                    feature_names=model_features
                )

            if shap_plot_type == "Bar Plot":
                fig, _ = plt.subplots()
                shap.plots.bar(explanation, show=False)
                st.pyplot(fig)
            elif shap_plot_type == "Waterfall":
                fig, _ = plt.subplots()
                shap.plots.waterfall(explanation, show=False)
                st.pyplot(fig)
            elif shap_plot_type == "Force Plot":
                st_shap(
                    shap.force_plot(
                        base_value_for_plot,
                        shap_values_for_plot,
                        features=explanation.data,
                        feature_names=explanation.feature_names,
                    )
                )
                fig = None
            elif shap_plot_type == "Summary Plot":
                fig = plt.figure()
                shap_values_bg = explainer.shap_values(X_background)
                if isinstance(shap_values_bg, list):
                    shap.summary_plot(shap_values_bg[1], X_background, show=False)
                else:
                    shap.summary_plot(shap_values_bg, X_background, show=False)
                st.pyplot(fig)
            if shap_plot_type != "Force Plot":
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                st.download_button(
                    "Download plot",
                    buf.getvalue(),
                    file_name="shap_plot.png",
                    mime="image/png",
                )
        except Exception as e:
            st.error(f"SHAP error: {e}")
    else: # LIME Explanation
        st.header(f"LIME Explanation: {selected_lime_model_name}")
        try:
            with st.spinner("Computing LIME explanation..."):
                model = models[selected_lime_model_name]
                # LIME explainer
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data=X_background.values,
                    feature_names=model_features,
                    class_names=['Not Subscribed', 'Subscribed'],
                    mode='classification'
                )

                # Explain the instance
                # LIME expects a 1D numpy array for the instance to explain
                explanation = explainer.explain_instance(
                    data_row=user_data_aligned.iloc[0].values,
                    predict_fn=model.predict_proba,
                    num_features=lime_feature_limit
                )

            # Display LIME explanation
            fig = explanation.as_pyplot_figure()
            st.pyplot(fig)
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            st.download_button(
                "Download plot",
                buf.getvalue(),
                file_name="lime_plot.png",
                mime="image/png",
            )

            st.markdown("---")
            st.subheader("LIME Explanation Details")
            lime_pairs = explanation.as_list()[:lime_feature_limit]
            if normalize_lime:
                denom = sum(abs(weight) for _, weight in lime_pairs) or 1
                lime_pairs = [(feature, weight / denom) for feature, weight in lime_pairs]
                st.caption("Weights normalized to sum of absolute contributions = 1.0")
            for feature, weight in lime_pairs:
                st.write(f"- **{feature}**: {weight:.4f}")

        except Exception as e:
            st.error(f"LIME error: {e}")

with tab3:
    st.header("Batch Predictions")
    uploaded_file = st.file_uploader(
        "Upload CSV", type="csv", help="File must include the same feature columns as sample_data.csv"
    )
    if uploaded_file is not None:
        raw_batch_df = pd.read_csv(uploaded_file)
        st.caption("Preview of uploaded dataset")
        st.dataframe(raw_batch_df.head())

        with st.status("Validating batch file...", expanded=False) as status:
            try:
                batch_df = prepare_features(raw_batch_df, model_features)
            except ValueError as exc:
                status.update(state="error", label="Validation failed")
                st.error(f"Schema validation failed: {exc}")
                batch_df = None
            else:
                status.update(state="complete", label="Validation passed")

        if batch_df is not None:
            report = validate_schema(batch_df, model_features)
            if report.row_errors:
                st.error(
                    "Row-level validation errors detected. Please fix the dataset and re-upload."
                )
                st.json(report.row_errors)
            else:
                st.success("Batch file validated successfully.")
                results_df = batch_df.copy()
                for model_name in selected_models:
                    model = models[model_name]
                    try:
                        probas = model.predict_proba(batch_df)[:, 1]
                        results_df[f"{model_name}_prob"] = probas
                        results_df[f"{model_name}_pred"] = np.where(
                            probas >= confidence_threshold, "Subscribed", "Not Subscribed"
                        )
                    except Exception as e:
                        st.error(f"Error with {model_name}: {e}")
                st.dataframe(results_df, use_container_width=True)
                csv_res = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download results", csv_res, "batch_predictions.csv", "text/csv"
                )
    else:
        st.info("Upload a CSV file to perform batch predictions.")

with tab4:
    st.header("Data Exploration")
    explore_upload = st.file_uploader(
        "Upload dataset", type="csv", key="explore_upload"
    )
    if explore_upload is not None:
        explore_df = pd.read_csv(explore_upload)
    else:
        explore_df = sample_data.copy()

    st.subheader("Summary Statistics")
    st.dataframe(explore_df.describe())

    st.subheader("Correlation Heatmap")
    corr = explore_df.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, ax=ax)
    st.pyplot(fig)

    st.subheader("Feature Distribution")
    feature = st.selectbox("Select Feature", explore_df.columns)
    fig2, ax2 = plt.subplots()
    sns.histplot(explore_df[feature], bins=20, ax=ax2)
    st.pyplot(fig2)

with tab5:
    st.header("Model Training & Evaluation")
    train_upload = st.file_uploader(
        "Upload labeled dataset", type="csv", key="train_upload"
    )
    if train_upload is not None:
        train_df = pd.read_csv(train_upload)
        target_col = st.selectbox(
            "Target column", train_df.columns, key="target_col"
        )
        feature_cols = [c for c in train_df.columns if c != target_col]
        binary_candidates = [
            c for c in feature_cols if set(train_df[c].dropna().unique()).issubset({0, 1})
        ]
        protected_attr = st.selectbox(
            "Protected attribute for fairness checks",
            options=binary_candidates or ["(none)"]
        )
        n_estimators = st.slider("Number of Trees", 50, 500, 100, 50)
        max_depth = st.slider("Max depth (0 = unlimited)", 0, 20, 0, 1)
        min_samples_leaf = st.slider("Min samples per leaf", 1, 10, 1)
        cv_folds = st.slider("Cross-validation Folds", 2, 10, 5, 1)
        balance_classes = st.checkbox("Balance class weights", value=True)

        target_counts = train_df[target_col].value_counts(normalize=True)
        st.caption("Class distribution")
        st.dataframe(target_counts.rename("share"))

        if st.button("Run Training", key="run_training"):
            X = train_df[feature_cols]
            y = train_df[target_col]
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=42,
                max_depth=None if max_depth == 0 else max_depth,
                min_samples_leaf=min_samples_leaf,
                class_weight="balanced" if balance_classes else None,
            )
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
            st.write(
                f"Accuracy: {scores.mean():.3f} ± {scores.std():.3f}"
            )
            preds = cross_val_predict(model, X, y, cv=cv)
            cm = confusion_matrix(y, preds)
            fig3, ax3 = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3)
            ax3.set_xlabel("Predicted")
            ax3.set_ylabel("Actual")
            st.subheader("Confusion Matrix")
            st.pyplot(fig3)
            probas = cross_val_predict(
                model, X, y, cv=cv, method="predict_proba"
            )[:, 1]
            fpr, tpr, _ = roc_curve(y, probas)
            roc_auc = auc(fpr, tpr)
            fig4, ax4 = plt.subplots()
            ax4.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
            ax4.plot([0, 1], [0, 1], "--")
            ax4.set_xlabel("False Positive Rate")
            ax4.set_ylabel("True Positive Rate")
            ax4.legend(loc="lower right")
            st.subheader("ROC Curve")
            st.pyplot(fig4)
            precision, recall, _ = precision_recall_curve(y, probas)
            fig5, ax5 = plt.subplots()
            ax5.plot(recall, precision)
            ax5.set_xlabel("Recall")
            ax5.set_ylabel("Precision")
            st.subheader("Precision-Recall Curve")
            st.pyplot(fig5)
            if protected_attr != "(none)":
                st.subheader("Fairness Check")
                fairness = compute_group_metrics(
                    train_df,
                    protected_attr,
                    preds.astype(int),
                    labels=y.to_numpy(),
                )
                st.table(pd.DataFrame([to_readable(fairness)]))
    else:
        st.info("Upload a labeled dataset to train and evaluate models.")

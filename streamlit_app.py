import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os

# --- Helper Function for SHAP Force Plots (Define before use) ---
def st_shap(plot, height=None):
    """
    Helper function to display SHAP plots in Streamlit.
    This function embeds the SHAP plot's HTML into the Streamlit app.
    """
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)

# --- 1. Load Pretrained Models with Robust Error Handling ---
@st.cache_resource
def load_models():
    """Loads all models from the 'models' directory with error handling."""
    models_dir = "models"
    # Stop if the directory doesn't exist in the repository
    if not os.path.isdir(models_dir):
        st.error(f"Error: The '{models_dir}' directory was not found. Please make sure it's in your GitHub repository.")
        st.stop()

    model_paths = {
        "Decision Tree": os.path.join(models_dir, "decision_tree_model.pkl"),
        "Random Forest": os.path.join(models_dir, "random_forest_model.pkl"),
        "Naive Bayes": os.path.join(models_dir, "naive_bayes_model.pkl"),
        "K-Nearest Neighbors": os.path.join(models_dir, "best_knn_model.pkl"),
    }
    
    loaded_models = {}
    for name, path in model_paths.items():
        try:
            loaded_models[name] = joblib.load(path)
        except FileNotFoundError:
            st.error(f"Error: Model file not found at '{path}'. Please ensure this file exists and the name is correct.")
            st.info(f"Check if '{os.path.basename(path)}' is in the '{models_dir}' folder in your GitHub repository.")
            st.stop() # Stop execution if a model is missing
        except Exception as e:
            st.error(f"Error loading {name} model from {path}: {e}")
            st.stop()
    return loaded_models

# --- 2. Load Background Data with Robust Error Handling ---
@st.cache_data
def load_sample_data(path="sample_data.csv"):
    """Loads sample data for SHAP KernelExplainer background."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"Error: Sample data file not found at '{path}'. Please ensure it's in your GitHub repository.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading sample data from {path}: {e}")
        st.stop()

# --- Page Setup and Main App Logic ---
st.set_page_config(
    page_title="Bank Marketing Predictor 🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data and models first
models = load_models()
sample_data = load_sample_data()

# Define model features and validate sample data
model_features = ["duration", "campaign", "contact_cellular", "month_may", "poutcome_success"]
if not all(feature in sample_data.columns for feature in model_features):
    st.error("Sample data does not contain all required features for SHAP explanation.")
    st.stop()
X_background = sample_data[model_features]

st.title("Bank Marketing Campaign Predictor 📈")
st.markdown("""
Welcome! This app helps predict whether a bank customer will subscribe to a term deposit
based on various input features. You can also explore the explanations behind the predictions.
""")

# --- Sidebar Controls ---
st.sidebar.header("App Controls")
selected_models = st.sidebar.multiselect(
    "Choose Models for Prediction:",
    list(models.keys()),
    default=list(models.keys())
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold for 'Subscribed':",
    0.0, 1.0, 0.5, 0.01
)

st.sidebar.header("Explanation Controls")
shap_plot_type = st.sidebar.selectbox(
    "Choose SHAP Plot Type:",
    ("Bar Plot", "Waterfall", "Force Plot")
)
selected_shap_model_name = st.sidebar.selectbox(
    "Select Model for SHAP Explanation:",
    list(models.keys())
)

# --- Input Form for User Data ---
st.header("Customer Information Input")
col1, col2, col3 = st.columns(3)

with col1:
    duration = st.slider("Duration (seconds):", 0, 1200, 200)
    campaign = st.number_input("Campaign (contacts this campaign):", 1, 60, 2)
with col2:
    contact_cellular = st.checkbox("Contacted via Cellular?", True)
    month_may = st.checkbox("Contacted in May?", False)
with col3:
    poutcome_success = st.checkbox("Previous Outcome was Success?", False)

user_data = pd.DataFrame({
    "duration": [duration],
    "campaign": [campaign],
    "contact_cellular": [1 if contact_cellular else 0],
    "month_may": [1 if month_may else 0],
    "poutcome_success": [1 if poutcome_success else 0]
})

user_data_aligned = user_data.reindex(columns=model_features, fill_value=0)

# --- Tabs for Predictions and Explanations ---
tab1, tab2 = st.tabs(["📊 Predictions", "🔍 SHAP Explanation"])

with tab1:
    st.header("Prediction Results")
    if not selected_models:
        st.warning("Please select at least one model to see predictions.")
    else:
        predictions_df = pd.DataFrame(columns=["Model", "Prediction", "Confidence (P=Subscribed)"])
        confidences = []
        individual_predictions = []

        for model_name in selected_models:
            model = models[model_name]
            try:
                proba = model.predict_proba(user_data_aligned)[:, 1][0]
                prediction_label = "Subscribed" if proba >= confidence_threshold else "Not Subscribed"
                predictions_df.loc[len(predictions_df)] = [model_name, prediction_label, f"{proba:.2f}"]
                confidences.append(proba)
                individual_predictions.append(1 if proba >= confidence_threshold else 0)
            except Exception as e:
                st.error(f"Error predicting with {model_name}: {e}")

        st.dataframe(predictions_df, use_container_width=True)

        if confidences:
            ensemble_vote = "Subscribed" if sum(individual_predictions) >= len(individual_predictions) / 2 else "Not Subscribed"
            st.markdown(f"**Ensemble Majority Vote Result:** <span style='color: {'green' if ensemble_vote == 'Subscribed' else 'red'}; font-weight: bold;'>{ensemble_vote}</span>", unsafe_allow_html=True)
            avg_confidence = np.mean(confidences)
            st.markdown(f"**Average Ensemble Confidence (P=Subscribed):** `{avg_confidence:.2f}`")
            st.subheader("Model Confidence Comparison")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=selected_models, y=confidences, ax=ax, palette="viridis")
            ax.set_ylim(0, 1)
            ax.set_ylabel("Confidence (P=Subscribed)")
            ax.set_title("Individual Model Confidence")
            st.pyplot(fig)

with tab2:
    st.header(f"SHAP Explanation for {selected_shap_model_name}")
    if not selected_shap_model_name:
        st.warning("Please select a model for explanation from the sidebar.")
    else:
        model_to_explain = models[selected_shap_model_name]
        st.subheader(f"Explanation Plot Type: {shap_plot_type}")
        try:
            if "Tree" in selected_shap_model_name or "Forest" in selected_shap_model_name:
                explainer = shap.TreeExplainer(model_to_explain)
            else:
                explainer = shap.KernelExplainer(model_to_explain.predict_proba, X_background)

            shap_values = explainer.shap_values(user_data_aligned)
            
            if isinstance(shap_values, list):
                explanation = shap.Explanation(values=shap_values[1][0], base_values=explainer.expected_value[1], data=user_data_aligned.iloc[0], feature_names=model_features)
            else:
                explanation = shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=user_data_aligned.iloc[0], feature_names=model_features)

            if shap_plot_type == "Bar Plot":
                fig, ax = plt.subplots()
                shap.plots.bar(explanation, show=False)
                st.pyplot(fig)

            elif shap_plot_type == "Waterfall":
                fig, ax = plt.subplots()
                shap.plots.waterfall(explanation, show=False)
                st.pyplot(fig)

            elif shap_plot_type == "Force Plot":
                st_shap(shap.force_plot(explanation.base_values, explanation.values, explanation.data, feature_names=explanation.feature_names))

        except Exception as e:
            st.error(f"Error generating SHAP plot: {e}")

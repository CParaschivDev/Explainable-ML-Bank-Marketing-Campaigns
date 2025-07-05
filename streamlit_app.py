import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os

# --- 1. Load Pretrained Models ---
@st.cache_resource
def load_models():
    models_dir = "models"
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
            st.error(f"Error: Model file not found at {path}. Please ensure all model files are in the 'models' directory.")
            st.stop()
        except Exception as e:
            st.error(f"Error loading {name} model from {path}: {e}")
            st.stop()
    return loaded_models

models = load_models()

# --- Dummy or Cached Background Data for SHAP ---
@st.cache_data
def load_sample_data(path="sample_data.csv"):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"Error: Sample data file not found at {path}. Please ensure 'sample_data.csv' exists.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading sample data from {path}: {e}")
        st.stop()

sample_data = load_sample_data()
# Ensure sample_data has the expected features for SHAP background
model_features = ["duration", "campaign", "contact_cellular", "month_may", "poutcome_success"]
if not all(feature in sample_data.columns for feature in model_features):
    st.error("Sample data does not contain all required features for SHAP explanation.")
    st.stop()
X_background = sample_data[model_features]


# --- 2. Streamlit Page Setup ---
st.set_page_config(
    page_title="Bank Marketing Predictor 🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

shap_plot_type = st.sidebar.selectbox(
    "Choose SHAP Plot Type:",
    ("Bar Plot", "Waterfall", "Force Plot")
)

# --- Input Form for User Data ---
st.header("Customer Information Input")
col1, col2, col3 = st.columns(3)

with col1:
    duration = st.slider("Duration (seconds):", 0, 1200, 200)
    campaign = st.number_input("Campaign (number of contacts performed during this campaign):", 1, 60, 2)
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

# Align columns to model's expected features
user_data_aligned = user_data.reindex(columns=model_features, fill_value=0)

# --- Two Tabs: Prediction vs Explanation ---
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
                # Predict probabilities for the positive class (1)
                proba = model.predict_proba(user_data_aligned)[:, 1][0]
                prediction_label = "Subscribed" if proba >= confidence_threshold else "Not Subscribed"
                
                predictions_df.loc[len(predictions_df)] = [model_name, prediction_label, f"{proba:.2f}"]
                confidences.append(proba)
                individual_predictions.append(1 if proba >= confidence_threshold else 0)
            except Exception as e:
                st.error(f"Error predicting with {model_name}: {e}")

        st.dataframe(predictions_df, use_container_width=True)

        if confidences:
            # Ensemble Majority Vote
            ensemble_vote = "Subscribed" if sum(individual_predictions) >= len(individual_predictions) / 2 else "Not Subscribed"
            st.markdown(f"**Ensemble Majority Vote Result:** <span style='color: {'green' if ensemble_vote == 'Subscribed' else 'red'}; font-weight: bold;'>{ensemble_vote}</span>", unsafe_allow_html=True)

            # Average Ensemble Confidence
            avg_confidence = np.mean(confidences)
            st.markdown(f"**Average Ensemble Confidence (P=Subscribed):** `{avg_confidence:.2f}`")

            # Bar plot of confidences
            st.subheader("Model Confidence Comparison")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=selected_models, y=confidences, ax=ax, palette="viridis")
            ax.set_ylim(0, 1)
            ax.set_ylabel("Confidence (P=Subscribed)")
            ax.set_title("Individual Model Confidence")
            st.pyplot(fig)

            # --- 6. Download Button ---
            csv_predictions = predictions_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=csv_predictions,
                file_name="bank_marketing_predictions.csv",
                mime="text/csv",
            )

with tab2:
    st.header("SHAP Explanation")
    selected_shap_model_name = st.selectbox(
        "Select Model for SHAP Explanation:",
        list(models.keys())
    )

    if selected_shap_model_name:
        model_to_explain = models[selected_shap_model_name]
        
        try:
            # Determine explainer type
            if "Tree" in selected_shap_model_name or "Forest" in selected_shap_model_name:
                explainer = shap.TreeExplainer(model_to_explain)
            else:
                # For KNN and Naive Bayes, use KernelExplainer
                # Ensure X_background is suitable (e.g., a small sample of training data)
                explainer = shap.KernelExplainer(model_to_explain.predict_proba, X_background)

            shap_values = explainer.shap_values(user_data_aligned)

            st.subheader(f"SHAP Values for {selected_shap_model_name} ({shap_plot_type})")

            # SHAP plots
            if shap_plot_type == "Bar Plot":
                # For classification, shap_values will be a list of arrays (one for each class)
                # We usually want to explain the positive class (index 1)
                if isinstance(shap_values, list):
                    shap.plots.bar(shap.Explanation(values=shap_values[1][0], 
                                                    base_values=explainer.expected_value[1], 
                                                    data=user_data_aligned.iloc[0], 
                                                    feature_names=model_features), show=False)
                else: # For regression or single output models
                    shap.plots.bar(shap.Explanation(values=shap_values[0], 
                                                    base_values=explainer.expected_value, 
                                                    data=user_data_aligned.iloc[0], 
                                                    feature_names=model_features), show=False)
                st.pyplot(plt.gcf())
                plt.clf() # Clear the current figure to prevent overlap

            elif shap_plot_type == "Waterfall":
                if isinstance(shap_values, list):
                    shap.plots.waterfall(shap.Explanation(values=shap_values[1][0], 
                                                          base_values=explainer.expected_value[1], 
                                                          data=user_data_aligned.iloc[0], 
                                                          feature_names=model_features), show=False)
                else:
                    shap.plots.waterfall(shap.Explanation(values=shap_values[0], 
                                                          base_values=explainer.expected_value, 
                                                          data=user_data_aligned.iloc[0], 
                                                          feature_names=model_features), show=False)
                st.pyplot(plt.gcf())
                plt.clf()

            elif shap_plot_type == "Force Plot":
                # Force plot requires the JS component, use st_shap wrapper
                # For classification, explain the positive class (index 1)
                if isinstance(shap_values, list):
                    st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1][0], user_data_aligned.iloc[0], feature_names=model_features))
                else:
                    st_shap(shap.force_plot(explainer.expected_value, shap_values[0], user_data_aligned.iloc[0], feature_names=model_features))

        except Exception as e:
            st.error(f"Error generating SHAP plot: {e}. Please ensure the selected model and data are compatible with the chosen SHAP explainer and plot type.")
            st.info("For KernelExplainer, ensure `X_background` is representative and small enough for performance.")

# Helper function for st_shap (from SHAP documentation)
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)

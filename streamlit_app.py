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

# --- FIX 1: Define ALL features the model was trained on ---
# The list is based on your error message. You might need to adjust it.
model_features = [
    'age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 
    'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 
    'job_blue-collar', 'job_entrepreneur', 'job_housemaid', 'job_management', 
    'job_retired', 'job_self-employed', 'job_services', 'job_student', 
    'job_technician', 'job_unemployed', 'marital_married', 'marital_single', 
    'education_basic.6y', 'education_basic.9y', 'education_high.school', 
    'education_illiterate', 'education_professional.course', 'education_university.degree', 
    'default_unknown', 'housing_yes', 'loan_yes', 'contact_telephone', 
    'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar', 'month_may', 
    'month_nov', 'month_oct', 'month_sep', 'day_of_week_mon', 'day_of_week_thu', 
    'day_of_week_tue', 'day_of_week_wed', 'poutcome_nonexistent', 'poutcome_success', 
    'campaign_log' # This is a preprocessed feature
]

# Create a smaller list for SHAP explanations for clarity if desired
# Using all features is also fine
shap_features = ["duration", "campaign_log", "age", "poutcome_success", "cons.price.idx", "euribor3m"]

if not all(feature in sample_data.columns for feature in model_features):
    st.error("Sample data does not contain all required features for the model.")
    st.info("Please ensure your `sample_data.csv` has all the columns from the training set.")
    st.stop()
    
X_background = sample_data[model_features].head(100) # Use a sample for SHAP background

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

# --- FIX 2: Add UI Inputs for the missing features ---
st.header("Customer Information Input")
st.info("Please fill in the details below. These inputs should match the features your model was trained on.")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age:", 18, 100, 40)
    duration = st.slider("Duration of last contact (seconds):", 0, 5000, 300)
    campaign = st.number_input("Number of contacts this campaign:", 1, 60, 2)
    pdays = st.number_input("Days since last contacted (999 if never):", 0, 999, 999)
    previous = st.number_input("Number of contacts before this campaign:", 0, 10, 0)
    
with col2:
    emp_var_rate = st.number_input("Employment Variation Rate:", -4.0, 2.0, 1.1, step=0.1)
    cons_price_idx = st.number_input("Consumer Price Index:", 92.0, 95.0, 93.9, step=0.1)
    cons_conf_idx = st.number_input("Consumer Confidence Index:", -51.0, -26.0, -42.7, step=0.1)
    euribor3m = st.number_input("Euribor 3 Month Rate:", 0.5, 5.5, 4.8, step=0.1)
    nr_employed = st.number_input("Number of Employees:", 4900.0, 5300.0, 5191.0, step=0.1)

with col3:
    contact_telephone = not st.checkbox("Contacted via Cellular?", True) # Inferred from 'contact_telephone'
    poutcome_success = st.checkbox("Previous Outcome was Success?", False)
    # The rest of the features are categorical (one-hot encoded). 
    # For a production app, you would use dropdowns for 'job', 'marital', etc.,
    # and then one-hot encode them. For this example, we'll set them to 0.
    st.write("Other categorical features are set to their default (0) for this demo.")


# --- FIX 3: Create the full feature set and apply preprocessing ---
# Start with a dictionary of zeros for all features
user_input_dict = {feature: 0 for feature in model_features}

# Update the dictionary with user inputs
user_input_dict.update({
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
    'contact_telephone': 1 if contact_telephone else 0,
    'poutcome_success': 1 if poutcome_success else 0
})

# IMPORTANT: Replicate preprocessing steps from training
# Add a small epsilon to avoid log(0)
user_input_dict['campaign_log'] = np.log(user_input_dict['campaign'] + 1e-6)

# Create the final DataFrame
user_data = pd.DataFrame([user_input_dict])

# Align columns to ensure the order is exactly as the model expects
user_data_aligned = user_data[model_features]


# --- Tabs for Predictions and Explanations ---
tab1, tab2 = st.tabs(["📊 Predictions", "🔍 SHAP Explanation"])

with tab1:
    st.header("Prediction Results")
    if not selected_models:
        st.warning("Please select at least one model to see predictions.")
    else:
        # Display the data being sent to the model for verification
        st.write("Data sent for prediction:")
        st.dataframe(user_data_aligned)
        
        predictions_df = pd.DataFrame(columns=["Model", "Prediction", "Confidence (P=Subscribed)"])
        confidences = []
        individual_predictions = []

        for model_name in selected_models:
            model = models[model_name]
            try:
                # Use the fully-featured and aligned DataFrame
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
                explainer = shap.TreeExplainer(model_to_explain, X_background)
            else:
                # Use the predict_proba function and the background dataset
                explainer = shap.KernelExplainer(model_to_explain.predict_proba, X_background)

            shap_values = explainer.shap_values(user_data_aligned)
            
            # Handling different output shapes from SHAP explainers
            if isinstance(shap_values, list): # For classifiers with two outputs (e.g., KernelExplainer)
                shap_values_for_class_1 = shap_values[1]
                expected_value = explainer.expected_value[1] if hasattr(explainer, 'expected_value') and isinstance(explainer.expected_value, list) else explainer.expected_value
            else: # For TreeExplainer on classifiers
                shap_values_for_class_1 = shap_values
                expected_value = explainer.expected_value
            
            # Create SHAP explanation object
            explanation = shap.Explanation(
                values=shap_values_for_class_1[0], 
                base_values=expected_value, 
                data=user_data_aligned.iloc[0], 
                feature_names=model_features
            )

            # Display plots
            if shap_plot_type == "Bar Plot":
                fig, ax = plt.subplots()
                shap.plots.bar(explanation, show=False)
                st.pyplot(fig)

            elif shap_plot_type == "Waterfall":
                fig, ax = plt.subplots()
                shap.plots.waterfall(explanation, show=False)
                st.pyplot(fig)

            elif shap_plot_type == "Force Plot":
                st_shap(shap.force_plot(explanation.base_values, explanation.values, features=explanation.data, feature_names=explanation.feature_names))

        except Exception as e:
            st.error(f"Error generating SHAP plot: {e}")
            st.info("This can happen if the background data format doesn't match the model's expectations or due to explainer-specific issues.")

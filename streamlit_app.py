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
    """Helper function to display SHAP plots in Streamlit."""
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)

# --- 1. Load Pretrained Models ---
@st.cache_resource
def load_models():
    """Loads all models from the 'models' directory."""
    models_dir = "models"
    if not os.path.isdir(models_dir):
        st.error(f"Error: The '{models_dir}' directory was not found. Please ensure it is in your repository.")
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
            st.error(f"Error: Model file not found at '{path}'.")
            st.stop()
        except Exception as e:
            st.error(f"Error loading {name} model from {path}: {e}")
            st.stop()
    return loaded_models

# --- 2. Load Background Data for SHAP ---
@st.cache_data
def load_sample_data(path="sample_data.csv"):
    """Loads sample data for SHAP KernelExplainer background."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"Error: Sample data file not found at '{path}'. Please ensure it is in your repository.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading sample data from {path}: {e}")
        st.stop()

# --- Page Setup and Main App Logic ---
st.set_page_config(page_title="Bank Marketing Predictor 🏦", layout="wide", initial_sidebar_state="expanded")

models = load_models()
sample_data = load_sample_data()

# --- FIX 1: Correct feature list to exactly match the trained model ---
# This list is derived from the error messages and likely matches your training script.
model_features = [
    'age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx',
    'cons.conf.idx', 'euribor3m', 'nr.employed', 'job_blue-collar',
    'job_entrepreneur', 'job_housemaid', 'job_management', 'job_retired',
    'job_self-employed', 'job_services', 'job_student', 'job_technician',
    'job_unemployed', 'job_unknown', 'marital_married', 'marital_single',
    'education_basic.6y', 'education_basic.9y', 'education_high.school',
    'education_illiterate', 'education_professional.course',
    'education_university.degree', 'education_unknown', 'default', 'housing', 'loan',
    'contact_telephone', 'month_aug', 'month_dec', 'month_jul', 'month_jun',
    'month_mar', 'month_may', 'month_nov', 'month_oct', 'month_sep',
    'day_of_week_mon', 'day_of_week_thu', 'day_of_week_tue', 'day_of_week_wed',
    'poutcome_nonexistent', 'poutcome_success', 'campaign_log', 'duration_log'
]

# Validate that the sample_data.csv has the correct columns
if not all(feature in sample_data.columns for feature in model_features):
    st.error("Your `sample_data.csv` is missing required columns. Please update it to match the training data.")
    st.info(f"Expected columns: {model_features}")
    st.stop()

X_background = sample_data[model_features].head(100)

st.title("Bank Marketing Campaign Predictor 📈")
st.markdown("This app predicts whether a bank customer will subscribe to a term deposit.")

# --- Sidebar Controls ---
st.sidebar.header("App Controls")
selected_models = st.sidebar.multiselect("Choose Models:", list(models.keys()), default=list(models.keys()))
confidence_threshold = st.sidebar.slider("Confidence Threshold:", 0.0, 1.0, 0.5, 0.01)

st.sidebar.header("Explanation Controls")
shap_plot_type = st.sidebar.selectbox("SHAP Plot Type:", ("Bar Plot", "Waterfall", "Force Plot"))
selected_shap_model_name = st.sidebar.selectbox("Model for Explanation:", list(models.keys()))

# --- FIX 2: Expand UI to include all necessary inputs ---
st.header("Customer Information Input")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age:", 18, 100, 40)
    job_options = ['admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown']
    job = st.selectbox("Job:", job_options)
    duration = st.slider("Duration of last contact (seconds):", 0, 5000, 300)
    campaign = st.number_input("Number of contacts this campaign:", 1, 60, 2)
    pdays = st.number_input("Days since last contacted (999 if never):", 0, 999, 999)
    previous = st.number_input("Contacts before this campaign:", 0, 10, 0)

with col2:
    poutcome = st.selectbox("Previous Outcome:", ['nonexistent', 'failure', 'success'])
    emp_var_rate = st.number_input("Employment Variation Rate:", -4.0, 2.0, 1.1, step=0.1)
    cons_price_idx = st.number_input("Consumer Price Index:", 92.0, 95.0, 93.9, step=0.1)
    cons_conf_idx = st.number_input("Consumer Confidence Index:", -51.0, -26.0, -42.7, step=0.1)
    euribor3m = st.number_input("Euribor 3 Month Rate:", 0.5, 5.5, 4.8, step=0.1)
    nr_employed = st.number_input("Number of Employees:", 4900.0, 5300.0, 5191.0, step=0.1)

with col3:
    # Based on error message, these were label-encoded (not one-hot)
    default_map = {'no': 0, 'yes': 1, 'unknown': 2}
    housing_map = {'no': 0, 'yes': 1, 'unknown': 2}
    loan_map = {'no': 0, 'yes': 1, 'unknown': 2}
    default = st.selectbox("Has credit in default?", list(default_map.keys()))
    housing = st.selectbox("Has housing loan?", list(housing_map.keys()))
    loan = st.selectbox("Has personal loan?", list(loan_map.keys()))
    contact_telephone = not st.checkbox("Contact by Cellular?", True)
    marital = st.selectbox("Marital Status:", ['single', 'married', 'divorced', 'unknown'])


# --- FIX 3: Create the full feature set with correct preprocessing ---
# Start with a dictionary of zeros for all expected features
user_input_dict = {feature: 0 for feature in model_features}

# Update with numerical and label-encoded inputs
user_input_dict.update({
    'age': age, 'campaign': campaign, 'pdays': pdays, 'previous': previous,
    'emp.var.rate': emp_var_rate, 'cons.price.idx': cons_price_idx,
    'cons.conf.idx': cons_conf_idx, 'euribor3m': euribor3m, 'nr.employed': nr_employed,
    'default': default_map[default], 'housing': housing_map[housing], 'loan': loan_map[loan],
    'contact_telephone': 1 if contact_telephone else 0
})

# Handle one-hot encoded features
job_col = f"job_{job.replace('.', '')}" # Clean 'admin.' to 'admin' if necessary
if job_col in user_input_dict:
    user_input_dict[job_col] = 1

marital_col = f"marital_{marital}"
if marital_col in user_input_dict:
    user_input_dict[marital_col] = 1

poutcome_col = f"poutcome_{poutcome}"
if poutcome_col in user_input_dict:
    user_input_dict[poutcome_col] = 1

# Apply log transformations to create the new features the model expects
user_input_dict['duration_log'] = np.log1p(duration)
user_input_dict['campaign_log'] = np.log1p(campaign)

# Create the final DataFrame, ensuring column order is exactly as the model expects
user_data_aligned = pd.DataFrame([user_input_dict])[model_features]

# --- Tabs for Predictions and Explanations ---
tab1, tab2 = st.tabs(["📊 Predictions", "🔍 SHAP Explanation"])

with tab1:
    st.header("Prediction Results")
    st.write("Data sent for prediction (after preprocessing):")
    st.dataframe(user_data_aligned)

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
        
        if not predictions_df.empty:
            st.dataframe(predictions_df, use_container_width=True)
            # ... (rest of your prediction display logic is fine) ...

with tab2:
    st.header(f"SHAP Explanation for {selected_shap_model_name}")
    if selected_shap_model_name:
        model_to_explain = models[selected_shap_model_name]
        try:
            if "Tree" in selected_shap_model_name or "Forest" in selected_shap_model_name:
                explainer = shap.TreeExplainer(model_to_explain)
            else:
                explainer = shap.KernelExplainer(model_to_explain.predict_proba, X_background)

            shap_values = explainer.shap_values(user_data_aligned)
            
            if isinstance(shap_values, list):
                shap_values_for_class_1 = shap_values[1]
                expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
            else:
                shap_values_for_class_1 = shap_values
                expected_value = explainer.expected_value
            
            explanation = shap.Explanation(
                values=shap_values_for_class_1[0], 
                base_values=expected_value, 
                data=user_data_aligned.iloc[0], 
                feature_names=model_features
            )

            if shap_plot_type == "Bar Plot":
                fig, ax = plt.subplots(); shap.plots.bar(explanation, show=False); st.pyplot(fig)
            elif shap_plot_type == "Waterfall":
                fig, ax = plt.subplots(); shap.plots.waterfall(explanation, show=False); st.pyplot(fig)
            elif shap_plot_type == "Force Plot":
                st_shap(shap.force_plot(explanation.base_values, explanation.values, features=explanation.data, feature_names=explanation.feature_names))
        except Exception as e:
            st.error(f"Error generating SHAP plot: {e}")

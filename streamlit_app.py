import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os

# --- SHAP Helper Function ---
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)

# --- Load Models ---
@st.cache_resource
def load_models():
    models_dir = "models"
    if not os.path.isdir(models_dir):
        st.error(f"Missing '{models_dir}' directory.")
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
        except Exception as e:
            st.error(f"Could not load {name}: {e}")
            st.stop()
    return loaded_models

# --- Load Sample Data ---
@st.cache_data
def load_sample_data(path="sample_data.csv"):
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Could not load sample data: {e}")
        st.stop()

# --- Setup ---
st.set_page_config("Bank Marketing Predictor 🏦", layout="wide")
models = load_models()
sample_data = load_sample_data()

model_features = list(sample_data.columns)
X_background = sample_data[model_features].head(100)

st.title("Bank Marketing Campaign Predictor 📈")
st.markdown("""
This app predicts whether a customer will subscribe to a term deposit, and explains model decisions with SHAP.
""")

# --- Sidebar ---
st.sidebar.header("Controls")
selected_models = st.sidebar.multiselect("Models:", list(models.keys()), default=list(models.keys()))
confidence_threshold = st.sidebar.slider("Confidence Threshold:", 0.0, 1.0, 0.5, 0.01)
shap_plot_type = st.sidebar.selectbox("SHAP Plot Type:", ["Bar Plot", "Waterfall", "Force Plot"])
selected_shap_model_name = st.sidebar.selectbox("Model for SHAP:", list(models.keys()))

# --- User Inputs ---
st.header("Customer Information")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 100, 40)
    duration = st.slider("Last contact duration (s)", 0, 5000, 300)
    campaign = st.number_input("This campaign contacts", 1, 60, 2)
    pdays = st.number_input("Days since last contact", 0, 999, 999)
    previous = st.number_input("Previous contacts", 0, 10, 0)

with col2:
    emp_var_rate = st.number_input("Employment variation rate", -4.0, 2.0, 1.1, step=0.1)
    cons_price_idx = st.number_input("Consumer price index", 92.0, 95.0, 93.9, step=0.1)
    cons_conf_idx = st.number_input("Consumer confidence index", -51.0, -26.0, -42.7, step=0.1)
    euribor3m = st.number_input("Euribor 3m", 0.5, 5.5, 4.8, step=0.1)
    nr_employed = st.number_input("Number of employees", 4900.0, 5300.0, 5191.0, step=0.1)

with col3:
    contact_telephone = not st.checkbox("Contacted via Cellular", True)
    poutcome_success = st.checkbox("Previous Outcome: Success", False)
    st.write("Categorical fields like job, marital, etc. default to 0")

user_input = {feature: 0 for feature in model_features}
user_input.update({
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
    'poutcome_success': 1 if poutcome_success else 0,
    'campaign_log': np.log(campaign + 1e-6)
})

user_df = pd.DataFrame([user_input]).reindex(columns=model_features, fill_value=0)

# --- Tabs ---
tab1, tab2 = st.tabs(["📊 Predictions", "🔍 SHAP Explanation"])

with tab1:
    st.subheader("Model Predictions")
    st.dataframe(user_df)

    results = []
    votes, probs = [], []

    for name in selected_models:
        try:
            model = models[name]
            p = model.predict_proba(user_df)[0, 1]
            label = "Subscribed" if p >= confidence_threshold else "Not Subscribed"
            results.append([name, label, f"{p:.2f}"])
            probs.append(p)
            votes.append(label == "Subscribed")
        except Exception as e:
            st.error(f"{name} failed: {e}")

    df_result = pd.DataFrame(results, columns=["Model", "Prediction", "Confidence"])
    st.dataframe(df_result)

    if probs:
        ensemble = "Subscribed" if sum(votes) >= len(votes)/2 else "Not Subscribed"
        avg_conf = np.mean(probs)
        st.markdown(f"**Ensemble Prediction:** {ensemble}")
        st.markdown(f"**Average Confidence:** `{avg_conf:.2f}`")

        fig, ax = plt.subplots()
        sns.barplot(x=selected_models, y=probs, ax=ax)
        ax.set_ylim(0, 1)
        st.pyplot(fig)

with tab2:
    st.subheader("SHAP Explanation")
    try:
        model = models[selected_shap_model_name]
        explainer = (shap.TreeExplainer(model, X_background)
                     if "Tree" in selected_shap_model_name or "Forest" in selected_shap_model_name
                     else shap.KernelExplainer(model.predict_proba, X_background))

        shap_vals = explainer.shap_values(user_df)
        val = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
        base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value

        explanation = shap.Explanation(values=val, base_values=base_val, data=user_df.iloc[0], feature_names=model_features)

        if shap_plot_type == "Bar Plot":
            fig, _ = plt.subplots()
            shap.plots.bar(explanation, show=False)
            st.pyplot(fig)
        elif shap_plot_type == "Waterfall":
            fig, _ = plt.subplots()
            shap.plots.waterfall(explanation, show=False)
            st.pyplot(fig)
        elif shap_plot_type == "Force Plot":
            st_shap(shap.force_plot(base_val, val, user_df.iloc[0]))

    except Exception as e:
        st.error(f"SHAP failed: {e}")

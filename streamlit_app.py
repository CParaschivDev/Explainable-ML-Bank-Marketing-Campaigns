import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import lime
import lime.lime_tabular
import io

# --- Helper Function for SHAP Force Plots ---
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)

# --- 1. Load Pretrained Models ---
@st.cache_resource
def load_models():
    models_dir = "models"
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
        data['campaign_log'] = np.log(data['campaign'] + 1e-6)
        data['pdays_log'] = np.log(data['pdays'] + 1e-6)
        data['previous_log'] = np.log(data['previous'] + 1e-6)
        data['duration_log'] = np.log(data['duration'] + 1e-6)
        if 'y' in data.columns:
            data = data.drop(columns=['y'])

        # Convert boolean columns stored as strings to numeric values
        bool_cols = data.select_dtypes(include=['object']).columns
        if len(bool_cols) > 0:
            data[bool_cols] = data[bool_cols].replace({'True': 1, 'False': 0})

        data = data.apply(pd.to_numeric, errors='coerce')
        data = data.astype('float64')
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

st.sidebar.header("Explanation Controls")
explanation_type = st.sidebar.selectbox("Explanation Type:", ("SHAP", "LIME"))

if explanation_type == "SHAP":
    shap_plot_type = st.sidebar.selectbox(
        "SHAP Plot Type:", ("Bar Plot", "Waterfall", "Force Plot", "Summary Plot")
    )
    selected_shap_model_name = st.sidebar.selectbox("Model for SHAP:", list(models.keys()))
else:  # LIME
    selected_lime_model_name = st.sidebar.selectbox("Model for LIME:", list(models.keys()))

st.sidebar.header("Appearance")
theme_choice = st.sidebar.selectbox("Theme", ["Light", "Dark"])
if theme_choice == "Dark":
    st.markdown(
        """
        <style>
        .stApp {background-color: #0e1117; color: white;}
        </style>
        """,
        unsafe_allow_html=True,
    )

if st.sidebar.button("Load Example Profile"):
    example_values = {
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
    for k, v in example_values.items():
        st.session_state[k] = v

# --- Input Form ---
st.header("Customer Input")
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

# --- Preprocess Input ---
user_input_dict = {feature: 0 for feature in model_features}
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
    'poutcome_success': 1 if poutcome_success else 0,
    'campaign_log': np.log(campaign + 1e-6),
    'pdays_log': np.log(pdays + 1e-6),
    'previous_log': np.log(previous + 1e-6),
    'duration_log': np.log(duration + 1e-6),
    'day_of_week_mon': 1 if day_of_week_mon else 0,
    'education_basic.6y': 1 if education_basic_6y else 0,
    'job_management': 1 if job_management else 0,
    'marital_single': 1 if marital_single else 0
})

user_data = pd.DataFrame([user_input_dict])
user_data_aligned = user_data.reindex(columns=model_features, fill_value=0)
user_data_aligned = user_data_aligned.apply(pd.to_numeric, errors='coerce').astype('float64')

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Predictions", "🔍 Explanation", "📁 Batch Predictions"])

with tab1:
    st.header("Prediction Results")
    if not selected_models:
        st.warning("Select at least one model.")
    else:
        st.dataframe(user_data_aligned)
        predictions_df = pd.DataFrame(columns=["Model", "Prediction", "Confidence"])
        confidences, predictions = [], []

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
            vote = "Subscribed" if sum(predictions) >= len(predictions)/2 else "Not Subscribed"
            st.markdown(f"**Ensemble Vote:** `{vote}`")
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
    if explanation_type == "SHAP":
        st.header(f"SHAP Explanation: {selected_shap_model_name}")
        try:
            model = models[selected_shap_model_name]
            if "Tree" in selected_shap_model_name or "Forest" in selected_shap_model_name:
                explainer = shap.TreeExplainer(model, X_background)
            else:
                # Define a wrapper function for predict_proba to avoid potential binding issues
                # and ensure it returns probabilities for the positive class
                def predict_proba_wrapper(X):
                    return model.predict_proba(X)[:, 1]

                explainer = shap.KernelExplainer(predict_proba_wrapper, X_background)

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
                num_features=len(model_features)
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
            for feature, weight in explanation.as_list():
                st.write(f"- **{feature}**: {weight:.4f}")

        except Exception as e:
            st.error(f"LIME error: {e}")

with tab3:
    st.header("Batch Predictions")
    uploaded_file = st.file_uploader(
        "Upload CSV", type="csv", help="File must include the same feature columns as sample_data.csv"
    )
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        for col in ["campaign", "pdays", "previous", "duration"]:
            if col in batch_df.columns:
                batch_df[f"{col}_log"] = np.log(batch_df[col] + 1e-6)
        bool_cols = batch_df.select_dtypes(include=["object"]).columns
        if len(bool_cols) > 0:
            batch_df[bool_cols] = batch_df[bool_cols].replace({"True": 1, "False": 0})
        batch_df = batch_df.reindex(columns=model_features, fill_value=0)
        batch_df = batch_df.apply(pd.to_numeric, errors="coerce").astype("float64")
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

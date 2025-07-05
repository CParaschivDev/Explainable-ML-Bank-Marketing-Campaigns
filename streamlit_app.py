import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

# --- Load models from 'models/' directory ---
@st.cache_resource
def load_models():
    return {
        "Decision Tree": joblib.load("models/decision_tree_model.pkl"),
        "Random Forest": joblib.load("models/random_forest_model.pkl"),
        "Naive Bayes": joblib.load("models/naive_bayes_model.pkl"),
        "KNN": joblib.load("models/best_knn_model.pkl")
    }

models = load_models()

st.title("📈 Explainable ML - Bank Marketing Prediction")
st.markdown("Predict whether a customer will subscribe to a term deposit using multiple machine learning models.")

# --- User input form ---
with st.form("input_form"):
    st.subheader("📝 Customer Information")
    duration = st.slider("Duration of Last Contact (seconds)", 0, 1200, 350)
    campaign = st.number_input("Number of Contacts in Current Campaign", 1, 50, 1)
    contact_cellular = st.checkbox("Contact Type: Cellular")
    month_may = st.checkbox("Was the last contact in May?")
    poutcome_success = st.checkbox("Previous Outcome was 'Success'?")
    submitted = st.form_submit_button("Predict")

# --- Prepare input for model ---
def prepare_input():
    input_data = pd.DataFrame([{
        "duration": duration,
        "campaign": campaign,
        "contact_cellular": int(contact_cellular),
        "month_may": int(month_may),
        "poutcome_success": int(poutcome_success)
    }])
    # Match column order used in training
    input_data = input_data.reindex(columns=models["Decision Tree"].feature_names_in_, fill_value=0)
    return input_data

# --- Perform Predictions ---
if submitted:
    st.subheader("🔍 Model Predictions")
    input_df = prepare_input()
    results = {}

    for name, model in models.items():
        prediction = int(model.predict(input_df)[0])
        confidence = float(model.predict_proba(input_df)[0][1])
        results[name] = {
            "prediction": prediction,
            "label": "Subscribed" if prediction == 1 else "Not Subscribed",
            "confidence": confidence
        }

    # --- Ensemble majority vote ---
    vote_preds = [r["prediction"] for r in results.values()]
    ensemble_prediction = int(sum(vote_preds) >= 2)
    ensemble_confidence = np.mean([r["confidence"] for r in results.values()])

    # --- Display Ensemble Result ---
    st.markdown(f"**✅ Ensemble Prediction:** {'Subscribed' if ensemble_prediction else 'Not Subscribed'}")
    st.markdown(f"**📊 Average Confidence:** {ensemble_confidence:.2%}")

    # --- Display Table ---
    results_df = pd.DataFrame.from_dict(results, orient="index")
    results_df["confidence %"] = results_df["confidence"].apply(lambda x: f"{x*100:.1f}%")
    st.dataframe(results_df[["label", "confidence %"]].rename(columns={"label": "Prediction"}))

    # --- Plot Bar Chart ---
    st.subheader("📈 Confidence by Model")
    fig, ax = plt.subplots()
    sns.barplot(x=results_df.index, y=results_df["confidence"], ax=ax)
    ax.set_ylabel("Confidence (P = Subscribed)")
    ax.set_ylim(0, 1)
    st.pyplot(fig)
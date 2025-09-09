# Explainable ML for Bank Marketing Campaigns

This project implements and explains **Machine Learning models** for predicting customer subscription to bank term deposits, with explanations powered by **LIME** and **SHAP**.
It provides an **interactive Streamlit dashboard** with built-in model interpretability to help understand which features drive predictions.

SHAP (SHapley Additive exPlanations) values quantify the contribution of each feature to a prediction, offering both global and local insight into model behavior.

LIME (Local Interpretable Model-Agnostic Explanations) approximates the model locally to highlight the features that influence individual predictions.

---

## 🚀 Features
- Trained models:
  - Decision Tree
  - Random Forest
  - Naive Bayes
  - K-Nearest Neighbors (KNN)
- 📊 Model comparison, ensemble vote, and average confidence across selected models
- 🔍 Local and global interpretability with **LIME** and **SHAP** (bar, waterfall, force, and summary plots)
- 🖥️ Interactive **Streamlit dashboard** for predictions and visualizations
- ✍️ Form-based input with dedicated **Predict** button, example profile loader, and input reset
- 🎛️ Confidence threshold slider, input presets, and light/dark/colorblind-friendly themes with custom accent color
- 🎉 Optional animations (balloons or snow) based on predictions
- 📁 Batch prediction via CSV uploads
- 📥 Downloadable predictions, confidence charts, and explanation plots
- 🧪 Data exploration with summary statistics, correlation heatmaps, and feature distributions
- 🛠️ Model training & evaluation with cross-validation, confusion matrices, ROC and precision–recall curves, and basic fairness checks
- 📂 Pre-trained models available in the `models/` directory

---

## 📂 Project Structure

```
├── models/              # Pre-trained ML models (.pkl)
├── sample_data.csv      # Example dataset
├── streamlit_app.py     # Streamlit dashboard entry point
├── requirements.txt     # Dependencies
├── LICENSE              # License file
└── README.md            # Project documentation
```

The sample dataset is available at: https://www.kaggle.com/datasets/yufengsui/portuguese-bank-marketing-data-set

---

## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Explainable-ML-Bank-Marketing-Campaigns.git
   cd Explainable-ML-Bank-Marketing-Campaigns
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## ▶️ Usage
1. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```
2. Open the app in your browser:
   ```text
   http://localhost:8501
   ```
3. Upload your dataset or use the provided `sample_data.csv` to:
    - Generate single or batch predictions
    - Compare model outputs and confidence scores
    - Visualize interpretability with LIME and SHAP
    - Explore datasets via summary statistics, correlation heatmaps, and feature distribution plots
    - Train and evaluate models with cross-validation metrics, confusion matrices, and ROC/PR curves

## 🤝 Contributing
Contributions are welcome!

Feel free to open an issue or submit a pull request to enhance features, improve models, or add datasets.

## 📜 License
This project is licensed under the MIT License.

## 📚 References
- [LIME Documentation](https://marcotcr.github.io/lime/)
- [SHAP Documentation](https://shap.readthedocs.io/)

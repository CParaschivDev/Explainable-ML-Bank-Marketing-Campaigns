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
- 📊 Model comparison with accuracy scores
- 🔍 Local and global interpretability with **LIME** and **SHAP** for feature importance
- 🖥️ Interactive **Streamlit dashboard** for predictions and visualizations
- 📁 Batch prediction via CSV uploads
 - 🎛️ Input presets and light/dark/colorblind-friendly theme toggle
- ✅ Form-based input with dedicated **Predict** button and progress spinners
- 📥 Downloadable prediction and explanation outputs
- 📂 Pre-trained models available in the `models/` directory

---

## 📂 Project Structure

### ├── models/ # Pre-trained ML models (.pkl)

### ├── sample_data.csv # Example dataset. Available at: https://www.kaggle.com/datasets/yufengsui/portuguese-bank-marketing-data-set

### ├── streamlit_app.py # Streamlit dashboard entry point

### ├── requirements.txt # Dependencies

### ├── LICENSE # License file

### └── README.md # Project documentation


---

## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Explainable-ML-Bank-Marketing-Campaigns.git
   cd Explainable-ML-Bank-Marketing-Campaigns
   
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt

## ▶️ Usage

1. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py

2. Open the app in your browser at:
   ```arduino
   http://localhost:8501

3. Upload your dataset or use the provided sample_data.csv to:
   - Train & evaluate models
   - Compare results
   - Visualize interpretability with LIME and SHAP

## 🤝 Contributing
Contributions are welcome!

Feel free to open an issue or submit a pull request to enhance features, improve models, or add datasets.

## 📜 License
This project is licensed under the MIT License.

## 📚 References
- [LIME Documentation](https://marcotcr.github.io/lime/)
- [SHAP Documentation](https://shap.readthedocs.io/)

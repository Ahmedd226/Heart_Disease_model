# 🫀 Heart Disease Risk Predictor (95.5% AUC)

An end-to-end Machine Learning web application that predicts the probability of heart disease using a state-of-the-art ensemble of Gradient Boosting models.

## 🚀 Live Demo
[Check out the Live App here!](https://heart-risk-predictor-22.streamlit.app/)

## 📊 Performance & Model
This project achieved a **Top-Tier Score (95.5%)** by combining three powerful algorithms into a weighted ensemble:
* **XGBoost:** Level-wise tree growth for general patterns.
* **CatBoost:** Optimized handling of categorical features (Chest Pain, Thallium).

### Key Features:
- **Seed Averaging:** Neutralized model variance by averaging predictions across multiple random seeds (42, 2026, 777).
- **Feature Engineering:** Includes custom clinical ratios like `BP_to_Age_Ratio`, `HR_Deficit`, and `Risk_Score`.
- **Deployment:** Hosted on Streamlit Cloud with a Python-based CI/CD pipeline.

## 🛠️ Tech Stack
- **Language:** Python 3.10+
- **Modeling:** XGBoost, CatBoost, LightGBM, Scikit-Learn
- **Interface:** Streamlit
- **API/Server:** Joblib for serialization

## 🏃 How to Run Locally
1. Clone the repo: `git clone https://github.com/YOUR_USERNAME/REPO_NAME.git`
2. Install requirements: `pip install -r requirements.txt`
3. Run the app: `streamlit run app2.py`

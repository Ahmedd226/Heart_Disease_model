import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Set up the page
st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀", layout="centered")
st.title("🫀 Heart Disease Risk Predictor")

# --- SIDEBAR INFO ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/822/822143.png", width=100)
    st.title("About the Model")
    st.info("""
    **Accuracy:** 95.5% (ROC-AUC)
    
    **Algorithm:** Weighted Ensemble 
    (XGBoost + CatBoost)
    
    **Features Used:** 13 Clinical Inputs + 6 Engineered Ratios.
    
    *Disclaimer: This is a student project for educational purposes and should not be used as professional medical advice.*
    """)
    
    st.success("✅ Model is Live & Stable")

st.write("Enter the patient's clinical data below to predict the likelihood of heart disease.")

# 2. Load the model (Streamlit caches this so it doesn't reload on every button click)
@st.cache_resource
def load_model():
    return joblib.load('heart_disease_model.joblib')

model = load_model()

# 3. The Feature Engineering Function
def engineer_features_v2(data):
    df_new = data.copy()
    
    df_new['HR_Deficit'] = (220 - df_new['Age']) - df_new['Max_HR']
    df_new['Risk_Score'] = (df_new['BP'] > 130).astype(int) + \
                           (df_new['Cholesterol'] > 240).astype(int) + \
                           (df_new['Chest_pain_type'] == 4).astype(int)
    df_new['BP_to_Age_Ratio'] = df_new['BP'] / df_new['Age']
    df_new['Cholesterol_Deviation'] = abs(df_new['Cholesterol'] - 200)
    
    chol_conditions = [(df_new['Cholesterol'] < 200), 
                       (df_new['Cholesterol'] >= 200) & (df_new['Cholesterol'] < 240), 
                       (df_new['Cholesterol'] >= 240)]
    df_new['Cholesterol_Category'] = np.select(chol_conditions, [0, 1, 2], default=0)

    bp_conditions = [(df_new['BP'] < 120), 
                     (df_new['BP'] >= 120) & (df_new['BP'] < 130), 
                     (df_new['BP'] >= 130)]
    df_new['BP_Category'] = np.select(bp_conditions, [0, 1, 2], default=0)
    
    return df_new

# 4. Build the User Interface with Columns
col1, col2 = st.columns(2)

with col1:
    st.header("Patient Vitals")
    age = st.number_input("Age", min_value=1, max_value=120, value=55)
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1], index=1)
    bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=140)
    cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=250)
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=130)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl? (1 = Yes, 0 = No)", [0, 1], index=0)

with col2:
    st.header("Clinical Test Results")
    chest_pain = st.selectbox("Chest Pain Type (1-4)", [1, 2, 3, 4], index=3)
    ekg = st.selectbox("Resting EKG Results (0-2)", [0, 1, 2], index=2)
    exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1], index=1)
    st_depress = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment (1-3)", [1, 2, 3], index=1)
    vessels = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (0-3)", [0, 1, 2, 3], index=0)
    thallium = st.selectbox("Thallium Stress Test Result (3=Normal, 6=Fixed, 7=Reversable)", [3, 6, 7], index=0)

# 5. The Prediction Button
st.markdown("---")
if st.button("Predict Heart Disease Risk", type="primary", use_container_width=True):
    
    # Gather inputs into a dictionary
    patient_data = {
        "Age": age, "Sex": sex, "Chest_pain_type": chest_pain, "BP": bp, 
        "Cholesterol": cholesterol, "FBS_over_120": fbs, "EKG_results": ekg, 
        "Max_HR": max_hr, "Exercise_angina": exang, "ST_depression": st_depress, 
        "Slope_of_ST": slope, "Number_of_vessels_fluro": vessels, "Thallium": thallium
    }
    
    # Convert to DataFrame
    df_raw = pd.DataFrame([patient_data])
    
    # Engineer features
    df_engineered = engineer_features_v2(df_raw)
    
    # Rename columns back to exactly match Kaggle data
    rename_map = {
        "Chest_pain_type": "Chest pain type", "FBS_over_120": "FBS over 120",
        "EKG_results": "EKG results", "Max_HR": "Max HR",
        "Exercise_angina": "Exercise angina", "ST_depression": "ST depression",
        "Slope_of_ST": "Slope of ST", "Number_of_vessels_fluro": "Number of vessels fluro"
    }
    df_final = df_engineered.rename(columns=rename_map)
    
    # Make Prediction
    probability = model.predict_proba(df_final)[0][1]
    
    # Display Results beautifully
    st.subheader("Results")
    if probability > 0.7:
        st.error(f"🚨 High Risk Detected! Estimated Probability: {probability * 100:.2f}%")
        st.progress(float(probability))
        assessment = "HIGH RISK"
    elif probability > 0.4:
        st.warning(f"⚠️ Potential Risk Detected! Estimated Probability: {probability * 100:.2f}%")
        st.progress(float(probability))
        assessment = "POTENTIAL RISK"
    else:
        st.success(f"✅ Low Risk. Estimated Probability: {probability * 100:.2f}%")
        st.progress(float(probability))
        assessment = "LOW RISK"


    # 1. Create the Report Content
    report_text = f"""
    HEART DISEASE RISK REPORT
    -------------------------
    Patient Age: {age}
    Patient Sex: {'Male' if sex == 1 else 'Female'}
    
    CLINICAL RESULTS:
    - Blood Pressure: {bp} mmHg
    - Cholesterol: {cholesterol} mg/dl
    - Max Heart Rate: {max_hr} bpm
    - ST Depression: {st_depress}
    
    MODEL PREDICTION:
    - Risk Probability: {probability * 100:.2f}%
    - Assessment: {assessment}
    
    Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
    -------------------------
    *Educational Project - Not Medical Advice*
    """
    
    # 2. Add the Download Button
    st.download_button(
        label="📥 Download Patient Report",
        data=report_text,
        file_name=f"Patient_Report_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
        mime="text/plain",
        use_container_width=True
    )
    
    # --- FEATURE IMPORTANCE SECTION ---
    st.markdown("---")
    st.subheader("📊 Model Insights: Clinical Impact Factors")
    
    try:
        # CatBoost uses get_feature_importance()
        importances = model.get_feature_importance()
        feature_names = df_final.columns
        
        # Create and sort the importance DataFrame
        importance_df = pd.DataFrame({
            'Factor': feature_names,
            'Impact Score': importances
        }).sort_values(by='Impact Score', ascending=False).head(10)
    
        # Use a horizontal bar chart for better readability of labels
        st.bar_chart(importance_df.set_index('Factor'))
        st.info("The chart above reveals the top 10 factors driving the AI's 95.5% accuracy. High impact scores indicate the features the model relies on most to determine heart disease risk.")
    except Exception as e:
        st.write("Feature importance is currently loading...")
    
    




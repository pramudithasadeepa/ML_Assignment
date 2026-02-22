import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

# Page configuration
st.set_page_config(page_title="Sri Lanka House Price Predictor", page_icon="🏠", layout="wide")

# Paths
MODELS_DIR = "models"
PLOTS_DIR = "plots"

@st.cache_resource
def load_assets():
    model = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))
    le_location = joblib.load(os.path.join(MODELS_DIR, "le_location.pkl"))
    le_house_type = joblib.load(os.path.join(MODELS_DIR, "le_house_type.pkl"))
    
    try:
        metrics = joblib.load(os.path.join(MODELS_DIR, "test_metrics.pkl"))
    except FileNotFoundError:
        metrics = {"RMSE": 0, "MAE": 0, "R2": 0}
        
    try:
        explainer = joblib.load(os.path.join(MODELS_DIR, "shap_explainer.pkl"))
    except FileNotFoundError:
        explainer = None
        
    return model, le_location, le_house_type, metrics, explainer

try:
    model, le_location, le_house_type, metrics, explainer = load_assets()
except Exception as e:
    st.error(f"Error loading models. Please ensure the pipeline has been executed. Error: {e}")
    st.stop()

# --- SIDEBAR INPUTS ---
st.sidebar.title("House Features")

# Get classes from encoders
locations = list(le_location.classes_)
house_types = list(le_house_type.classes_)

selected_location = st.sidebar.selectbox("Location", options=locations)
selected_type = st.sidebar.selectbox("House Type", options=house_types)
bedrooms = st.sidebar.slider("Number of Bedrooms", min_value=1, max_value=6, value=3)
bathrooms = st.sidebar.slider("Number of Bathrooms", min_value=1, max_value=4, value=2)

predict_clicked = st.sidebar.button("Predict Price", type="primary")

# --- MAIN PAGE ---
st.title("🏠 Sri Lanka House Price Predictor")
st.markdown("""
Welcome to the House Price Predictor! This tool uses an XGBoost Machine Learning model trained on 
real estate data to estimate house prices in Sri Lanka. It also provides transparency into *why* a 
specific price was predicted using SHAP values.
""")

if predict_clicked:
    st.divider()
    
    # 1. Prediction
    loc_encoded = le_location.transform([selected_location])[0]
    type_encoded = le_house_type.transform([selected_type])[0]
    
    # Feature array matches training order: location_encoded, house_type_encoded, bedrooms, bathrooms
    features = np.array([[loc_encoded, type_encoded, bedrooms, bathrooms]])
    
    prediction_lakhs = model.predict(features)[0]
    
    st.markdown(f"### **Estimated Price: Rs. {prediction_lakhs:,.2f} Lakhs**")
    
    # 2. Model Metrics
    st.subheader("Model Performance on Test Data")
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{metrics['R2']:.3f}", help="Higher is better (max 1.0)")
    col2.metric("MAE", f"{metrics['MAE']:.2f} Lakhs", help="Mean Absolute Error (Lower is better)")
    col3.metric("RMSE", f"{metrics['RMSE']:.2f} Lakhs", help="Root Mean Squared Error (Lower is better)")
    
    st.divider()
    
    # 3. Local Explanation (SHAP Waterfall)
    st.subheader("🔍 Why this price? (Local Explanation)")
    st.markdown("""
    **What is this?** This *SHAP Waterfall Plot* explains exactly how the model arrived at this specific prediction. 
    It starts from the average house price at the bottom and adds or subtracts value based on your specific selections 
    (Location, Bedrooms, etc.) to reach the final estimated price at the top. Red arrows push the price higher, while blue arrows push it lower.
    """)
    
    if explainer is not None:
        # Generate SHAP values for the single instance
        shap_values_local = explainer(features)
        
        # We need to map feature names for the plot to look nice
        feature_names = ['Location', 'House Type', 'Bedrooms', 'Bathrooms']
        shap_values_local.feature_names = feature_names
        
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(shap_values_local[0], show=False)
        st.pyplot(fig)
    else:
        st.warning("SHAP explainer not found. Run src/explain.py first.")
        
    st.divider()
    
    # 4. Global Insights and Metrics Plots
    st.subheader("🌍 Overall Model Insights (Global Explanations)")
    st.markdown("These plots show patterns across all the houses the model has learned from.")
    
    tab1, tab2 = st.tabs(["Feature Importance (SHAP Bar)", "Predicted vs Actual"])
    
    with tab1:
        st.markdown("**Which features matter the most overall?** The longer the bar, the more impactful the feature is across the entire dataset.")
        bar_path = os.path.join(PLOTS_DIR, "shap_bar_importance.png")
        if os.path.exists(bar_path):
            st.image(bar_path)
        else:
            st.warning("SHAP bar chart not found.")
            
    with tab2:
        st.markdown("**How accurate is the model?** This plot shows how closely the model's predictions align with actual prices in our test dataset. Points closer to the red dashed line are more accurate.")
        scatter_path = os.path.join(PLOTS_DIR, "predicted_vs_actual.png")
        if os.path.exists(scatter_path):
            st.image(scatter_path)
        else:
            st.warning("Predicted vs Actual plot not found.")
else:
    st.info("👈 Please select house features from the sidebar and click **Predict Price**.")

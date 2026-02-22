import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
import json
from src.utils import MODELS_DIR, PLOTS_DIR, OUTPUTS_DIR

# Page configuration
st.set_page_config(page_title="Sri Lanka House Price Predictor", layout="wide")

st.title("🏡 Sri Lanka House Price Prediction (XGBoost)")

# Sidebar - Model Info
st.sidebar.title("Model Dashboard")
try:
    with open(os.path.join(OUTPUTS_DIR, "metrics.json"), "r") as f:
        metrics = json.load(f)
    st.sidebar.write("### Model Performance")
    st.sidebar.metric("R2 Score", f"{metrics['R2 Score']:.3f}")
    st.sidebar.metric("MAE", f"Rs {metrics['MAE']:,.0f}")
    st.sidebar.metric("RMSE", f"Rs {metrics['RMSE']:,.0f}")
except FileNotFoundError:
    st.sidebar.warning("Model metrics not found. Run evaluation first.")

# Load models and encoders
@st.cache_resource
def load_assets():
    model = joblib.load(os.path.join(MODELS_DIR, "xgb_model.joblib"))
    encoders = joblib.load(os.path.join(MODELS_DIR, "encoders.joblib"))
    return model, encoders

try:
    model, encoders = load_assets()
except Exception as e:
    st.error(f"Error loading model or encoders. Have you run the training script? Error: {e}")
    st.stop()

# User Input Layout
st.subheader("Enter House Details")
col1, col2, col3 = st.columns(3)

with col1:
    location = st.selectbox("Location", options=encoders['location'].classes_)
    house_model = st.selectbox("House Type (Model)", options=encoders['house_model'].classes_)

with col2:
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=20, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=20, value=2)

with col3:
    # Placeholders as requested by user prompt columns
    house_size = st.number_input("House Size (sqft)", min_value=100, max_value=20000, value=2000)
    land_size = st.number_input("Land Size (perches)", min_value=1, max_value=1000, value=10)
    condition = st.selectbox("Condition", options=encoders['condition'].classes_ if 'condition' in encoders else ["N/A"])

# Prediction Logic
if st.button("Predict Price", type="primary"):
    # Encode inputs
    loc_encoded = encoders['location'].transform([location])[0]
    model_encoded = encoders['house_model'].transform([house_model])[0]
    
    # Prepare feature vector (Core features used in training)
    # Note: If you add house_size/land_size to training, they should be included here
    features = np.array([[bedrooms, bathrooms, loc_encoded, model_encoded]])
    prediction = model.predict(features)[0]
    
    st.success(f"### Predicted Price: Rs {prediction:,.2f}")
    
    # Local Explanation
    st.subheader("🔍 Local Explanation (Why this price?)")
    st.write("The SHAP Waterfall plot shows how each feature contributed to this specific prediction.")
    
    # Create explainer and shap values for this single instance
    # We use a wrapper or the explainer directly
    explainer = shap.TreeExplainer(model)
    shap_v = explainer(features)
    
    # Show waterfall plot for the instance
    fig_local, ax_local = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(shap_v[0], show=False)
    st.pyplot(fig_local)
    
    st.info("**Water Fall Plot interpretation:** Red arrows increase the price, blue arrows decrease the price from the average model prediction.")

# Global Explanations
st.divider()
st.subheader("🌍 Global Model Explanations")
tab1, tab2 = st.tabs(["Feature Impact (SHAP Summary)", "Feature Importance"])

with tab1:
    st.write("This plot shows the overall impact of features across the entire dataset.")
    summary_plot_path = os.path.join(PLOTS_DIR, "shap_summary_plot.png")
    if os.path.exists(summary_plot_path):
        st.image(summary_plot_path, caption="SHAP Summary Plot")
    else:
        st.warning("SHAP summary plot image not found.")
    st.write("**Interpretation:** The further right a feature's dots are, the more it increases the predicted price.")

with tab2:
    st.write("Relative importance of each feature in the XGBoost model.")
    importance_plot_path = os.path.join(PLOTS_DIR, "feature_importance_bar.png")
    if os.path.exists(importance_plot_path):
        st.image(importance_plot_path, caption="Feature Importance Bar Chart")
    else:
        st.warning("Feature importance plot image not found.")
    st.write("**Interpretation:** Higher bars mean the model relies more on that feature for making predictions.")

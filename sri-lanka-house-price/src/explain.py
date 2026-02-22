import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
import matplotlib

def run_shap_analysis(data_dir="data", models_dir="models", plots_dir="plots"):
    print("Loading training data and XGBoost model for SHAP analysis...")
    # Use training data for building the explainer background, 
    # but we can explain the test set or train set. We'll use train set here.
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    
    model_path = os.path.join(models_dir, "xgboost_model.pkl")
    model = joblib.load(model_path)
    
    # Create the explainer
    print("Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_train)
    
    # Save the explainer and shap_values for Streamlit
    joblib.dump(explainer, os.path.join(models_dir, "shap_explainer.pkl"))
    
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. SHAP Beeswarm Summary Plot
    print("Generating SHAP beeswarm summary plot...")
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, show=False)
    plt.title("SHAP Beeswarm Summary Plot")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "shap_summary_beeswarm.png"))
    plt.close()
    
    # 2. SHAP Bar chart (Mean Absolute Value)
    print("Generating SHAP bar chart (mean absolute value)...")
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, show=False)
    plt.title("SHAP Feature Importance (Mean |SHAP Value|)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "shap_bar_importance.png"))
    plt.close()
    
    # 3. SHAP Dependence plot for 'location'
    print("Generating SHAP dependence plot for location...")
    plt.figure(figsize=(8, 6))
    # 'location_encoded' is the feature name we mapped
    if 'location_encoded' in X_train.columns:
        shap.dependence_plot(
            "location_encoded", 
            shap_values.values, 
            X_train, 
            show=False, 
            interaction_index=None
        )
        plt.title("SHAP Dependence Plot: Location")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "shap_dependence_location.png"))
        plt.close()
    else:
        print("Warning: 'location_encoded' column not found.")
        
    # 4. SHAP Dependence plot for 'bedrooms'
    print("Generating SHAP dependence plot for bedrooms...")
    plt.figure(figsize=(8, 6))
    if 'bedrooms' in X_train.columns:
        shap.dependence_plot(
            "bedrooms", 
            shap_values.values, 
            X_train, 
            show=False, 
            interaction_index=None
        )
        plt.title("SHAP Dependence Plot: Bedrooms")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "shap_dependence_bedrooms.png"))
        plt.close()
    else:
        print("Warning: 'bedrooms' column not found.")
        
    print("SHAP analysis completed. Explainer and plots saved.")

if __name__ == "__main__":
    run_shap_analysis()

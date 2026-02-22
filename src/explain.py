import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
from src.utils import logger, MODELS_DIR, PLOTS_DIR, ensure_dirs

def explain_model(train_x_path, model_path):
    logger.info("Loading model and training data for SHAP")
    model = joblib.load(model_path)
    X_train = pd.read_csv(train_x_path)
    
    # SHAP TreeExplainer for XGBoost
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    ensure_dirs()
    
    # 1. SHAP Summary Plot
    # Shows the impact of each feature on the model output. 
    # High values are shown in red, low values in blue.
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_train, show=False)
    plt.title("SHAP Summary Plot: Feature Impact on Price")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "shap_summary_plot.png"))
    plt.close()
    
    # 2. SHAP Dependence Plot for the most important feature
    # We find the most important feature by calculating mean absolute SHAP values
    importance = np.abs(shap_values).mean(0)
    most_important_idx = np.argmax(importance)
    most_important_feature = X_train.columns[most_important_idx]
    
    # Shows how a single feature relates to the model prediction.
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(most_important_feature, shap_values, X_train, show=False)
    plt.title(f"SHAP Dependence Plot for {most_important_feature}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "shap_dependence_plot.png"))
    plt.close()
    
    # 3. Feature Importance Bar Chart
    # Standard feature importance showing relative weight of each feature in the model.
    plt.figure(figsize=(10, 6))
    feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
    feature_importances.sort_values().plot(kind='barh', color='skyblue')
    plt.title("Model Feature Importance")
    plt.xlabel("Relative Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_importance_bar.png"))
    plt.close()
    
    logger.info("Saved SHAP and importance plots to outputs/plots/")

if __name__ == "__main__":
    explain_model("data/X_train.csv", "models/xgb_model.joblib")

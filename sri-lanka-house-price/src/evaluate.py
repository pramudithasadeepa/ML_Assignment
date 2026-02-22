import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(data_dir="data", models_dir="models", plots_dir="plots"):
    print("Loading test data and model...")
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    
    model_path = os.path.join(models_dir, "xgboost_model.pkl")
    model = joblib.load(model_path)
    
    # Predictions
    print("Generating predictions...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Evaluation Metrics (Test Set) ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}")
    print("-------------------------------------")
    
    # Save metrics for Streamlit app
    metrics = {"RMSE": rmse, "MAE": mae, "R2": r2}
    joblib.dump(metrics, os.path.join(models_dir, "test_metrics.pkl"))
    
    # Generate Plots
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Predicted vs Actual
    print("Generating predicted vs actual scatter plot...")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
    
    # Perfect fit line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Fit")
    
    plt.title("Predicted House Price vs Actual House Price")
    plt.xlabel("Actual Price (Lakhs)")
    plt.ylabel("Predicted Price (Lakhs)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "predicted_vs_actual.png"))
    plt.close()
    
    # 2. Residual Histogram
    print("Generating residual histogram...")
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color='green')
    plt.title("Histogram of Prediction Residuals")
    plt.xlabel("Residuals (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "residual_histogram.png"))
    plt.close()
    
    # 3. XGBoost Feature Importance
    print("Generating feature importance bar chart...")
    plt.figure(figsize=(10, 6))
    
    # Get feature importance from XGBoost model
    # (if using pd.DataFrame during training, feature names are maintained)
    feature_importances = pd.Series(model.feature_importances_, index=X_test.columns)
    feature_importances = feature_importances.sort_values(ascending=True)
    
    feature_importances.plot(kind='barh', color='skyblue')
    plt.title("XGBoost Internal Feature Importance")
    plt.xlabel("Relative Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "feature_importance_xgboost.png"))
    plt.close()
    
    print("Evaluation completed. Plots saved to plots/ directory.")

if __name__ == "__main__":
    evaluate_model()

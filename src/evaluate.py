import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.utils import logger, OUTPUTS_DIR, PLOTS_DIR, ensure_dirs

def evaluate_model(test_x_path, test_y_path, model_path):
    logger.info("Loading model and test data")
    model = joblib.load(model_path)
    X_test = pd.read_csv(test_x_path)
    y_test = pd.read_csv(test_y_path).values.ravel()
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    metrics = {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2 Score": float(r2)
    }
    
    logger.info(f"Evaluation Metrics: {metrics}")
    
    # Save metrics
    ensure_dirs()
    with open(os.path.join(OUTPUTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(OUTPUTS_DIR, "metrics_table.csv"), index=False)
    
    # Plots
    logger.info("Generating plots")
    
    # Predicted vs Actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title("Predicted House Price vs Actual House Price")
    plt.xlabel("Actual House Price (Rs)")
    plt.ylabel("Predicted House Price (Rs)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "predicted_vs_actual.png"))
    plt.close()
    
    # Residual Histogram
    residuals = y_test - predictions
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color='green')
    plt.title("Histogram of Prediction Residuals")
    plt.xlabel("Residual (Error)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "residual_histogram.png"))
    plt.close()
    
    logger.info("Saved metrics and plots to outputs/")

if __name__ == "__main__":
    evaluate_model(
        "data/X_test.csv", "data/y_test.csv",
        "models/xgb_model.joblib"
    )

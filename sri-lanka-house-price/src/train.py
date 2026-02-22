import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import joblib
import os

def train_model(data_dir="data", models_dir="models"):
    print("Loading training and validation data...")
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    X_val = pd.read_csv(os.path.join(data_dir, "X_val.csv"))
    y_val = pd.read_csv(os.path.join(data_dir, "y_val.csv")).values.ravel()
    
    # Define XGBoost Regressor
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    
    # Hyperparameter Grid
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0]
    }
    
    print("Starting RandomizedSearchCV for hyperparameter tuning...")
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=15, # Try 15 random combinations
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    # Fit RandomizedSearchCV (no early stopping here to allow standard CV behavior)
    random_search.fit(X_train, y_train)
    
    best_params = random_search.best_params_
    print(f"\nBest hyperparameters found:\n{best_params}")
    
    # Train final model with early stopping on validation set
    print("\nTraining final model with early stopping on validation set...")
    final_model = xgb.XGBRegressor(
        **best_params,
        objective='reg:squarederror',
        random_state=42,
        early_stopping_rounds=50 # Setup early stopping here
    )
    
    # Note: starting with xgboost 2.0+, early_stopping_rounds is passed to fit via kwargs
    # or specified in init and fit called with eval_set
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    print(f"Best iteration: {final_model.best_iteration}")
    
    # Save the model
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "xgboost_model.pkl")
    joblib.dump(final_model, model_path)
    
    print(f"\nModel saved to {model_path}")
    return final_model

if __name__ == "__main__":
    train_model()

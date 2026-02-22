import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import joblib
import os
from src.utils import logger, MODELS_DIR, ensure_dirs

def train_model(train_x_path, train_y_path, val_x_path, val_y_path):
    logger.info("Loading training and validation data")
    X_train = pd.read_csv(train_x_path)
    y_train = pd.read_csv(train_y_path).values.ravel()
    X_val = pd.read_csv(val_x_path)
    y_val = pd.read_csv(val_y_path).values.ravel()
    
    # Define XGBoost Regressor
    # We use RandomizedSearchCV because it's more efficient than GridSearchCV 
    # for high-dimensional hyperparameter spaces, providing a good trade-off 
    # between computation time and performance.
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    
    param_grid = {
        'n_estimators': [100, 500, 1000],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2]
    }
    
    logger.info("Starting hyperparameter tuning with RandomizedSearchCV")
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=10,
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    
    best_params = random_search.best_params_
    logger.info(f"Best Hyperparameters: {best_params}")
    
    # Final training with best parameters and early stopping
    logger.info("Training final model with early stopping")
    final_model = xgb.XGBRegressor(
        **best_params,
        objective='reg:squarederror',
        random_state=42
    )
    
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    # Save the model
    ensure_dirs()
    model_path = os.path.join(MODELS_DIR, "xgb_model.json")
    final_model.save_model(model_path)
    # Also save with joblib for easier loading in some contexts
    joblib.dump(final_model, os.path.join(MODELS_DIR, "xgb_model.joblib"))
    
    logger.info(f"Model saved to {model_path}")
    print(f"Final chosen hyperparameters: {best_params}")
    
    return final_model

if __name__ == "__main__":
    train_model(
        "data/X_train.csv", "data/y_train.csv",
        "data/X_val.csv", "data/y_val.csv"
    )

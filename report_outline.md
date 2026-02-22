# Project Report: Sri Lanka House Price Prediction

## 1. Introduction
- Title: Sri Lanka house Price Prediction
- Objective: To build a regression model that predicts house prices based on various features.

## 2. Dataset Description
- Data Source: Scraped from ikman.lk
- Features: Location, Bedrooms, Bathrooms, House Type (extracted from title).
- Target Variable: Price (numeric).

## 3. Methodology
### 3.1 Data Preprocessing
- Cleaning currency formats (Rs, Lakh, Mn).
- Extracting house categories from titles using Regex.
- Handling missing data and outliers.
- Encoding categorical features for the model.

### 3.2 Model Training
- Algorithm: XGBoost Regressor.
- Tuning: RandomizedSearchCV for optimized hyperparameters.
- Early Stopping: Used to prevent overfitting.

## 4. Evaluation Results
- Metrics: RMSE, MAE, R2 Score.
- Visualizations: Predicted vs Actual Scatter Plot, Residual Histogram.

## 5. Model Explainability
- SHAP (SHapley Additive exPlanations):
  - Summary Plot: Global feature contribution.
  - Dependence Plot: Single feature relationship.
  - Waterfall Plot: Individual prediction breakdown.

## 6. Conclusion
- Summary of model performance.
- Future improvements (e.g., adding more features like house size, land size).

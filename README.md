# 🏡 Sri Lanka House Price Prediction

Welcome to the end-to-end Machine Learning pipeline for predicting house prices in Sri Lanka! This complete project was built from scratch and includes data scraping, preprocessing, XGBoost model training, SHAP explainability, and a fully interactive Streamlit web application.

## 🚀 Features
- **Ethical Web Scraping:** Uses `requests` and `BeautifulSoup` to scrape real estate data from ikman.lk.
- **Robust Preprocessing:** Handles complex currency conversions, extracts features via Regex, and encodes categorical variables.
- **XGBoost Regressor:** Utilizes advanced gradient boosting with hyperparameter tuning via `RandomizedSearchCV` and early stopping.
- **Explainable AI (XAI):** Uses `SHAP` (SHapley Additive exPlanations) to provide Global and Local explanations for predictions.
- **Interactive UI:** A highly polished `Streamlit` application for real-time predictions and visualizations.

## 📂 Project Structure
There are two main components of this repository. The final comprehensive assignment is located in the `sri-lanka-house-price/` directory.

```text
sri-lanka-house-price/
├── scrape.py           # Web scraper (with synthetic fallback)
├── src/
│   ├── preprocess.py   # Cleans data and splits into Train/Val/Test
│   ├── train.py        # Trains the XGBoost model
│   ├── evaluate.py     # Calculates metrics and generates plots
│   └── explain.py      # Generates SHAP explainability plots
├── app/
│   └── streamlit_app.py # The Streamlit front-end
├── data/               # Raw and processed CSV datasets
├── models/             # Saved XGBoost models and encoders
├── plots/              # All generated charts and visual explanations
├── requirements.txt    # Project dependencies
└── report.md           # Comprehensive academic assignment report
```

## 🛠️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/pramudithasadeepa/ML_Assignment.git
   cd ML_Assignment/sri-lanka-house-price
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ How to Run the Pipeline
To execute the pipeline from scratch and generate new models/plots, run the scripts in this exact order from the `sri-lanka-house-price/` directory:

1. **Scrape Data:** `python scrape.py`
2. **Preprocess:** `python src/preprocess.py`
3. **Train Model:** `python src/train.py`
4. **Evaluate:** `python src/evaluate.py`
5. **Explainability:** `python src/explain.py`
6. **Launch Web App:** `streamlit run app/streamlit_app.py`

## 📊 Documentation
A fully detailed academic report covering the problem definition, algorithm selection, evaluation results, and critical limitations can be found in `sri-lanka-house-price/report.md`.

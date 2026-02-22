# Sri Lanka Houses Price Prediction using XGBoost

## Project Overview
This project aims to predict house prices in Sri Lanka using web-scraped data from ikman.lk. It uses the XGBoost regression algorithm and provides model explainability using SHAP. A Streamlit web application is included for interactive predictions.

## Folder Structure
```
scrape.py           # Web scraping script
src/
  preprocess.py     # Data cleaning and feature engineering
  train.py          # Model training and hyperparameter tuning
  evaluate.py       # Model evaluation and metrics
  explain.py        # Model explainability using SHAP
  utils.py          # Logging and directory utilities
models/             # Saved models and encoders
outputs/            # Metrics and plots
  plots/            # Generated visualizations
app/
  streamlit_app.py  # Streamlit frontend app
requirements.txt    # Project dependencies
README.md           # Instructions
report_outline.md   # Report structure
```

## Installation Steps
1. Clone or download this project.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run Commands
Follow these steps in order to run the project end-to-end:

1. **Scrape Data (Optional if you have houses_for_sale.csv):**
   ```bash
   python scrape.py
   ```

2. **Preprocess Data:**
   ```bash
   python src/preprocess.py
   ```

3. **Train Model:**
   ```bash
   python src/train.py
   ```

4. **Evaluate Model:**
   ```bash
   python src/evaluate.py
   ```

5. **Generate Explanations:**
   ```bash
   python src/explain.py
   ```

6. **Run Streamlit App:**
   ```bash
   streamlit run app/streamlit_app.py
   ```

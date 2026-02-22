import pandas as pd
import numpy as np
import re
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.utils import logger, MODELS_DIR, ensure_dirs

def clean_currency(price_str):
    if pd.isna(price_str) or price_str == "N/A":
        return np.nan
    
    # Remove "Rs", commas, and whitespace
    clean_str = re.sub(r'[Rs, \s]', '', str(price_str), flags=re.IGNORECASE)
    
    try:
        if 'Lakh' in str(price_str):
            val = float(re.sub(r'[^\d.]', '', clean_str))
            return val * 100000
        elif 'Mn' in str(price_str):
            val = float(re.sub(r'[^\d.]', '', clean_str))
            return val * 1000000
        else:
            return float(clean_str)
    except ValueError:
        return np.nan

def extract_model_from_title(title):
    if pd.isna(title):
        return "Standard"
    
    title = title.lower()
    if "luxury" in title or "luxurious" in title:
        return "Luxury"
    elif "brand new" in title:
        return "Brand New"
    elif "story" in title or "storey" in title:
        match = re.search(r'(\d+)\s*(story|storey)', title)
        if match:
            return f"{match.group(1)} Story"
        return "Multi-Story"
    elif "villa" in title:
        return "Villa"
    elif "bungalow" in title:
        return "Bungalow"
    return "Standard"

def preprocess_data(input_path, output_dir="data"):
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    
    # Target detection
    target_col = 'price'
    if target_col not in df.columns:
        # Simple detection: find column with 'price' in name
        for col in df.columns:
            if 'price' in col.lower():
                logger.info(f"Automatically detected target column: {col}")
                target_col = col
                break
    
    if target_col not in df.columns:
        logger.error("Target column 'price' not found and could not be detected.")
        return

    # Cleaning price
    df['price_numeric'] = df[target_col].apply(clean_currency)
    df = df.dropna(subset=['price_numeric'])
    
    # Feature Engineering
    df['house_model'] = df['title'].apply(extract_model_from_title)
    
    # Handling numeric features (bedrooms, bathrooms)
    # Convert to numeric, handle errors
    for col in ['bedrooms', 'bathrooms']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill NAs for bedrooms/bathrooms with median
    df['bedrooms'] = df['bedrooms'].fillna(df['bedrooms'].median())
    df['bathrooms'] = df['bathrooms'].fillna(df['bathrooms'].median())
    
    # House size / Land size might not be present in all datasets
    # But we'll try to find them if they exist
    potential_cols = ['house size', 'Land size', 'condition']
    for p_col in potential_cols:
        if p_col not in df.columns:
            df[p_col] = "N/A" # Placeholder if missing
            logger.warning(f"Column '{p_col}' not found, using 'N/A' as placeholder.")

    # Encoding
    categorical_cols = ['location', 'house_model', 'condition']
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    # Save encoders
    ensure_dirs()
    joblib.dump(encoders, os.path.join(MODELS_DIR, "encoders.joblib"))
    logger.info("Saved LabelEncoders to models/encoders.joblib")
    
    # Prepare features and target
    features = ['bedrooms', 'bathrooms', 'location', 'house_model'] # Core features
    # Add optional features if they were actually in the dataset (not N/A)
    # For this assignment, we use what was requested
    X = df[features]
    y = df['price_numeric']
    
    # Train/Validation/Test split
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15/0.85, random_state=42)
    
    logger.info(f"Split sizes - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Save split datasets
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_val.to_csv(os.path.join(output_dir, "X_val.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_val.to_csv(os.path.join(output_dir, "y_val.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    
    logger.info("Saved processed datasets to data/")

if __name__ == "__main__":
    preprocess_data("houses_for_sale.csv")

import os
import re
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def clean_price(price_str):
    """Normalize messy price strings into pure floats in Lakhs."""
    if pd.isna(price_str) or not isinstance(price_str, str):
        return np.nan
        
    # Remove "Rs", commas, spaces
    clean_str = re.sub(r'[Rs, \s]', '', price_str, flags=re.IGNORECASE)
    
    try:
        # Check for Lakh
        if 'lakh' in price_str.lower():
            val = float(re.sub(r'[^\d.]', '', clean_str))
            return val
        # Check for Mn
        elif 'mn' in price_str.lower():
            val = float(re.sub(r'[^\d.]', '', clean_str))
            return val * 10
        else:
            # Assume it's a raw number in rupees, convert to Lakhs
            val = float(clean_str)
            return val / 100000
    except ValueError:
        return np.nan

def extract_house_type(title):
    """Extract house type from title using Regex."""
    if pd.isna(title):
        return "Standard"
        
    title = title.lower()
    if re.search(r'luxury|luxurious', title):
        return "Luxury"
    elif re.search(r'brand new', title):
        return "Brand New"
    elif re.search(r'story|storey', title):
        return "Multi-Story"
    elif re.search(r'villa', title):
        return "Villa"
    elif re.search(r'modern', title):
        return "Modern"
    return "Standard"

def preprocess_data(input_csv="data/houses_raw.csv", output_dir="data", models_dir="models"):
    print(f"Loading data from {input_csv}...")
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found. Run scrape.py first.")
        return
        
    df = pd.read_csv(input_csv)
    
    # 1. Normalize Price
    df['price_lkh'] = df['price'].apply(clean_price)
    
    # Drop rows with missing or invalid prices
    initial_len = len(df)
    df.dropna(subset=['price_lkh'], inplace=True)
    df = df[df['price_lkh'] > 0]
    print(f"Dropped {initial_len - len(df)} rows with invalid prices.")
    
    # 2. Extract House Type
    df['house_type'] = df['title'].apply(extract_house_type)
    
    # 3. Ensure numerical columns
    df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce').fillna(3)
    df['bathrooms'] = pd.to_numeric(df['bathrooms'], errors='coerce').fillna(2)
    
    # Keep only required features
    features = ['location', 'house_type', 'bedrooms', 'bathrooms', 'price_lkh']
    df = df[features]
    df.dropna(inplace=True) # Drop any remaining NaNs
    
    # 4. Encode Categorical Columns
    os.makedirs(models_dir, exist_ok=True)
    
    le_location = LabelEncoder()
    df['location_encoded'] = le_location.fit_transform(df['location'])
    
    le_house_type = LabelEncoder()
    df['house_type_encoded'] = le_house_type.fit_transform(df['house_type'])
    
    # Save Encoders
    joblib.dump(le_location, os.path.join(models_dir, 'le_location.pkl'))
    joblib.dump(le_house_type, os.path.join(models_dir, 'le_house_type.pkl'))
    print("Saved LabelEncoders to models/")
    
    # Prepare X and y
    X = df[['location_encoded', 'house_type_encoded', 'bedrooms', 'bathrooms']]
    y = df['price_lkh']
    
    # 5. Train/Val/Test Split (70/15/15)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15/0.85, random_state=42)
    
    # Save splits
    os.makedirs(output_dir, exist_ok=True)
    
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_val.to_csv(os.path.join(output_dir, "X_val.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_val.to_csv(os.path.join(output_dir, "y_val.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    
    print(f"Preprocessing complete. Splits saved to data/ directory.")
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

if __name__ == "__main__":
    preprocess_data()

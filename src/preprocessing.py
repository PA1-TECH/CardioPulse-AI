import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]
TARGET_COLUMN = 'target'

def load_data(data_path="dataset/heart_clean.csv"):
    if not os.path.isabs(data_path):
        data_path = os.path.abspath(data_path)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Sanity checks
    missing = df.isnull().sum().sum()
    if missing > 0:
        df = df.fillna(df.median())
        
    return df

def prepare_data(data_path="dataset/heart_clean.csv", test_size=0.2, val_size=0.15, random_state=42):
    """
    Loads dataset, splits into Train / Validation / Test sets, applies StandardScaler,
    and returns scaled numpy arrays, original DataFrames, and fitted scaler.
    """
    df = load_data(data_path)
    
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    
    # First split: Train+Val vs Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: Train vs Validation
    val_relative_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative_size, random_state=random_state, stratify=y_train_val
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    data_dict = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_val_scaled': X_val_scaled,
        'X_test_scaled': X_test_scaled,
        'scaler': scaler,
        'feature_names': FEATURE_COLUMNS
    }
    return data_dict

def save_preprocessor(scaler, output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)
    scaler_path = os.path.join(output_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    
    meta_path = os.path.join(output_dir, "feature_names.json")
    with open(meta_path, "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)
    print(f"Saved scaler to {scaler_path} and feature names to {meta_path}")

def load_preprocessor(models_dir="models"):
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    meta_path = os.path.join(models_dir, "feature_names.json")
    
    if not os.path.exists(scaler_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Preprocessor artifacts missing from models directory.")
        
    scaler = joblib.load(scaler_path)
    with open(meta_path, "r") as f:
        features = json.load(f)
    return scaler, features

if __name__ == "__main__":
    data = prepare_data()
    print("Train shape:", data['X_train_scaled'].shape)
    print("Val shape:", data['X_val_scaled'].shape)
    print("Test shape:", data['X_test_scaled'].shape)
    save_preprocessor(data['scaler'])

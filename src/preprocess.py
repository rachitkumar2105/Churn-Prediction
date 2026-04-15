import pandas as pd
import numpy as np
import joblib

def preprocess_input(data: dict):
    df = pd.DataFrame([data])
    
    # 1. Handle missing / incorrect types
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
        
    # Feature Engineering
    if 'tenure' in df.columns and 'TotalCharges' in df.columns:
        df["AvgCharges"] = df["TotalCharges"] / (df["tenure"] + 1)
        
    if 'tenure' in df.columns:
        df["TenureGroup"] = pd.cut(
            df["tenure"],
            bins=[-1, 12, 24, 48, 60, 1000],
            labels=[0, 1, 2, 3, 4]
        ).astype(int)
        
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    
    # Load feature names and scaling
    try:
        feature_names = joblib.load('model/feature_names.pkl')
        scaler = joblib.load('model/scaler.pkl')
        numerical_cols = joblib.load('model/numerical_cols.pkl')
    except Exception as e:
        print(f"Error loading preprocessors: {e}")
        return None

    # Reindex to ensure all columns exist
    df = df.reindex(columns=feature_names, fill_value=0)
    
    # Scale
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    return df
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

def load_and_preprocess_data():
    df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # 1. Handle missing values and types
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0) # or mean
    
    df.drop('customerID', axis=1, inplace=True)
    
    # Feature Engineering
    df["AvgCharges"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 60, 1000],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)
    
    # Separate target
    X = df.drop('Churn', axis=1)
    y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['number']).columns
    
    # Encoding & Scaling
    # Using simple one-hot encoding for categorical or label encoding
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Save the feature names
    feature_names = X.columns.tolist()
    
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Ensure model dir
    os.makedirs('model', exist_ok=True)
    
    # Save preprocessors
    joblib.dump(scaler, 'model/scaler.pkl')
    joblib.dump(numerical_cols, 'model/numerical_cols.pkl')
    joblib.dump(feature_names, 'model/feature_names.pkl')
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y), feature_names

def train_xgboost(X_train, y_train, X_test, y_test):
    print("Training XGBoost...")
    # Initialize basic model
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    
    # Simple randomized search
    params = {
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    search = RandomizedSearchCV(xgb_model, param_distributions=params, n_iter=10, scoring='recall', cv=3, random_state=42, verbose=1)
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    
    # Evaluate
    preds = best_model.predict(X_test)
    probs = best_model.predict_proba(X_test)[:, 1]
    
    print("\n[XGBoost Performance]")
    print(classification_report(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, probs))
    
    joblib.dump(best_model, 'model/xgboost_churn_model.pkl')
    return best_model

def train_nn(X_train, y_train, X_test, y_test):
    print("\nTraining Neural Network...")
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Recall(name='recall')])
    
    callbacks = [
        EarlyStopping(monitor='val_recall', mode='max', patience=10, restore_best_weights=True),
        ModelCheckpoint('model/nn_churn_model.h5', save_best_only=True, monitor='val_recall', mode='max')
    ]
    
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=callbacks, verbose=1)
    
    # Evaluate
    probs = model.predict(X_test).flatten()
    preds = (probs > 0.5).astype(int)
    
    print("\n[Neural Network Performance]")
    print(classification_report(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, probs))
    
    return model

if __name__ == "__main__":
    (X_train, X_test, y_train, y_test), feature_names = load_and_preprocess_data()
    train_xgboost(X_train, y_train, X_test, y_test)
    train_nn(X_train, y_train, X_test, y_test)
    print("Training complete. Models are saved in 'model/' directory.")

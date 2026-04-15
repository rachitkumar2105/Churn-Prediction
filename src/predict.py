import joblib
import numpy as np
import shap
import json
import base64
import matplotlib
import io
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.preprocess import preprocess_input
from src.db import log_prediction

# Load models lazily to avoid startup delay if only one is requested, or at module level
xgb_model = None
nn_model = None
explainer = None

def get_xgb():
    global xgb_model, explainer
    if xgb_model is None:
        xgb_model = joblib.load("model/xgboost_churn_model.pkl")
        try:
            # Create explainer for XGBoost model
            explainer = shap.TreeExplainer(xgb_model)
        except Exception as e:
            print("SHAP explainer init failed:", e)
    return xgb_model

def get_nn():
    global nn_model
    if nn_model is None:
        from tensorflow.keras.models import load_model
        nn_model = load_model("model/nn_churn_model.h5", compile=False)
    return nn_model

def categorize_risk(probability: float):
    if probability < 0.3:
        return "Low"
    elif probability < 0.7:
        return "Medium"
    else:
        return "High"

def get_retention_recommendation(risk: str):
    if risk == "High":
        return "Offer a 20% discount on 1-year contract; proactive outreach by retention team."
    elif risk == "Medium":
        return "Send automated email with a smaller perk or bundle upgrade offer."
    else:
        return "No immediate action required, maintain regular engagement."

def predict_churn(data: dict, model_type: str = "xgboost", explain: bool = False):
    processed = preprocess_input(data)
    if processed is None:
        raise ValueError("Error preprocessing input.")
        
    probability = 0.0
    
    if model_type == "xgboost":
        model = get_xgb()
        probability = float(model.predict_proba(processed)[0, 1])
    elif model_type == "nn":
        model = get_nn()
        probability = float(model.predict(processed)[0][0])
    else:
        raise ValueError("Invalid model type. Choose 'xgboost' or 'nn'.")
        
    risk_category = categorize_risk(probability)
    recommendation = get_retention_recommendation(risk_category)
    prediction = 1 if probability > 0.5 else 0
    
    # Log prediction to DB
    log_prediction(data, model_type, prediction, probability, risk_category)
    
    result = {
        "churn_prediction": int(prediction),
        "churn_probability": probability,
        "risk_category": risk_category,
        "recommendation": recommendation,
        "model_used": model_type
    }
    
    if explain and model_type == "xgboost" and explainer is not None:
        shap_values = explainer(processed)
        # Create a waterfall plot and return as base64 string
        fig = plt.figure()
        shap.plots.waterfall(shap_values[0], show=False)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        result["explanation_image_base64"] = img_str
        
        # Approximate feature impacts
        feature_names = processed.columns
        importances = shap_values.values[0]
        impacts = [{"feature": str(f), "impact": float(i)} for f, i in zip(feature_names, importances)]
        # Sort by absolute impact
        impacts.sort(key=lambda x: abs(x['impact']), reverse=True)
        result["top_factors"] = impacts[:5]

    return result

def bulk_predict(data_list: list, model_type: str = "xgboost"):
    # This could be batched, but for simplicity we do it via list comp
    results = [predict_churn(d, model_type, explain=False) for d in data_list]
    return results
from src.predict import predict_churn

# The pipeline automatically handles feature engineering and scaling!
# Pass the raw values exactly as you would see them in the dataset.
sample_input = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 400.0
}

# You can choose 'xgboost' or 'nn', and turn on 'explain=True' to get the SHAP visual data
result = predict_churn(sample_input, model_type="xgboost", explain=True)
print("\n--- Churn Prediction Result ---")
print(f"Risk Category: {result['risk_category']}")
print(f"Churn Probability: {round(result['churn_probability'] * 100, 2)}%")
print(f"Recommendation: {result['recommendation']}")
print("\nTop Contributing Factors:")
for factor in result.get('top_factors', []):
    print(f" - {factor['feature']}: {round(factor['impact'], 3)}")

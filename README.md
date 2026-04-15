---
title: Churn Prediction API
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# IntelliChurn AI - Intelligent Customer Churn Prediction Platform

IntelliChurn AI is a complete, end-to-end production-grade system that leverages machine learning (XGBoost), deep learning (TensorFlow), and Explainable AI (SHAP) to predict customer churn and provide actionable business insights.

## Features
- **Data Pipeline:** End-to-end preprocessing, feature engineering, and scaling.
- **Multi-Model Inference:** Seamlessly switch between XGBoost and Neural Networks.
- **Explainable AI (XAI):** Integrated SHAP explanations (visual waterfall plots and feature impacts) to build trust.
- **REST API Backend:** Built with FastAPI for high performance, supporting single and bulk predictions.
- **React UI:** A highly premium, dynamic glassmorphism dark-themed UI.
- **Database Logging:** SQLite integrated for tracking API telemetry and predictions over time.

## Quickstart

### 1. Backend Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run the backend API (Runs on http://localhost:8000)
python api/app.py
```

### 2. Frontend Setup

```bash
cd frontend
# Install Node dependencies
npm install

# Run the frontend dev server
npm run dev
```

## Architecture
- `src/train.py` handles model training and artifact generation.
- `src/preprocess.py` correctly aligns and scales inputs using the pre-fitted components.
- `src/predict.py` manages inference, SHAP evaluation, and SQLite logging.
- `api/app.py` exposes REST APIs `/predict`, `/bulk_predict`, and `/analytics`.
- `frontend/` contains the React dashboard using Vite.

## Deployment Options
- **Backend (Render / AWS):** The backend is structured to be instantly deployed as a container or native python app. Point your host setup to `api.app:app` via `uvicorn`.
- **Frontend (Vercel / Netlify):** Simply point Vercel to `frontend/` and it will automatically build (`npm run build`) and deploy the React application.

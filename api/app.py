from fastapi import FastAPI, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import pandas as pd
import io

from src.predict import predict_churn, bulk_predict
from src.db import get_analytics

app = FastAPI(title="Intelligent Customer Churn API", description="Predicts customer churn using ML and Deep Learning")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChurnRequest(BaseModel):
    customer_data: dict
    model_type: Optional[str] = "xgboost"
    explain: Optional[bool] = False

@app.get("/")
def home():
    return {"message": "Intelligent Customer Churn API is running. Check /docs for endpoints."}

@app.post("/predict")
def predict_single(req: ChurnRequest):
    try:
        result = predict_churn(req.customer_data, model_type=req.model_type, explain=req.explain)
        return {"status": "success", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/bulk_predict")
async def predict_bulk(file: UploadFile = File(...), model_type: str = Form("xgboost")):
    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        records = df.to_dict(orient='records')
        
        results = bulk_predict(records, model_type=model_type)
        
        # Merge results with input identifiers if possible, for now just return results
        # Attach prediction to the df and return as json
        df['churn_prediction'] = [r['churn_prediction'] for r in results]
        df['churn_probability'] = [r['churn_probability'] for r in results]
        df['risk_category'] = [r['risk_category'] for r in results]
        
        return {"status": "success", "data": df.to_dict(orient='records')}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/analytics")
def analytics():
    try:
        stats = get_analytics()
        return {"status": "success", "data": stats}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
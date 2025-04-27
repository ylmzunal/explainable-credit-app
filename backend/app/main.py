# backend/app/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from app.predict import predict_credit
import numpy as np
from app.explain import shap_explain

app = FastAPI(
    title="Credit Approval Prediction API",
    description="API for predicting credit approvals based on 23 features.",
    version="1.0.0"
)

class PredictRequest(BaseModel):
    """
    Defines the JSON schema for /predict:
      { "features": [f1, f2, ..., f23] }
    
    All 23 features must be provided in the correct order expected by the model.
    Example: [0.1, -0.2, 0.3, ..., 0.5]
    """
    features: list[float]
    
    @validator('features')
    def validate_feature_count(cls, v):
        if len(v) != 23:
            raise ValueError(f"Expected 23 features, but got {len(v)}. Please provide all required features.")
        return v

@app.post("/predict", 
    summary="Predict credit approval",
    description="Makes a prediction based on 23 input features and returns the class prediction (0 or 1) and its probability."
)
async def predict_endpoint(req: PredictRequest):
    """
    Receives a PredictRequest, calls predict_credit, and returns JSON:
      {
        "prediction": <0 or 1>,
        "probability": <float between 0 and 1>
      }
    """
    arr = np.array(req.features, dtype=float).reshape(1, -1)
    class_pred, proba = predict_credit(arr)
    return {
        "prediction": class_pred,
        "probability": proba
    }

@app.get("/", 
    summary="API Info",
    description="Returns information about the API"
)
async def root():
    return {
        "message": "Welcome to the Credit Approval Prediction API",
        "instructions": "Send a POST request to /predict with 23 features to get a prediction",
        "documentation": "Visit /docs for interactive API documentation"
    }
    
@app.post("/explain/shap")
async def shap_endpoint(req: PredictRequest):
    """
    Receives a PredictRequest, computes SHAP values, and returns a list:
      [
        {"feature_index": 0, "shap_value": -0.25},
        {"feature_index": 1, "shap_value": 0.13},
        ...
      ]
    """
    shap_result = shap_explain(req.features)
    return {"explanation": shap_result}
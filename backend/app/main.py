# backend/app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from app.predict import predict_credit
from app.explain import shap_explain

app = FastAPI()

class PredictRequest(BaseModel):
    """
    Defines the JSON schema for /predict:
      { "features": [f1, f2, ..., fN] }
    """
    features: list[float]

@app.post("/predict")
async def predict_endpoint(req: PredictRequest):
    """
    Receives a PredictRequest, calls predict_credit, and returns JSON:
      {
        "prediction": <0 or 1>,
        "probability": <float between 0 and 1>
      }
    """
    class_pred, proba = predict_credit(req.features)
    return {
        "prediction": class_pred,
        "probability": proba
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
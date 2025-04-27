# backend/app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from app.predict import predict_credit

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
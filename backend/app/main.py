from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Temporarysimple endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to Explainable Credit Scoring API"}

# For testing : defien an input schima
class UserInfo(BaseModel):
    income: float
    age: int
    credit_history: float
    debt_to_income_ratio: float
    
# Dummy predict endpoint
@app.post("/predict")
async def predict(user_info: UserInfo):
    # Dummy plogic: prdict approval if income > 50k
    if user_info.income > 50000:
        prediction = "Approved"
    else:
        prediction = "Rejected"
    
    return {
        "prediction": prediction,
        "input": user_info.dict()
    }
    
    
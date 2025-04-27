# backend/app/predict.py

import joblib
import numpy as np
from pathlib import Path

# 1. Define the path to the saved model relative to this file
MODEL_PATH = Path(__file__).parent / "model.pkl"

# 2. Load the model at import time so it's cached in memory
model = joblib.load(MODEL_PATH)

def predict_credit(features: list[float]) -> tuple[int, float]:
    """
    Given a list of numeric feature values in the correct order, 
    returns:
      - class_pred (0 or 1)
      - proba      (probability of that class)
    """
    # 3. Convert Python list to a NumPy array of shape (1, n_features)
    arr = np.array(features, dtype=float).reshape(1, -1)

    # 4. model.predict returns an array like [0] or [1]
    class_pred = int(model.predict(arr)[0])

    # 5. model.predict_proba returns array [[p0, p1]]; pick probability of predicted class
    proba = float(model.predict_proba(arr)[0, class_pred])

    return class_pred, proba
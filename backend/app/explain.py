# backend/app/explain.py

import shap
import numpy as np
from app.predict import model  # reuse the loaded model
from pathlib import Path
import joblib

# 1. Prepare SHAP explainer object
explainer = shap.Explainer(model.named_steps['clf'], feature_names=None)

def shap_explain(features: list[float]) -> list[dict]:
    """
    Given a list of numeric feature values, returns a list of feature contributions.
    Each contribution shows how much each feature pushed the model output higher or lower.
    """
    # 2. Convert input features into the right shape
    arr = np.array(features, dtype=float).reshape(1, -1)

    # 3. Calculate SHAP values for the single prediction
    shap_values = explainer(arr)

    # 4. Extract values
    shap_value = shap_values.values[0]  # For the first (and only) instance

    # 5. Map features to their SHAP values
    contributions = [
        {"feature_index": i, "shap_value": float(value)}
        for i, value in enumerate(shap_value)
    ]

    return contributions
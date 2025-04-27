# backend/app/explain.py

import shap
import numpy as np
from app.predict import model  # your pipeline
from pathlib import Path
import joblib

# 1. Create a dummy sample input for the masker
# We'll create a random sample similar to training data (scaled)
# (Better practice later: save X_train mean/std and load it)

dummy_input = np.random.normal(0, 1, size=(100, 23))  # 100 samples, 23 features

# 2. Create LinearExplainer for logistic regression
explainer = shap.LinearExplainer(
    model.named_steps['clf'],
    masker=dummy_input,
    feature_perturbation="interventional"
)

def shap_explain(features: list[float]) -> list[dict]:
    """
    Given a list of numeric feature values, returns a list of feature contributions.
    Each contribution shows how much each feature pushed the model output higher or lower.
    """
    arr = np.array(features, dtype=float).reshape(1, -1)
    scaled_arr = model.named_steps['scaler'].transform(arr)

    # 3. Calculate SHAP values
    shap_values = explainer(scaled_arr)

    shap_value = shap_values.values[0]  # For the first (and only) instance

    contributions = [
        {"feature_index": i, "shap_value": float(value)}
        for i, value in enumerate(shap_value)
    ]

    return contributions
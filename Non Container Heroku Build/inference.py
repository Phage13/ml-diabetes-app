import os
import joblib
import pandas as pd
import numpy as np

# 1. Load the model
def model_fn(model_dir):
    """Load the trained model from the model_dir."""
    model_path = os.path.join(model_dir, "best_diabetes_model.pkl")
    model = joblib.load(model_path)
    return model

# 2. Parse input
def input_fn(request_body, request_content_type):
    """Deserialize request body into a Pandas DataFrame."""
    if request_content_type == "application/json":
        # Expecting JSON like: {"age": 45, "bmi": 28.1, "HbA1c_level": 6.2, ...}
        data = pd.DataFrame([request_body])
        return data
    elif request_content_type == "text/csv":
        # Expecting a CSV row
        data = pd.read_csv(request_body)
        return data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

# 3. Make prediction
def predict_fn(input_data, model):
    """Apply model to the input data."""
    prediction = model.predict(input_data)
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_data)[:, 1]
    return {"prediction": prediction.tolist(), "probability": proba.tolist() if proba is not None else None}

# 4. Format output
def output_fn(prediction, response_content_type):
    """Serialize prediction output."""
    if response_content_type == "application/json":
        return prediction
    elif response_content_type == "text/csv":
        # Return CSV string
        pred = prediction["prediction"]
        return ",".join(map(str, pred))
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")
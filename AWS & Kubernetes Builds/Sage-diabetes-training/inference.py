# inference.py
import pandas as pd
import io
import joblib
import os

def model_fn(model_dir):
    model_path = os.path.join(model_dir, "best_diabetes_model.pkl")
    print(f"DEBUG: Loading model from {model_path}")
    return joblib.load(model_path)

def input_fn(request_body, content_type):
    if content_type == "text/csv":
        df = pd.read_csv(io.StringIO(request_body), header=None)
        df.columns = [
            "gender","age","hypertension","heart_disease",
            "smoking_history","bmi","HbA1c_level","blood_glucose_level"
        ]
        return df
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, accept):
    return str(prediction.tolist())
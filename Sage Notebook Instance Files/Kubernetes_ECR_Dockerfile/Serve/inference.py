import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = os.path.join(os.getcwd(), "best_diabetes_model.pkl")
model = joblib.load(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if request.content_type == "application/json":
            data = request.get_json()["data"]
            df = pd.DataFrame(data, columns=[
                "gender","age","hypertension","heart_disease",
                "smoking_history","bmi","HbA1c_level","blood_glucose_level"
            ])
        elif request.content_type == "text/csv":
            df = pd.read_csv(request.data, header=None)
            df.columns = [
                "gender","age","hypertension","heart_disease",
                "smoking_history","bmi","HbA1c_level","blood_glucose_level"
            ]
        else:
            return jsonify({"error":"Unsupported content type"}), 400

        # ✅ Map gender strings to numeric values
        gender_map = {"male":1, "female":0}
        df["gender"] = df["gender"].str.strip().str.lower().map(gender_map)

        prediction = model.predict(df).tolist()
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok"})
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

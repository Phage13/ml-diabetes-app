from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("models/best_diabetes_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract form inputs
        age = float(request.form.get("age"))
        bmi = float(request.form.get("bmi"))
        hba1c = float(request.form.get("HbA1c_level"))
        glucose = float(request.form.get("blood_glucose_level"))
        gender = int(request.form.get("gender"))
        smoking = request.form.get("smoking_history")

        # Validation checks
        if not (1 <= age <= 120):
            return render_template("index.html", prediction_text="Error: Age must be between 1 and 120 years.")
        if not (10 <= bmi <= 60):
            return render_template("index.html", prediction_text="Error: BMI must be between 10 and 60.")
        if not (3 <= hba1c <= 15):
            return render_template("index.html", prediction_text="Error: HbA1c must be between 3 and 15%.")
        if not (50 <= glucose <= 400):
            return render_template("index.html", prediction_text="Error: Blood glucose must be between 50 and 400 mg/dL.")
        if gender not in [0, 1]:
            return render_template("index.html", prediction_text="Error: Gender must be 0 (Female) or 1 (Male).")
        if smoking not in ["No Info", "never", "former", "current", "not current", "ever"]:
            return render_template("index.html", prediction_text="Error: Invalid smoking history category.")

        # Prepare features for model
        features = np.array([age, bmi, hba1c, glucose, gender, smoking]).reshape(1, -1)

        # Predict
        prediction = model.predict(features)[0]
        result = "Diabetic" if prediction == 1 else "Non-Diabetic"

        return render_template("index.html", prediction_text=f"Prediction: {result}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error processing input: {e}")

if __name__ == "__main__":
    app.run(debug=True)
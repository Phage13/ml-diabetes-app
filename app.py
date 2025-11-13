from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained pipeline (with ColumnTransformer + model)
model = joblib.load("models/best_diabetes_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect inputs from the form
        gender = int(request.form.get("gender"))
        age = float(request.form.get("age"))
        hypertension = int(request.form.get("hypertension"))
        heart_disease = int(request.form.get("heart_disease"))
        smoking = request.form.get("smoking_history")
        bmi = float(request.form.get("bmi"))
        hba1c = float(request.form.get("HbA1c_level"))
        glucose = float(request.form.get("blood_glucose_level"))

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
        if hypertension not in [0, 1]:
            return render_template("index.html", prediction_text="Error: Hypertension must be 0 (No) or 1 (Yes).")
        if heart_disease not in [0, 1]:
            return render_template("index.html", prediction_text="Error: Heart disease must be 0 (No) or 1 (Yes).")
        if smoking not in ["No Info", "never", "former", "current", "not current", "ever"]:
            return render_template("index.html", prediction_text="Error: Invalid smoking history category.")

        # Build DataFrame with correct column names
        input_df = pd.DataFrame([{
            "gender": gender,
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "smoking_history": smoking,
            "bmi": bmi,
            "HbA1c_level": hba1c,
            "blood_glucose_level": glucose
        }])

        # Debugging: print to console
        print("Features received:\n", input_df)

        # Predict
        prediction = model.predict(input_df)[0]
        result = "Diabetic" if prediction == 1 else "Non-Diabetic"

        return render_template("index.html", prediction_text=f"Prediction: {result}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error processing input: {e}")

if __name__ == "__main__":
    app.run(debug=True)
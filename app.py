import pandas as pd

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect inputs
        gender = int(request.form.get("gender"))
        age = float(request.form.get("age"))
        hypertension = int(request.form.get("hypertension"))
        heart_disease = int(request.form.get("heart_disease"))
        smoking = request.form.get("smoking_history")
        bmi = float(request.form.get("bmi"))
        hba1c = float(request.form.get("HbA1c_level"))
        glucose = float(request.form.get("blood_glucose_level"))

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

        # Predict
        prediction = model.predict(input_df)[0]
        result = "Diabetic" if prediction == 1 else "Non-Diabetic"

        return render_template("index.html", prediction_text=f"Prediction: {result}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error processing input: {e}")
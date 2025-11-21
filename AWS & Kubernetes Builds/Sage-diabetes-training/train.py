# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
import os

def main():
    # SageMaker training channel
    input_path = "/opt/ml/input/data/train/diabetes_prediction_dataset.csv"
    df = pd.read_csv(input_path)

    # Clean gender values
    df['gender'] = df['gender'].str.strip().str.title()
    df = df[df['gender'].isin(['Male','Female'])]

    # Features and target
    X = df.drop('diabetes', axis=1)
    y = df['diabetes']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocessing
    numeric_features = ['age','bmi','HbA1c_level','blood_glucose_level']
    categorical_features = ['smoking_history','gender']  # include gender here

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'  # keep hypertension, heart_disease
    )

    # Model pipeline
    gb_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(random_state=42))
    ])

    gb_model.fit(X_train, y_train)

    # Evaluation
    y_pred = gb_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, gb_model.predict_proba(X_test)[:,1]))

    # Save model to SageMaker output dir
    model_dir = "/opt/ml/model"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(gb_model, os.path.join(model_dir, "best_diabetes_model.pkl"))
    print("Model saved.")

if __name__ == "__main__":
    main()
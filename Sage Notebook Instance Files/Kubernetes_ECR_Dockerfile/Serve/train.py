import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
import os
import argparse

def main(args):
    # Load dataset
    df = pd.read_csv(args.input_path)

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
    categorical_features = ['smoking_history','gender']

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

    # Save model
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "best_diabetes_model.pkl")
    joblib.dump(gb_model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, default="diabetes_prediction_dataset.csv",
                        help="Path to training dataset")
    parser.add_argument("--model-dir", type=str, default="./model",
                        help="Directory to save trained model")
    args = parser.parse_args()
    main(args)
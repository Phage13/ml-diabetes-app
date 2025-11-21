import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib

# Local CSV path
csv_path = r"C:\Users\Nanoo\OneDrive\Desktop\ANA-680\ml-diabetes-app\Sage Notebook Instance Files\Kubernetes_ECR_Dockerfile\Train\diabetes_prediction_dataset.csv"

# Load dataset
df = pd.read_csv(csv_path)

# Basic cleaning
df['gender'] = df['gender'].str.strip().str.title()
df = df[df['gender'].isin(['Male','Female'])]
df['gender'] = df['gender'].map({'Male':1, 'Female':0})

# Features and target
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing
numeric_features = ['age','bmi','HbA1c_level','blood_glucose_level']
categorical_features = ['smoking_history']

numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

# Gradient Boosting pipeline
gb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# Train
gb_model.fit(X_train, y_train)

# Evaluate
y_pred = gb_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, gb_model.predict_proba(X_test)[:,1]))

# Save model
joblib.dump(gb_model, "best_diabetes_model.pkl", compress=3)
print("Model saved to best_diabetes_model.pkl")
"""
train_model.py
---------------
Standalone script that reproduces the training steps from
`diabetes_ml_pipeline.ipynb` and saves `diabetes_model.joblib`.

Only needed if you don't already have diabetes_model.joblib — app.py will
also auto-run this same logic on first launch if diabetes.csv is present.

Usage:
    python train_model.py
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "diabetes.csv"
MODEL_PATH = "diabetes_model.joblib"


def main():
    df = pd.read_csv(DATA_PATH)
    df_clean = df.copy()

    # Zeros in these medical columns are impossible -> treat as missing, fill with median
    medical_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in medical_cols:
        df_clean[col] = df_clean[col].replace(0, np.nan)
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    X = df_clean.drop(columns=["Outcome"])
    y = df_clean["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_pred_rounded = np.round(np.clip(y_pred, 0, 1)).astype(int)
    accuracy = (y_pred_rounded == y_test.values).mean() * 100
    print(f"Test accuracy: {accuracy:.2f}%")

    joblib.dump(
        {"model": model, "scaler": scaler, "columns": list(X.columns)},
        MODEL_PATH,
    )
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()

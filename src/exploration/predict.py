"""
Prediction pipeline for Remaining Useful Life (RUL).

This module loads the trained XGBoost model and performs inference
on new engine telemetry data to estimate remaining useful life.
"""

import pandas as pd
import joblib


model = joblib.load("models/model.pkl")

df = pd.read_csv("data/engineered_test.csv")

X = df.drop(columns=["engine_id", "cycle"])

df["predicted_RUL"] = model.predict(X)

df.to_csv("data/results.csv", index=False)

print("Predictions saved to results.csv")

if __name__ == "__main__":
    print("Running RUL prediction pipeline...")
import pandas as pd
import joblib


model = joblib.load("models/model.pkl")

df = pd.read_csv("data/engineered_test.csv")

X = df.drop(columns=["engine_id", "cycle"])

df["predicted_RUL"] = model.predict(X)

df.to_csv("data/results.csv", index=False)

print("Predictions saved to results.csv")

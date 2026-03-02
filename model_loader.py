import joblib
import pandas as pd

from src.feature_engineering import FeatureEngineer


# Load model once (efficient)
model = joblib.load("models/model.pkl")

# Initialize feature engineer
fe = FeatureEngineer()


# ==============================
# RUL Prediction Function
# ==============================
def predict_rul(sensor_data: dict):

    try:

        # Convert incoming JSON → DataFrame
        df = pd.DataFrame([sensor_data])

        # Apply feature engineering
        df = fe.rolling_features(df)
        df = fe.health_index(df)

        df.dropna(inplace=True)

        # Drop non-feature columns if present
        drop_cols = ["engine_id", "cycle", "RUL"]

        X = df.drop(
            columns=[col for col in drop_cols if col in df.columns],
            errors="ignore"
        )

        prediction = model.predict(X)[0]

        return float(prediction)

    except Exception as e:

        print("Prediction error:", e)

        return None


# ==============================
# Risk Classification
# ==============================
def classify_risk(rul):

    if rul is None:
        return "UNKNOWN"

    if rul > 120:
        return "LOW"

    elif rul > 60:
        return "MEDIUM"

    else:
        return "HIGH"


# ==============================
# Decision Logic
# ==============================
def get_decision(risk):

    if risk == "LOW":
        return "Normal operation"

    elif risk == "MEDIUM":
        return "Schedule maintenance"

    elif risk == "HIGH":
        return "Immediate maintenance required"

    else:
        return "Check system"

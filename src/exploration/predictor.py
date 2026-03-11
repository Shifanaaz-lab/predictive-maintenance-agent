import joblib
import pandas as pd

from src.feature_engineering import FeatureEngineer


class RULPredictor:

    def __init__(self):

        self.model = joblib.load("models/model.pkl")

        self.fe = FeatureEngineer()


    def predict(self, sensor_data: dict):

        # Convert JSON → DataFrame
        df = pd.DataFrame([sensor_data])

        # Apply feature engineering
        df = self.fe.rolling_features(df)
        df = self.fe.health_index(df)

        df.dropna(inplace=True)

        # Extract health index
        health_index = float(df["health_index"].iloc[-1])

        # Prepare features
        drop_cols = ["engine_id", "cycle", "RUL"]

        X = df.drop(
            columns=[col for col in drop_cols if col in df.columns],
            errors="ignore"
        )

        predicted_rul = float(self.model.predict(X)[0])

        return {

            "engine_id": int(sensor_data["engine_id"]),

            "cycle": int(sensor_data["cycle"]),

            "predicted_RUL": predicted_rul,

            "health_index": health_index
        }

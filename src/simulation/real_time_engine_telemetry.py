import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple

import numpy as np
import pandas as pd
from pymongo import MongoClient
from xgboost import XGBRegressor
import joblib


ROLLING_WINDOW = 5
NUM_ENGINES = 100
BATCH_INTERVAL_SECONDS = 3


def _safe_positive(values: np.ndarray) -> np.ndarray:
    return np.clip(values, a_min=0.0, a_max=None)


@dataclass
class EngineState:
    engine_id: int
    cycle: int = 0
    base_sensors: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    degradation_rate: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    history: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=ROLLING_WINDOW))
    design_life: int = 3000

    def initialize_random(self, rng: np.random.Generator) -> None:
        # Three synthetic sensors with different nominal ranges
        self.base_sensors = rng.uniform([20.0, 600.0, 900.0], [50.0, 800.0, 1100.0])
        # Per‑cycle degradation
        self.degradation_rate = rng.uniform([0.001, 0.02, 0.03], [0.01, 0.05, 0.06])
        self.design_life = int(rng.integers(2000, 4000))

    def next_reading(self, rng: np.random.Generator) -> Tuple[int, np.ndarray, np.ndarray]:
        self.cycle += 1

        degradation = self.degradation_rate * self.cycle
        noise = rng.normal(loc=0.0, scale=[0.2, 1.0, 1.5], size=3)
        sensors = self.base_sensors - degradation + noise
        sensors = _safe_positive(sensors)

        # Operational settings (3)
        op_settings = np.array(
            [
                rng.uniform(0.0, 1.0),      # setting 1
                rng.uniform(20.0, 40.0),    # setting 2 (e.g. temp)
                rng.uniform(900.0, 1100.0), # setting 3 (e.g. RPM)
            ],
            dtype=float,
        )

        self.history.append(sensors.copy())
        return self.cycle, sensors, op_settings


class FeatureEngineer:
    """
    Applies feature engineering consistent with model training.

    Features per sensor s{i}:
      - rolling mean/std over last ROLLING_WINDOW cycles
      - lag1, lag2
      - change = current - lag1
      - trend = slope from linear regression over window

    Global per‑engine features:
      - health_index: normalized combination of sensors
      - life_ratio: cycle / design_life
    """

    SENSOR_COLS = ["s1", "s2", "s3"]
    SETTING_COLS = ["setting1", "setting2", "setting3"]

    def __init__(self, rolling_window: int = ROLLING_WINDOW) -> None:
        self.window = rolling_window

    def _compute_sensor_features(self, engine: EngineState, current_sensors: np.ndarray) -> Dict[str, float]:
        hist = np.array(engine.history)
        features: Dict[str, float] = {}

        for i, s_name in enumerate(self.SENSOR_COLS):
            series = hist[:, i] if hist.shape[0] > 0 else np.array([], dtype=float)

            # rolling mean/std
            if series.size >= 1:
                w = min(self.window, series.size)
                window_vals = series[-w:]
                features[f"{s_name}_roll_mean"] = float(window_vals.mean())
                features[f"{s_name}_roll_std"] = float(window_vals.std(ddof=0))
            else:
                features[f"{s_name}_roll_mean"] = 0.0
                features[f"{s_name}_roll_std"] = 0.0

            # lags
            features[f"{s_name}_lag1"] = float(series[-1]) if series.size >= 1 else 0.0
            features[f"{s_name}_lag2"] = float(series[-2]) if series.size >= 2 else 0.0

            # change
            features[f"{s_name}_change"] = float(current_sensors[i] - features[f"{s_name}_lag1"])

            # trend (linear regression slope)
            if series.size >= 2:
                x = np.arange(series.size)
                coeffs = np.polyfit(x, series, 1)
                features[f"{s_name}_trend"] = float(coeffs[0])
            else:
                features[f"{s_name}_trend"] = 0.0

        # Health index: lower sensors => lower health; normalize by initial bases
        base_sum = float(engine.base_sensors.sum())
        current_sum = float(current_sensors.sum())
        if base_sum > 0:
            health_index = current_sum / base_sum
        else:
            health_index = 1.0
        features["health_index"] = float(np.clip(health_index, 0.0, 1.5))

        # Life ratio: cycle relative to design life
        life_ratio = engine.cycle / max(engine.design_life, 1)
        features["life_ratio"] = float(np.clip(life_ratio, 0.0, 2.0))

        return features

    def build_feature_row(
        self,
        engine: EngineState,
        cycle: int,
        sensors: np.ndarray,
        op_settings: np.ndarray,
    ) -> Dict[str, float]:
        row: Dict[str, float] = {
            "engine_id": float(engine.engine_id),
            "cycle": float(cycle),
        }

        # raw sensor and setting values
        for i, name in enumerate(self.SENSOR_COLS):
            row[name] = float(sensors[i])
        for i, name in enumerate(self.SETTING_COLS):
            row[name] = float(op_settings[i])

        row.update(self._compute_sensor_features(engine, sensors))
        return row


class ModelPredictor:
    """
    Wraps XGBRegressor for RUL prediction and derives a simple failure probability.
    """

    def __init__(self, model_path: str, expected_features: List[str] | None = None) -> None:
        """
        Load an XGBoost model saved either as:
        - native XGBoost JSON/binary (for XGBRegressor)  -> use load_model
        - Python pickle / joblib (.pkl, .pickle, .joblib) -> use joblib.load
        """
        model_path_lower = model_path.lower()
        if model_path_lower.endswith((".pkl", ".pickle", ".joblib")):
            # sklearn-style pickled model
            self.model = joblib.load(model_path)
        else:
            # native XGBoost model file
            self.model = XGBRegressor()
            self.model.load_model(model_path)

        # Determine feature order
        booster_feature_names: List[str] = []
        try:
            booster = self.model.get_booster()
            booster_feature_names = booster.feature_names or []
        except Exception:
            booster_feature_names = []

        if expected_features is not None:
            self.feature_order = expected_features
        elif booster_feature_names:
            self.feature_order = list(booster_feature_names)
        else:
            raise ValueError(
                "Could not infer feature names from model. "
                "Please pass expected_features that matches training."
            )

    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        aligned = df.reindex(columns=self.feature_order, fill_value=0.0)
        return aligned.astype(float)

    def predict_batch(self, features: List[Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray]:
        df = pd.DataFrame(features)
        X = self._align_features(df)

        rul_pred = self.model.predict(X)

        # Simple heuristic failure probability: higher when RUL is low
        # Assumes RUL in same scale as training; adjust as needed.
        failure_prob = np.clip(1.0 - rul_pred / (rul_pred.max() + 1e-6), 0.0, 1.0)
        return rul_pred, failure_prob


class MongoSink:
    def __init__(self, uri: str, db_name: str, collection_name: str = "live_predictions") -> None:
        self.client = MongoClient(uri)
        self.collection = self.client[db_name][collection_name]

    def write_batch(
        self,
        batch_features: List[Dict[str, float]],
        rul: np.ndarray,
        failure_prob: np.ndarray,
    ) -> None:
        ts = time.time()
        docs = []
        for feat, r, p in zip(batch_features, rul, failure_prob):
            docs.append(
                {
                    "_id": str(uuid.uuid4()),
                    "timestamp": ts,
                    "engine_id": int(feat.get("engine_id", -1)),
                    "cycle": int(feat.get("cycle", -1)),
                    "predicted_rul": float(r),
                    "failure_probability": float(p),
                    "features": feat,
                }
            )
        if docs:
            self.collection.insert_many(docs, ordered=False)


class TelemetrySimulator:
    def __init__(
        self,
        num_engines: int,
        model_path: str,
        mongo_uri: str,
        mongo_db: str,
        expected_features: List[str] | None = None,
        batch_interval_seconds: int = BATCH_INTERVAL_SECONDS,
    ) -> None:
        self.rng = np.random.default_rng(seed=42)
        self.engines = [
            self._create_engine(engine_id=i) for i in range(1, num_engines + 1)
        ]
        self.fe = FeatureEngineer()
        self.predictor = ModelPredictor(model_path=model_path, expected_features=expected_features)
        self.sink = MongoSink(uri=mongo_uri, db_name=mongo_db)
        self.batch_interval_seconds = batch_interval_seconds

    def _create_engine(self, engine_id: int) -> EngineState:
        eng = EngineState(engine_id=engine_id)
        eng.initialize_random(self.rng)
        # Warm‑up history with initial reading (cycle 0 treated as initial state)
        for _ in range(ROLLING_WINDOW):
            _, sensors, _ = eng.next_reading(self.rng)
            # history already updated in next_reading
        return eng

    def _generate_batch(self) -> List[Dict[str, float]]:
        batch_rows: List[Dict[str, float]] = []
        for eng in self.engines:
            cycle, sensors, op_settings = eng.next_reading(self.rng)
            row = self.fe.build_feature_row(eng, cycle, sensors, op_settings)
            batch_rows.append(row)
        return batch_rows

    def run_forever(self) -> None:
        print("Starting real‑time engine telemetry simulation...")
        batch_idx = 0
        while True:
            batch_idx += 1
            start = time.time()

            features = self._generate_batch()
            rul_pred, failure_prob = self.predictor.predict_batch(features)
            self.sink.write_batch(features, rul_pred, failure_prob)

            # Print summary
            mean_rul = float(rul_pred.mean())
            min_rul = float(rul_pred.min())
            max_rul = float(rul_pred.max())
            mean_fail = float(failure_prob.mean())
            high_risk = int((failure_prob > 0.8).sum())

            print(
                f"[Batch {batch_idx}] engines={len(self.engines)} "
                f"mean_RUL={mean_rul:.2f} min_RUL={min_rul:.2f} max_RUL={max_rul:.2f} "
                f"mean_failure_prob={mean_fail:.3f} high_risk_engines={high_risk}"
            )

            elapsed = time.time() - start
            sleep_time = max(0.0, self.batch_interval_seconds - elapsed)
            time.sleep(sleep_time)


def load_expected_features_from_env() -> List[str] | None:
    """
    Optionally provide EXACT feature order via environment variable
    EXPECTED_FEATURES as a comma‑separated list.
    This should match the features used during model training.
    """
    raw = os.getenv("EXPECTED_FEATURES")
    if not raw:
        return None
    return [f.strip() for f in raw.split(",") if f.strip()]


def main() -> None:
    model_path = os.getenv("XGB_MODEL_PATH", "xgb_rul_model.json")
    mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    mongo_db = os.getenv("MONGODB_DB", "engine_telemetry")

    expected_features = load_expected_features_from_env()

    sim = TelemetrySimulator(
        num_engines=NUM_ENGINES,
        model_path=model_path,
        mongo_uri=mongo_uri,
        mongo_db=mongo_db,
        expected_features=expected_features,
        batch_interval_seconds=BATCH_INTERVAL_SECONDS,
    )
    sim.run_forever()


if __name__ == "__main__":
    main()


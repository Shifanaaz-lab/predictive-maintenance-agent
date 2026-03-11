"""
Train an XGBoost RUL model that is 100% compatible with the real‑time
simulator in `real_time_engine_telemetry.py`.

This script:
  - Uses the same `EngineState` and `FeatureEngineer` classes as the live simulator
  - Simulates many engines and cycles offline
  - Defines RUL labels as (design_life - cycle)
  - Trains an XGBRegressor on those features
  - Saves the model as `xgb_rul_model.json` with feature names embedded

After running this once, point the simulator at the new model:

  $env:XGB_MODEL_PATH = "E:\\ROBOTICS\\syn-dataset\\xgb_rul_model.json"

and (optionally) clear EXPECTED_FEATURES so it uses the model's own feature names:

  Remove-Item Env:EXPECTED_FEATURES
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from real_time_engine_telemetry import (
    EngineState,
    FeatureEngineer,
    ROLLING_WINDOW,
)


NUM_TRAIN_ENGINES = 200
RUL_HORIZON_MULTIPLIER = 1.2  # simulate beyond nominal design life


def generate_training_data(
    num_engines: int = NUM_TRAIN_ENGINES,
    random_state: int = 0,
) -> Tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(random_state)
    fe = FeatureEngineer()

    rows = []
    labels = []

    for engine_id in range(1, num_engines + 1):
        eng = EngineState(engine_id=engine_id)
        eng.initialize_random(rng)

        # Warm‑up history to be consistent with the live simulator
        for _ in range(ROLLING_WINDOW):
            _, sensors, op_settings = eng.next_reading(rng)

        max_cycles = int(eng.design_life * RUL_HORIZON_MULTIPLIER)

        for _ in range(max_cycles):
            cycle, sensors, op_settings = eng.next_reading(rng)
            feat = fe.build_feature_row(eng, cycle, sensors, op_settings)

            # Remaining useful life label (clipped at 0)
            rul = max(eng.design_life - cycle, 0)

            rows.append(feat)
            labels.append(rul)

    X = pd.DataFrame(rows)
    y = np.asarray(labels, dtype=float)
    return X, y


def train_and_save_model(
    output_path: str = "xgb_rul_model.json",
) -> None:
    print("Generating synthetic training data...")
    X, y = generate_training_data()

    print(f"Training data: {X.shape[0]:,} rows, {X.shape[1]} features")

    # Basic XGBoost regression model for RUL
    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
    )

    print("Training XGBoost model...")
    model.fit(X, y)

    # Save native XGBoost model with feature names embedded
    model.save_model(output_path)
    abs_path = os.path.abspath(output_path)
    print(f"Saved trained model to {abs_path}")

    print("\nFeature order used during training (for reference):")
    print(",".join(X.columns))


if __name__ == "__main__":
    train_and_save_model()


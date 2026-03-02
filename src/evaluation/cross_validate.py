# src/cross_validate.py

import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from xgboost import callback


print("Loading dataset...")
df = pd.read_csv("data/engineered_train.csv")

engine_ids = df["engine_id"].unique()

np.random.seed(42)
np.random.shuffle(engine_ids)

k = 5
fold_size = len(engine_ids) // k

results = []

for fold in range(k):

    print(f"\nFOLD {fold + 1}")

    start = fold * fold_size
    end = start + fold_size

    test_engines = engine_ids[start:end]
    train_engines = np.concatenate((engine_ids[:start], engine_ids[end:]))

    train_df = df[df["engine_id"].isin(train_engines)]
    test_df = df[df["engine_id"].isin(test_engines)]

    X_train = train_df.drop(columns=["engine_id", "cycle", "RUL"])
    y_train = train_df["RUL"]

    X_test = test_df.drop(columns=["engine_id", "cycle", "RUL"])
    y_test = test_df["RUL"]

    model = XGBRegressor(
   	n_estimators=500,
    	learning_rate=0.05,
    	max_depth=6,
    	subsample=0.8,
    	colsample_bytree=0.8,
    	random_state=42,
    	n_jobs=-1,
    	tree_method="hist",
    	eval_metric="rmse",
    	early_stopping_rounds=50
   )    
    model.fit(
    	X_train,
    	y_train,
    	eval_set=[(X_test, y_test)],
    	verbose=False
)    
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)

    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R²:", round(r2, 4))

    results.append(r2)

print("\nAverage R² across folds:", round(np.mean(results), 4))
print("Std deviation:", round(np.std(results), 4))
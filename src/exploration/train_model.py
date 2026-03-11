# src/train_model.py

import pandas as pd
import joblib
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


print("Loading dataset...")

df = pd.read_csv("data/engineered_train.csv")


# Engine split
engine_ids = df["engine_id"].unique()

np.random.seed(42)
np.random.shuffle(engine_ids)

train_ids = engine_ids[:80]
test_ids = engine_ids[80:]

train_df = df[df["engine_id"].isin(train_ids)]
test_df = df[df["engine_id"].isin(test_ids)]


X_train = train_df.drop(columns=["engine_id", "cycle", "RUL"])
y_train = train_df["RUL"]

X_test = test_df.drop(columns=["engine_id", "cycle", "RUL"])
y_test = test_df["RUL"]


print("Training samples:", X_train.shape)


model = XGBRegressor(

    n_estimators=3000,

    learning_rate=0.005,

    max_depth=8,

    min_child_weight=3,

    subsample=0.9,
    colsample_bytree=0.9,

    gamma=0.01,

    reg_alpha=0.1,
    reg_lambda=1.5,

    objective="reg:squarederror",

    random_state=42,
    n_jobs=-1
)


model.fit(

    X_train,
    y_train,

    eval_set=[(X_test, y_test)],
    early_stopping_rounds=200,
    verbose=100
)


pred = model.predict(X_test)


print("\nPERFORMANCE")

print("MAE:", mean_absolute_error(y_test, pred))
print("RMSE:", mean_squared_error(y_test, pred) ** 0.5)
print("R²:", r2_score(y_test, pred))


joblib.dump(model, "models/model.pkl")

print("Model saved.")

import matplotlib.pyplot as plt
import numpy as np

feature_importance = model.feature_importances_
feature_names = X_train.columns

indices = np.argsort(feature_importance)[-20:]

plt.figure(figsize=(10, 8))
plt.barh(range(len(indices)), feature_importance[indices])
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.title("Top 20 Important Features")
plt.tight_layout()
plt.savefig("models/feature_importance.png")
plt.close()
import pandas as pd
import joblib

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


print("Loading engineered training data...")

df = pd.read_csv("data/engineered_train.csv")


print("Loading model...")

model = joblib.load("models/model.pkl")


print("Preparing features...")

X = df.drop(columns=["engine_id", "cycle", "RUL"])

y = df["RUL"]


print("Predicting...")

predictions = model.predict(X)


mae = mean_absolute_error(y, predictions)

rmse = mean_squared_error(y, predictions) ** 0.5

r2 = r2_score(y, predictions)


print("\nXGBoost Model Performance:")

print("MAE:", round(mae, 2))

print("RMSE:", round(rmse, 2))

print("R² Score:", round(r2, 4))

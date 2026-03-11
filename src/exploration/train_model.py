# src/exploration/train_model.py

import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


def load_data():
    print("Loading dataset...")
    return pd.read_csv("data/processed/engineered_train.csv")


def split_data(df):
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

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, X_test, y_test):

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

    return model


def evaluate_model(model, X_test, y_test):

    pred = model.predict(X_test)

    print("\nPERFORMANCE")
    print("MAE:", mean_absolute_error(y_test, pred))
    print("RMSE:", mean_squared_error(y_test, pred) ** 0.5)
    print("R²:", r2_score(y_test, pred))


def save_model(model):
    joblib.dump(model, "models/model.pkl")
    print("Model saved.")


def plot_feature_importance(model, feature_names):

    feature_importance = model.feature_importances_
    indices = np.argsort(feature_importance)[-20:]

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), feature_importance[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title("Top 20 Important Features")
    plt.tight_layout()
    plt.savefig("models/feature_importance.png")
    plt.close()


def main():

    df = load_data()

    X_train, X_test, y_train, y_test = split_data(df)

    print("Training samples:", X_train.shape)

    model = train_model(X_train, y_train, X_test, y_test)

    evaluate_model(model, X_test, y_test)

    save_model(model)

    plot_feature_importance(model, X_train.columns)


if __name__ == "__main__":
    main()
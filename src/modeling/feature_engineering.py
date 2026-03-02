# src/feature_engineering.py

import pandas as pd
import numpy as np


class FeatureEngineer:

    def __init__(self, window_size=10):

        self.window_size = window_size

        self.columns = (
            ["engine_id", "cycle"]
            + [f"setting{i}" for i in range(1, 4)]
            + [f"sensor{i}" for i in range(1, 22)]
        )

        self.sensors = [f"sensor{i}" for i in range(1, 22)]


    # -------------------------
    # Load NASA dataset
    # -------------------------
    def load_data(self, path):

        df = pd.read_csv(path, sep="\s+", header=None)
        df.columns = self.columns

        return df


    # -------------------------
    # Calculate RUL
    # -------------------------
    def calculate_rul(self, df):

        max_cycle = df.groupby("engine_id")["cycle"].max()

        df = df.merge(max_cycle.rename("max_cycle"), on="engine_id")

        df["RUL"] = df["max_cycle"] - df["cycle"]

        # IMPORTANT: cap RUL (NASA best practice)
        df["RUL"] = df["RUL"].clip(upper=125)

        df.drop(columns=["max_cycle"], inplace=True)

        return df


    # -------------------------
    # Lifecycle normalization
    # -------------------------
    def lifecycle_features(self, df):

        max_cycle = df.groupby("engine_id")["cycle"].transform("max")

        df["life_ratio"] = df["cycle"] / max_cycle

        df["life_remaining_ratio"] = 1 - df["life_ratio"]

        return df


    # -------------------------
    # Rolling statistical features
    # -------------------------
    def rolling_features(self, df):

        for sensor in self.sensors:

            grouped = df.groupby("engine_id")[sensor]

            df[f"{sensor}_mean"] = grouped.rolling(self.window_size).mean().reset_index(0, drop=True)

            df[f"{sensor}_std"] = grouped.rolling(self.window_size).std().reset_index(0, drop=True)

            df[f"{sensor}_ema"] = grouped.transform(
                lambda x: x.ewm(span=self.window_size).mean()
            )

        return df


    # -------------------------
    # Degradation slope feature (CRITICAL)
    # -------------------------
    def slope_features(self, df):

        for sensor in self.sensors:

            df[f"{sensor}_slope"] = (
                df.groupby("engine_id")[sensor]
                .transform(lambda x: x.diff().rolling(self.window_size).mean())
            )

        return df


    # -------------------------
    # Sensor deviation from healthy state
    # -------------------------
    def deviation_features(self, df):

        healthy_baseline = df.groupby("engine_id")[self.sensors].transform("first")

        for sensor in self.sensors:

            df[f"{sensor}_dev"] = df[sensor] - healthy_baseline[sensor]

        return df


    # -------------------------
    # Health index
    # -------------------------
    def health_index(self, df):

        df["health_index"] = df[self.sensors].mean(axis=1)

        return df


    # -------------------------
    # Full pipeline
    # -------------------------
    def transform_file(self, input_path, output_path, is_train=True):

        print("Loading data...")
        df = self.load_data(input_path)

        if is_train:
            print("Calculating RUL...")
            df = self.calculate_rul(df)

        print("Lifecycle features...")
        df = self.lifecycle_features(df)

        print("Rolling features...")
        df = self.rolling_features(df)

        print("Slope features...")
        df = self.slope_features(df)

        print("Deviation features...")
        df = self.deviation_features(df)

        print("Health index...")
        df = self.health_index(df)

        print("Cleaning...")
        df.dropna(inplace=True)

        print("Saving...")
        df.to_csv(output_path, index=False)

        print("Final shape:", df.shape)

        return df
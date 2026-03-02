from pymongo import MongoClient
from datetime import datetime

# MongoDB connection
client = MongoClient("mongodb://localhost:27017")

db = client["predictive_maintenance"]

sensor_collection = db["sensor_data"]
prediction_collection = db["predictions"]


def save_prediction(row):
    prediction_collection.insert_one({
        "engine_id": int(row["engine_id"]),
        "cycle": int(row["cycle"]),
        "predicted_RUL": float(row["predicted_RUL"]),
        "health_index": float(row["health_index"]),
        "timestamp": datetime.utcnow()
    })

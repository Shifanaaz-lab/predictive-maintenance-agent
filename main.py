from fastapi import FastAPI
from datetime import datetime
from bson import ObjectId

from database import sensor_collection, prediction_collection
from src.predictor import RULPredictor

app = FastAPI()

# Initialize predictor
predictor = RULPredictor()


# ==============================
# SAFE CONVERSION FUNCTION
# ==============================
def convert_mongo_doc(doc):

    safe_doc = {}

    for key, value in doc.items():

        if isinstance(value, ObjectId):
            safe_doc[key] = str(value)

        elif isinstance(value, datetime):
            safe_doc[key] = value.isoformat()

        elif isinstance(value, dict):
            safe_doc[key] = convert_mongo_doc(value)

        elif isinstance(value, list):
            safe_doc[key] = [
                convert_mongo_doc(item) if isinstance(item, dict)
                else str(item) if isinstance(item, ObjectId)
                else item
                for item in value
            ]

        else:
            safe_doc[key] = value

    return safe_doc


# ==============================
# HOME ENDPOINT
# ==============================
@app.get("/")
def home():
    return {"message": "Predictive Maintenance API running"}


# ==============================
# SENSOR DATA ENDPOINT
# ==============================
@app.post("/sensor-data")
def receive_sensor_data(data: dict):

    try:

        # Store raw sensor data
        data_copy = data.copy()
        data_copy["timestamp"] = datetime.utcnow()

        sensor_collection.insert_one(data_copy)

        # ==========================
        # Predict using model
        # ==========================

        result = predictor.predict(data)

        engine_id = result["engine_id"]
        cycle = result["cycle"]
        predicted_RUL = result["predicted_RUL"]
        health_index = result["health_index"]

        # ==========================
        # Risk logic
        # ==========================

        if predicted_RUL < 30:

            risk = "HIGH"
            decision = "Immediate Maintenance Required"

        elif predicted_RUL < 80:

            risk = "MEDIUM"
            decision = "Schedule Maintenance"

        else:

            risk = "LOW"
            decision = "Normal Operation"

        # ==========================
        # MongoDB structured insert
        # ==========================

        prediction_doc = {

            "engine_id": engine_id,

            "cycle": cycle,

            "predicted_RUL": predicted_RUL,

            "health_index": health_index,

            "risk_level": risk,

            "decision": decision,

            "timestamp": datetime.utcnow()

        }

        insert_result = prediction_collection.insert_one(prediction_doc)

        prediction_doc["_id"] = str(insert_result.inserted_id)

        prediction_doc["timestamp"] = prediction_doc["timestamp"].isoformat()

        return prediction_doc


    except Exception as e:

        return {"error": str(e)}
        
# ==============================
# GET PREDICTIONS ENDPOINT
# ==============================
@app.get("/predictions")
def get_predictions():

    try:

        cursor = prediction_collection.find()

        predictions = []

        for doc in cursor:

            safe_doc = convert_mongo_doc(doc)

            predictions.append(safe_doc)

        return predictions

    except Exception as e:

        print("ERROR IN /predictions:", e)

        return {"error": str(e)}

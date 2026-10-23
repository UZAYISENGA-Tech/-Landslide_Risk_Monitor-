
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "landslide_random_forest.joblib"

model = joblib.load(MODEL_PATH)

app = FastAPI(title="Landslide Risk Monitor API")


class SensorData(BaseModel):
    soil_moisture: float = Field(ge=0, le=100)
    vibration: float = Field(ge=0)
    rainfall: float = Field(ge=0)


@app.get("/")
def home():
    return {"message": "Landslide Risk Monitor API is running"}


@app.get("/health")
def health():
    return {"status": "running", "model_loaded": True}


@app.post("/predict")
def predict(data: SensorData):
    features = pd.DataFrame(
        [[data.soil_moisture, data.vibration, data.rainfall]],
        columns=["Soil_Moisture", "Vibration", "Rainfall"],
    )

    prediction = model.predict(features)[0]
    return {"risk_level": str(prediction)}

"""
Day 26 - Optimized FastAPI ML Microservice

Improvements:
✔ Model loads once at startup (performance optimized)
✔ Clean API contract using Pydantic
✔ Fast inference (<500ms)
✔ Production-ready structure
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
import pandas as pd
import joblib

app = FastAPI()

# ==============================
# GLOBAL VARIABLES (loaded once)
# ==============================
model = None
label_encoders = None


# ==============================
# LOAD MODEL AT STARTUP
# ==============================
@app.on_event("startup")
def load_model():
    global model, label_encoders

    model = joblib.load("model/best_titanic_model.pkl")
    label_encoders = joblib.load("model/label_encoders.pkl")


# ==============================
# REQUEST MODEL (API CONTRACT)
# ==============================
class PassengerData(BaseModel):
    Pclass: int = Field(..., ge=1, le=3)
    Sex: str = Field(..., pattern="^(male|female)$")
    Age: float = Field(..., ge=0.42, le=80)
    SibSp: int = Field(..., ge=0, le=8)
    Parch: int = Field(..., ge=0, le=6)
    Fare: float = Field(..., ge=0, le=512.3292)
    Embarked: str = Field(..., pattern="^(C|Q|S)$")


# ==============================
# RESPONSE MODEL
# ==============================
class PredictionResponse(BaseModel):
    survival_prediction: int


# ==============================
# HEALTH CHECK
# ==============================
@app.get("/health-check")
async def health_check():
    return {"status": "API is live"}


# ==============================
# PREDICTION ENDPOINT
# ==============================
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PassengerData):

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    input_data = request.model_dump()

    # ==============================
    # OOD VALIDATION (Security)
    # ==============================
    if not (1 <= input_data["Pclass"] <= 3):
        raise HTTPException(status_code=400, detail="Invalid Pclass")

    if not (0.42 <= input_data["Age"] <= 80):
        raise HTTPException(status_code=400, detail="Invalid Age")

    if not (0 <= input_data["SibSp"] <= 8):
        raise HTTPException(status_code=400, detail="Invalid SibSp")

    if not (0 <= input_data["Parch"] <= 6):
        raise HTTPException(status_code=400, detail="Invalid Parch")

    if not (0 <= input_data["Fare"] <= 512.3292):
        raise HTTPException(status_code=400, detail="Invalid Fare")

    # ==============================
    # ENCODING
    # ==============================
    input_data["Sex"] = int(
        label_encoders["Sex"].transform([input_data["Sex"]])[0]
    )

    input_data["Embarked"] = int(
        label_encoders["Embarked"].transform([input_data["Embarked"]])[0]
    )

    # ==============================
    # DATAFRAME
    # ==============================
    input_df = pd.DataFrame([input_data])

    # ==============================
    # PREDICTION (async safe)
    # ==============================
    prediction = await run_in_threadpool(
        model.predict,
        input_df
    )

    return PredictionResponse(
        survival_prediction=int(prediction[0])
    )
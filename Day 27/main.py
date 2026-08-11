"""
Day 27 - Advanced Pydantic Validation for FastAPI ML Microservice

Security and reliability improvements:
- Strict Pydantic field constraints
- Custom field validators
- Input sanitization
- Out-of-distribution checks
- Async FastAPI endpoints
- Non-blocking CPU-bound model inference
- Structured prediction response
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool
import pandas as pd
import joblib
import math


app = FastAPI(
    title="Titanic ML Prediction API",
    description="Secure FastAPI Machine Learning Microservice",
    version="2.0"
)


# ============================================================
# GLOBAL VARIABLES
# Model is loaded once during application startup.
# ============================================================

model = None
label_encoders = None


# ============================================================
# LOAD MODEL AT STARTUP
# ============================================================

@app.on_event("startup")
def load_model():
    global model, label_encoders

    model = joblib.load("model/best_titanic_model.pkl")
    label_encoders = joblib.load("model/label_encoders.pkl")


# ============================================================
# REQUEST MODEL
# API CONTRACT + ADVANCED VALIDATION
# ============================================================

class PassengerData(BaseModel):

    Pclass: int = Field(
        ...,
        ge=1,
        le=3,
        description="Passenger class: 1, 2, or 3"
    )

    Sex: str = Field(
        ...,
        min_length=4,
        max_length=6,
        description="Passenger sex: male or female"
    )

    Age: float = Field(
        ...,
        ge=0.42,
        le=80,
        description="Passenger age"
    )

    SibSp: int = Field(
        ...,
        ge=0,
        le=8,
        description="Number of siblings/spouses aboard"
    )

    Parch: int = Field(
        ...,
        ge=0,
        le=6,
        description="Number of parents/children aboard"
    )

    Fare: float = Field(
        ...,
        ge=0,
        le=512.3292,
        description="Passenger fare"
    )

    Embarked: str = Field(
        ...,
        min_length=1,
        max_length=1,
        description="Embarkation port: C, Q, or S"
    )

    # ========================================================
    # CUSTOM SEX VALIDATOR
    # ========================================================

    @field_validator("Sex")
    @classmethod
    def validate_sex(cls, value: str) -> str:

        value = value.strip().lower()

        if value not in {"male", "female"}:
            raise ValueError("Sex must be either 'male' or 'female'.")

        return value

    # ========================================================
    # CUSTOM EMBARKED VALIDATOR
    # ========================================================

    @field_validator("Embarked")
    @classmethod
    def validate_embarked(cls, value: str) -> str:

        value = value.strip().upper()

        if value not in {"C", "Q", "S"}:
            raise ValueError("Embarked must be one of: C, Q, or S.")

        return value

    # ========================================================
    # FINITE NUMBER VALIDATION
    # Prevent NaN and Infinity values.
    # ========================================================

    @field_validator("Age", "Fare")
    @classmethod
    def validate_finite_numbers(cls, value: float) -> float:

        if not math.isfinite(value):
            raise ValueError("Value must be a finite number.")

        return value


# ============================================================
# RESPONSE MODEL
# ============================================================

class PredictionResponse(BaseModel):
    survival_prediction: int


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health-check")
async def health_check():
    return {
        "status": "API is live"
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
async def predict(request: PassengerData):

    # --------------------------------------------------------
    # Ensure model is available
    # --------------------------------------------------------

    if model is None or label_encoders is None:
        raise HTTPException(
            status_code=500,
            detail="Model is not loaded."
        )

    input_data = request.model_dump()

    # ========================================================
    # OOD VALIDATION
    # Additional protection before model inference.
    # ========================================================

    if not (1 <= input_data["Pclass"] <= 3):
        raise HTTPException(
            status_code=400,
            detail="Pclass is outside the training distribution."
        )

    if not (0.42 <= input_data["Age"] <= 80):
        raise HTTPException(
            status_code=400,
            detail="Age is outside the training distribution."
        )

    if not (0 <= input_data["SibSp"] <= 8):
        raise HTTPException(
            status_code=400,
            detail="SibSp is outside the training distribution."
        )

    if not (0 <= input_data["Parch"] <= 6):
        raise HTTPException(
            status_code=400,
            detail="Parch is outside the training distribution."
        )

    if not (0 <= input_data["Fare"] <= 512.3292):
        raise HTTPException(
            status_code=400,
            detail="Fare is outside the training distribution."
        )

    # ========================================================
    # ENCODE CATEGORICAL FEATURES
    # ========================================================

    input_data["Sex"] = int(
        label_encoders["Sex"].transform(
            [input_data["Sex"]]
        )[0]
    )

    input_data["Embarked"] = int(
        label_encoders["Embarked"].transform(
            [input_data["Embarked"]]
        )[0]
    )

    # ========================================================
    # CREATE MODEL INPUT
    # ========================================================

    input_df = pd.DataFrame([input_data])

    # ========================================================
    # NON-BLOCKING MODEL INFERENCE
    # CPU-bound prediction is dispatched to a worker thread.
    # ========================================================

    prediction = await run_in_threadpool(
        model.predict,
        input_df
    )

    # ========================================================
    # STRUCTURED RESPONSE
    # ========================================================

    return PredictionResponse(
        survival_prediction=int(prediction[0])
    )
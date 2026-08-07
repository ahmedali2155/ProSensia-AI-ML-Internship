"""
Day 24 - Security Hardening for FastAPI AI Microservice

This project implements multiple security layers before user input reaches
the Machine Learning model.

Security Features:
1. Pydantic Field Validation
   - Validates numeric ranges.
   - Validates required fields.
   - Uses regex patterns for categorical values.
   - Rejects malformed JSON requests.

2. Out-of-Distribution (OOD) Detection
   - Ensures input values remain within the training distribution.
   - Rejects abnormal values with HTTP 400 Bad Request.

3. Prompt Injection Protection
   - Unexpected text values fail regex validation.
   - Malicious strings never reach the Machine Learning model.

4. Secure API Design
   - Invalid requests return HTTP 400 or HTTP 422.
   - Prevents unhandled HTTP 500 Internal Server Errors.
"""

from starlette.concurrency import run_in_threadpool
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import numpy as np

model = joblib.load("best_titanic_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

app = FastAPI()


class PassengerData(BaseModel):
    Pclass: int = Field(..., ge=1, le=3)
    Sex: str = Field(..., pattern="^(male|female)$")
    Age: float = Field(..., ge=0.42, le=80)
    SibSp: int = Field(..., ge=0, le=8)
    Parch: int = Field(..., ge=0, le=6)
    Fare: float = Field(..., ge=0, le=512.3292)
    Embarked: str = Field(..., pattern="^(C|Q|S)$")

class PredictionResponse(BaseModel):
    survival_prediction: int


@app.get("/health-check")
async def health_check():
    return {
        "status": "API is live"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PassengerData):

    input_data = request.model_dump()
    # ======================================================
    # Out-of-Distribution (OOD) Validation
    # These limits are based on the Titanic training dataset.
    # Requests outside these ranges are rejected before
    # reaching the Machine Learning model.
    # ======================================================-----------------------------

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

    # Encode categorical columns
    input_data["Sex"] = int(
        label_encoders["Sex"].transform([input_data["Sex"]])[0]
    )

    input_data["Embarked"] = int(
        label_encoders["Embarked"].transform([input_data["Embarked"]])[0]
    )

    # Convert to DataFrame
    

    input_array = np.array([[
        input_data["Pclass"],
        input_data["Sex"],
        input_data["Age"],
        input_data["SibSp"],
        input_data["Parch"],
        input_data["Fare"],
        input_data["Embarked"]
    ]])

    # Predict
    prediction = await run_in_threadpool(
      model.predict,
      input_array
)

    return PredictionResponse(
    survival_prediction=int(prediction[0])
)
   
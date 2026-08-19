from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool
import pandas as pd
import joblib
import math

from drift_detector import log_request, detect_drift

app = FastAPI(
    title="Titanic ML Prediction API",
    description="Secure FastAPI Machine Learning Microservice",
    version="5.0"
)

# ============================================================
# GLOBAL VARIABLES
# ============================================================

model = None
pipeline = None

# ============================================================
# LOAD MODEL & PIPELINE
# ============================================================

@app.on_event("startup")
def load_model():

    global model, pipeline

    model = joblib.load("model/best_titanic_model.pkl")
    pipeline = joblib.load("model/pipeline.pkl")


# ============================================================
# OOD BOUNDARY INTERCEPTOR
# ============================================================

def check_ood(data: dict):

    boundaries = {
        "Pclass": (1, 3),
        "Age": (1, 75),
        "SibSp": (0, 5),
        "Parch": (0, 5),
        "Fare": (0, 300),
    }

    for feature, (minimum, maximum) in boundaries.items():

        value = data[feature]

        if value < minimum or value > maximum:

            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Data Out of Bounds",
                    "feature": feature,
                    "value": value,
                    "allowed_range": [minimum, maximum],
                },
            )


# ============================================================
# REQUEST MODEL
# ============================================================

class PassengerData(BaseModel):

    Pclass: int = Field(..., ge=1, le=3)
    Sex: str = Field(...)
    Age: float = Field(..., ge=0.42, le=80)
    SibSp: int = Field(..., ge=0, le=8)
    Parch: int = Field(..., ge=0, le=6)
    Fare: float = Field(..., ge=0, le=512.3292)
    Embarked: str = Field(...)

    @field_validator("Sex")
    @classmethod
    def validate_sex(cls, value):

        value = value.strip().lower()

        if value not in {"male", "female"}:
            raise ValueError("Sex must be 'male' or 'female'.")

        return value

    @field_validator("Embarked")
    @classmethod
    def validate_embarked(cls, value):

        value = value.strip().upper()

        if value not in {"C", "Q", "S"}:
            raise ValueError("Embarked must be C, Q or S.")

        return value

    @field_validator("Age", "Fare")
    @classmethod
    def validate_numbers(cls, value):

        if not math.isfinite(value):
            raise ValueError("Must be a finite number.")

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
# DRIFT METRICS ENDPOINT
# ============================================================

@app.get("/metrics/drift")
async def drift_metrics():

    return detect_drift()


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
async def predict(request: PassengerData):

    if model is None or pipeline is None:

        raise HTTPException(
            status_code=500,
            detail="Model or preprocessing pipeline not loaded."
        )

    input_data = request.model_dump()

    # ========================================================
    # Log incoming production request
    # ========================================================

    log_request(input_data)

    # ========================================================
    # OOD Boundary Check
    # ========================================================

    check_ood(input_data)

    # ========================================================
    # Convert to DataFrame
    # ========================================================

    input_df = pd.DataFrame([input_data])

    # ========================================================
    # Automatic preprocessing
    # ========================================================

    processed_data = pipeline.transform(input_df)

    # ========================================================
    # Prediction
    # ========================================================

    prediction = await run_in_threadpool(
        model.predict,
        processed_data
    )

    # ========================================================

    return PredictionResponse(
        survival_prediction=int(prediction[0])
    )
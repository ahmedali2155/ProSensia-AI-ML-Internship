
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool
import pandas as pd
import joblib
import math

app = FastAPI(
    title="Titanic ML Prediction API",
    description="Secure FastAPI Machine Learning Microservice",
    version="3.0"
)

# ============================================================
# GLOBAL VARIABLES
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
# OOD BOUNDARY INTERCEPTOR
# ============================================================

def check_ood(data: dict):
    """
    Reject requests outside the statistical boundaries
    of the Titanic training dataset before model inference.
    """

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

    @field_validator("Sex")
    @classmethod
    def validate_sex(cls, value: str):

        value = value.strip().lower()

        if value not in {"male", "female"}:
            raise ValueError("Sex must be either 'male' or 'female'.")

        return value

    @field_validator("Embarked")
    @classmethod
    def validate_embarked(cls, value: str):

        value = value.strip().upper()

        if value not in {"C", "Q", "S"}:
            raise ValueError("Embarked must be one of: C, Q, or S.")

        return value

    @field_validator("Age", "Fare")
    @classmethod
    def validate_finite_numbers(cls, value: float):

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

    if model is None or label_encoders is None:
        raise HTTPException(
            status_code=500,
            detail="Model is not loaded."
        )

    input_data = request.model_dump()

    # ========================================================
    # OOD Boundary Interceptor
    # ========================================================

    check_ood(input_data)

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
    # ========================================================

    prediction = await run_in_threadpool(
        model.predict,
        input_df
    )

    # ========================================================
    # RESPONSE
    # ========================================================

    return PredictionResponse(
        survival_prediction=int(prediction[0])
    )
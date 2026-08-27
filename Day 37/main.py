from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

import pandas as pd
import joblib
import math
import threading
import subprocess
import sys
import time

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from slowapi.middleware import SlowAPIMiddleware

from security import verify_api_key
from drift_detector import log_request, detect_drift

from router import (
    choose_model,
    log_latency,
    get_metrics
)

is_retraining = False

app = FastAPI(
    title="Titanic ML Prediction API",
    description="Secure Self-Healing FastAPI Machine Learning Microservice",
    version="6.0"
)

# ============================================================
# RATE LIMITER
# ============================================================

limiter = Limiter(key_func=get_remote_address)

app.state.limiter = limiter

app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)

app.add_middleware(SlowAPIMiddleware)

# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# GLOBAL VARIABLES
# ============================================================

champion_model = None
challenger_model = None
pipeline = None

model_lock = threading.Lock()

# ============================================================
# LOAD MODEL & PIPELINE
# ============================================================

@app.on_event("startup")
def startup():

    global champion_model, challenger_model, pipeline

    champion_model = joblib.load("model/model_v1.pkl")
    print("✅ Champion model loaded.")

    challenger_model = joblib.load("model/model_v2.pkl")
    print("✅ Challenger model loaded.")

    pipeline = joblib.load("model/pipeline.pkl")
    print("✅ Pipeline loaded.")

# ============================================================
# RETRAIN + HOT SWAP
# ============================================================

def retrain_and_reload():

    global challenger_model
    global is_retraining

    if is_retraining:
        return

    is_retraining = True

    try:

        print("Starting retraining...")

        subprocess.run(
            [sys.executable, "retrain.py"],
            check=True
        )

        with model_lock:
            challenger_model = joblib.load("model/model_v2.pkl")

        print("Hot swap complete.")

    finally:
        is_retraining = False

# ============================================================
# OOD CHECK
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
# DRIFT METRICS
# ============================================================

@app.get("/metrics/drift")
async def drift_metrics(
    request: Request,
    api_key: str = Depends(verify_api_key)
):

    return detect_drift()

# ============================================================
# A/B METRICS
# ============================================================

@app.get("/ab/metrics")
async def ab_metrics(
    request: Request,
    api_key: str = Depends(verify_api_key)
):

    return get_metrics()

# ============================================================
# PREDICTION
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
@limiter.limit("10/minute")
async def predict(
    request: Request,
    passenger: PassengerData,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):

    if (
        champion_model is None
        or challenger_model is None
        or pipeline is None
    ):

        raise HTTPException(
            status_code=500,
            detail="Model or preprocessing pipeline not loaded."
        )

    # Convert Pydantic model to dictionary
    input_data = passenger.model_dump()

    # Log production request
    log_request(input_data)

    # OOD validation
    check_ood(input_data)

    # Check drift
    drift = detect_drift()

    if drift["drift_detected"]:
        background_tasks.add_task(retrain_and_reload)

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Automatic preprocessing
    processed_data = pipeline.transform(input_df)

    # ========================================================
    # A/B Routing
    # ========================================================

    selected_model = choose_model()

    start_time = time.perf_counter()

    with model_lock:

        if selected_model == "champion":

            prediction = await run_in_threadpool(
                champion_model.predict,
                processed_data
            )

        else:

            prediction = await run_in_threadpool(
                challenger_model.predict,
                processed_data
            )

    latency = time.perf_counter() - start_time

    log_latency(
        selected_model,
        latency
    )

    return PredictionResponse(
        survival_prediction=int(prediction[0])
    )
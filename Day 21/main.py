from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

model = joblib.load("best_titanic_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

app = FastAPI()


class PassengerData(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str


@app.get("/health-check")
def health_check():
    return {
        "status": "API is live"
    }


@app.post("/predict")
def predict(request: PassengerData):

    input_data = request.model_dump()
    # -----------------------------
    # Out-of-Distribution (OOD) Validation
    # -----------------------------

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
    input_df = pd.DataFrame([input_data])

    # Predict
    prediction = model.predict(input_df)

    return {
        "survival_prediction": int(prediction[0])
    }
   
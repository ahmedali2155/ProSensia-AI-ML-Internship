from fastapi import FastAPI
from pydantic import BaseModel
import joblib

model = joblib.load("best_loan_prediction_model.pkl")

app = FastAPI()


class PredictionRequest(BaseModel):
    data: dict


@app.get("/health-check")
def health_check():
    return {"status": "API is live"}


@app.post("/predict")
def predict(request: PredictionRequest):

    print(request.data)

    return {
        "message": "Data received successfully",
        "received_data": request.data
    }
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

model = joblib.load("best_loan_prediction_model.pkl")
encoders = joblib.load("label_encoders.pkl")

app = FastAPI()


class LoanApplication(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str


@app.get("/health-check")
def health_check():
    return {"status": "API is live"}


@app.post("/predict")
def predict(request: LoanApplication):

    input_data = request.model_dump()

    categorical_columns = [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "Property_Area"
    ]

    for col in categorical_columns:
        input_data[col] = encoders[col].transform([input_data[col]])[0]

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)

    return {
        "loan_prediction": int(prediction[0])
    }
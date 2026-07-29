from fastapi import FastAPI, HTTPException
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

# Out-of-Distribution (OOD) Validation
# These limits are based on the training dataset.
# If any value is outside this range, reject the request.

    if not (150 <= input_data["ApplicantIncome"] <= 81000):
     raise HTTPException(
        status_code=400,
        detail="Data Out of Bounds: ApplicantIncome is outside the training distribution."
    )

    if not (0 <= input_data["CoapplicantIncome"] <= 41667):
      raise HTTPException(
        status_code=400,
        detail="Data Out of Bounds: CoapplicantIncome is outside the training distribution."
    )

    if not (9 <= input_data["LoanAmount"] <= 700):
     raise HTTPException(
        status_code=400,
        detail="Data Out of Bounds: LoanAmount is outside the training distribution."
    )

    if not (12 <= input_data["Loan_Amount_Term"] <= 480):
      raise HTTPException(
        status_code=400,
        detail="Data Out of Bounds: Loan_Amount_Term is outside the training distribution."
    )

    if not (0 <= input_data["Credit_History"] <= 1):
      raise HTTPException(
        status_code=400,
        detail="Data Out of Bounds: Credit_History is outside the training distribution."
    )

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)

    return {
    "loan_prediction": int(prediction[0])
}
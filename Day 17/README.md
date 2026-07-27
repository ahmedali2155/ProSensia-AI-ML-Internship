# Day 17 – FastAPI Loan Prediction API

## Overview

This project deploys a Machine Learning Loan Default Prediction model using FastAPI.

The API accepts loan applicant details, validates the input using Pydantic, converts categorical values into numerical values using saved Label Encoders, and predicts whether the loan will be approved.

---

## Technologies Used

- Python
- FastAPI
- Pydantic
- Pandas
- Scikit-learn
- Joblib
- Uvicorn

---

## Files

- main.py
- best_loan_prediction_model.pkl
- label_encoders.pkl
- requirements.txt

---

## API Endpoints

### GET /

Returns a welcome message.

### GET /health-check

Checks whether the API is running.

### POST /predict

Predicts the loan approval status.

Example Request:

```json
{
  "Gender": "Male",
  "Married": "Yes",
  "Dependents": "0",
  "Education": "Graduate",
  "Self_Employed": "No",
  "ApplicantIncome": 5000,
  "CoapplicantIncome": 2000,
  "LoanAmount": 120,
  "Loan_Amount_Term": 360,
  "Credit_History": 1,
  "Property_Area": "Urban"
}
```

Example Response:

```json
{
  "loan_prediction": 1
}
```
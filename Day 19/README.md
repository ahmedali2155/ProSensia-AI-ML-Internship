# Day 18 – FastAPI OOD Guardrails & Postman Testing

## Overview

This project extends the Day 17 Loan Prediction API by implementing Out-of-Distribution (OOD) Guardrails. The API now validates numerical inputs against the ranges used during model training before generating predictions.

## Features

- FastAPI REST API
- Pydantic request validation
- Machine Learning Loan Prediction
- Label Encoding for categorical features
- Out-of-Distribution (OOD) validation
- Custom HTTP 400 Bad Request errors
- Postman API testing
- Swagger UI testing

## Project Files

```
main.py
best_loan_prediction_model.pkl
label_encoders.pkl
requirements.txt
README.md
```

## API Endpoints

### GET /health-check

Returns API status.

### POST /predict

Predicts whether a loan will be approved.

## OOD Validation

The API checks whether numerical values fall within the same ranges as the training dataset.

| Feature | Min | Max |
|---------|----:|----:|
| ApplicantIncome | 150 | 81000 |
| CoapplicantIncome | 0 | 41667 |
| LoanAmount | 9 | 700 |
| Loan_Amount_Term | 12 | 480 |
| Credit_History | 0 | 1 |

If any value is outside these limits, the API returns:

```

400 Bad Request

```

instead of sending invalid data to the Machine Learning model.

## Testing

The API was tested using:

- Swagger UI
- Postman

Valid requests returned:

```

200 OK

```

Invalid requests returned:

```

400 Bad Request

```

using FastAPI HTTPException.

## Technologies

- Python
- FastAPI
- Pydantic
- Scikit-learn
- Pandas
- Joblib
- Uvicorn
- Postman
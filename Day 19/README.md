# Day 19 – FastAPI Response Models

## Overview

This project extends the Day 18 Loan Prediction API by implementing FastAPI Response Models using Pydantic. The API now returns predictions in a standardized format with a confidence score while maintaining input validation and Out-of-Distribution (OOD) guardrails.

## Features

- FastAPI REST API
- Pydantic Request Validation
- Pydantic Response Models
- Machine Learning Loan Prediction
- Label Encoding
- OOD Guardrails
- Confidence Score using predict_proba()
- Swagger UI Documentation
- Postman API Testing

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

Predicts whether a loan application will be approved.

## Response Format

```json
{
    "prediction": 1,
    "confidence_score": 0.7733
}
```

## OOD Validation

The API validates numerical inputs against the ranges used during model training before making predictions.

## Testing

The API was successfully tested using:

- Swagger UI
- Postman

## Technologies

- Python
- FastAPI
- Pydantic
- Pandas
- Scikit-learn
- Joblib
- Uvicorn
- Postman
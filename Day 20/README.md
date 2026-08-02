# Titanic Survival Prediction AI Microservice

## Project Overview

This project is a production-ready AI Microservice built using FastAPI and Machine Learning. It predicts whether a passenger would survive the Titanic disaster based on passenger information provided through a REST API.

---

## Features

- FastAPI REST API
- Machine Learning Prediction
- Pydantic Request Validation
- Out-of-Distribution (OOD) Guardrails
- Swagger UI Documentation
- Health Check Endpoint
- Prediction Endpoint
- HTTP 400 Error Handling
- HTTP 422 Validation Handling

---

## Dataset

Titanic Dataset

---

## Machine Learning Workflow

- Data Cleaning
- Missing Value Handling
- Label Encoding
- Train-Test Split
- Logistic Regression Model
- Model Evaluation
- Model Serialization using Joblib

---

## Model Performance

- Model: Logistic Regression
- Accuracy: **81.01%**

---

## API Endpoints

### Health Check

```
GET /health-check
```

Response

```json
{
  "status": "API is live"
}
```

---

### Predict Survival

```
POST /predict
```

Example Request

```json
{
  "Pclass": 3,
  "Sex": "male",
  "Age": 22,
  "SibSp": 1,
  "Parch": 0,
  "Fare": 7.25,
  "Embarked": "S"
}
```

Example Response

```json
{
  "survival_prediction": 0
}
```

---

## OOD Guardrails

The API validates incoming numerical values before making predictions.

If values fall outside the training dataset distribution, the API returns:

```
HTTP 400 Bad Request
```

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

## Project Structure

```
Day 20/
│── best_titanic_model.pkl
│── label_encoders.pkl
│── titanic.csv
│── train_model.py
│── main.py
│── requirements.txt
│── README.md
```

---

## Author

Ahmed Ali

BS Artificial Intelligence

University of Haripur
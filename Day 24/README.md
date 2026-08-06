# Titanic Survival Prediction - Docker Compose AI Microservice

## Project Overview

This project is a Dockerized Machine Learning API built with FastAPI and managed using Docker Compose. It predicts whether a Titanic passenger would survive based on passenger information.

---

## Project Structure

```
Day 22/
│── main.py
│── Dockerfile
│── docker-compose.yml
│── best_titanic_model.pkl
│── label_encoders.pkl
│── requirements.txt
└── README.md
```

---

## Build and Run

```bash
docker compose up --build
```

---

## Stop Containers

```bash
docker compose down
```

---

## Swagger UI

```
http://127.0.0.1:8000/docs
```

---

## API Endpoints

### Health Check

```
GET /health-check
```

### Prediction

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

---

## Services

- **FastAPI Service** – Hosts the Machine Learning prediction API.
- **Logger Service** – Demonstrates a supporting service in the Docker Compose setup.

---

## Networking

A custom Docker bridge network (`ai-network`) allows the services to communicate securely.

---

## Volumes

A named volume (`model-storage`) is used for persistent storage so data remains available even if containers are recreated.

---

## Technologies Used

- Python
- FastAPI
- Docker
- Docker Compose
- Pydantic
- Pandas
- Scikit-learn
- Joblib
- Uvicorn

## Performance Benchmark

API Endpoint Tested:
POST /predict

Tool Used:
PowerShell Measure-Command

Result:

Response Time:
420.06 ms

Target:
Below 500 ms

Status:
Passed

## Security Features

### Pydantic Field Validation

The API validates all incoming JSON requests using Pydantic Field constraints. Invalid data types, missing fields, invalid numeric ranges, and incorrect categorical values are rejected before reaching the Machine Learning model.

### Out-of-Distribution (OOD) Detection

The API checks whether incoming numerical values fall within the expected training data distribution. Requests outside these limits are rejected with HTTP 400 Bad Request.

### Prompt Injection Protection

Regex validation prevents unexpected or malicious text inputs such as SQL injection, JavaScript injection, and prompt injection attempts from reaching the model.

### Security Testing

The API was tested against:

- SQL Injection
- Script Injection
- Prompt Injection
- Invalid Numeric Values
- Missing Fields
- Invalid Categories

All invalid requests returned HTTP 400 or HTTP 422 without generating HTTP 500 Internal Server Errors.
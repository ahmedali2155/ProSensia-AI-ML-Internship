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
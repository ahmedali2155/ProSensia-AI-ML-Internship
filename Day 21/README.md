# Titanic Survival Prediction - Dockerized AI Microservice

## Project Overview

This project is a Dockerized Machine Learning API built with FastAPI. It predicts whether a Titanic passenger would survive based on passenger information.

---

## Project Structure

```
Day 21/
│── main.py
│── best_titanic_model.pkl
│── label_encoders.pkl
│── Dockerfile
│── requirements.txt
└── README.md
```

---

## Build the Docker Image

```bash
docker build -t titanic-api .
```

---

## Run the Docker Container

```bash
docker run -p 8000:8000 titanic-api
```

---

## Test the API

Swagger UI:

```
http://127.0.0.1:8000/docs
```

Health Check:

```
GET /health-check
```

Prediction Endpoint:

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

## Technologies Used

- Python
- FastAPI
- Docker
- Pydantic
- Pandas
- Scikit-learn
- Joblib
- Uvicorn

---

## Author

Ahmed Ali

BS Artificial Intelligence

University of Haripur
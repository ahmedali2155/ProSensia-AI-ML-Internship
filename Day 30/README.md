# Day 30 - FastAPI ML Microservice Integration Testing

## Project Overview

This project extends the Dockerized FastAPI Machine Learning microservice by adding automated integration testing using FastAPI TestClient and pytest. The goal is to verify API reliability, proper validation, error handling, and stable prediction responses before deployment.

---

# Project Structure

```
Day 30/
│── model/
│   ├── best_titanic_model.pkl
│   └── label_encoders.pkl
│
│── main.py
│── test_integration.py
│── Dockerfile
│── requirements.txt
│── README.md
│── .dockerignore
```

---

# Features

- FastAPI Machine Learning API
- Pydantic request validation
- Custom field validators
- Out-of-Distribution (OOD) boundary checking
- Asynchronous prediction endpoint
- Automated integration testing
- Dockerized deployment
- Structured API responses

---

# Running the FastAPI Application

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn main:app --reload
```

Open Swagger UI:

```
http://127.0.0.1:8000/docs
```

Health Check:

```
http://127.0.0.1:8000/health-check
```

---

# Running Integration Tests

Execute the automated integration tests using pytest:

```bash
python -m pytest -v
```

The test suite verifies:

- Health Check endpoint
- Valid prediction requests
- Invalid input validation (422)
- Out-of-Distribution input handling (400)
- Malformed request handling

Example Result:

```
5 passed
```

---

# Running with Docker

Build the Docker image:

```bash
docker build -t prosensia-ml-service:v3 .
```

Run the container:

```bash
docker run -p 8000:8000 prosensia-ml-service:v3
```

Open:

```
http://localhost:8000/docs
```

---

# Testing Strategy

The API was tested using FastAPI TestClient and pytest to ensure:

- Correct prediction responses
- Proper HTTP status codes
- Robust input validation
- Graceful handling of malformed requests
- Stable API behavior during integration testing

---

# API Endpoints

## Health Check

```
GET /health-check
```

Returns:

```json
{
  "status": "API is live"
}
```

---

## Prediction Endpoint

```
POST /predict
```

Example Request:

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

Example Response:

```json
{
  "survival_prediction": 0
}
```

---

# Integration Testing Results

- Health Check Passed
- Prediction Endpoint Passed
- Validation Tests Passed
- OOD Boundary Tests Passed
- Malformed Request Tests Passed

Overall Result:

```
5/5 Tests Passed
```

---

# Technologies Used

- Python
- FastAPI
- Pydantic
- Pandas
- Scikit-learn
- Joblib
- Pytest
- FastAPI TestClient
- Docker

---

# Learning Outcomes

- Implemented automated integration testing
- Verified API stability using pytest
- Improved API reliability with validation and error handling
- Tested Dockerized FastAPI services
- Built a production-ready Machine Learning API

---

## Author

**Ahmed Ali**

ProSensia AI/ML Internship – Week 6 Day 30
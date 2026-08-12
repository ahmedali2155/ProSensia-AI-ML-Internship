# Day 28 – OOD Boundary Interceptor for FastAPI ML Microservice

## Project Overview

This project extends the Titanic Machine Learning FastAPI microservice by implementing an Out-of-Distribution (OOD) Boundary Interceptor. Incoming requests are validated using Pydantic and checked against predefined statistical boundaries before reaching the Machine Learning model.

The objective is to prevent predictions on abnormal or unexpected input data, improving the reliability and security of the API.

---

## Features

- FastAPI REST API
- Model loaded once during application startup
- Async prediction endpoint
- Pydantic request and response models
- Custom field validators
- Statistical OOD Boundary Interceptor
- Structured HTTP error responses
- Dockerized application
- Health Check endpoint

---

## Project Structure

```
Day 28/
│
├── model/
│   ├── best_titanic_model.pkl
│   └── label_encoders.pkl
│
├── main.py
├── Dockerfile
├── requirements.txt
├── .dockerignore
└── README.md
```

---

## Validation Strategy

The API validates all incoming requests using Pydantic before processing.

Validation includes:

- Numeric value constraints
- String validation
- Custom field validators
- Finite numeric value checks
- Required field validation

Invalid requests automatically return HTTP 422 Unprocessable Entity.

---

## OOD Boundary Interceptor

Before sending data to the Machine Learning model, every request is checked against statistical boundaries based on the Titanic training dataset.

Example boundaries:

| Feature | Allowed Range |
|----------|---------------|
| Pclass | 1–3 |
| Age | 1–75 |
| SibSp | 0–5 |
| Parch | 0–5 |
| Fare | 0–300 |

If any feature falls outside these ranges, the request is rejected with:

- HTTP 400 Bad Request
- Structured "Data Out of Bounds" response

This prevents abnormal inputs from reaching the model.

---

## Asynchronous Processing

The prediction endpoint uses:

```python
run_in_threadpool()
```

This allows CPU-intensive prediction tasks to execute without blocking the FastAPI event loop, improving API responsiveness.

---

## API Endpoints

### Health Check

```
GET /health-check
```

Example Response:

```json
{
    "status": "API is live"
}
```

---

### Prediction

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

## Docker

### Build

```bash
docker build -t prosensia-ml-service:v3 .
```

### Run

```bash
docker run -p 8000:8000 prosensia-ml-service:v3
```

---

## Testing

The API was tested using Postman with:

- Valid requests
- Invalid data types
- Missing fields
- Invalid categorical values
- Out-of-Distribution (OOD) values

The application correctly returned:

- 200 OK
- 400 Bad Request
- 422 Unprocessable Entity

without crashing.

---

## Technologies Used

- Python 3.11
- FastAPI
- Pydantic
- Pandas
- Scikit-learn
- Joblib
- Docker
- Uvicorn

---

## Learning Outcomes

Through this project, I learned how to:

- Build secure Machine Learning APIs
- Validate user input using Pydantic
- Implement Out-of-Distribution protection
- Prevent invalid requests from reaching the ML model
- Use asynchronous inference in FastAPI
- Deploy ML APIs using Docker

---
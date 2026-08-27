# Day 36 – Multi-Worker Gunicorn Deployment & Microservice Memory Optimization

## Project Overview

This project upgrades the existing FastAPI Machine Learning microservice from a single-process Uvicorn deployment to a production-ready multi-worker Gunicorn architecture. The application serves a Titanic survival prediction model while supporting preprocessing pipelines, drift detection, automated retraining, model hot-swapping, A/B testing, and concurrent request handling.

---

# Features

- FastAPI Machine Learning API
- Scikit-Learn Pipeline & ColumnTransformer
- Serialized preprocessing (pipeline.pkl)
- Drift Detection using KS Test
- Automated Background Retraining
- Dynamic Model Hot-Swapping
- Champion vs Challenger A/B Testing
- Multi-Worker Gunicorn Deployment
- Docker Containerization
- Locust Load Testing
- Pytest Integration Tests

---

# Project Structure

```
Day 36/
│
├── model/
│   ├── best_titanic_model.pkl
│   ├── model_v1.pkl
│   ├── model_v2.pkl
│   ├── pipeline.pkl
│   └── titanic.csv
│
├── screenshot/
│
├── main.py
├── preprocessing.py
├── drift_detector.py
├── retrain.py
├── router.py
├── gunicorn_conf.py
├── locustfile.py
├── train_model.py
├── test_integration.py
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# Gunicorn Multi-Worker Configuration

The application is deployed using Gunicorn with Uvicorn workers.

Configuration:

- Worker Class:
  - uvicorn.workers.UvicornWorker

- Worker Formula:

```
Workers = (2 × CPU Cores) + 1
```

This allows multiple worker processes to serve concurrent requests efficiently.

---

# Memory Optimization

The application uses:

- Serialized Joblib models
- Serialized preprocessing pipeline
- Gunicorn preload_app option
- Shared application loading strategy

These techniques reduce unnecessary model loading overhead across worker processes and improve memory efficiency.

---

# Docker Deployment

Build the Docker image:

```bash
docker build -t prosensia-ml-service:v8 .
```

Run the container:

```bash
docker run -d -p 8000:8000 --name day36-container prosensia-ml-service:v8
```

Verify:

```bash
docker ps
```

Monitor container resources:

```bash
docker stats
```

Stop the container gracefully:

```bash
docker stop day36-container
```

---

# API Endpoints

## Health Check

```
GET /health-check
```

---

## Prediction

```
POST /predict
```

Example Request

```json
{
    "Pclass": 1,
    "Sex": "male",
    "Age": 28,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 72.5,
    "Embarked": "S"
}
```

---

## Drift Monitoring

```
GET /metrics/drift
```

Returns:

- Drift Status
- Drifted Features
- KS Statistics
- p-values
- Production Sample Count

---

## A/B Metrics

```
GET /ab/metrics
```

Returns:

- Routing Strategy
- Champion Requests
- Challenger Requests
- Average Latency
- Total Requests

---

# Load Testing

Locust was used to simulate concurrent traffic.

Run:

```bash
locust -f locustfile.py
```

Configuration:

- Users: 100
- Spawn Rate: 10
- Host:
```

http://localhost:8000

```

The application successfully handled concurrent prediction requests without worker crashes.

---

# Testing

Run integration tests:

```bash
pytest -v
```

Tests include:

- Health Check
- Prediction Endpoint
- Invalid Request Handling
- Pydantic Validation
- Drift Endpoint
- A/B Metrics Endpoint
- Malformed Request Handling

---

# Technologies Used

- Python
- FastAPI
- Gunicorn
- Uvicorn
- Scikit-Learn
- Pandas
- NumPy
- SciPy
- Joblib
- Docker
- Locust
- Pytest

---

# Learning Outcomes

Through this project, I learned:

- Deploying FastAPI using Gunicorn and Uvicorn workers
- Configuring dynamic multi-worker architectures
- Understanding the worker formula (2 × CPU Cores + 1)
- Handling concurrent ML inference requests
- Monitoring container memory usage
- Performing load testing using Locust
- Deploying scalable Machine Learning microservices using Docker

---

# Author

Ahmed

AI & Machine Learning Internship – ProSensia
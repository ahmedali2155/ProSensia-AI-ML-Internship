# Week 7 – End-to-End Automated MLOps Pipeline

## Project Overview

This project demonstrates a production-oriented Machine Learning microservice built with FastAPI and Docker. The application predicts Titanic passenger survival while implementing a complete MLOps workflow, including automated preprocessing, data drift monitoring, background model retraining, zero-downtime model hot-swapping, and A/B deployment.

---

# Features

- FastAPI Machine Learning API
- Scikit-Learn Pipeline & ColumnTransformer
- Automated preprocessing
- Data validation using Pydantic
- Out-of-Distribution (OOD) detection
- Statistical Data Drift Detection (KS Test)
- Drift telemetry endpoint
- Background model retraining
- Model versioning
- Dynamic model hot-swapping
- Champion–Challenger A/B routing
- Docker containerization
- Automated integration testing using Pytest

---

# Project Structure

```
Day 35
│
├── model
│   ├── best_titanic_model.pkl
│   ├── model_v1.pkl
│   ├── model_v2.pkl
│   ├── pipeline.pkl
│   └── titanic.csv
│
├── screenshot
│
├── drift_detector.py
├── preprocessing.py
├── retrain.py
├── router.py
├── train_model.py
├── main.py
├── test_integration.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# Technologies Used

- Python
- FastAPI
- Scikit-Learn
- Pandas
- NumPy
- SciPy
- Joblib
- Docker
- Pytest
- Uvicorn

---

# MLOps Pipeline

## 1. Data Preprocessing

A Scikit-Learn Pipeline with a ColumnTransformer performs:

- Missing value imputation
- Numerical feature scaling
- Categorical feature encoding

The fitted preprocessing pipeline is serialized as:

```
pipeline.pkl
```

and reused during inference.

---

## 2. Prediction Workflow

Incoming JSON request

↓

Pydantic Validation

↓

OOD Boundary Check

↓

Pipeline Transformation

↓

Model Prediction

↓

JSON Response

---

## 3. Data Drift Detection

Incoming production requests are logged.

The Kolmogorov–Smirnov (KS) Test compares production feature distributions against the training dataset.

If:

```
p-value < 0.05
```

the feature is considered drifted.

Drift metrics are available through:

```
GET /metrics/drift
```

---

## 4. Automated Background Retraining

When significant drift is detected:

- BackgroundTasks trigger `retrain.py`
- Fresh data is loaded
- Pipeline preprocessing is applied
- The model is retrained
- F1-score is evaluated
- A new versioned model (`model_v2.pkl`) is generated

Retraining runs asynchronously without blocking prediction requests.

---

## 5. Zero-Downtime Model Hot-Swapping

After successful validation:

- The latest model is loaded into memory
- Thread-safe model replacement is performed
- API requests continue without restarting the server

---

## 6. Champion–Challenger A/B Deployment

Two models are maintained:

- Champion Model (Production)
- Challenger Model (Candidate)

Traffic is routed using an 80/20 A/B strategy.

Performance metrics include:

- Request distribution
- Average latency
- Routing statistics

Available at:

```
GET /ab/metrics
```

---

# API Endpoints

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | `/health-check` | Health check |
| POST | `/predict` | Survival prediction |
| GET | `/metrics/drift` | Drift telemetry |
| GET | `/ab/metrics` | A/B routing metrics |

---

# Docker

## Build

```bash
docker build -t prosensia-ml-service:v6 .
```

## Run

```bash
docker run -d -p 8000:8000 --name day35-container prosensia-ml-service:v6
```

## Verify

```bash
docker ps
```

---

# Automated Testing

Integration tests were written using Pytest to verify:

- Health endpoint
- Prediction endpoint
- Input validation
- Pydantic validation
- Drift detection
- A/B metrics
- Malformed request handling

Run:

```bash
python -m pytest -v
```

Result:

```
7 Passed
```

---

# Screenshots

The repository includes screenshots demonstrating:

- Successful prediction
- Drift detection
- A/B metrics
- Background retraining
- Docker container execution
- Clean container shutdown

---

# Results

The completed pipeline successfully demonstrates:

- Automated preprocessing
- Real-time statistical drift monitoring
- Background model retraining
- Zero-downtime model hot-swapping
- Champion–Challenger deployment
- Dockerized production-ready FastAPI service

---

# Future Improvements

- Persistent telemetry storage
- Prometheus and Grafana monitoring
- Kubernetes deployment
- CI/CD pipeline using GitHub Actions
- Model registry integration (MLflow)

---

# Author

**Ahmed**

AI & Machine Learning Intern

ProSensia
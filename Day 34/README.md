# Titanic ML Prediction API – Day 34

## Project Overview

This project extends the Titanic Machine Learning FastAPI microservice by implementing **Shadow Deployment and A/B Traffic Splitting** for safe production model validation.

The application serves predictions through FastAPI while comparing a Champion model and a Challenger model using controlled traffic routing. It also integrates preprocessing, data drift monitoring, automated retraining, and Docker deployment.

---

# Features

- FastAPI REST API
- Champion & Challenger Models
- 80/20 A/B Traffic Splitting
- Scikit-Learn Pipeline
- ColumnTransformer Preprocessing
- Automatic Data Preprocessing
- Data Drift Monitoring (KS Test)
- Background Model Retraining
- Dynamic Model Hot-Swapping
- Docker Deployment
- Request Logging
- Out-of-Distribution (OOD) Detection

---

# Project Structure

```
Day 34/
│
├── model/
│   ├── model_v1.pkl
│   ├── model_v2.pkl
│   ├── pipeline.pkl
│   └── titanic.csv
│
├── main.py
├── router.py
├── drift_detector.py
├── retrain.py
├── requirements.txt
├── Dockerfile
└── README.md
```

---

# Champion vs Challenger Models

## Champion Model

The Champion model is the stable production model that serves most user requests.

```
model_v1.pkl
```

## Challenger Model

The Challenger model is a newly retrained model used for evaluation before replacing the Champion model.

```
model_v2.pkl
```

---

# A/B Traffic Splitting

The API implements **80/20 A/B Testing**.

- 80% of requests are routed to the Champion model.
- 20% of requests are routed to the Challenger model.
- Request counts and latency metrics are recorded for both models.

This enables safe evaluation of the new model before production deployment.

---

# Shadow Deployment

The Challenger model can be evaluated alongside the Champion model without replacing the production model.

This approach allows safe comparison of prediction behavior while keeping production stable.

---

# Data Drift Monitoring

Incoming prediction requests are logged and compared against the original training dataset.

The application uses the **Kolmogorov-Smirnov (KS) Test** to detect feature distribution changes.

Features with a **p-value below 0.05** are flagged as drifted.

The monitoring endpoint is available at:

```
GET /metrics/drift
```

---

# A/B Metrics Endpoint

The API exposes routing statistics through:

```
GET /ab/metrics
```

Example metrics include:

- Routing strategy
- Total requests
- Champion requests
- Challenger requests
- Average latency
- Traffic distribution

---

# Automated Retraining

When significant drift is detected:

- Background retraining starts automatically.
- A new model is trained.
- The model is evaluated using the F1-score.
- The new model is saved as:

```
model_v2.pkl
```

After validation, the model is hot-swapped into memory without restarting the API.

---

# Preprocessing Pipeline

The project uses:

- Scikit-Learn Pipeline
- ColumnTransformer
- SimpleImputer
- StandardScaler
- OneHotEncoder

The pipeline is serialized as:

```
pipeline.pkl
```

Incoming JSON requests are automatically transformed before prediction.

---

# Docker

## Build Image

```bash
docker build -t prosensia-ml-service:v6 .
```

## Run Container

```bash
docker run -d -p 8000:8000 --name day34-container prosensia-ml-service:v6
```

## Verify

```bash
docker ps
```

---

# API Endpoints

## Health Check

```
GET /health-check
```

## Prediction

```
POST /predict
```

## Drift Monitoring

```
GET /metrics/drift
```

## A/B Metrics

```
GET /ab/metrics
```

---

# Technologies Used

- Python
- FastAPI
- Scikit-Learn
- Pandas
- NumPy
- Joblib
- SciPy
- Docker
- Uvicorn
- Pydantic

---

# Testing

The application was verified by:

- Testing prediction endpoint
- Running Champion and Challenger models
- Verifying A/B routing
- Monitoring request latency
- Detecting data drift
- Running automated retraining
- Testing hot-swapping
- Deploying inside Docker

---

# Version

Day 34 – Shadow Deployment & A/B Traffic Splitting
# Day 32 - Real-Time Data Drift Monitoring & Telemetry

## Overview

This project extends the existing Dockerized FastAPI Machine Learning microservice by adding real-time Data Drift Monitoring and Telemetry.

Incoming prediction requests are logged, compared with the original training dataset using the Kolmogorov-Smirnov (KS) Test, and exposed through a dedicated monitoring endpoint.

The objective is to detect changes in production data before they negatively impact model performance.

---

## Features

- FastAPI Machine Learning Microservice
- Pydantic Request Validation
- Out-of-Distribution (OOD) Boundary Checks
- Serialized Scikit-Learn Preprocessing Pipeline
- Real-Time Inference Logging
- Statistical Data Drift Detection
- KS Test-based Feature Comparison
- Dedicated `/metrics/drift` Monitoring Endpoint
- Dockerized Deployment

---

## Project Structure

```
Day 32/
│
├── main.py
├── drift_detector.py
├── requirements.txt
├── Dockerfile
├── README.md
├── model/
│   ├── best_titanic_model.pkl
│   └── pipeline.pkl
└── screenshots/
```

---

## How Drift Monitoring Works

1. The FastAPI service loads:
   - Trained ML model
   - Serialized preprocessing pipeline

2. Every prediction request is logged in an in-memory buffer.

3. The Drift Detector compares incoming production data with the original training dataset.

4. The Kolmogorov-Smirnov (KS) Test is applied to numerical features.

5. If the p-value is less than **0.05**, the feature is flagged as drifted.

6. The monitoring results are available through:

```
GET /metrics/drift
```

---

## API Endpoints

### Health Check

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

Example Response

```json
{
  "survival_prediction": 0
}
```

---

### Drift Monitoring

```
GET /metrics/drift
```

Example Response

```json
{
  "drift_detected": true,
  "drifted_features": [
    "Age",
    "Fare"
  ],
  "statistics": {
    "Age": {
      "ks_statistic": 0.75,
      "p_value": 0.00
    }
  },
  "production_samples": 64
}
```

---

## Docker Commands

### Build

```bash
docker build -t prosensia-ml-service:v5 .
```

### Run

```bash
docker run -d -p 8000:8000 --name day32-container prosensia-ml-service:v5
```

### Verify Running Container

```bash
docker ps
```

---

## Testing

The application was tested using Postman.

Tests performed:

- Health Check Endpoint
- Valid Prediction Request
- Invalid Input Validation
- OOD Boundary Validation
- Drift Monitoring Endpoint
- Docker Deployment Verification

Approximately 50+ prediction requests with intentionally shifted values were sent to simulate production drift.

The `/metrics/drift` endpoint successfully detected feature distribution drift using the KS Test.

---

## Dependencies

- FastAPI
- Uvicorn
- Scikit-Learn
- Pandas
- NumPy
- SciPy
- Joblib
- Pydantic

---

## Key Learning Outcomes

- Implemented real-time telemetry for ML inference.
- Applied statistical drift detection using the Kolmogorov-Smirnov Test.
- Logged production inference requests.
- Exposed monitoring metrics through a dedicated API endpoint.
- Integrated drift monitoring into a Dockerized FastAPI ML service.
- Improved production readiness by continuously monitoring incoming data quality.

---

## Author

Ahmed Ali

AI & ML Internship – ProSensia
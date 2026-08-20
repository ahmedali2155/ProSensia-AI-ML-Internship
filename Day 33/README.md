# Day 33 - Automated Self-Healing ML Pipeline

## Project Overview

This project extends the Titanic Survival Prediction FastAPI microservice by implementing an automated self-healing Machine Learning pipeline. The application detects data drift, automatically retrains the model in the background, creates versioned model artifacts, and hot-swaps the updated model without restarting the API.

---

## Features

- FastAPI ML microservice
- Dockerized deployment
- Automated preprocessing using Scikit-Learn Pipeline
- Data drift detection using the Kolmogorov-Smirnov (KS) Test
- Background retraining using FastAPI BackgroundTasks
- Automatic model versioning
- Zero-downtime model hot-swapping
- Pydantic request validation
- Out-of-distribution (OOD) input validation
- Health check endpoint

---

## Project Structure

```
Day 33/
│
├── main.py
├── retrain.py
├── drift_detector.py
├── pipeline.py
├── train_model.py
├── requirements.txt
├── Dockerfile
├── README.md
│
├── model/
│   ├── best_titanic_model.pkl
│   ├── model_v2.pkl
│   └── pipeline.pkl
│
└── screenshots/
```

---

## API Endpoints

### Health Check

```
GET /health-check
```

Returns the API status.

---

### Prediction

```
POST /predict
```

Accepts raw passenger information, preprocesses the input automatically, performs prediction, logs inference requests, detects drift, and triggers background retraining when required.

---

### Drift Monitoring

```
GET /metrics/drift
```

Returns:

- Drift status
- Drifted features
- KS statistics
- p-values
- Production sample count

---

## Automated Retraining Pipeline

When significant statistical drift is detected:

1. BackgroundTasks starts retraining.
2. retrain.py loads new data.
3. Existing preprocessing pipeline is applied.
4. A new model is trained.
5. F1-score is evaluated.
6. The new model is saved as:

```
model/model_v2.pkl
```

---

## Dynamic Model Hot-Swapping

After successful retraining:

- The latest model is loaded into memory.
- Thread locking ensures safe model replacement.
- Prediction requests continue without restarting FastAPI.

---

## Running the Project

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start FastAPI

```bash
uvicorn main:app --reload
```

---

## Docker

Build:

```bash
docker build -t prosensia-ml-service:v5 .
```

Run:

```bash
docker run -d -p 8000:8000 --name day33-container prosensia-ml-service:v5
```

Verify:

```bash
docker ps
```

---

## Testing

### Health Check

```
GET /health-check
```

### Prediction

```
POST /predict
```

### Drift Monitoring

```
GET /metrics/drift
```

Send multiple shifted prediction requests to simulate production drift and verify automatic retraining and hot-swapping.

---

## Technologies Used

- Python
- FastAPI
- Scikit-Learn
- Pandas
- NumPy
- Joblib
- SciPy
- Docker
- Uvicorn

---

## Observations

- Automated preprocessing ensures consistent feature transformation.
- Data drift is detected using the KS Test.
- Background retraining prevents API downtime.
- Model versioning improves maintainability.
- Dynamic hot-swapping enables continuous deployment of updated models.
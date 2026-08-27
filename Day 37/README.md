
# Day 37 – FastAPI ML Microservice Security (API Key Authentication, Rate Limiting & CORS)

## Project Overview

This project enhances the existing FastAPI Machine Learning microservice by implementing production-level security features. The application now protects inference endpoints using API Key Authentication, prevents abuse through Rate Limiting, and restricts unauthorized cross-origin requests using strict CORS policies.

The project continues to support all previously implemented MLOps capabilities including:

- FastAPI ML Microservice
- Docker Deployment
- ColumnTransformer Preprocessing
- Data Drift Detection
- Background Model Retraining
- Dynamic Model Hot-Swapping
- A/B Traffic Routing
- Multi-worker Gunicorn Deployment
- API Security

---

# Features

## Machine Learning

- Titanic Survival Prediction
- Scikit-learn Pipeline
- Automatic Feature Preprocessing
- Random Forest Classifier
- Joblib Model Serialization

---

## Security Features

### API Key Authentication

- Uses `fastapi.security.APIKeyHeader`
- API key stored securely in `.env`
- Environment variables loaded using `pydantic-settings`
- Protected endpoints require `X-API-Key`
- Unauthorized requests return **401 Unauthorized**

Example Header:

```

X-API-Key: prosensia-secret-key

```

---

### Rate Limiting

Implemented using **SlowAPI**.

Configuration:

- Maximum **10 requests per minute**
- IP-based limiting
- Prevents abuse of expensive ML inference endpoints

Exceeding the limit returns:

```

429 Too Many Requests

```

---

### CORS Protection

Configured using FastAPI CORSMiddleware.

Allowed Origins:

```

[http://localhost:3000](http://localhost:3000)
[http://127.0.0.1:3000](http://127.0.0.1:3000)

```

Wildcard origins (`*`) are intentionally disabled for better security.

---

# Previous MLOps Features

## Data Drift Detection

- KS Test
- Real-time production monitoring
- Drift metrics endpoint

```

GET /metrics/drift

```

---

## Background Retraining

When drift is detected:

- Background retraining starts
- New model is trained
- Model validation performed
- New model saved automatically

---

## Model Hot Swapping

- Thread-safe model loading
- Zero downtime deployment
- No server restart required

---

## A/B Deployment

Traffic Routing:

- 80% Champion Model
- 20% Challenger Model

Metrics Endpoint:

```

GET /ab/metrics

````

Shows:

- Request distribution
- Average latency
- Champion requests
- Challenger requests

---

## Docker Deployment

Build Image

```bash
docker build -t prosensia-ml-service:v9 .
````

Run Container

```bash
docker run -d -p 8000:8000 --env-file .env --name day37-container prosensia-ml-service:v9
```

Verify

```bash
docker ps
```

Stop

```bash
docker stop day37-container
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

Requires:

* Valid API Key

Rate Limited:

* 10 requests/minute

Example Header

```

X-API-Key: prosensia-secret-key

```

---

## Drift Monitoring

```

GET /metrics/drift

```

Returns:

* Drift status
* Drifted features
* KS statistics
* P-values

---

## A/B Metrics

```

GET /ab/metrics

```

Returns:

* Champion requests
* Challenger requests
* Average latency
* Traffic distribution

---

# Project Structure

```

Day 37/
│
├── model/
│   ├── best_titanic_model.pkl
│   ├── model_v1.pkl
│   ├── model_v2.pkl
│   ├── pipeline.pkl
│   └── titanic.csv
│
├── main.py
├── security.py
├── drift_detector.py
├── retrain.py
├── router.py
├── preprocessing.py
├── gunicorn_conf.py
├── locustfile.py
├── requirements.txt
├── Dockerfile
├── .env.example
├── README.md
└── screenshot/

```

---

# Installation

Clone Repository

```bash
git clone <repository-url>
```

Create Virtual Environment

```bash
python -m venv venv
```

Activate

Windows

```bash
venv\Scripts\activate
```

Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file.

Example:

```env
API_KEY=prosensia-secret-key
```

Never commit your `.env` file to GitHub.

---

# Running the Application

Development

```bash
uvicorn main:app --reload
```

Production

```bash
gunicorn -c gunicorn_conf.py main:app
```

---

# Testing

## Authentication Test

Without API Key

Expected Response

```
401 Unauthorized
```

---

With Valid API Key

Expected Response

```
200 OK
```

---

## Rate Limiting Test

Send approximately 15 rapid requests.

Expected Response

```
429 Too Many Requests
```

---

## Docker Verification

Build

```bash
docker build -t prosensia-ml-service:v9 .
```

Run

```bash
docker run -d -p 8000:8000 --env-file .env --name day37-container prosensia-ml-service:v9
```

Verify

```bash
docker ps
```

---

# Technologies Used

* Python 3.11
* FastAPI
* Scikit-learn
* Pandas
* NumPy
* Joblib
* SciPy
* Gunicorn
* Uvicorn
* SlowAPI
* Pydantic
* Pydantic Settings
* Python-dotenv
* Docker

---

# Learning Outcomes

This project demonstrates how to:

* Secure ML inference APIs using API Key Authentication.
* Store secrets safely with environment variables.
* Protect expensive ML endpoints using rate limiting.
* Configure strict CORS policies.
* Detect production data drift.
* Trigger automatic background retraining.
* Perform zero-downtime model hot-swapping.
* Implement Champion–Challenger A/B deployment.
* Deploy scalable ML microservices using Docker and Gunicorn.

---

# Author

**Ahmed**

**AI & ML Internship – ProSensia**

```


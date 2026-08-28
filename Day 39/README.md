# Day 38 - Prometheus Metrics Instrumentation & Grafana Observability Stack

## Project Overview

This project extends the existing secured FastAPI Machine Learning microservice by integrating production-grade observability using Prometheus and Grafana. The application now exposes real-time performance metrics, enabling continuous monitoring of inference activity, latency, throughput, and API health.

---

# Objectives

- Integrate Prometheus monitoring with FastAPI.
- Expose application metrics through the `/metrics` endpoint.
- Monitor prediction throughput and inference latency.
- Visualize metrics using Grafana dashboards.
- Deploy the complete monitoring stack using Docker Compose.

---

# Technologies Used

- Python
- FastAPI
- Gunicorn
- Uvicorn
- Scikit-Learn
- Pandas
- Joblib
- Docker
- Docker Compose
- Prometheus
- Grafana
- prometheus-fastapi-instrumentator
- prometheus-client

---

# Features

## Machine Learning API

- Titanic Survival Prediction
- Champion/Challenger Models
- Automated Preprocessing Pipeline
- Drift Detection
- Background Retraining
- Zero-Downtime Model Hot Swapping
- A/B Traffic Routing

---

## Security

- API Key Authentication
- Environment Variable Configuration
- Rate Limiting using SlowAPI
- CORS Policy Restriction

---

## Observability

Prometheus collects real-time application metrics through the `/metrics` endpoint.

Custom metrics include:

- model_predictions_total
- model_inference_latency_seconds

Additional FastAPI metrics include:

- HTTP Request Count
- Request Duration
- Response Status Codes
- Throughput

---

# Docker Compose Architecture

The project uses three containers:

- FastAPI ML Service
- Prometheus
- Grafana

Prometheus continuously scrapes metrics from the FastAPI application while Grafana visualizes them using dashboards.

---

# Grafana Dashboard

The dashboard displays:

- Prediction Throughput
- P95 Inference Latency
- Requests Per Second (RPS)
- API Error Rate
- HTTP Request Statistics

---

# Prometheus Metrics

Example custom metrics:

- model_predictions_total
- model_inference_latency_seconds

These metrics help monitor model usage and inference performance in production.

---

# API Endpoints

| Endpoint | Description |
|-----------|-------------|
| GET /health-check | Service health |
| POST /predict | ML prediction |
| GET /metrics | Prometheus metrics |
| GET /metrics/drift | Drift detection metrics |
| GET /ab/metrics | A/B routing metrics |

---

# Running the Project

## Build Docker Image

```bash
docker build -t prosensia-ml-service:v10 .
```

## Start Complete Monitoring Stack

```bash
docker compose up -d
```

## Verify Containers

```bash
docker compose ps
```

---

# Monitoring URLs

FastAPI

```
http://localhost:8000
```

Swagger

```
http://localhost:8000/docs
```

Prometheus

```
http://localhost:9090
```

Grafana

```
http://localhost:3000
```

---

# Testing

The system was verified by:

- Sending approximately 100 prediction requests.
- Confirming prediction metrics increased.
- Monitoring Prometheus target status.
- Visualizing live metrics in Grafana.
- Verifying Docker Compose deployment.

---

# Project Structure

```
Day 38/
│
├── model/
├── main.py
├── router.py
├── security.py
├── drift_detector.py
├── retrain.py
├── preprocessing.py
├── gunicorn_conf.py
├── Dockerfile
├── docker-compose.yml
├── prometheus.yml
├── requirements.txt
├── .env.example
├── README.md
```

---

# Key Learning Outcomes

- Production observability for ML systems.
- Prometheus metric instrumentation.
- Grafana dashboard creation.
- Counter vs Histogram metrics.
- P95 latency monitoring.
- Docker Compose multi-service deployment.
- Monitoring ML inference performance in real time.

---

# Conclusion

This project demonstrates how a production-ready Machine Learning microservice can be monitored using Prometheus and Grafana. By collecting real-time inference metrics, monitoring latency and throughput, and visualizing system behavior through dashboards, the application becomes significantly more reliable, observable, and maintainable for production environments.
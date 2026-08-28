# ProSensia AI/ML Internship

## Day 39 – CI/CD Pipeline for FastAPI Machine Learning Microservice

This repository contains my work completed during the ProSensia AI/ML Internship.

The Day 39 project implements a production-ready Machine Learning microservice using FastAPI with a complete CI/CD pipeline powered by GitHub Actions. Every push to the repository automatically validates the application through automated testing, builds a Docker image, and publishes it to GitHub Container Registry (GHCR).

---

## Features

- FastAPI Machine Learning API
- API Key Authentication
- Rate Limiting
- Input Validation
- Champion–Challenger Model Architecture
- Automatic Model Retraining
- Hot Model Swapping
- Model Drift Monitoring
- Prometheus Metrics
- Grafana Dashboard
- Docker Containerization
- GitHub Actions CI/CD Pipeline
- GitHub Container Registry (GHCR) Deployment

---

## Project Structure

```text
ProSensia-AI-ML-Internship
│
├── Day 39
│   ├── model/
│   ├── screenshot/
│   ├── .github/workflows/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── main.py
│   ├── router.py
│   ├── security.py
│   ├── test_pipeline.py
│   ├── requirements.txt
│   └── ...
```

---

## Tech Stack

- Python
- FastAPI
- Scikit-learn
- Pandas
- NumPy
- Uvicorn
- Gunicorn
- Docker
- Docker Compose
- Prometheus
- Grafana
- Pytest
- GitHub Actions
- GitHub Container Registry (GHCR)

---

## Automated Testing

The CI pipeline validates:

- Authentication (401 Unauthorized)
- Invalid Request Validation (422)
- Out-of-Bounds Input (400)
- Successful Prediction (200)

Tests are executed automatically using Pytest before Docker image creation.

---

## CI/CD Workflow

GitHub Actions automatically performs:

1. Checkout Repository
2. Setup Python Environment
3. Install Dependencies
4. Run Automated Tests
5. Build Docker Image
6. Push Docker Image to GitHub Container Registry (GHCR)

If any test fails, the Docker image is not published.

---

## Docker Image

Latest image:

```bash
docker pull ghcr.io/ahmedali2155/prosensia-ml-service:latest
```

---

## GitHub Actions

Every push to the `main` branch automatically triggers:

- Automated Testing
- Docker Build
- Container Registry Deployment

---

## Monitoring

The API exposes Prometheus metrics for:

- Request Count
- Response Time
- P95/P99 Latency
- Prediction Count
- Model Inference Latency

Metrics are visualized through Grafana dashboards.

---

## Repository

https://github.com/ahmedali2155/ProSensia-AI-ML-Internship

---

## Author

Ahmed Ali

BS Artificial Intelligence Student

Python | Machine Learning | FastAPI | Docker | MLOps | CI/CD
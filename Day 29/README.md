# Day 29 - FastAPI ML Microservice Load Testing with Locust

## Project Overview

This project extends the containerized FastAPI Machine Learning microservice by performing load testing using Locust. The objective is to evaluate API performance under concurrent requests and ensure the service meets production-level latency and reliability requirements.

---

## Project Structure

```
Day 29/
│── main.py
│── Dockerfile
│── requirements.txt
│── README.md
│── locustfile.py
│── benchmark_results.txt
│── locust_dashboard.png
│── .dockerignore
│
└── model/
    ├── best_titanic_model.pkl
    └── label_encoders.pkl
```

---

## Running the FastAPI Service

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

---

## Running with Docker

Build the Docker image:

```bash
docker build -t prosensia-ml-service:v4 .
```

Run the container:

```bash
docker run -p 8000:8000 prosensia-ml-service:v4
```

---

## Running the Locust Load Test

Start the FastAPI service first.

Run Locust:

```bash
locust -f locustfile.py
```

Open the Locust dashboard:

```
http://localhost:8089
```

Use the following configuration:

- Users: **50**
- Spawn Rate: **5 users/second**
- Host: **http://localhost:8000**

Click **Start Swarming** to begin the benchmark.

---

## Benchmark Methodology

The API was tested using Locust by simulating 50 concurrent users sending POST requests to the `/predict` endpoint with valid JSON payloads.

The benchmark measured:

- Requests Per Second (RPS)
- Average Response Time
- P95 Latency
- Failure Rate
- Overall API Stability

---

## Performance Results

| Metric | Result |
|--------|---------|
| Concurrent Users | 50 |
| Spawn Rate | 5 users/sec |
| Requests Per Second (RPS) | 23.9 |
| Average Response Time | 49.08 ms |
| P95 Latency | 53 ms |
| Failure Rate | 0% |

---

## Optimizations Applied

- Model loaded once during application startup.
- Asynchronous FastAPI endpoints.
- Non-blocking inference using `run_in_threadpool`.
- Input validation with Pydantic.
- Out-of-Distribution (OOD) boundary checking before model inference.
- Dockerized deployment for consistent execution.

---

## Conclusion

The FastAPI Machine Learning microservice successfully handled concurrent requests with zero failures while maintaining a P95 latency well below the required 500 ms threshold. The benchmark demonstrates that the service is stable, responsive, and suitable for production-style deployment scenarios.
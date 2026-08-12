# Day 27 - Advanced Validation & Async FastAPI ML API

## Project Overview

This project is a secure and containerized Machine Learning API built with FastAPI.

Day 27 focuses on:

- Advanced Pydantic validation
- Custom field validators
- Input boundary validation
- Graceful error handling
- Asynchronous FastAPI endpoints
- Non-blocking Machine Learning inference
- Docker-based deployment
- API testing with valid and invalid requests

The API uses a trained Titanic Machine Learning model to predict passenger survival.

---

## Project Structure

```text
Day 27/
│
├── main.py
├── model/
│   ├── best_titanic_model.pkl
│   └── label_encoders.pkl
│
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
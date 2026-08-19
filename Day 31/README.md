# Day 31 - Modular Preprocessing Pipeline with Scikit-Learn Pipeline & ColumnTransformer

## Project Overview

This project upgrades the Titanic Survival Prediction FastAPI microservice by integrating a reusable Scikit-Learn preprocessing pipeline. The preprocessing steps are serialized using `joblib` and automatically applied to incoming requests before prediction.

The implementation prevents data leakage by fitting preprocessing only on the training dataset and provides a production-ready preprocessing workflow for deployment with FastAPI and Docker.

---

# Features

- FastAPI Machine Learning REST API
- Scikit-Learn Pipeline
- ColumnTransformer
- Automatic preprocessing
- Missing value handling
- Numerical feature scaling
- Categorical feature encoding
- Data leakage prevention
- Serialized preprocessing pipeline (`pipeline.pkl`)
- Serialized trained model (`best_titanic_model.pkl`)
- Pydantic request validation
- Out-of-Distribution (OOD) boundary checking
- Asynchronous prediction endpoint
- Dockerized deployment
- Production-ready project structure

---

# Project Structure

```
Day 31/
│
├── model/
│   ├── best_titanic_model.pkl
│   └── pipeline.pkl
│
├── screenshot/
│
├── preprocessing.py
├── train_model.py
├── main.py
├── test_integration.py
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```

---

# Preprocessing Pipeline

The preprocessing pipeline is implemented using Scikit-Learn Pipeline and ColumnTransformer.

### Numerical Features

- Age
- SibSp
- Parch
- Fare

Applied preprocessing:

- SimpleImputer (Median)
- StandardScaler

---

### Categorical Features

- Pclass
- Sex
- Embarked

Applied preprocessing:

- SimpleImputer (Most Frequent)
- OneHotEncoder

---

# Data Leakage Prevention

To prevent data leakage:

- Dataset is split into training and testing sets.
- The preprocessing pipeline is fitted only on `X_train`.
- Validation and test data are transformed using the fitted pipeline.
- Incoming FastAPI requests are transformed using the serialized pipeline without retraining.

---

# Serialized Artifacts

The following files are generated after training:

- `model/pipeline.pkl`
- `model/best_titanic_model.pkl`

Both artifacts are loaded automatically when the FastAPI application starts.

---

# FastAPI Workflow

Incoming Request

↓

Pydantic Validation

↓

OOD Boundary Validation

↓

Convert JSON to Pandas DataFrame

↓

pipeline.transform()

↓

ML Model Prediction

↓

JSON Response

---

# API Endpoint

## Health Check

```
GET /health-check
```

Response

```json
{
  "status": "API is live"
}
```

---

## Prediction

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

# Running the Project

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Train the Model

```bash
python train_model.py
```

This generates:

- pipeline.pkl
- best_titanic_model.pkl

---

## Run FastAPI

```bash
uvicorn main:app --reload
```

Swagger UI

```
http://127.0.0.1:8000/docs
```

---

# Docker

## Build

```bash
docker build -t prosensia-ml-service:v4 .
```

## Run

```bash
docker run -d -p 8000:8000 --name day31-container prosensia-ml-service:v4
```

Verify

```bash
docker ps
```

---

# API Testing

The API was tested using Postman with:

- Valid request (200 OK)
- Invalid request (422 Unprocessable Entity)
- Out-of-Distribution request (400 Bad Request)

Screenshots are included in the `screenshot` folder.

---

# Technologies Used

- Python
- FastAPI
- Scikit-Learn
- Pandas
- NumPy
- Joblib
- Pydantic
- Uvicorn
- Docker
- Postman

---

# Key Learning Outcomes

- Built a reusable preprocessing pipeline.
- Implemented ColumnTransformer for mixed feature types.
- Prevented data leakage by fitting preprocessing only on training data.
- Serialized preprocessing using Joblib.
- Integrated preprocessing into a FastAPI ML microservice.
- Deployed the application using Docker.
- Tested the API with raw input data through Postman.

---

# Author

**Ahmed Ali**

ProSensia AI/ML Internship – Day 31
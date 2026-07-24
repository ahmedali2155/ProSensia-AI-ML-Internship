# Day 16 - FastAPI Model Deployment

## Objective
Deploy a trained machine learning model using FastAPI.

## Project Structure

```
Day 16/
│── main.py
│── best_loan_prediction_model.pkl
│── requirements.txt
│── README.md
```

## Features

- Loads trained loan prediction model
- Health check endpoint
- Placeholder prediction endpoint
- Interactive Swagger documentation

## API Endpoints

### GET /health-check

Returns API status.

Example Response

```json
{
  "status": "API is live"
}
```

### POST /predict

Accepts JSON data and returns the received data.

Example Request

```json
{
  "data": {
    "ApplicantIncome": 5000,
    "LoanAmount": 120,
    "Credit_History": 1
  }
}
```

Run the server

```bash
uvicorn main:app --reload
```

Open Swagger UI

```
http://127.0.0.1:8000/docs
```
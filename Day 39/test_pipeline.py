from fastapi.testclient import TestClient
from main import app
import os

API_KEY = os.getenv("API_KEY", "prosensia-secret-key")

HEADERS = {
    "X-API-Key": API_KEY
}

VALID_PAYLOAD = {
    "Pclass": 3,
    "Sex": "male",
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S",
}


def test_missing_api_key():
    """
    Should return 401 when API key is missing.
    """
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=VALID_PAYLOAD
        )

        assert response.status_code == 401


def test_invalid_input():
    """
    Invalid payload should return 422.
    """
    payload = VALID_PAYLOAD.copy()
    payload["Age"] = "invalid"

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            headers=HEADERS,
            json=payload
        )

        assert response.status_code == 422


def test_out_of_bounds():
    """
    Passes Pydantic but fails custom OOD validation.
    """
    payload = VALID_PAYLOAD.copy()
    payload["Age"] = 76

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            headers=HEADERS,
            json=payload
        )

        print(response.status_code)
        print(response.json())

        assert response.status_code == 400


def test_successful_prediction():
    """
    Valid payload should return prediction.
    """
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            headers=HEADERS,
            json=VALID_PAYLOAD
        )

        print(response.status_code)
        print(response.json())

        assert response.status_code == 200

        body = response.json()

        assert "survival_prediction" in body
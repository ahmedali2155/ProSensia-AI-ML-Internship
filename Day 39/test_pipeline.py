import os

from fastapi.testclient import TestClient

from main import app

API_KEY = os.getenv("API_KEY", "prosensia-secret-key")

HEADERS = {"X-API-Key": API_KEY}

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
    with TestClient(app) as client:
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 401


def test_invalid_input():
    payload = VALID_PAYLOAD.copy()
    payload["Age"] = "invalid"

    with TestClient(app) as client:
        response = client.post("/predict", headers=HEADERS, json=payload)
        assert response.status_code == 422


def test_out_of_bounds():
    payload = VALID_PAYLOAD.copy()
    payload["Age"] = 76

    with TestClient(app) as client:
        response = client.post("/predict", headers=HEADERS, json=payload)
        assert response.status_code == 400


def assert_prediction_response(body):
    assert body["survival_prediction"] in {0, 1}
    assert body["prediction"] in {"Survived", "Did Not Survive"}
    assert body["model"] in {"Champion", "Challenger"}
    assert isinstance(body["prediction_id"], str)
    assert isinstance(body["latency_ms"], (int, float))
    assert body["latency_ms"] >= 0
    assert isinstance(body["timestamp"], str)

    if body["confidence"] is not None:
        assert 0 <= body["confidence"] <= 1


def test_successful_prediction():
    with TestClient(app) as client:
        response = client.post("/predict", headers=HEADERS, json=VALID_PAYLOAD)

        assert response.status_code == 200
        assert_prediction_response(response.json())


def test_public_demo_prediction():
    with TestClient(app) as client:
        response = client.post("/demo/predict", json=VALID_PAYLOAD)

        assert response.status_code == 200

        body = response.json()
        assert_prediction_response(body)
        assert body["model"] == "Champion"

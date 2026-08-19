from fastapi.testclient import TestClient
from main import app


def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health-check")
        assert response.status_code == 200
        assert response.json() == {"status": "API is live"}


def test_valid_prediction():
    payload = {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "survival_prediction" in response.json()


def test_invalid_input_422():
    payload = {
        "Pclass": 3,
        "Sex": "invalid",
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_out_of_bounds_400():
    payload = {
        "Pclass": 3,
        "Sex": "male",
        "Age": 76,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 400


def test_malformed_request():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            data="This is not JSON",
            headers={"Content-Type": "application/json"}
        )

    assert response.status_code in [400, 422]
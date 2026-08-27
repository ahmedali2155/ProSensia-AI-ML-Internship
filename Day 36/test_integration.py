from fastapi.testclient import TestClient
from main import app
import pytest


# ============================================================
# TEST CLIENT
# ============================================================

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ============================================================
# HEALTH CHECK
# ============================================================

def test_health_check(client):

    response = client.get("/health-check")

    assert response.status_code == 200
    assert response.json()["status"] == "API is live"


# ============================================================
# VALID PREDICTION
# ============================================================

def test_valid_prediction(client):

    payload = {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "survival_prediction" in response.json()


# ============================================================
# INVALID INPUT
# ============================================================

def test_invalid_prediction(client):

    payload = {
        "Pclass": 3,
        "Sex": "unknown",
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


# ============================================================
# PYDANTIC VALIDATION
# ============================================================

def test_pydantic_validation(client):

    payload = {
        "Pclass": 3,
        "Sex": "male",
        "Age": 100,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
    }

    response = client.post("/predict", json=payload)

    # Pydantic rejects Age > 80 before reaching check_ood()
    assert response.status_code == 422


# ============================================================
# DRIFT METRICS
# ============================================================

def test_drift_endpoint(client):

    response = client.get("/metrics/drift")

    assert response.status_code == 200

    body = response.json()

    assert "drift_detected" in body

    if body["drift_detected"]:
        assert "drifted_features" in body
        assert "statistics" in body
        assert "production_samples" in body
    else:
        assert "message" in body


# ============================================================
# A/B METRICS
# ============================================================

def test_ab_metrics(client):

    response = client.get("/ab/metrics")

    assert response.status_code == 200

    body = response.json()

    assert "routing_strategy" in body
    assert "total_requests" in body
    assert "champion_requests" in body
    assert "challenger_requests" in body


# ============================================================
# MALFORMED REQUEST
# ============================================================

def test_malformed_request(client):

    response = client.post(
        "/predict",
        data="Not JSON",
        headers={
            "Content-Type": "application/json"
        }
    )

    assert response.status_code in [400, 422]
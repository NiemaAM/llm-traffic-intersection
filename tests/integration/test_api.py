"""
test_api.py
Integration tests for the FastAPI service (no real LLM calls).
Uses a mock LLM to verify API behavior without API key dependency.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi.testclient import TestClient

# ─── Mock LLM fixture ─────────────────────────────────────────────────────────

MOCK_RESPONSE = {
    "is_conflict": "yes",
    "number_of_conflicts": 1,
    "conflict_vehicles": [{"vehicle1_id": "V1001", "vehicle2_id": "V1002"}],
    "decisions": ["Potential conflict: Vehicle V1002 must yield to Vehicle V1001"],
    "priority_order": {"V1001": 1, "V1002": 2},
    "waiting_times": {"V1001": 0, "V1002": 3},
}


@pytest.fixture
def client():
    mock_llm = MagicMock()
    mock_llm.model = "gpt-4o-mini-mock"
    mock_llm.few_shot = True
    mock_llm.predict.return_value = MOCK_RESPONSE

    with patch("src.api.app.get_llm", return_value=mock_llm):
        from src.api.app import app

        with TestClient(app) as c:
            yield c


# ─── Health endpoint ──────────────────────────────────────────────────────────


class TestHealth:
    def test_health_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"

    def test_health_has_model_field(self, client):
        r = client.get("/health")
        assert "model" in r.json()

    def test_root_endpoint(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "message" in r.json()


# ─── Predict endpoint ─────────────────────────────────────────────────────────

VALID_SCENARIO = {
    "vehicles": [
        {
            "vehicle_id": "V1001",
            "lane": 1,
            "speed": 60.0,
            "distance_to_intersection": 50.0,
            "direction": "north",
            "destination": "A",
        },
        {
            "vehicle_id": "V1002",
            "lane": 1,
            "speed": 55.0,
            "distance_to_intersection": 45.0,
            "direction": "south",
            "destination": "C",
        },
    ]
}


class TestPredict:
    def test_predict_returns_200(self, client):
        r = client.post("/predict", json=VALID_SCENARIO)
        assert r.status_code == 200

    def test_predict_has_conflict_field(self, client):
        r = client.post("/predict", json=VALID_SCENARIO)
        data = r.json()
        assert "is_conflict" in data
        assert data["is_conflict"] in ("yes", "no")

    def test_predict_has_decisions(self, client):
        r = client.post("/predict", json=VALID_SCENARIO)
        assert "decisions" in r.json()

    def test_predict_has_latency(self, client):
        r = client.post("/predict", json=VALID_SCENARIO)
        assert "latency_ms" in r.json()

    def test_predict_requires_min_2_vehicles(self, client):
        bad = {"vehicles": [VALID_SCENARIO["vehicles"][0]]}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422  # validation error

    def test_predict_rejects_invalid_direction(self, client):
        bad = {
            "vehicles": [
                {**VALID_SCENARIO["vehicles"][0], "direction": "diagonal"},
                VALID_SCENARIO["vehicles"][1],
            ]
        }
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_predict_rejects_negative_speed(self, client):
        bad = {
            "vehicles": [
                {**VALID_SCENARIO["vehicles"][0], "speed": -10.0},
                VALID_SCENARIO["vehicles"][1],
            ]
        }
        r = client.post("/predict", json=bad)
        assert r.status_code == 422


# ─── Batch predict endpoint ───────────────────────────────────────────────────


class TestBatchPredict:
    def test_batch_predict_returns_200(self, client):
        r = client.post("/predict/batch", json=[VALID_SCENARIO, VALID_SCENARIO])
        assert r.status_code == 200

    def test_batch_result_count(self, client):
        r = client.post("/predict/batch", json=[VALID_SCENARIO, VALID_SCENARIO])
        data = r.json()
        assert data["count"] == 2
        assert len(data["results"]) == 2

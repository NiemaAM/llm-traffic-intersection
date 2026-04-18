"""
serving_pipeline.py  —  Milestone 5
--------------------------------------
ZenML pipeline: model packaging, API health check, and serving registration.

Steps
-----
  1. load_best_model       Read best model config from MLflow model-comparison run
  2. validate_api          POST a test request to the FastAPI service; verify response
  3. run_api_tests         Run the full pytest integration test suite against live API
  4. register_serving      Log the serving endpoint URL + model config to MLflow
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Annotated

import mlflow
from zenml import pipeline, step
from zenml.logger import get_logger

logger = get_logger(__name__)


def _mlflow_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def _api_url() -> str:
    return os.environ.get("API_URL", "http://localhost:8000")


# ─── Step 1: Load best model config ──────────────────────────────────────────

@step
def load_best_model() -> Annotated[dict, "model_config"]:
    """
    Read the best model configuration from the MLflow model-comparison run.
    Falls back to defaults if no comparison run exists yet.
    """
    mlflow.set_tracking_uri(_mlflow_uri())

    try:
        client  = mlflow.tracking.MlflowClient()
        exp     = client.get_experiment_by_name("traffic-intersection-llm")
        if exp:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string="tags.mlflow.runName = 'model-comparison'",
                order_by=["start_time DESC"],
                max_results=1,
            )
            if runs:
                run    = runs[0]
                config = {
                    "best_config":         run.data.params.get("best_config",         "few_shot"),
                    "best_f1":             float(run.data.params.get("best_f1",       0.0)),
                    "fine_tuned_model_id": run.data.params.get("fine_tuned_model_id", ""),
                    "model_name":          os.environ.get("MODEL_NAME", "gpt-4o-mini"),
                }
                logger.info(f"✅ Loaded model config from MLflow: best={config['best_config']}  F1={config['best_f1']:.4f}")
                return config
    except Exception as exc:
        logger.warning(f"⚠️  Could not read MLflow config ({exc}) — using defaults")

    # Default config
    config = {
        "best_config":         "few_shot",
        "best_f1":             0.0,
        "fine_tuned_model_id": os.environ.get("FINE_TUNED_MODEL_ID", ""),
        "model_name":          os.environ.get("MODEL_NAME", "gpt-4o-mini"),
    }
    logger.info(f"Using default config: {config}")
    return config


# ─── Step 2: Validate live API ────────────────────────────────────────────────

@step
def validate_api(
    model_config: dict,
    api_url: str = "",
) -> Annotated[dict, "api_health"]:
    """
    Send a test prediction request to the live FastAPI service.
    Verifies the endpoint is up, accepts requests, and returns valid JSON.
    """
    import requests

    url = api_url or _api_url()
    health_result = {"api_url": url, "health": False, "predict": False, "latency_ms": None}

    # 1. Health check
    try:
        r = requests.get(f"{url}/health", timeout=10)
        health_result["health"] = r.status_code == 200
        health_result["model_reported"] = r.json().get("model", "unknown") if r.ok else "unknown"
        logger.info(f"Health check: {'✅' if health_result['health'] else '❌'}  ({r.status_code})")
    except Exception as exc:
        logger.warning(f"Health check failed: {exc}")
        return health_result

    # 2. Test prediction
    test_scenario = {
        "vehicles": [
            {"vehicle_id": "TEST001", "lane": 1, "speed": 50,
             "distance_to_intersection": 100, "direction": "north", "destination": "A"},
            {"vehicle_id": "TEST002", "lane": 1, "speed": 50,
             "distance_to_intersection": 100, "direction": "south", "destination": "C"},
        ]
    }
    try:
        t0 = __import__("time").time()
        r  = requests.post(f"{url}/predict", json=test_scenario, timeout=30)
        latency_ms = (__import__("time").time() - t0) * 1000

        if r.status_code == 200:
            body = r.json()
            health_result["predict"]    = True
            health_result["latency_ms"] = round(latency_ms, 1)
            health_result["is_conflict"] = body.get("is_conflict", "unknown")
            logger.info(f"Predict test: ✅  latency={latency_ms:.0f}ms  conflict={body.get('is_conflict')}")
        else:
            logger.warning(f"Predict returned {r.status_code}: {r.text[:200]}")
    except Exception as exc:
        logger.warning(f"Predict test failed: {exc}")

    return health_result


# ─── Step 3: Run API integration tests ───────────────────────────────────────

@step
def run_api_tests(
    api_health: dict,
) -> Annotated[dict, "test_results"]:
    """
    Run the pytest integration test suite against the live API.
    Tests health endpoint, predict endpoint, input validation, and batch predict.
    """
    import subprocess

    if not api_health.get("health"):
        logger.warning("⚠️  API not healthy — skipping integration tests")
        return {"skipped": True, "reason": "API not reachable"}

    result = subprocess.run(
        ["python", "-m", "pytest",
         "tests/integration/test_api.py",
         "-v", "--tb=short", "--no-header",
         f"--base-url={api_health['api_url']}"],
        capture_output=True, text=True, timeout=120,
        env={**__import__("os").environ, "PYTHONPATH": "."},
    )

    passed  = result.stdout.count(" PASSED")
    failed  = result.stdout.count(" FAILED")
    errors  = result.stdout.count(" ERROR")
    total   = passed + failed + errors

    test_results = {
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "total":  total,
        "pass_rate": passed / total if total > 0 else 0.0,
        "output": result.stdout[-2000:],
    }

    status = "✅ PASSED" if failed == 0 and errors == 0 else "❌ FAILED"
    logger.info(f"Integration tests: {status}  ({passed}/{total} passed)")
    return test_results


# ─── Step 4: Register serving endpoint ───────────────────────────────────────

@step
def register_serving(
    model_config: dict,
    api_health:   dict,
    test_results: dict,
) -> None:
    """
    Log the serving configuration, API health, and test results to MLflow.
    Creates a 'serving-registration' run that documents the deployed endpoint.
    """
    mlflow.set_tracking_uri(_mlflow_uri())
    mlflow.set_experiment("traffic-intersection-llm")

    with mlflow.start_run(run_name="serving-registration"):
        # Model config
        mlflow.log_params({
            "serving_model":          model_config.get("model_name",          "gpt-4o-mini"),
            "serving_config":         model_config.get("best_config",         "few_shot"),
            "fine_tuned_model_id":    model_config.get("fine_tuned_model_id", "none"),
            "api_url":                api_health.get("api_url",               "unknown"),
        })

        # Health metrics
        mlflow.log_metrics({
            "api_healthy":     int(api_health.get("health",  False)),
            "predict_ok":      int(api_health.get("predict", False)),
            "latency_ms":      api_health.get("latency_ms", 0) or 0,
        })

        # Test results
        if not test_results.get("skipped"):
            mlflow.log_metrics({
                "integration_tests_passed":   test_results.get("passed",    0),
                "integration_tests_failed":   test_results.get("failed",    0),
                "integration_tests_total":    test_results.get("total",     0),
                "integration_pass_rate":      test_results.get("pass_rate", 0),
            })

    logger.info("✅ Serving registration logged to MLflow")
    logger.info(f"   API URL:    {api_health.get('api_url')}")
    logger.info(f"   Model:      {model_config.get('model_name')} ({model_config.get('best_config')})")
    logger.info(f"   Tests:      {test_results.get('passed', '?')}/{test_results.get('total', '?')} passed")


# ─── Pipeline definition ──────────────────────────────────────────────────────

@pipeline(name="m5_serving_pipeline", enable_cache=False)
def serving_pipeline(
    api_url: str = "",
):
    """
    Milestone 5 — Serving Pipeline

    Graph
    -----
    load_best_model ──► validate_api ──► run_api_tests ──► register_serving

    Verifies the live FastAPI service is working correctly,
    runs the integration test suite, and registers the serving
    configuration in MLflow for traceability.
    """
    model_config = load_best_model()
    api_health   = validate_api(model_config=model_config, api_url=api_url)
    test_results = run_api_tests(api_health=api_health)
    register_serving(
        model_config=model_config,
        api_health=api_health,
        test_results=test_results,
    )


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Milestone 5 — Serving Pipeline")
    parser.add_argument("--api-url", default="", help="API URL (default: from .env API_URL)")
    args = parser.parse_args()

    print("🚀 Starting Milestone 5 Serving Pipeline")
    print(f"   API URL: {args.api_url or os.environ.get('API_URL', 'http://localhost:8000')}")
    print()

    serving_pipeline(api_url=args.api_url)

"""
monitor.py
----------
Model monitoring: data drift, performance degradation, and continual learning triggers.
Milestone 6: Uses Evidently for drift, Prometheus for metrics, WhyLogs for logging.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Evidently drift monitoring ───────────────────────────────────────────────

def compute_data_drift(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
    output_path: str = "reports/drift_report.html",
) -> dict:
    """
    Compute data drift between reference and production data using Evidently.
    Falls back to a simple statistical drift check if Evidently is unavailable.
    """
    try:
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
        from evidently.report import Report

        report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
        report.run(reference_data=reference_df, current_data=production_df)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        report.save_html(output_path)
        print(f"📊 Drift report saved → {output_path}")

        result = report.as_dict()
        drift_detected = result["metrics"][0]["result"].get("dataset_drift", False)
        return {"drift_detected": drift_detected, "report_path": output_path}

    except ImportError:
        return _simple_drift_check(reference_df, production_df)


def _simple_drift_check(
    ref: pd.DataFrame,
    prod: pd.DataFrame,
    threshold: float = 0.1,
) -> dict:
    """
    Fallback drift detection using mean/std comparison.
    """
    numeric_cols = ref.select_dtypes(include=[np.number]).columns.tolist()
    drift_scores = {}

    for col in numeric_cols:
        if col in prod.columns:
            ref_mean = ref[col].mean()
            prod_mean = prod[col].mean()
            if ref_mean != 0:
                drift = abs(prod_mean - ref_mean) / abs(ref_mean)
                drift_scores[col] = drift

    drifted = {k: v for k, v in drift_scores.items() if v > threshold}
    return {
        "drift_detected": len(drifted) > 0,
        "drifted_features": drifted,
        "method": "simple_statistical",
    }


# ─── WhyLogs data logging ─────────────────────────────────────────────────────

def log_prediction(
    scenario: dict,
    prediction: dict,
    log_dir: str = "reports/whylogs",
) -> None:
    """
    Log a prediction using WhyLogs for ongoing profiling.
    Falls back to JSON logging if WhyLogs is unavailable.
    """
    try:
        import whylogs as why
        from whylogs.core.resolvers import STANDARD_RESOLVER

        row = {
            "num_vehicles": len(scenario.get("vehicles", [])),
            "is_conflict_pred": 1 if prediction.get("is_conflict") == "yes" else 0,
            "num_conflicts_pred": prediction.get("number_of_conflicts", 0),
        }
        result = why.log(row)
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        result.writer("local").option(base_dir=log_dir).write()

    except ImportError:
        _fallback_log(scenario, prediction, log_dir)


def _fallback_log(scenario: dict, prediction: dict, log_dir: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "num_vehicles": len(scenario.get("vehicles", [])),
        "prediction": prediction,
    }
    log_file = Path(log_dir) / "predictions_log.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ─── Prometheus metrics ───────────────────────────────────────────────────────

_prometheus_available = False
try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    _prometheus_available = True

    PREDICTION_COUNTER = Counter(
        "intersection_predictions_total",
        "Total predictions served",
        ["conflict_result"],
    )
    LATENCY_HISTOGRAM = Histogram(
        "intersection_prediction_latency_seconds",
        "Prediction latency in seconds",
    )
    DRIFT_GAUGE = Gauge(
        "intersection_drift_detected",
        "1 if data drift was detected in last check",
    )
    CONFLICT_RATE_GAUGE = Gauge(
        "intersection_conflict_rate",
        "Rolling conflict detection rate",
    )
except ImportError:
    pass


def record_prediction_metrics(
    conflict_detected: bool,
    latency_s: float,
) -> None:
    """Record prediction metrics to Prometheus if available."""
    if not _prometheus_available:
        return
    label = "conflict" if conflict_detected else "no_conflict"
    PREDICTION_COUNTER.labels(conflict_result=label).inc()
    LATENCY_HISTOGRAM.observe(latency_s)


def start_metrics_server(port: int = 9090) -> None:
    """Start Prometheus metrics HTTP server."""
    if not _prometheus_available:
        print("⚠️  prometheus_client not installed – metrics server not started")
        return
    start_http_server(port)
    print(f"📈 Prometheus metrics server started on :{port}")


# ─── A/B testing ──────────────────────────────────────────────────────────────

class ABTestRouter:
    """
    Simple A/B test router that splits traffic between two model variants.
    """

    def __init__(
        self,
        model_a_config: dict,
        model_b_config: dict,
        traffic_split: float = 0.5,
    ):
        self.model_a_config = model_a_config
        self.model_b_config = model_b_config
        self.traffic_split = traffic_split
        self._results_a = []
        self._results_b = []

    def route(self, request_id: str) -> str:
        """Returns 'A' or 'B' deterministically based on request_id."""
        import hashlib
        h = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        return "A" if (h % 100) < (self.traffic_split * 100) else "B"

    def get_config(self, request_id: str) -> dict:
        variant = self.route(request_id)
        return self.model_a_config if variant == "A" else self.model_b_config

    def record_outcome(self, request_id: str, correct: bool) -> None:
        variant = self.route(request_id)
        if variant == "A":
            self._results_a.append(int(correct))
        else:
            self._results_b.append(int(correct))

    def summary(self) -> dict:
        def acc(lst): return sum(lst) / len(lst) if lst else 0
        return {
            "model_a": {"n": len(self._results_a), "accuracy": acc(self._results_a)},
            "model_b": {"n": len(self._results_b), "accuracy": acc(self._results_b)},
        }


# ─── Continual learning trigger ───────────────────────────────────────────────

class ContinualLearningTrigger:
    """
    Monitors performance metrics and triggers retraining when thresholds are breached.
    Can be connected to Airflow or ZenML for pipeline orchestration.
    """

    def __init__(
        self,
        f1_threshold: float = 0.70,
        drift_trigger: bool = True,
        window_size: int = 100,
    ):
        self.f1_threshold = f1_threshold
        self.drift_trigger = drift_trigger
        self.window_size = window_size
        self._recent_preds: list[tuple[int, int]] = []

    def update(self, y_true: int, y_pred: int) -> None:
        self._recent_preds.append((y_true, y_pred))
        if len(self._recent_preds) > self.window_size:
            self._recent_preds.pop(0)

    def should_retrain(self, drift_detected: bool = False) -> tuple[bool, str]:
        if not self._recent_preds:
            return False, "No predictions yet"

        y_true = [x[0] for x in self._recent_preds]
        y_pred = [x[1] for x in self._recent_preds]

        from sklearn.metrics import f1_score
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if f1 < self.f1_threshold:
            return True, f"F1 dropped below threshold: {f1:.4f} < {self.f1_threshold}"
        if self.drift_trigger and drift_detected:
            return True, "Data drift detected"
        return False, f"Performance OK (F1={f1:.4f})"

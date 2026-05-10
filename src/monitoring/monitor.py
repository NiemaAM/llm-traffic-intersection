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
    reference_data: "pd.DataFrame",
    production_data: "pd.DataFrame",
    output_path: str = "reports/drift_report.html",
) -> dict:
    """
    Compute data drift between reference and production datasets.
    Uses evidently legacy Report API with scipy KS-test fallback.
    Compatible with Python 3.11 across all evidently versions.
    """
    import os
    import warnings

    import numpy as np
    import pandas as pd
    from scipy import stats

    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else "reports",
        exist_ok=True,
    )

    def _statistical_drift(ref: pd.DataFrame, prod: pd.DataFrame) -> dict:
        """KS-test drift detection — always works, no evidently needed."""
        numeric_cols = [
            c for c in ref.columns
            if ref[c].dtype in [np.float64, np.int64, float, int]
            and c in prod.columns
        ]
        drifted, drift_scores = [], {}
        for col in numeric_cols:
            try:
                _, pval = stats.ks_2samp(ref[col].dropna(), prod[col].dropna())
                drift_scores[col] = float(pval)
                if pval < 0.05:
                    drifted.append(col)
            except Exception:
                pass
        return {
            "drift_detected":   len(drifted) > 0,
            "drifted_columns":  drifted,
            "drift_scores":     drift_scores,
            "n_drifted":        len(drifted),
            "n_tested":         len(numeric_cols),
            "method":           "KS test (scipy)",
        }

    # ── Try evidently Report API (v0.4+) ─────────────────────────────────────
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from evidently.report import Report  # type: ignore
            from evidently.metric_preset import DataDriftPreset  # type: ignore

            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=reference_data, current_data=production_data)
            report.save_html(output_path)
            result = report.as_dict()

        metrics = result.get("metrics", [{}])[0].get("result", {})
        drift_share = float(metrics.get("share_of_drifted_columns", 0))
        n_drifted   = int(metrics.get("number_of_drifted_columns", 0))
        return {
            "drift_detected":       drift_share > 0.2,
            "drift_share":          drift_share,
            "n_drifted_columns":    n_drifted,
            "report_path":          output_path,
            "method":               "Evidently DataDriftPreset",
        }
    except Exception:
        pass

    # ── Fallback: pure scipy KS test + plain HTML report ─────────────────────
    stat_result = _statistical_drift(reference_data, production_data)

    html_rows = "".join(
        f"<tr><td>{c}</td><td>{v:.4f}</td>"
        f"<td style='color:{'red' if v < 0.05 else 'green'}'>"
        f"{'Yes' if v < 0.05 else 'No'}</td></tr>"
        for c, v in stat_result["drift_scores"].items()
    )
    html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<title>Data Drift Report</title>
<style>body{{font-family:sans-serif;padding:24px}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccc;padding:8px;text-align:left}}
th{{background:#f0f0f0}}</style></head><body>
<h2>Data Drift Report</h2>
<p><b>Method:</b> {stat_result["method"]}</p>
<p><b>Drift detected:</b> {stat_result["drift_detected"]}</p>
<p><b>Drifted columns:</b> {stat_result["n_drifted"]} / {stat_result["n_tested"]}</p>
<table><tr><th>Column</th><th>KS p-value</th><th>Drifted</th></tr>
{html_rows}</table></body></html>"""

    with open(output_path, "w") as f:
        f.write(html)

    stat_result["report_path"] = output_path
    return stat_result

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
        def acc(lst):
            return sum(lst) / len(lst) if lst else 0

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

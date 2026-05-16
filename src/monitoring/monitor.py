"""
monitor.py
----------
Production monitoring for the traffic-intersection LLM system.

Responsibilities
----------------
1. Structured prediction logging  → reports/monitoring/predictions.jsonl
2. Retraining trigger evaluation  → reports/monitoring/trigger_log.jsonl
   Triggers: FNR, recall, rule-agreement, JSON failure rate, data drift
3. Data drift detection            → reports/monitoring/drift_report.html
4. Prometheus metrics              → :9090/metrics

Ground truth
------------
No cameras are available. The rule-based engine (llm_model._build_full_decision,
which wraps conflict_detection_orig.py) is the authoritative reference — the
same engine used to label all training data. Ground truth is therefore derived
on-the-fly for every logged prediction at zero extra cost.
"""

import json
import sys
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]
MONITORING_DIR = REPO_ROOT / "reports" / "monitoring"
PREDICTIONS_LOG = MONITORING_DIR / "predictions.jsonl"
TRIGGER_LOG = MONITORING_DIR / "trigger_log.jsonl"

# ── Retraining thresholds ─────────────────────────────────────────────────────
# Any single breach is sufficient to trigger a retraining recommendation.

THRESHOLDS = {
    "fnr_max": 0.08,  # FNR > 8%  — missed conflicts are safety-critical
    "recall_min": 0.92,  # Recall < 92% — model misses real conflicts
    "json_failure_max": 0.02,  # > 2% invalid JSON — serving reliability
    "rule_agree_min": 0.75,  # < 75% rule agreement — right answer, wrong reason
    "window_size": 100,  # rolling window length (number of predictions)
}


# ── Rule-engine ground truth ──────────────────────────────────────────────────


def _rule_label(vehicles: list) -> Optional[str]:
    """
    Return the rule-based conflict label ('yes' | 'no') for a vehicle list.
    Uses the same engine that generated all training labels — no cameras needed.
    Returns None on import / runtime failure (treated as unknown ground truth).
    """
    try:
        sys.path.insert(0, str(REPO_ROOT / "src"))
        sys.path.insert(0, str(REPO_ROOT / "src" / "poc"))
        from models.llm_model import _build_full_decision

        ref = _build_full_decision(vehicles, None)
        return ref.get("is_conflict", "no")
    except Exception:
        return None


# ── Structured prediction logging ─────────────────────────────────────────────


def log_prediction_event(
    scenario_id: str,
    vehicles: list,
    prediction: dict,
    model_version: str = "fine_tuned_v1",
    prompt_version: str = "prompt_v1",
    latency_ms: float = 0.0,
    json_valid: bool = True,
    fallback_used: bool = False,
) -> dict:
    """
    Log one prediction as a structured JSONL event to reports/monitoring/predictions.jsonl.

    ground_truth    — rule-engine label ('yes'|'no'|null if engine unavailable)
    rule_agreement  — True when LLM prediction matches rule engine

    Returns the event dict that was written.
    """
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)

    gt = _rule_label(vehicles)
    pred_label = prediction.get("is_conflict", "no")
    rule_agreement = (pred_label == gt) if gt is not None else None

    event = {
        "scenario_id": scenario_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "model_version": model_version,
        "prompt_version": prompt_version,
        "vehicles": vehicles,
        "prediction": {
            "is_conflict": pred_label,
            "waiting_times": prediction.get("waiting_times", {}),
            "priority_order": prediction.get("priority_order", {}),
        },
        "latency_ms": round(latency_ms, 1),
        "json_valid": json_valid,
        "fallback_used": fallback_used,
        "ground_truth": gt,
        "rule_agreement": rule_agreement,
    }

    with open(PREDICTIONS_LOG, "a") as f:
        f.write(json.dumps(event) + "\n")

    _update_prometheus(pred_label == "yes", latency_ms / 1000.0)
    return event


def log_prediction(scenario: dict, prediction: dict, log_dir: str = "") -> None:
    """Backward-compatible shim — prefer log_prediction_event()."""
    vehicles = scenario.get("vehicles", [])
    sid = scenario.get("scenario_id", f"legacy_{datetime.now().timestamp():.0f}")
    log_prediction_event(
        scenario_id=sid,
        vehicles=vehicles,
        prediction=prediction,
        json_valid=isinstance(prediction, dict),
    )


# ── Prometheus metrics ────────────────────────────────────────────────────────

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
    FNR_GAUGE = Gauge(
        "intersection_fnr",
        "Rolling false negative rate (safety-critical)",
    )
    RECALL_GAUGE = Gauge(
        "intersection_recall",
        "Rolling recall",
    )
    CONFLICT_RATE_GAUGE = Gauge(
        "intersection_conflict_rate",
        "Rolling conflict detection rate",
    )
    JSON_FAILURE_GAUGE = Gauge(
        "intersection_json_failure_rate",
        "Rolling JSON failure rate",
    )
    RULE_AGREE_GAUGE = Gauge(
        "intersection_rule_agreement_rate",
        "Rolling rule-engine agreement rate",
    )
    DRIFT_GAUGE = Gauge(
        "intersection_drift_detected",
        "1 if data drift was detected in last check",
    )
    RETRAIN_TRIGGER_GAUGE = Gauge(
        "intersection_retrain_triggered",
        "1 if retraining is currently triggered, 0 otherwise",
    )
except ImportError:
    pass


def _update_prometheus(conflict: bool, latency_s: float) -> None:
    if not _prometheus_available:
        return
    PREDICTION_COUNTER.labels(conflict_result="conflict" if conflict else "no_conflict").inc()
    LATENCY_HISTOGRAM.observe(latency_s)


def record_prediction_metrics(conflict_detected: bool, latency_s: float) -> None:
    """Legacy helper kept for external callers."""
    _update_prometheus(conflict_detected, latency_s)


def start_metrics_server(port: int = 9090) -> None:
    if not _prometheus_available:
        print("prometheus_client not installed — metrics server not started")
        return
    start_http_server(port)
    print(f"Prometheus metrics server started on :{port}")


# ── Data drift detection ──────────────────────────────────────────────────────


def compute_data_drift(
    reference_data: pd.DataFrame,
    production_data: pd.DataFrame,
    output_path: str = "",
) -> dict:
    """
    KS-test drift detection between reference and production datasets.
    Saves an HTML report; tries Evidently first, falls back to scipy.

    Default output: reports/monitoring/drift_report.html
    """
    import os
    import warnings

    from scipy import stats

    if not output_path:
        output_path = str(MONITORING_DIR / "drift_report.html")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def _ks_drift(ref: pd.DataFrame, prod: pd.DataFrame) -> dict:
        numeric_cols = [
            c
            for c in ref.columns
            if ref[c].dtype in [np.float64, np.int64, float, int] and c in prod.columns
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
            "drift_detected": len(drifted) > 0,
            "drifted_columns": drifted,
            "drift_scores": drift_scores,
            "n_drifted": len(drifted),
            "n_tested": len(numeric_cols),
            "method": "KS test (scipy)",
        }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from evidently.metric_preset import DataDriftPreset  # type: ignore
            from evidently.report import Report  # type: ignore

            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=reference_data, current_data=production_data)
            report.save_html(output_path)
            result = report.as_dict()

        metrics = result.get("metrics", [{}])[0].get("result", {})
        drift_share = float(metrics.get("share_of_drifted_columns", 0))
        drift_detected = drift_share > 0.2
        if _prometheus_available:
            DRIFT_GAUGE.set(1 if drift_detected else 0)
        return {
            "drift_detected": drift_detected,
            "drift_share": drift_share,
            "n_drifted_columns": int(metrics.get("number_of_drifted_columns", 0)),
            "report_path": output_path,
            "method": "Evidently DataDriftPreset",
        }
    except Exception:
        pass

    stat = _ks_drift(reference_data, production_data)
    html_rows = "".join(
        f"<tr><td>{c}</td><td>{v:.4f}</td>"
        f"<td style='color:{'red' if v < 0.05 else 'green'}'>"
        f"{'Yes' if v < 0.05 else 'No'}</td></tr>"
        for c, v in stat["drift_scores"].items()
    )
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>Data Drift Report</title>"
        "<style>body{font-family:sans-serif;padding:24px}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #ccc;padding:8px;text-align:left}"
        "th{background:#f0f0f0}</style></head><body>"
        "<h2>Data Drift Report</h2>"
        f"<p><b>Method:</b> {stat['method']}</p>"
        f"<p><b>Drift detected:</b> {stat['drift_detected']}</p>"
        f"<p><b>Drifted columns:</b> {stat['n_drifted']} / {stat['n_tested']}</p>"
        "<table><tr><th>Column</th><th>KS p-value</th><th>Drifted</th></tr>"
        f"{html_rows}</table></body></html>"
    )
    with open(output_path, "w") as f:
        f.write(html)
    stat["report_path"] = output_path
    if _prometheus_available:
        DRIFT_GAUGE.set(1 if stat["drift_detected"] else 0)
    return stat


# ── Continual learning trigger ────────────────────────────────────────────────


class ContinualLearningTrigger:
    """
    Reads the persisted prediction log and evaluates whether any safety or
    quality threshold has been breached.

    Trigger conditions (any one fires a retraining recommendation):
      FNR > 0.08          Missed conflicts are safety-critical
      Recall < 0.92       Model misses real conflicts
      JSON failure > 2%   Serving reliability degraded
      Rule agreement < 75%  Model may be right for the wrong reason
      Drift detected       Input distribution shifted from training data

    State is read from JSONL on every call — survives server restarts.
    Each evaluation is appended to trigger_log.jsonl for audit trail.
    """

    def __init__(self, window_size: int = THRESHOLDS["window_size"]):
        self.window_size = window_size

    def _load_window(self) -> list[dict]:
        if not PREDICTIONS_LOG.exists():
            return []
        lines = PREDICTIONS_LOG.read_text().strip().splitlines()
        events = []
        for line in lines[-self.window_size :]:
            try:
                events.append(json.loads(line))
            except Exception:
                pass
        return events

    def evaluate(self, drift_detected: bool = False) -> dict:
        """
        Evaluate all triggers on the last `window_size` predictions.
        Appends result to trigger_log.jsonl and updates Prometheus.
        Returns a trigger report dict.
        """
        events = self._load_window()
        n = len(events)

        if n == 0:
            return {
                "should_retrain": False,
                "reason": "No predictions logged yet",
                "triggers": [],
                "metrics": {},
                "n_window": 0,
                "timestamp": datetime.now(UTC).isoformat(),
            }

        triggers: list[str] = []
        metrics: dict = {"n_window": n}

        # ── FNR and Recall (requires rule-engine ground truth) ─────────────────
        labeled = [e for e in events if e.get("ground_truth") is not None]
        if labeled:
            y_true = [1 if e["ground_truth"] == "yes" else 0 for e in labeled]
            y_pred = [1 if e["prediction"]["is_conflict"] == "yes" else 0 for e in labeled]
            tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
            fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
            recall = tp / (tp + fn + 1e-9)
            fnr = fn / (fn + tp + 1e-9)
            metrics.update(
                {"n_labeled": len(labeled), "recall": round(recall, 4), "fnr": round(fnr, 4)}
            )
            if fnr > THRESHOLDS["fnr_max"]:
                triggers.append(
                    f"FNR={fnr:.3f} > threshold {THRESHOLDS['fnr_max']} (safety-critical)"
                )
            if recall < THRESHOLDS["recall_min"]:
                triggers.append(f"Recall={recall:.3f} < threshold {THRESHOLDS['recall_min']}")

        # ── JSON failure rate ──────────────────────────────────────────────────
        json_failure_rate = sum(1 for e in events if not e.get("json_valid", True)) / n
        metrics["json_failure_rate"] = round(json_failure_rate, 4)
        if json_failure_rate > THRESHOLDS["json_failure_max"]:
            triggers.append(
                f"JSON failure rate={json_failure_rate:.3f} > threshold {THRESHOLDS['json_failure_max']}"
            )

        # ── Rule-engine agreement ──────────────────────────────────────────────
        agree_events = [e for e in events if e.get("rule_agreement") is not None]
        if agree_events:
            agree_rate = sum(1 for e in agree_events if e["rule_agreement"]) / len(agree_events)
            metrics["rule_agreement_rate"] = round(agree_rate, 4)
            if agree_rate < THRESHOLDS["rule_agree_min"]:
                triggers.append(
                    f"Rule agreement={agree_rate:.3f} < threshold {THRESHOLDS['rule_agree_min']}"
                )

        # ── Conflict rate (informational) ──────────────────────────────────────
        conflict_rate = sum(1 for e in events if e["prediction"]["is_conflict"] == "yes") / n
        metrics["conflict_rate"] = round(conflict_rate, 4)

        # ── Drift (passed in from drift detection step) ────────────────────────
        if drift_detected:
            triggers.append("Data drift detected in input features")

        should_retrain = len(triggers) > 0
        result = {
            "should_retrain": should_retrain,
            "triggers": triggers,
            "reason": "; ".join(triggers) if triggers else "All metrics within thresholds",
            "metrics": metrics,
            "thresholds": THRESHOLDS,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        MONITORING_DIR.mkdir(parents=True, exist_ok=True)
        with open(TRIGGER_LOG, "a") as f:
            f.write(json.dumps(result) + "\n")

        if _prometheus_available:
            RETRAIN_TRIGGER_GAUGE.set(1 if should_retrain else 0)
            CONFLICT_RATE_GAUGE.set(conflict_rate)
            if "fnr" in metrics:
                FNR_GAUGE.set(metrics["fnr"])
            if "recall" in metrics:
                RECALL_GAUGE.set(metrics["recall"])
            JSON_FAILURE_GAUGE.set(json_failure_rate)
            if "rule_agreement_rate" in metrics:
                RULE_AGREE_GAUGE.set(metrics["rule_agreement_rate"])

        return result

    def update(self, y_true: int, y_pred: int) -> None:
        """Deprecated — predictions are now persisted via log_prediction_event()."""
        pass

    def should_retrain(self, drift_detected: bool = False) -> tuple[bool, str]:
        """Deprecated shim — use evaluate() instead."""
        result = self.evaluate(drift_detected)
        return result["should_retrain"], result["reason"]


# ── A/B test router ───────────────────────────────────────────────────────────


class ABTestRouter:
    """
    Deterministic traffic router for A/B testing two model variants.
    Route is stable per request_id (MD5-based) so the same request
    always goes to the same variant.
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
        self._results_a: list[int] = []
        self._results_b: list[int] = []

    def route(self, request_id: str) -> str:
        import hashlib

        h = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        return "A" if (h % 100) < int(self.traffic_split * 100) else "B"

    def get_config(self, request_id: str) -> dict:
        return self.model_a_config if self.route(request_id) == "A" else self.model_b_config

    def record_outcome(self, request_id: str, correct: bool) -> None:
        if self.route(request_id) == "A":
            self._results_a.append(int(correct))
        else:
            self._results_b.append(int(correct))

    def summary(self) -> dict:
        def _acc(lst: list) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        return {
            "model_a": {"n": len(self._results_a), "accuracy": _acc(self._results_a)},
            "model_b": {"n": len(self._results_b), "accuracy": _acc(self._results_b)},
        }

"""
evaluate.py
-----------
Comprehensive model evaluation: accuracy, robustness, bias auditing.
Milestone 6: Testing beyond accuracy with adversarial and behavioral tests.
"""

import json
import os
import random
from pathlib import Path
from typing import Any

import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────

# Repo root = two levels up from src/evaluation/evaluate.py
REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "reports"


def _ensure_reports_dir() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Standard evaluation ──────────────────────────────────────────────────────


def evaluate_model_masri(
    test_csv: str,
    model,
    max_scenarios: int = 100,
) -> dict:
    """Evaluate on masri-format test_scenarios.csv (scenario JSON column)."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    df = pd.read_csv(test_csv)
    rows = df.sample(min(max_scenarios, len(df)), random_state=42).iterrows()
    y_true, y_pred = [], []
    for _, row in rows:
        try:
            scenario = json.loads(row["scenario"])
            vehicles = scenario.get("vehicles_scenario", [])
            true_label = 1 if str(row["is_conflict"]).strip().lower() == "yes" else 0
            result = model.predict({"vehicles": vehicles})
            pred_label = 1 if result.get("is_conflict") == "yes" else 0
            y_true.append(true_label)
            y_pred.append(pred_label)
        except Exception:
            pass
    if not y_true:
        return {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "fnr": 1.0,
            "n_evaluated": 0,
        }
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall,
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "fnr": float(1 - recall),
        "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n_evaluated": len(y_true),
    }


def evaluate_model(
    raw_csv: str,
    model,
    max_scenarios: int = 100,
) -> dict[str, float]:
    """Standard accuracy evaluation on a held-out test set."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    df = pd.read_csv(raw_csv)
    scenarios = list(df.groupby("scenario_id"))
    random.shuffle(scenarios)
    scenarios = scenarios[:max_scenarios]

    y_true, y_pred = [], []
    for scenario_id, group in scenarios:
        vehicles = group[
            [
                "vehicle_id",
                "lane",
                "speed",
                "distance_to_intersection",
                "direction",
                "destination",
            ]
        ].to_dict(orient="records")
        true_label = 1 if group.iloc[0]["is_conflict"] == "yes" else 0
        try:
            result = model.predict({"vehicles": vehicles})
            pred_label = 1 if result.get("is_conflict") == "yes" else 0
            y_true.append(true_label)
            y_pred.append(pred_label)
        except Exception:
            pass

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fnr": fn / (fn + tp + 1e-10),
        "fpr": fp / (fp + tn + 1e-10),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n_evaluated": len(y_true),
    }


# ─── Robustness / adversarial testing ────────────────────────────────────────


class RobustnessTests:
    """
    Behavioral and adversarial tests to evaluate model robustness.
    Each test returns (passed: bool, description: str).
    """

    def __init__(self, model):
        self.model = model

    def _predict_conflict(self, vehicles: list[dict]) -> str:
        result = self.model.predict({"vehicles": vehicles})
        return result.get("is_conflict", "no")

    def test_obvious_conflict(self) -> tuple[bool, str]:
        """Two vehicles approaching from N and S at high speed very close to intersection."""
        vehicles = [
            {
                "vehicle_id": "ADV001",
                "lane": 1,
                "speed": 80,
                "distance_to_intersection": 30,
                "direction": "north",
                "destination": "A",
            },
            {
                "vehicle_id": "ADV002",
                "lane": 1,
                "speed": 75,
                "distance_to_intersection": 35,
                "direction": "south",
                "destination": "C",
            },
        ]
        pred = self._predict_conflict(vehicles)
        passed = pred == "yes"
        return passed, f"Obvious conflict (N-S, close, fast): predicted={pred}, expected=yes"

    def test_no_conflict_far(self) -> tuple[bool, str]:
        """Two vehicles far from intersection should not conflict."""
        vehicles = [
            {
                "vehicle_id": "ADV003",
                "lane": 1,
                "speed": 50,
                "distance_to_intersection": 500,
                "direction": "north",
                "destination": "A",
            },
            {
                "vehicle_id": "ADV004",
                "lane": 2,
                "speed": 45,
                "distance_to_intersection": 600,
                "direction": "east",
                "destination": "E",
            },
        ]
        pred = self._predict_conflict(vehicles)
        passed = pred == "no"
        return passed, f"No conflict (far from intersection): predicted={pred}, expected=no"

    def test_perturbation_stability(self) -> tuple[bool, str]:
        """Slightly perturbing speed should not flip the conflict prediction."""
        base_vehicles = [
            {
                "vehicle_id": "ADV005",
                "lane": 1,
                "speed": 60,
                "distance_to_intersection": 50,
                "direction": "north",
                "destination": "A",
            },
            {
                "vehicle_id": "ADV006",
                "lane": 1,
                "speed": 55,
                "distance_to_intersection": 45,
                "direction": "south",
                "destination": "C",
            },
        ]
        pred_base = self._predict_conflict(base_vehicles)

        perturbed = [v.copy() for v in base_vehicles]
        perturbed[0]["speed"] += 2  # tiny perturbation
        pred_perturbed = self._predict_conflict(perturbed)

        passed = pred_base == pred_perturbed
        return passed, f"Perturbation stability: base={pred_base}, perturbed={pred_perturbed}"

    def test_priority_consistency(self) -> tuple[bool, str]:
        """Faster vehicle should always get higher priority."""
        vehicles = [
            {
                "vehicle_id": "ADV007",
                "lane": 1,
                "speed": 80,
                "distance_to_intersection": 40,
                "direction": "north",
                "destination": "A",
            },
            {
                "vehicle_id": "ADV008",
                "lane": 1,
                "speed": 30,
                "distance_to_intersection": 50,
                "direction": "south",
                "destination": "C",
            },
        ]
        result = self.model.predict({"vehicles": vehicles})
        prio = result.get("priority_order", {})
        if prio.get("ADV007") is not None and prio.get("ADV008") is not None:
            passed = prio["ADV007"] < prio["ADV008"]  # lower rank = higher priority
        else:
            passed = False
        return (
            passed,
            f"Priority consistency: V007 prio={prio.get('ADV007')}, V008 prio={prio.get('ADV008')}",
        )

    def run_all(self) -> dict[str, Any]:
        tests = [
            self.test_obvious_conflict,
            self.test_no_conflict_far,
            self.test_perturbation_stability,
            self.test_priority_consistency,
        ]
        results = []
        for test_fn in tests:
            try:
                passed, desc = test_fn()
                results.append({"test": test_fn.__name__, "passed": passed, "description": desc})
            except Exception as exc:
                results.append({"test": test_fn.__name__, "passed": False, "description": str(exc)})

        n_passed = sum(1 for r in results if r["passed"])
        return {
            "passed": n_passed,
            "total": len(results),
            "pass_rate": n_passed / len(results),
            "details": results,
        }


# ─── Bias auditing ────────────────────────────────────────────────────────────


def audit_bias(
    raw_csv: str,
    model,
    protected_attribute: str = "direction",
    max_scenarios_per_group: int = 20,
    output_path: str | None = None,
) -> dict[str, Any]:
    """
    Audit model for bias across direction groups.
    Checks that conflict detection rate and accuracy are similar across all directions.

    Results are saved to reports/bias_audit_report.json (or output_path if provided)
    and logged to MLflow if an active run exists.
    """
    df = pd.read_csv(raw_csv)
    groups = df[protected_attribute].unique()
    group_metrics: dict[str, Any] = {}

    for group_val in groups:
        group_df = df[df[protected_attribute] == group_val]
        scenarios = list(group_df.groupby("scenario_id"))[:max_scenarios_per_group]

        y_true, y_pred = [], []
        for _, g in scenarios:
            vehicles = g[
                [
                    "vehicle_id",
                    "lane",
                    "speed",
                    "distance_to_intersection",
                    "direction",
                    "destination",
                ]
            ].to_dict(orient="records")
            true_label = 1 if g.iloc[0]["is_conflict"] == "yes" else 0
            try:
                result = model.predict({"vehicles": vehicles})
                pred_label = 1 if result.get("is_conflict") == "yes" else 0
                y_true.append(true_label)
                y_pred.append(pred_label)
            except Exception:
                pass

        if y_true:
            from sklearn.metrics import accuracy_score, f1_score, recall_score

            recall = recall_score(y_true, y_pred, zero_division=0)
            fn_count = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
            tp_count = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
            group_metrics[str(group_val)] = {
                "n": len(y_true),
                "accuracy": round(accuracy_score(y_true, y_pred), 4),
                "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
                "recall": round(recall, 4),
                "fnr": round(fn_count / (fn_count + tp_count + 1e-10), 4),
                "conflict_rate": round(sum(y_pred) / len(y_pred), 4),
            }

    # Compute disparity across groups
    if group_metrics:
        f1_values = [v["f1"] for v in group_metrics.values()]
        acc_values = [v["accuracy"] for v in group_metrics.values()]
        fnr_values = [v["fnr"] for v in group_metrics.values()]
        f1_disparity = round(max(f1_values) - min(f1_values), 4)
        acc_disparity = round(max(acc_values) - min(acc_values), 4)
        fnr_disparity = round(max(fnr_values) - min(fnr_values), 4)
        bias_detected = f1_disparity > 0.15
    else:
        f1_disparity = acc_disparity = fnr_disparity = None
        bias_detected = False

    report = {
        "protected_attribute": protected_attribute,
        "group_metrics": group_metrics,
        "f1_disparity": f1_disparity,
        "acc_disparity": acc_disparity,
        "fnr_disparity": fnr_disparity,
        "bias_detected": bias_detected,
        "bias_threshold": 0.15,
        "summary": (
            f"Bias {'DETECTED' if bias_detected else 'NOT detected'}: "
            f"max F1 gap across {protected_attribute} groups = {f1_disparity}"
        ),
    }

    # ── FIX: persist the report to disk ───────────────────────────────────────
    _ensure_reports_dir()
    save_path = Path(output_path) if output_path else REPORTS_DIR / "bias_audit_report.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[bias audit] Report saved → {save_path}")

    # ── FIX: log to MLflow if an active run exists ─────────────────────────────
    try:
        import mlflow

        if mlflow.active_run():
            mlflow.log_dict(report, "bias_audit_report.json")
            if f1_disparity is not None:
                mlflow.log_metric("bias_f1_disparity", f1_disparity)
                mlflow.log_metric("bias_acc_disparity", acc_disparity)
                mlflow.log_metric("bias_fnr_disparity", fnr_disparity)
                mlflow.log_metric("bias_detected", int(bias_detected))
            print("[bias audit] Metrics logged to MLflow")
    except ImportError:
        pass  # MLflow not installed — skip silently

    return report


# ─── CLI entrypoint ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    import mlflow

    parser = argparse.ArgumentParser(description="Run bias audit on the traffic intersection model")
    parser.add_argument(
        "--csv",
        default=str(REPO_ROOT / "data" / "raw" / "generated_dataset.csv"),
        help="Path to raw dataset CSV",
    )
    parser.add_argument(
        "--attribute",
        default="direction",
        choices=["direction", "lane"],
        help="Protected attribute to audit across (default: direction)",
    )
    parser.add_argument(
        "--max-per-group",
        type=int,
        default=20,
        help="Max scenarios evaluated per group (default: 20)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override output JSON path (default: reports/bias_audit_report.json)",
    )
    parser.add_argument(
        "--mlflow-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        help="MLflow tracking URI",
    )
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from src.models.llm_model import IntersectionLLM  # type: ignore

    model = IntersectionLLM()

    # ── Run inside an MLflow run ───────────────────────────────────────────────
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("traffic-intersection-llm")

    with mlflow.start_run(run_name=f"bias-audit-{args.attribute}"):
        mlflow.log_param("protected_attribute", args.attribute)
        mlflow.log_param("max_scenarios_per_group", args.max_per_group)
        mlflow.log_param("csv", args.csv)

        report = audit_bias(
            raw_csv=args.csv,
            model=model,
            protected_attribute=args.attribute,
            max_scenarios_per_group=args.max_per_group,
            output_path=args.output,
        )

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"BIAS AUDIT — {args.attribute.upper()}")
    print("=" * 60)
    for group, metrics in report["group_metrics"].items():
        print(
            f"  {group:10s}  n={metrics['n']:3d}  "
            f"acc={metrics['accuracy']:.3f}  "
            f"f1={metrics['f1']:.3f}  "
            f"recall={metrics['recall']:.3f}  "
            f"fnr={metrics['fnr']:.3f}  "
            f"conflict_rate={metrics['conflict_rate']:.3f}"
        )
    print("-" * 60)
    print(f"  F1 disparity : {report['f1_disparity']}")
    print(f"  Acc disparity: {report['acc_disparity']}")
    print(f"  FNR disparity: {report['fnr_disparity']}")
    print(f"  Bias detected: {report['bias_detected']}  (threshold = {report['bias_threshold']})")
    print("=" * 60)
    print(f"\nFull report: {REPORTS_DIR / 'bias_audit_report.json'}")

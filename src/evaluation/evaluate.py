"""
evaluate.py
-----------
Comprehensive model evaluation: accuracy, robustness, bias auditing.
Milestone 6: Testing beyond accuracy with adversarial and behavioral tests.
"""

import random
from typing import Any

import pandas as pd

# ─── Standard evaluation ──────────────────────────────────────────────────────


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
) -> dict[str, Any]:
    """
    Audit model for bias across direction groups.
    Checks that conflict detection rate and accuracy are similar across all directions.
    """
    df = pd.read_csv(raw_csv)
    groups = df[protected_attribute].unique()
    group_metrics = {}

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
            from sklearn.metrics import accuracy_score, f1_score

            group_metrics[str(group_val)] = {
                "n": len(y_true),
                "accuracy": accuracy_score(y_true, y_pred),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "conflict_rate": sum(y_pred) / len(y_pred),
            }

    # Compute disparity
    if group_metrics:
        f1_values = [v["f1"] for v in group_metrics.values()]
        disparity = max(f1_values) - min(f1_values)
        bias_detected = disparity > 0.15
    else:
        disparity = None
        bias_detected = False

    return {
        "protected_attribute": protected_attribute,
        "group_metrics": group_metrics,
        "f1_disparity": disparity,
        "bias_detected": bias_detected,
    }

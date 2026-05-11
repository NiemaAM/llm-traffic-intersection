"""
src/monitoring/explain.py
-------------------------
Model explainability and interpretability for the fine-tuned LLM conflict detector.

Three approaches:
1. Feature importance via perturbation (LIME-style)
2. Prompt-based chain-of-thought explanation
3. Attention proxy: which vehicle attributes most influence the decision

Usage:
  MLFLOW_TRACKING_URI=http://localhost:5000 \
  PYTHONPATH=. python src/monitoring/explain.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))  # Add project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))  # Add src directory

# Change to project root directory
os.chdir(Path(__file__).parent.parent.parent)

load_dotenv(Path(__file__).parent.parent / ".env", override=False)

from openai import OpenAI  # noqa: E402

# ── Config ──────────────────────────────────────────────────────────────────────

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT = "traffic-intersection-llm"
TEST_CSV = "data/masri_finetune/eval_only_masri.csv"
FT_MODEL_ID = os.environ.get(
    "FINE_TUNED_MODEL_ID",
    "ft:gpt-4o-mini-2024-07-18:personal::DX7kzKtB",
)
N_EXPLAIN = 5  # scenarios to explain
RANDOM_STATE = 42

SYSTEM_PROMPT = (
    "You are an Urban Intersection Traffic Conflict Detector, responsible for "
    "monitoring a four-way intersection with traffic coming from the north, east, "
    "south, and west. Each direction has two lanes guiding vehicles to different "
    "destinations:\n\n"
    "- North: Lane 1 directs vehicles to F and H, Lane 2 directs vehicles to E, D, and C.\n"
    "- East: Lane 3 leads to H and B, Lane 4 leads to G, E, and F.\n"
    "- South: Lane 5 directs vehicles to B and D, Lane 6 directs vehicles to A, G, and H.\n"
    "- West: Lane 7 directs vehicles to D and F, Lane 8 directs vehicles to B, C, and A.\n\n"
    "Analyze the traffic data from all directions and lanes, and determine if there "
    "is a potential conflict between vehicles at the intersection. "
    "Respond only with 'yes' or 'no'."
)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


# ── Helpers ─────────────────────────────────────────────────────────────────────


def vehicles_to_text(vehicles: list) -> str:
    parts = []
    for v in vehicles:
        parts.append(
            f"Vehicle {v['vehicle_id']} is in lane {v['lane']}, "
            f"moving {v['direction']} at a speed of {float(v['speed']):.2f} km/h, "
            f"and is {float(v['distance_to_intersection']):.2f} meters away from "
            f"the intersection, heading towards {v['destination']}."
        )
    return " ".join(parts)


def predict_raw(vehicles: list) -> str:
    text = vehicles_to_text(vehicles)
    resp = client.chat.completions.create(
        model=FT_MODEL_ID,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Analyze the following scenario and determine if there is a "
                    "conflict (Respond only with 'yes' or 'no'): " + text
                ),
            },
        ],
        max_tokens=5,
        temperature=0,
    )
    return resp.choices[0].message.content.strip().lower()


def explain_cot(vehicles: list) -> str:
    """Chain-of-thought: ask the model to explain its reasoning."""
    text = vehicles_to_text(vehicles)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",  # use base model for explanation
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert traffic conflict analyst. "
                    "Analyze the intersection scenario and explain step by step "
                    "whether the vehicles will conflict, considering: "
                    "1) their lanes and directions, "
                    "2) their speeds and distances, "
                    "3) their intended destinations. "
                    "End with a clear verdict: CONFLICT or NO CONFLICT."
                ),
            },
            {"role": "user", "content": text},
        ],
        max_tokens=300,
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def perturbation_importance(vehicles: list, base_pred: str) -> dict:
    """
    LIME-style feature importance: perturb each attribute and measure
    how often the prediction flips. Higher flip rate = more important.
    """
    importance = {}

    for i, v in enumerate(vehicles):
        vid = v["vehicle_id"]

        # Perturb speed (+/- 20 km/h)
        flips = 0
        for delta in [-20, -10, 10, 20]:
            perturbed = [dict(x) for x in vehicles]
            perturbed[i]["speed"] = max(10, v["speed"] + delta)
            pred = predict_raw(perturbed)
            if pred != base_pred:
                flips += 1
        importance[f"{vid}_speed"] = flips / 4

        # Perturb distance (+/- 50m)
        flips = 0
        for delta in [-50, -25, 25, 50]:
            perturbed = [dict(x) for x in vehicles]
            perturbed[i]["distance_to_intersection"] = max(5, v["distance_to_intersection"] + delta)
            pred = predict_raw(perturbed)
            if pred != base_pred:
                flips += 1
        importance[f"{vid}_distance"] = flips / 4

    return importance


# ── Main ────────────────────────────────────────────────────────────────────────


def main():
    df = pd.read_csv(TEST_CSV).sample(N_EXPLAIN, random_state=RANDOM_STATE)

    explanations = []

    for idx, (_, row) in enumerate(df.iterrows(), 1):
        scenario = json.loads(row["scenario"])
        vehicles = scenario["vehicles_scenario"]
        true_label = str(row["is_conflict"]).strip().lower()

        print(f"\n{'='*60}")
        print(f"Scenario {idx}/{N_EXPLAIN}  |  true={true_label}")
        print(f"{'='*60}")

        # 1. Base prediction
        base_pred = predict_raw(vehicles)
        correct = base_pred == true_label
        print(f"Prediction: {base_pred}  {'✓' if correct else '✗'}")

        # 2. Chain-of-thought explanation
        print("\n[Chain-of-thought explanation]")
        cot = explain_cot(vehicles)
        print(cot[:400] + ("..." if len(cot) > 400 else ""))

        # 3. Perturbation-based feature importance
        print("\n[Perturbation feature importance]")
        importance = perturbation_importance(vehicles, base_pred)
        sorted_imp = sorted(importance.items(), key=lambda x: -x[1])
        for feat, score in sorted_imp:
            bar = "█" * int(score * 10)
            print(f"  {feat:<35} {bar} {score:.2f}")

        most_important = sorted_imp[0][0] if sorted_imp else "N/A"
        print(f"\n  Most influential feature: {most_important}")

        explanations.append(
            {
                "scenario_idx": idx,
                "true_label": true_label,
                "predicted_label": base_pred,
                "correct": correct,
                "cot_explanation": cot,
                "feature_importance": importance,
                "most_important_feature": most_important,
            }
        )

    # ── Summary ────────────────────────────────────────────────────────────────
    accuracy = sum(e["correct"] for e in explanations) / len(explanations)
    print(f"\n{'='*60}")
    print(f"EXPLAINABILITY SUMMARY ({N_EXPLAIN} scenarios)")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.2f}")

    # Aggregate feature importance
    all_features = {}
    for e in explanations:
        for feat, score in e["feature_importance"].items():
            # Normalize to attribute type (speed vs distance)
            attr = feat.split("_")[-1]
            all_features[attr] = all_features.get(attr, 0) + score
    print("\nAggregate feature importance:")
    for attr, total in sorted(all_features.items(), key=lambda x: -x[1]):
        print(f"  {attr:<15} {total:.2f}")

    # Save report
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/explainability_report.json"
    with open(report_path, "w") as f:
        json.dump(
            {
                "model": FT_MODEL_ID,
                "n_scenarios": N_EXPLAIN,
                "accuracy": accuracy,
                "explanations": explanations,
                "aggregate_importance": all_features,
            },
            f,
            indent=2,
        )
    print(f"\nReport saved → {report_path}")

    # ── Log to MLflow ──────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="explainability"):
        mlflow.log_params(
            {
                "model": FT_MODEL_ID,
                "n_scenarios": N_EXPLAIN,
                "methods": "chain-of-thought, perturbation-importance",
            }
        )
        mlflow.log_metrics(
            {
                "explanation_accuracy": accuracy,
                "speed_importance": all_features.get("speed", 0),
                "distance_importance": all_features.get("distance", 0),
            }
        )
        mlflow.log_artifact(report_path)
    print(f"✅ Logged to MLflow: {MLFLOW_URI}")


if __name__ == "__main__":
    main()

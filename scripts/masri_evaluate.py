"""
scripts/masri_evaluate.py
--------------------------
Exact reproduction of Masri et al. (2025) evaluation pipeline.
Evaluates a fine-tuned model on the test set and logs results to MLflow.

Usage:
  # Use model ID saved by masri_finetune.py
  MLFLOW_TRACKING_URI=http://localhost:5000 \
  PYTHONPATH=. python scripts/masri_evaluate.py

  # Or specify model ID directly
  MLFLOW_TRACKING_URI=http://localhost:5000 \
  PYTHONPATH=. python scripts/masri_evaluate.py \
    --model-id ft:gpt-4o-mini-2024-07-18:personal::DWL89pFu

  # Use a different test set
  MLFLOW_TRACKING_URI=http://localhost:5000 \
  PYTHONPATH=. python scripts/masri_evaluate.py \
    --model-id ft:gpt-4o-mini-2024-07-18:personal::DWL89pFu \
    --test-data data/masri_finetune/test_scenarios.csv
"""

from __future__ import annotations

import argparse
import json
import os

import mlflow
import pandas as pd
from openai import OpenAI
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ── Config ─────────────────────────────────────────────────────────────────────

MAX_TOKENS  = 5
TEMPERATURE = 0.0

MLFLOW_URI        = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT = "traffic-intersection-llm"

OUT_DIR        = "data/masri_finetune"
DEFAULT_TEST   = "data/masri_finetune/test_scenarios.csv"
DEFAULT_ID_FILE = "data/masri_finetune/fine_tuned_model_id.txt"

# Exact system prompt from Masri et al.
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


# ── Helpers ─────────────────────────────────────────────────────────────────────

def scenario_to_text(scenario_json: str) -> str:
    """Exact parse_scenario_to_string() from Masri et al. prepare_data.py."""
    data = json.loads(scenario_json)
    parts = []
    for v in data.get("vehicles_scenario", []):
        parts.append(
            f"Vehicle {v['vehicle_id']} is in lane {v['lane']}, "
            f"moving {v['direction']} at a speed of {float(v['speed']):.2f} km/h, "
            f"and is {float(v['distance_to_intersection']):.2f} meters away from the "
            f"intersection, heading towards {v['destination']}."
        )
    return " ".join(parts)


def evaluate(client: OpenAI, df: pd.DataFrame, model_id: str) -> dict:
    """Exact predict_and_evaluate() + generate_evaluation_report() from Masri et al."""
    y_true, y_pred = [], []
    correct = 0

    for i, (_, row) in enumerate(df.iterrows(), 1):
        text       = scenario_to_text(row["scenario"])
        true_label = row["is_conflict"].strip().lower()

        resp = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": (
                    "Analyze the following scenario and determine if there is a "
                    "conflict (Respond only with 'yes' or 'no'): " + text
                )},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        pred_label = resp.choices[0].message.content.strip().lower()

        y_true.append(true_label)
        y_pred.append(pred_label)
        if pred_label == true_label:
            correct += 1
        print(f"  [{i}/{len(df)}] true={true_label}  pred={pred_label}  "
              f"running_acc={correct/i*100:.1f}%")

    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label="yes", zero_division=0)
    recall    = recall_score(y_true, y_pred, pos_label="yes", zero_division=0)
    f1        = f1_score(y_true, y_pred, pos_label="yes", zero_division=0)
    fnr       = 1 - recall
    cm        = confusion_matrix(y_true, y_pred, labels=["yes", "no"])

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(classification_report(y_true, y_pred,
          target_names=["Conflict: Yes", "Conflict: No"]))
    print("Confusion Matrix (rows=true, cols=pred):")
    print(f"          yes   no")
    print(f"  yes  {cm[0][0]:6d} {cm[0][1]:4d}")
    print(f"  no   {cm[1][0]:6d} {cm[1][1]:4d}")
    print(f"\nAccuracy:  {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"FNR:       {fnr:.4f}")
    print("=" * 60)

    return dict(accuracy=accuracy, precision=precision,
                recall=recall, f1=f1, fnr=fnr)


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Masri et al. evaluation")
    parser.add_argument("--model-id",  default="",
                        help="Fine-tuned model ID (reads from fine_tuned_model_id.txt if omitted)")
    parser.add_argument("--test-data", default=DEFAULT_TEST,
                        help="Path to test_scenarios.csv produced by masri_finetune.py")
    parser.add_argument("--external-data", default="",
                        help="Path to a raw per-vehicle CSV (never seen during training) "
                             "for leakage-free evaluation. Overrides --test-data.")
    parser.add_argument("--out-dir",   default=OUT_DIR)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    # Resolve model ID
    model_id = args.model_id
    if not model_id:
        if not os.path.exists(DEFAULT_ID_FILE):
            raise ValueError(
                f"No --model-id provided and {DEFAULT_ID_FILE} not found. "
                "Run masri_finetune.py first or pass --model-id."
            )
        with open(DEFAULT_ID_FILE) as f:
            model_id = f.read().strip()
    print(f"\nModel: {model_id}")

    # Load test set — external raw CSV takes priority (no data leakage)
    if args.external_data:
        if not os.path.exists(args.external_data):
            raise FileNotFoundError(f"External data not found: {args.external_data}")
        print(f"\nLoading external evaluation data (leakage-free): {args.external_data}")
        raw_df = pd.read_csv(args.external_data)
        rows = []
        import json as _json
        for sid, grp in raw_df.groupby("scenario_id"):
            vehicles = []
            for _, r in grp.iterrows():
                vehicles.append({
                    "vehicle_id":               str(r["vehicle_id"]),
                    "lane":                     int(r["lane"]),
                    "speed":                    float(r["speed"]),
                    "distance_to_intersection": float(r["distance_to_intersection"]),
                    "direction":                str(r["direction"]),
                    "destination":              str(r["destination"]),
                })
            rows.append({
                "scenario_id": sid,
                "scenario":    _json.dumps({"vehicles_scenario": vehicles}),
                "is_conflict": str(grp["is_conflict"].iloc[0]).strip().lower(),
            })
        test_df = pd.DataFrame(rows)
        print(f"Loaded {len(test_df)} scenarios from {len(raw_df)} vehicle rows")
    else:
        if not os.path.exists(args.test_data):
            raise FileNotFoundError(
                f"Test data not found: {args.test_data}\n"
                "Run masri_finetune.py first to generate it."
            )
        test_df = pd.read_csv(args.test_data)
        print(f"\nLoading pre-split test set: {args.test_data}")
    print(f"Test scenarios: {len(test_df)}")
    print(f"Conflict distribution:\n{test_df['is_conflict'].value_counts().to_string()}")

    client = OpenAI(api_key=api_key)
    os.makedirs(args.out_dir, exist_ok=True)

    # Evaluate
    print(f"\n[1] Evaluating on {len(test_df)} test scenarios")
    metrics = evaluate(client, test_df, model_id)

    # Save results
    results_path = os.path.join(args.out_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump({**metrics, "model_id": model_id}, f, indent=2)
    print(f"\n  Results saved -> {results_path}")

    # Log to MLflow
    print("\n[2] Logging to MLflow")
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name="masri-exact-eval"):
            mlflow.log_params({
                "model_id":        model_id,
                "test_scenarios":  len(test_df),
                "temperature":     TEMPERATURE,
                "eval_mode":       "masri_exact_reproduction",
            })
            mlflow.log_metrics({
                "accuracy":  metrics["accuracy"],
                "precision": metrics["precision"],
                "recall":    metrics["recall"],
                "f1":        metrics["f1"],
                "fnr":       metrics["fnr"],
            })
            mlflow.log_artifact(results_path)
        print(f"  Logged to MLflow experiment: {MLFLOW_EXPERIMENT}")
        print(f"  View at: {MLFLOW_URI}")
    except Exception as e:
        print(f"  Warning: MLflow logging failed ({e})")
        print(f"  Results still saved to {results_path}")


if __name__ == "__main__":
    main()
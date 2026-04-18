"""
train.py
--------
Model training entry-point.
Runs prompt-based fine-tuning (OpenAI) or evaluates zero/few-shot performance.
Tracks all experiments with MLflow.
"""

import os
import time
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import pandas as pd

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
EXPERIMENT_NAME     = "traffic-intersection-llm"


# ─── MLflow experiment setup ──────────────────────────────────────────────────

def setup_mlflow(experiment_name: str = EXPERIMENT_NAME) -> str:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_name)
    return experiment_id


# ─── Evaluation helpers ───────────────────────────────────────────────────────

def compute_metrics(y_true: list, y_pred: list) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "fnr":       1 - recall_score(y_true, y_pred, zero_division=0),
    }


# ─── Training run ─────────────────────────────────────────────────────────────

def run_evaluation(
    model_name: str = "gpt-4o-mini",
    few_shot: bool = True,
    fine_tuned_model_id: str | None = None,
    processed_csv: str = "data/processed/features.csv",
    raw_csv: str = "data/raw/generated_dataset.csv",
    max_eval_scenarios: int = 50,
    run_name: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate the LLM on held-out scenarios and log results to MLflow.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.llm_model import IntersectionLLM

    setup_mlflow()

    raw_df = pd.read_csv(raw_csv)

    # Build a balanced sample: equal number of conflict and no-conflict scenarios
    # This prevents the model from gaming accuracy by always predicting "no"
    import random
    all_scenarios = list(raw_df.groupby("scenario_id"))
    conflict_scenarios    = [(sid, g) for sid, g in all_scenarios if g.iloc[0]["is_conflict"] == "yes"]
    no_conflict_scenarios = [(sid, g) for sid, g in all_scenarios if g.iloc[0]["is_conflict"] == "no"]

    half = max_eval_scenarios // 2
    random.seed(42)
    sampled = (
        random.sample(conflict_scenarios,    min(half, len(conflict_scenarios))) +
        random.sample(no_conflict_scenarios, min(half, len(no_conflict_scenarios)))
    )
    random.shuffle(sampled)
    scenarios = sampled[:max_eval_scenarios]

    params = {
        "model_name":          fine_tuned_model_id or model_name,
        "few_shot":            few_shot,
        "fine_tuned":          fine_tuned_model_id is not None,
        "eval_scenarios":      len(scenarios),
    }

    with mlflow.start_run(run_name=run_name or f"eval-{model_name}"):
        mlflow.log_params(params)

        llm = IntersectionLLM(
            model=model_name,
            few_shot=few_shot,
            fine_tuned_model_id=fine_tuned_model_id,
        )

        y_true, y_pred = [], []
        latencies = []
        errors = 0

        for scenario_id, group in scenarios:
            vehicles = group[[
                "vehicle_id", "lane", "speed",
                "distance_to_intersection", "direction", "destination"
            ]].to_dict(orient="records")
            scenario = {"vehicles": vehicles}
            true_label = 1 if group.iloc[0]["is_conflict"] == "yes" else 0

            t0 = time.time()
            try:
                pred = llm.predict(scenario)
                pred_label = 1 if pred.get("is_conflict") == "yes" else 0
                y_true.append(true_label)
                y_pred.append(pred_label)
                latencies.append(time.time() - t0)
            except Exception as exc:
                errors += 1
                print(f"⚠️  Error on {scenario_id}: {exc}")

        if y_true:
            metrics = compute_metrics(y_true, y_pred)
            metrics["avg_latency_s"] = sum(latencies) / len(latencies)
            metrics["error_rate"]    = errors / len(scenarios)

            mlflow.log_metrics(metrics)
            print("\n📊 Evaluation Results:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")

            # Save predictions artifact
            preds_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
            preds_path = "reports/predictions.csv"
            Path(preds_path).parent.mkdir(parents=True, exist_ok=True)
            preds_df.to_csv(preds_path, index=False)
            mlflow.log_artifact(preds_path)

            return metrics
        else:
            print("⚠️  No successful predictions – check API key and model settings")
            return {}


# ─── Fine-tuning launcher ─────────────────────────────────────────────────────

def launch_finetune(
    training_file_path: str = "data/processed/finetune_train.jsonl",
    model: str = "gpt-4o-mini",
    n_epochs: int = 3,
) -> str | None:
    """
    Upload training data and launch an OpenAI fine-tuning job.
    Returns the fine-tuning job ID.
    """
    try:
        from openai import OpenAI
        client = OpenAI()
    except ImportError:
        print("❌ openai package not installed")
        return None

    with open(training_file_path, "rb") as f:
        upload_response = client.files.create(file=f, purpose="fine-tune")
    file_id = upload_response.id
    print(f"📤 Training file uploaded: {file_id}")

    job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model=model,
        hyperparameters={"n_epochs": n_epochs},
    )
    print(f"🚀 Fine-tuning job started: {job.id}")

    # Log to MLflow
    setup_mlflow()
    with mlflow.start_run(run_name=f"finetune-{model}"):
        mlflow.log_params({
            "base_model":    model,
            "n_epochs":      n_epochs,
            "training_file": training_file_path,
            "job_id":        job.id,
        })

    return job.id


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["eval", "finetune"], default="eval")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--few-shot", action="store_true", default=False)
    parser.add_argument("--fine-tuned-id", default=None)
    parser.add_argument("--max-scenarios", type=int, default=50)
    args = parser.parse_args()

    if args.mode == "eval":
        run_evaluation(
            model_name=args.model,
            few_shot=args.few_shot,
            fine_tuned_model_id=args.fine_tuned_id,
            max_eval_scenarios=args.max_scenarios,
        )
    else:
        launch_finetune(model=args.model)

"""
training_pipeline.py  —  Milestone 4
--------------------------------------
ZenML pipeline: LLM prompt training, fine-tuning, evaluation, versioning.

Steps
-----
  1. prepare_finetune_data   CSV → OpenAI JSONL training file
  2. evaluate_baseline       Zero-shot inference → metrics → MLflow
  3. evaluate_few_shot       Few-shot inference  → metrics → MLflow
  4. train_model             Fine-tune GPT-4o-mini (OpenAI API)
                             Polls until job completes; returns model ID
                             *** This is the model.fit() equivalent ***
  5. evaluate_finetuned      Fine-tuned model eval → metrics → MLflow
  6. compare_and_register    Pick winner; log comparison run to MLflow
  7. measure_energy          CodeCarbon CO₂ measurement

Skip fine-tuning (no cost) by setting skip_training=True — the pipeline
will still run all evaluation and comparison steps using the base model.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Annotated, Optional

import mlflow
from zenml import pipeline, step
from zenml.logger import get_logger

logger = get_logger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _src():
    p = str(Path(__file__).parent.parent)
    import sys
    if p not in sys.path:
        sys.path.insert(0, p)


def _mlflow_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


# ─── Step 1: Prepare fine-tuning data ────────────────────────────────────────

@step
def prepare_finetune_data(
    raw_csv:      str = "data/raw/generated_dataset.csv",
    output_path:  str = "data/processed/finetune_train.jsonl",
    max_examples: int = 500,
) -> Annotated[str, "finetune_path"]:
    """
    Convert the raw intersection CSV into an OpenAI fine-tuning JSONL file.
    Each line is one training example:
      { messages: [ {system}, {user: scenario_json}, {assistant: decision_json} ] }
    """
    _src()
    from models.llm_model import prepare_finetune_dataset

    out = prepare_finetune_dataset(raw_csv, output_path, max_examples)
    logger.info(f"✅ Fine-tuning data ready → {out}")
    return str(out)


# ─── Step 2: Evaluate zero-shot baseline ─────────────────────────────────────

@step
def evaluate_baseline(
    raw_csv:       str = "data/raw/generated_dataset.csv",
    model_name:    str = "gpt-4o-mini",
    max_scenarios: int = 20,
) -> Annotated[dict, "baseline_metrics"]:
    """
    Zero-shot evaluation: no examples in the prompt.
    Establishes the lower-bound performance before any prompt engineering.
    All metrics logged to MLflow under run 'zero-shot-baseline'.
    """
    _src()
    from models.train import run_evaluation

    mlflow.set_tracking_uri(_mlflow_uri())
    metrics = run_evaluation(
        model_name=model_name,
        few_shot=False,
        raw_csv=raw_csv,
        max_eval_scenarios=max_scenarios,
        run_name="zero-shot-baseline",
    )
    logger.info(f"Zero-shot → accuracy={metrics.get('accuracy',0):.3f}  F1={metrics.get('f1',0):.3f}")
    return metrics or {}


# ─── Step 3: Evaluate few-shot prompting ─────────────────────────────────────

@step
def evaluate_few_shot(
    raw_csv:       str = "data/raw/generated_dataset.csv",
    model_name:    str = "gpt-4o-mini",
    max_scenarios: int = 20,
) -> Annotated[dict, "fewshot_metrics"]:
    """
    Few-shot evaluation: 2 worked examples injected into the prompt.
    This is 'prompt training' — improving model behavior without updating weights.
    All metrics logged to MLflow under run 'few-shot-eval'.
    """
    _src()
    from models.train import run_evaluation

    mlflow.set_tracking_uri(_mlflow_uri())
    metrics = run_evaluation(
        model_name=model_name,
        few_shot=True,
        raw_csv=raw_csv,
        max_eval_scenarios=max_scenarios,
        run_name="few-shot-eval",
    )
    logger.info(f"Few-shot  → accuracy={metrics.get('accuracy',0):.3f}  F1={metrics.get('f1',0):.3f}")
    return metrics or {}


# ─── Step 4: Train model (fine-tuning) ───────────────────────────────────────

@step
def train_model(
    finetune_path:  str  = "data/processed/finetune_train.jsonl",
    base_model:     str  = "gpt-4o-mini",
    n_epochs:       int  = 3,
    skip_training:  bool = True,
) -> Annotated[str, "trained_model_id"]:
    """
    *** The model.fit() equivalent for LLMs ***

    Submits a supervised fine-tuning job to the OpenAI API:
      1. Upload the JSONL training file
      2. Start a fine-tuning job
      3. Poll every 30s until the job status is 'succeeded' or 'failed'
      4. Return the resulting fine-tuned model ID (e.g. ft:gpt-4o-mini:org::abc)

    The fine-tuned model ID is then used in evaluate_finetuned() and
    logged to MLflow so it can be reproduced later.

    Set skip_training=True to skip this step (costs ~$1-2 and takes 10-30 min).
    The pipeline will fall back to evaluating the best prompt-engineered variant.
    """
    if skip_training:
        # Check if a fine-tuned model ID is already set in the environment
        existing_id = os.environ.get("FINE_TUNED_MODEL_ID", "").strip()
        if existing_id:
            logger.info(f"✅ Using existing fine-tuned model from env: {existing_id}")
            return existing_id
        logger.info("⏭️  Fine-tuning skipped (skip_training=True, no FINE_TUNED_MODEL_ID set)")
        return ""

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.warning("⚠️  OPENAI_API_KEY not set — skipping fine-tuning")
        return ""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except ImportError:
        logger.warning("⚠️  openai not installed — skipping fine-tuning")
        return ""

    # ── Upload training file ──────────────────────────────────────────────────
    logger.info(f"📤 Uploading training file: {finetune_path}")
    with open(finetune_path, "rb") as f:
        upload = client.files.create(file=f, purpose="fine-tune")
    file_id = upload.id
    logger.info(f"   File ID: {file_id}")

    # ── Start fine-tuning job ─────────────────────────────────────────────────
    logger.info(f"🚀 Starting fine-tuning job: {base_model}, epochs={n_epochs}")
    job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model=base_model,
        hyperparameters={"n_epochs": n_epochs},
    )
    job_id = job.id
    logger.info(f"   Job ID: {job_id}")

    # Log job start to MLflow
    mlflow.set_tracking_uri(_mlflow_uri())
    mlflow.set_experiment("traffic-intersection-llm")
    with mlflow.start_run(run_name="fine-tuning-job"):
        mlflow.log_params({
            "base_model":     base_model,
            "n_epochs":       n_epochs,
            "training_file":  finetune_path,
            "file_id":        file_id,
            "job_id":         job_id,
        })

    # ── Poll until complete ───────────────────────────────────────────────────
    logger.info("⏳ Polling for job completion (this may take 10–30 minutes)...")
    poll_interval = 30   # seconds between checks
    max_wait      = 3600 # 1 hour timeout

    elapsed = 0
    while elapsed < max_wait:
        time.sleep(poll_interval)
        elapsed += poll_interval

        status = client.fine_tuning.jobs.retrieve(job_id)
        current_status = status.status
        logger.info(f"   [{elapsed:4d}s] Status: {current_status}")

        if current_status == "succeeded":
            model_id = status.fine_tuned_model
            logger.info(f"✅ Fine-tuning complete! Model ID: {model_id}")

            # Update MLflow run with result
            mlflow.set_experiment("traffic-intersection-llm")
            with mlflow.start_run(run_name="fine-tuning-job"):
                mlflow.log_param("fine_tuned_model_id", model_id)
                mlflow.log_param("status", "succeeded")
                mlflow.log_metric("training_duration_s", elapsed)

            return model_id

        elif current_status in ("failed", "cancelled"):
            logger.error(f"❌ Fine-tuning {current_status}: {status.error}")
            return ""

    logger.warning(f"⚠️  Fine-tuning timed out after {max_wait}s")
    return ""


# ─── Step 5: Evaluate fine-tuned model ───────────────────────────────────────

@step
def evaluate_finetuned(
    trained_model_id: str,
    base_model_name:  str = "gpt-4o-mini",
    raw_csv:          str = "data/raw/generated_dataset.csv",
    max_scenarios:    int = 20,
) -> Annotated[dict, "finetuned_metrics"]:
    """
    Evaluate the fine-tuned model on the test set.
    If no fine-tuned model ID is available, returns empty metrics.
    Logs results to MLflow under run 'fine-tuned-eval'.
    """
    if not trained_model_id:
        logger.info("⏭️  No fine-tuned model ID — skipping fine-tuned evaluation")
        return {}

    _src()
    from models.train import run_evaluation

    mlflow.set_tracking_uri(_mlflow_uri())
    metrics = run_evaluation(
        model_name=base_model_name,
        few_shot=True,
        fine_tuned_model_id=trained_model_id,
        raw_csv=raw_csv,
        max_eval_scenarios=max_scenarios,
        run_name="fine-tuned-eval",
    )
    logger.info(f"Fine-tuned → accuracy={metrics.get('accuracy',0):.3f}  F1={metrics.get('f1',0):.3f}")
    return metrics or {}


# ─── Step 6: Compare all variants and register best ──────────────────────────

@step
def compare_and_register(
    baseline_metrics:  dict,
    fewshot_metrics:   dict,
    finetuned_metrics: dict,
    trained_model_id:  str,
) -> Annotated[str, "best_config"]:
    """
    Compare all three model variants and register the best one in MLflow.

    Selection criteria: highest F1 score (safety-critical — maximise recall).
    Logs a 'model-comparison' run to MLflow with all metrics side by side.
    """
    candidates = {
        "zero_shot": baseline_metrics,
        "few_shot":  fewshot_metrics,
    }
    if finetuned_metrics:
        candidates["fine_tuned"] = finetuned_metrics

    best_name    = max(candidates, key=lambda k: candidates[k].get("f1", 0))
    best_metrics = candidates[best_name]
    best_f1      = best_metrics.get("f1", 0)

    logger.info(f"🏆 Best configuration: {best_name}  (F1={best_f1:.4f})")

    # Build comparison table for logging
    mlflow.set_tracking_uri(_mlflow_uri())
    mlflow.set_experiment("traffic-intersection-llm")
    with mlflow.start_run(run_name="model-comparison"):
        for variant, metrics in candidates.items():
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"{variant}_{metric_name}", value)

        mlflow.log_param("best_config",         best_name)
        mlflow.log_param("best_f1",             best_f1)
        mlflow.log_param("fine_tuned_model_id", trained_model_id or "none")

        # Log comparison table as artifact
        import json
        from pathlib import Path
        table = {v: {k: round(m.get(k, 0), 4) for k in ["accuracy","f1","recall","fnr","avg_latency_s"]}
                 for v, m in candidates.items() if m}
        Path("reports").mkdir(exist_ok=True)
        with open("reports/model_comparison.json", "w") as f:
            json.dump({"best": best_name, "results": table}, f, indent=2)
        mlflow.log_artifact("reports/model_comparison.json")

    # Print comparison table to logs
    header = f"{'Variant':<15} {'Accuracy':>10} {'F1':>8} {'Recall':>8} {'FNR':>8}"
    logger.info(f"\n{header}")
    logger.info("-" * len(header))
    for variant, metrics in candidates.items():
        flag = " ← BEST" if variant == best_name else ""
        logger.info(
            f"{variant:<15} "
            f"{metrics.get('accuracy',0):>10.4f} "
            f"{metrics.get('f1',0):>8.4f} "
            f"{metrics.get('recall',0):>8.4f} "
            f"{metrics.get('fnr',0):>8.4f}{flag}"
        )

    return best_name


# ─── Step 7: Energy measurement ───────────────────────────────────────────────

@step
def measure_energy(
    best_config: str,
) -> Annotated[dict, "energy_report"]:
    """
    Measure CO₂ equivalent emissions using CodeCarbon.
    Logs the result to MLflow. Falls back gracefully if not installed.
    """
    try:
        from codecarbon import EmissionsTracker
        tracker = EmissionsTracker(
            project_name="traffic-intersection-llm",
            log_level="error",
            output_dir="reports/",
        )
        tracker.start()
        _ = sum(i ** 2 for i in range(200_000))   # representative compute workload
        emissions_kg = tracker.stop()

        report = {"co2_kg": emissions_kg, "best_config": best_config}
        logger.info(f"🌿 Emissions: {emissions_kg:.8f} kg CO₂")

        mlflow.set_tracking_uri(_mlflow_uri())
        mlflow.set_experiment("traffic-intersection-llm")
        with mlflow.start_run(run_name="energy-measurement"):
            mlflow.log_metric("co2_kg",    emissions_kg)
            mlflow.log_metric("co2_g",     emissions_kg * 1000)
            mlflow.log_param("best_config", best_config)

    except Exception as exc:
        logger.warning(f"⚠️  CodeCarbon unavailable ({exc}) — skipping energy measurement")
        report = {"co2_kg": None, "best_config": best_config}

    return report


# ─── Pipeline definition ──────────────────────────────────────────────────────

@pipeline(name="m4_training_pipeline", enable_cache=False)
def training_pipeline(
    model_name:     str  = "gpt-4o-mini",
    max_scenarios:  int  = 20,
    skip_training:  bool = True,
    max_examples:   int  = 500,
    n_epochs:       int  = 3,
):
    """
    Milestone 4 — Full LLM Training Pipeline

    Graph
    -----
    prepare_finetune_data
          │
          ├──► evaluate_baseline   (zero-shot)  ──┐
          │                                        │
          ├──► evaluate_few_shot   (few-shot)   ──►├──► compare_and_register ──► measure_energy
          │                                        │
          └──► train_model ──► evaluate_finetuned ─┘

    Parameters
    ----------
    model_name      Base OpenAI model (default: gpt-4o-mini)
    max_scenarios   Scenarios per evaluation run (lower = cheaper)
    skip_training   If True, skip fine-tuning (no cost). Set False to fine-tune.
    max_examples    Max JSONL examples for fine-tuning dataset
    n_epochs        Fine-tuning epochs (OpenAI default: 3)
    """
    # Step 1 — Prepare training data
    finetune_path = prepare_finetune_data(
        max_examples=max_examples,
    )

    # Steps 2 & 3 — Evaluate prompt-only variants (always run)
    baseline_metrics = evaluate_baseline(
        model_name=model_name,
        max_scenarios=max_scenarios,
    )
    fewshot_metrics = evaluate_few_shot(
        model_name=model_name,
        max_scenarios=max_scenarios,
    )

    # Step 4 — Train (fine-tune) the model
    trained_model_id = train_model(
        finetune_path=finetune_path,
        base_model=model_name,
        n_epochs=n_epochs,
        skip_training=skip_training,
    )

    # Step 5 — Evaluate fine-tuned model
    finetuned_metrics = evaluate_finetuned(
        trained_model_id=trained_model_id,
        base_model_name=model_name,
        max_scenarios=max_scenarios,
    )

    # Step 6 — Compare all variants and register best
    best_config = compare_and_register(
        baseline_metrics=baseline_metrics,
        fewshot_metrics=fewshot_metrics,
        finetuned_metrics=finetuned_metrics,
        trained_model_id=trained_model_id,
    )

    # Step 7 — Energy measurement
    measure_energy(best_config=best_config)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Milestone 4 — Training Pipeline")
    parser.add_argument("--model",          default="gpt-4o-mini")
    parser.add_argument("--max-scenarios",  type=int,  default=20)
    parser.add_argument("--max-examples",   type=int,  default=500)
    parser.add_argument("--n-epochs",       type=int,  default=3)
    parser.add_argument("--train",          action="store_true",
                        help="Enable fine-tuning (costs money, takes 10-30 min)")
    args = parser.parse_args()

    print("🚀 Starting Milestone 4 Training Pipeline")
    print(f"   Model:          {args.model}")
    print(f"   Max scenarios:  {args.max_scenarios}")
    print(f"   Fine-tuning:    {'✅ Enabled' if args.train else '⏭️  Skipped (pass --train to enable)'}")
    print()

    training_pipeline(
        model_name=args.model,
        max_scenarios=args.max_scenarios,
        skip_training=not args.train,
        max_examples=args.max_examples,
        n_epochs=args.n_epochs,
    )

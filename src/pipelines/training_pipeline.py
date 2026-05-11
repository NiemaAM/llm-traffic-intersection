"""
src/pipelines/training_pipeline.py  —  Milestone 4
----------------------------------------------------
ZenML pipeline: data validation, fine-tuning only.
Evaluation is handled by src/pipelines/evaluation_pipeline.py.

Steps
-----
  1. check_data_balance      Audit class balance; generate balanced dataset if needed
  2. prepare_finetune_data   CSV → Masri et al. JSONL (masri_finetune.py logic)
  3. train_model             Fine-tune via OpenAI API (auto-skip if FINE_TUNED_MODEL_ID set)
  4. register_model          Register best model ID to MLflow
  5. measure_energy          CodeCarbon CO2 measurement

Usage:
  export $(grep -v '^#' .env | grep -v '^$' | xargs)
  MLFLOW_TRACKING_URI=http://localhost:5000 \\
  PYTHONPATH=. python src/pipelines/training_pipeline.py

  # Force fine-tuning even if model exists
  PYTHONPATH=. python src/pipelines/training_pipeline.py --force-train
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated

import mlflow
import pandas as pd
from zenml import pipeline, step
from zenml.logger import get_logger

logger = get_logger(__name__)

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass


def _mlflow_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


# ─── Step 1: Data balance check ───────────────────────────────────────────────


@step
def check_data_balance(
    raw_csv: str = "data/raw/generated_dataset.csv",
    min_balance_ratio: float = 0.40,
    target_size: int = 5000,
) -> Annotated[str, "training_csv"]:
    """
    Audit class balance of the training dataset.
    - If balanced (minority class >= min_balance_ratio): use as-is.
    - If imbalanced: generate a new balanced dataset and undersample.
    Returns path to a balanced CSV ready for fine-tuning.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
    except ImportError:
        pass

    balanced_path = "data/raw/generated_large_balanced.csv"

    if not os.path.exists(raw_csv):
        logger.warning(f"Raw CSV not found: {raw_csv} — generating...")
        subprocess.run(
            [
                sys.executable,
                "src/data/generate_data.py",
                "--num-records",
                str(target_size),
                "--output",
                raw_csv,
                "--seed",
                "42",
            ],
            check=True,
        )

    df = pd.read_csv(raw_csv)
    counts = df["is_conflict"].value_counts()
    total = len(df)
    yes_count = counts.get("yes", 0)
    no_count = counts.get("no", 0)
    minority = min(yes_count, no_count)
    ratio = minority / total if total > 0 else 0

    logger.info(
        f"Dataset: {total} rows — yes={yes_count}, no={no_count}, " f"minority_ratio={ratio:.3f}"
    )

    mlflow.set_tracking_uri(_mlflow_uri())
    mlflow.set_experiment("traffic-intersection-llm")
    with mlflow.start_run(run_name="data-balance-check"):
        mlflow.log_metrics(
            {
                "total_rows": float(total),
                "conflict_yes": float(yes_count),
                "conflict_no": float(no_count),
                "minority_ratio": float(ratio),
                "is_balanced": float(ratio >= min_balance_ratio),
            }
        )
        mlflow.log_param("raw_csv", raw_csv)
        mlflow.log_param("min_balance_ratio", min_balance_ratio)

    if ratio >= min_balance_ratio:
        logger.info(f"Dataset is balanced (ratio={ratio:.3f} >= {min_balance_ratio})")
        return raw_csv

    # Generate + balance
    logger.warning(f"Imbalanced dataset (ratio={ratio:.3f} < {min_balance_ratio}) — balancing")
    raw_large = "data/raw/generated_large_raw.csv"
    subprocess.run(
        [
            sys.executable,
            "src/data/generate_data.py",
            "--num-records",
            str(target_size),
            "--output",
            raw_large,
            "--seed",
            "123",
        ],
        check=True,
    )

    df_large = pd.read_csv(raw_large)
    scenarios = df_large.groupby("scenario_id")["is_conflict"].first().reset_index()
    yes_ids = scenarios[scenarios["is_conflict"] == "yes"]
    no_ids = scenarios[scenarios["is_conflict"] == "no"].sample(len(yes_ids), random_state=42)
    ids = pd.concat([yes_ids, no_ids])["scenario_id"]
    df_bal = df_large[df_large["scenario_id"].isin(ids)]
    df_bal.to_csv(balanced_path, index=False)

    new_counts = df_bal["is_conflict"].value_counts()
    logger.info(
        f"Balanced dataset saved: {balanced_path} "
        f"yes={new_counts.get('yes',0)} no={new_counts.get('no',0)}"
    )
    return balanced_path


# ─── Step 2: Prepare fine-tuning data ─────────────────────────────────────────


@step
def prepare_finetune_data(
    training_csv: str,
    out_dir: str = "data/masri_finetune",
) -> Annotated[dict, "finetune_paths"]:
    """
    Convert balanced CSV to Masri et al. JSONL format.
    Uses exact masri_finetune.py logic: 70/15/15 stratified split.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
    except ImportError:
        pass

    from sklearn.model_selection import train_test_split

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

    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(training_csv)

    rows = []
    for sid, grp in df.groupby("scenario_id"):
        vehicles = [
            {
                "vehicle_id": str(r["vehicle_id"]),
                "lane": int(r["lane"]),
                "speed": float(r["speed"]),
                "distance_to_intersection": float(r["distance_to_intersection"]),
                "direction": str(r["direction"]),
                "destination": str(r["destination"]),
            }
            for _, r in grp.iterrows()
        ]
        rows.append(
            {
                "scenario_id": sid,
                "scenario": json.dumps({"vehicles_scenario": vehicles}),
                "is_conflict": str(grp["is_conflict"].iloc[0]).strip().lower(),
            }
        )
    scenarios = pd.DataFrame(rows)
    logger.info(f"Loaded {len(scenarios)} scenarios from {len(df)} rows")

    train_df, temp_df = train_test_split(
        scenarios,
        test_size=0.30,
        random_state=42,
        stratify=scenarios["is_conflict"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["is_conflict"],
    )
    logger.info(f"Split: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    def _to_text(scenario_json: str) -> str:
        data = json.loads(scenario_json)
        return " ".join(
            f"Vehicle {v['vehicle_id']} is in lane {v['lane']}, "
            f"moving {v['direction']} at a speed of {float(v['speed']):.2f} km/h, "
            f"and is {float(v['distance_to_intersection']):.2f} meters away from the "
            f"intersection, heading towards {v['destination']}."
            for v in data.get("vehicles_scenario", [])
        )

    def _save_jsonl(df_split: pd.DataFrame, path: str) -> None:
        with open(path, "w") as f:
            for _, row in df_split.iterrows():
                entry = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                "Analyze the following scenario and determine if there is a "
                                "conflict (Respond only with 'yes' or 'no'): "
                                + _to_text(row["scenario"])
                            ),
                        },
                        {"role": "assistant", "content": row["is_conflict"]},
                    ]
                }
                f.write(json.dumps(entry) + "\n")

    train_path = os.path.join(out_dir, "train_data.jsonl")
    val_path = os.path.join(out_dir, "val_data.jsonl")
    test_path = os.path.join(out_dir, "test_scenarios.csv")
    _save_jsonl(train_df, train_path)
    _save_jsonl(val_df, val_path)
    test_df.to_csv(test_path, index=False)
    logger.info(f"JSONL files ready: {out_dir}")

    mlflow.set_tracking_uri(_mlflow_uri())
    mlflow.set_experiment("traffic-intersection-llm")
    with mlflow.start_run(run_name="prepare-finetune-data"):
        mlflow.log_params(
            {
                "training_csv": training_csv,
                "train_scenarios": len(train_df),
                "val_scenarios": len(val_df),
                "test_scenarios": len(test_df),
                "split": "70/15/15",
            }
        )
        mlflow.log_artifact(test_path)

    return {"train": train_path, "val": val_path, "test": test_path}


# ─── Step 3: Train model (fine-tuning) ────────────────────────────────────────


@step
def train_model(
    finetune_paths: dict,
    base_model: str = "gpt-4o-mini-2024-07-18",
    force_train: bool = False,
) -> Annotated[str, "trained_model_id"]:
    """
    Fine-tune GPT-4o-mini using Masri et al. exact JSONL format.

    Auto-skip logic:
      - If FINE_TUNED_MODEL_ID is set in .env and force_train=False: skip
      - Otherwise: upload JSONL and start OpenAI fine-tuning job

    Returns the fine-tuned model ID.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
    except ImportError:
        pass

    existing_id = os.environ.get("FINE_TUNED_MODEL_ID", "").strip()

    if existing_id and not force_train:
        logger.info(f"Fine-tuning skipped — existing model: {existing_id}")
        logger.info("  Pass --force-train or clear FINE_TUNED_MODEL_ID to retrain.")
        mlflow.set_tracking_uri(_mlflow_uri())
        mlflow.set_experiment("traffic-intersection-llm")
        with mlflow.start_run(run_name="train-model"):
            mlflow.log_param("fine_tuned_model_id", existing_id)
            mlflow.log_param("status", "skipped_existing")
            mlflow.log_param("base_model", base_model)
        return existing_id

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set — skipping fine-tuning")
        return ""

    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    train_path = finetune_paths["train"]
    val_path = finetune_paths["val"]
    n_train = sum(1 for _ in open(train_path))

    logger.info("Uploading training files to OpenAI...")
    with open(train_path, "rb") as f:
        train_file = client.files.create(file=f, purpose="fine-tune")
    with open(val_path, "rb") as f:
        val_file = client.files.create(file=f, purpose="fine-tune")
    logger.info(f"  train_file={train_file.id}  val_file={val_file.id}")

    logger.info(f"Starting fine-tuning job (base={base_model}, {n_train} examples)...")
    job = client.fine_tuning.jobs.create(
        training_file=train_file.id,
        validation_file=val_file.id,
        model=base_model,
    )
    logger.info(f"  Job ID: {job.id}")

    mlflow.set_tracking_uri(_mlflow_uri())
    mlflow.set_experiment("traffic-intersection-llm")
    with mlflow.start_run(run_name="train-model"):
        mlflow.log_params(
            {
                "base_model": base_model,
                "job_id": job.id,
                "train_file_id": train_file.id,
                "val_file_id": val_file.id,
                "n_train": n_train,
                "status": "running",
            }
        )

    logger.info("Polling for completion (10-40 min)...")
    elapsed = 0
    while True:
        time.sleep(60)
        elapsed += 60
        status = client.fine_tuning.jobs.retrieve(job.id)
        logger.info(f"  [{elapsed:4d}s] {status.status}")
        if status.status == "succeeded":
            model_id = status.fine_tuned_model
            logger.info(f"Fine-tuning complete: {model_id}")
            os.makedirs("data/masri_finetune", exist_ok=True)
            with open("data/masri_finetune/fine_tuned_model_id.txt", "w") as f:
                f.write(model_id)
            mlflow.set_tracking_uri(_mlflow_uri())
            mlflow.set_experiment("traffic-intersection-llm")
            with mlflow.start_run(run_name="train-model"):
                mlflow.log_params(
                    {
                        "fine_tuned_model_id": model_id,
                        "status": "succeeded",
                        "duration_s": elapsed,
                    }
                )
            return model_id
        if status.status in ("failed", "cancelled"):
            logger.error(f"Fine-tuning {status.status}")
            return ""
        if elapsed > 7200:
            logger.warning("Timed out after 2h")
            return ""


# ─── Step 4: Register model ────────────────────────────────────────────────────


@step
def register_model(
    trained_model_id: str,
) -> Annotated[str, "registered_model_id"]:
    """Register the fine-tuned model in MLflow."""
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
    except ImportError:
        pass

    model_id = trained_model_id or os.environ.get("FINE_TUNED_MODEL_ID", "")

    mlflow.set_tracking_uri(_mlflow_uri())
    mlflow.set_experiment("traffic-intersection-llm")
    with mlflow.start_run(run_name="model-registration"):
        mlflow.log_params(
            {
                "fine_tuned_model_id": model_id or "none",
                "base_model": "gpt-4o-mini-2024-07-18",
                "eval_pipeline": "src/pipelines/evaluation_pipeline.py",
                "status": "registered" if model_id else "no_model",
            }
        )

    if model_id:
        logger.info(f"Model registered: {model_id}")
        logger.info("Next step → run evaluation pipeline:")
        logger.info("  PYTHONPATH=. python src/pipelines/evaluation_pipeline.py")
    else:
        logger.warning("No fine-tuned model to register")

    return model_id


# ─── Step 5: Energy measurement ────────────────────────────────────────────────


@step
def measure_energy(
    registered_model_id: str,
) -> Annotated[dict, "energy_report"]:
    """Measure CO2 equivalent emissions using CodeCarbon."""
    try:
        from codecarbon import EmissionsTracker

        tracker = EmissionsTracker(
            project_name="traffic-intersection-llm",
            log_level="error",
            output_dir="reports/",
        )
        tracker.start()
        _ = sum(i**2 for i in range(200_000))
        emissions_kg = tracker.stop()
        report = {"co2_kg": emissions_kg, "model_id": registered_model_id}
        logger.info(f"Emissions: {emissions_kg:.8f} kg CO2")
        mlflow.set_tracking_uri(_mlflow_uri())
        mlflow.set_experiment("traffic-intersection-llm")
        with mlflow.start_run(run_name="energy-measurement"):
            mlflow.log_metric("co2_kg", emissions_kg)
            mlflow.log_metric("co2_g", emissions_kg * 1000)
            mlflow.log_param("model_id", registered_model_id or "none")
    except Exception as exc:
        logger.warning(f"CodeCarbon unavailable ({exc}) — skipping")
        report = {"co2_kg": None, "model_id": registered_model_id}
    return report


# ─── Pipeline ──────────────────────────────────────────────────────────────────


@pipeline(name="m4_training_pipeline", enable_cache=False)
def training_pipeline(
    raw_csv: str = "data/raw/generated_dataset.csv",
    force_train: bool = False,
    target_size: int = 5000,
):
    """
    Milestone 4 — Training Pipeline (training only, no evaluation).
    Evaluation is in src/pipelines/evaluation_pipeline.py.

    check_data_balance -> prepare_finetune_data -> train_model
                       -> register_model -> measure_energy
    """
    training_csv = check_data_balance(raw_csv=raw_csv, target_size=target_size)
    finetune_paths = prepare_finetune_data(training_csv=training_csv)
    trained_id = train_model(finetune_paths=finetune_paths, force_train=force_train)
    registered_id = register_model(trained_model_id=trained_id)
    measure_energy(registered_model_id=registered_id)


# ─── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Milestone 4 — Training Pipeline")
    parser.add_argument("--raw-csv", default="data/raw/generated_dataset.csv")
    parser.add_argument("--target-size", type=int, default=5000)
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Force fine-tuning even if FINE_TUNED_MODEL_ID is set",
    )
    args = parser.parse_args()

    print("Milestone 4 — Training Pipeline")
    print(f"  Raw CSV:      {args.raw_csv}")
    print(f"  Force train:  {args.force_train}")
    print(f"  Existing ID:  {os.environ.get('FINE_TUNED_MODEL_ID', '(none)')}")
    print()

    training_pipeline(
        raw_csv=args.raw_csv,
        force_train=args.force_train,
        target_size=args.target_size,
    )

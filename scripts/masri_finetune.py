"""
scripts/masri_finetune.py
--------------------------
Exact reproduction of Masri et al. (2025) GPT fine-tuning pipeline.
Handles data preparation and fine-tuning only.

Usage:
  export OPENAI_API_KEY=sk-...
  PYTHONPATH=. python scripts/masri_finetune.py
  PYTHONPATH=. python scripts/masri_finetune.py --data data/raw/generated_large_balanced.csv
"""

from __future__ import annotations

import argparse
import json
import os
import time

import pandas as pd
from openai import OpenAI
from sklearn.model_selection import train_test_split

# ── Config ─────────────────────────────────────────────────────────────────────

BASE_MODEL   = "gpt-4o-mini-2024-07-18"
RANDOM_STATE = 42
TRAIN_RATIO  = 0.70   # 70% train
VAL_RATIO    = 0.15   # 15% val  → 15% test (used by masri_evaluate.py)

DATA_PATH = "data/raw/generated_dataset.csv"
OUT_DIR   = "data/masri_finetune"

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


# ── Dataset helpers ─────────────────────────────────────────────────────────────

def load_as_scenarios(csv_path: str) -> pd.DataFrame:
    """
    Load this project's per-vehicle CSV and group into one row per scenario.
    Output columns: scenario_id, scenario (JSON string), is_conflict
    """
    df = pd.read_csv(csv_path)
    rows = []
    for sid, grp in df.groupby("scenario_id"):
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
            "scenario":    json.dumps({"vehicles_scenario": vehicles}),
            "is_conflict": str(grp["is_conflict"].iloc[0]).strip().lower(),
        })
    scenarios = pd.DataFrame(rows)
    print(f"  Loaded {len(scenarios)} scenarios from {len(df)} vehicle rows")
    print(f"  Conflict distribution:\n{scenarios['is_conflict'].value_counts().to_string()}")
    return scenarios


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


def prepare_jsonl(df: pd.DataFrame, path: str) -> None:
    """Exact prepare_chat_jsonl_file() from Masri et al."""
    with open(path, "w") as f:
        for _, row in df.iterrows():
            text = scenario_to_text(row["scenario"])
            entry = {
                "messages": [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": (
                        "Analyze the following scenario and determine if there is a "
                        "conflict (Respond only with 'yes' or 'no'): " + text
                    )},
                    {"role": "assistant", "content": row["is_conflict"]},
                ]
            }
            f.write(json.dumps(entry) + "\n")
    print(f"  Saved {len(df)} examples -> {path}")


# ── OpenAI helpers ──────────────────────────────────────────────────────────────

def upload_file(client: OpenAI, path: str) -> str:
    with open(path, "rb") as f:
        resp = client.files.create(file=f, purpose="fine-tune")
    print(f"  Uploaded {os.path.basename(path)} -> {resp.id}")
    return resp.id


def poll_finetune(client: OpenAI, job_id: str) -> str:
    print(f"  Polling job {job_id} ...")
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(f"    status={job.status}")
        if job.status == "succeeded":
            print(f"  Fine-tuned model: {job.fine_tuned_model}")
            return job.fine_tuned_model
        if job.status in ("failed", "cancelled"):
            raise RuntimeError(f"Fine-tuning {job.status}")
        time.sleep(60)


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Masri et al. fine-tuning")
    parser.add_argument("--data",    default=DATA_PATH, help="Input CSV path")
    parser.add_argument("--out-dir", default=OUT_DIR,   help="Output directory")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load and group dataset
    print("\n[1] Loading dataset")
    scenarios = load_as_scenarios(args.data)

    # 2. Split 70/15/15 stratified
    print(f"\n[2] Splitting 70/15/15 (seed={RANDOM_STATE})")
    train_df, temp_df = train_test_split(
        scenarios, test_size=(1 - TRAIN_RATIO),
        random_state=RANDOM_STATE, stratify=scenarios["is_conflict"],
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5,
        random_state=RANDOM_STATE, stratify=temp_df["is_conflict"],
    )
    print(f"  Train={len(train_df)}  Val={len(val_df)}  Test={len(test_df)}")

    # 3. Prepare JSONL files (test set saved for masri_evaluate.py)
    print("\n[3] Preparing JSONL files")
    train_path = os.path.join(args.out_dir, "train_data.jsonl")
    val_path   = os.path.join(args.out_dir, "val_data.jsonl")
    test_path  = os.path.join(args.out_dir, "test_scenarios.csv")
    prepare_jsonl(train_df, train_path)
    prepare_jsonl(val_df,   val_path)
    test_df.to_csv(test_path, index=False)
    print(f"  Saved test set -> {test_path}  (use with masri_evaluate.py)")

    # 4. Upload to OpenAI
    print("\n[4] Uploading to OpenAI")
    train_id = upload_file(client, train_path)
    val_id   = upload_file(client, val_path)

    # 5. Start fine-tuning job
    print(f"\n[5] Starting fine-tuning (base={BASE_MODEL})")
    job = client.fine_tuning.jobs.create(
        training_file=train_id,
        validation_file=val_id,
        model=BASE_MODEL,
    )
    print(f"  Job ID: {job.id}")

    # 6. Poll until done
    print("\n[6] Waiting for completion ...")
    model_id = poll_finetune(client, job.id)

    # Save model ID for masri_evaluate.py
    id_file = os.path.join(args.out_dir, "fine_tuned_model_id.txt")
    with open(id_file, "w") as f:
        f.write(model_id)
    print(f"\n  Model ID saved -> {id_file}")
    print(f"  Fine-tuned model: {model_id}")
    print(f"\n  Now run evaluation:")
    print(f"  PYTHONPATH=. python scripts/masri_evaluate.py --model-id {model_id}")


if __name__ == "__main__":
    main()
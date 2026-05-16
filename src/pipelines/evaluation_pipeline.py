"""
src/pipelines/evaluation_pipeline.py
--------------------------------------
ZenML evaluation pipeline matching Masri et al. (2025) methodology.
Separated from training_pipeline.py — runs independently after training.

Steps
-----
  1. prepare_eval_data       Generate leakage-free eval dataset using src/data/generate_data.py
                             (seed=999, never used in training) OR load existing CSV
  2. evaluate_all_variants   3 models x 3 scenario types; logs per-step metrics to MLflow;
                             tracks carbon emissions (CodeCarbon) → emission_report.json
  3. generate_and_log_charts Accuracy comparison + confusion matrices

Models:    zero-shot | few-shot | fine-tuned (DX7kzKtB)
Scenarios: 4-vehicle | 8-vehicle | mixed (2-8 vehicles)

Output files (reports/offline_evaluation/):
  figures/accuracy_comparaison.png  grouped bar chart with accuracy formula
  figures/confusion_matrices.png    confusion matrices for fine-tuned model
  report.json                       metrics + MLflow experiment/run links
  emission_report.json              carbon traces (CodeCarbon, if installed)
  evaluation_data.json              raw per-run evaluation results

Usage:
  export $(grep -v '^#' .env | grep -v '^$' | xargs)

  MLFLOW_TRACKING_URI=http://localhost:5000 \\
  PYTHONPATH=. python src/pipelines/evaluation_pipeline.py \\
    --use-csv data/masri_finetune/eval_only_masri.csv

  MLFLOW_TRACKING_URI=http://localhost:5000 \\
  PYTHONPATH=. python src/pipelines/evaluation_pipeline.py --n-scenarios 50

  PYTHONPATH=. python src/pipelines/evaluation_pipeline.py \\
    --use-csv data/masri_finetune/eval_only_masri.csv --n-scenarios 10
"""

import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import mlflow
import pandas as pd
from zenml import pipeline, step
from zenml.logger import get_logger

logger = get_logger(__name__)

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "poc"))

OFFLINE_EVAL_DIR = ROOT / "reports" / "offline_evaluation"

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass

# ── Config ──────────────────────────────────────────────────────────────────────

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT = "traffic-intersection-llm"
BASE_MODEL = "gpt-4o-mini"
FT_MODEL = os.environ.get(
    "FINE_TUNED_MODEL_ID",
    "ft:gpt-4o-mini-2024-07-18:personal::DX7kzKtB",
)

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

FEW_SHOT_EXAMPLES = [
    {
        "user": (
            "Vehicle V001 is in lane 1, moving north at a speed of 60.00 km/h, "
            "and is 80.00 meters away from the intersection, heading towards F. "
            "Vehicle V002 is in lane 3, moving east at a speed of 60.00 km/h, "
            "and is 80.00 meters away from the intersection, heading towards H."
        ),
        "assistant": "yes",
    },
    {
        "user": (
            "Vehicle V003 is in lane 5, moving south at a speed of 50.00 km/h, "
            "and is 400.00 meters away from the intersection, heading towards B. "
            "Vehicle V004 is in lane 7, moving west at a speed of 50.00 km/h, "
            "and is 450.00 meters away from the intersection, heading towards D."
        ),
        "assistant": "no",
    },
    {
        "user": (
            "Vehicle V005 is in lane 2, moving north at a speed of 70.00 km/h, "
            "and is 90.00 meters away from the intersection, heading towards E. "
            "Vehicle V006 is in lane 6, moving south at a speed of 65.00 km/h, "
            "and is 95.00 meters away from the intersection, heading towards A."
        ),
        "assistant": "yes",
    },
    {
        "user": (
            "Vehicle V007 is in lane 4, moving east at a speed of 40.00 km/h, "
            "and is 500.00 meters away from the intersection, heading towards G. "
            "Vehicle V008 is in lane 8, moving west at a speed of 45.00 km/h, "
            "and is 480.00 meters away from the intersection, heading towards A."
        ),
        "assistant": "no",
    },
]

SCENARIO_TYPES = ["4-vehicle", "8-vehicle", "mixed"]
VARIANTS = [
    ("zero-shot", BASE_MODEL, "Zero-Shot"),
    ("few-shot", BASE_MODEL, "Few-Shot"),
    ("fine-tuned", FT_MODEL, "Fine-Tuned"),
]
TRAINING_SEEDS = {42, 123}
EVAL_SEED = 999


# ── Helpers ─────────────────────────────────────────────────────────────────────


def _vehicles_to_text(vehicles: list) -> str:
    return " ".join(
        f"Vehicle {v['vehicle_id']} is in lane {v['lane']}, "
        f"moving {v['direction']} at a speed of {float(v['speed']):.2f} km/h, "
        f"and is {float(v['distance_to_intersection']):.2f} meters away from "
        f"the intersection, heading towards {v['destination']}."
        for v in vehicles
    )


def _predict(vehicles: list, mode: str, model_id: str, client) -> str:
    text = _vehicles_to_text(vehicles)
    user_msg = (
        "Analyze the following scenario and determine if there is a "
        "conflict (Respond only with 'yes' or 'no'): " + text
    )
    if mode == "zero-shot":
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
    elif mode == "few-shot":
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for ex in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": ex["user"]})
            messages.append({"role": "assistant", "content": ex["assistant"]})
        messages.append({"role": "user", "content": user_msg})
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
    resp = client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=5,
        temperature=0,
    )
    return resp.choices[0].message.content.strip().lower()


def _load_masri_csv_as_scenarios(csv_path: str, stype: str, n: int, rng: random.Random) -> list:
    """Load scenarios from a masri-format CSV, grouped by vehicle count."""
    df = pd.read_csv(csv_path)

    if "scenario" in df.columns:
        rows = []
        for _, row in df.iterrows():
            s = json.loads(row["scenario"])
            vehicles = s.get("vehicles_scenario", [])
            nv = len(vehicles)
            if stype == "4-vehicle" and nv != 4:
                continue
            if stype == "8-vehicle" and nv != 8:
                continue
            rows.append(
                {
                    "vehicles": vehicles,
                    "is_conflict": str(row["is_conflict"]).strip().lower(),
                }
            )
        rng.shuffle(rows)
        yes = [r for r in rows if r["is_conflict"] == "yes"]
        no = [r for r in rows if r["is_conflict"] == "no"]
        half = min(len(yes), len(no), n // 2)
        return yes[:half] + no[:half]

    rows = []
    for sid, grp in df.groupby("scenario_id"):
        nv = len(grp)
        if stype == "4-vehicle" and nv != 4:
            continue
        if stype == "8-vehicle" and nv != 8:
            continue
        vehicles = grp[
            ["vehicle_id", "lane", "speed", "distance_to_intersection", "direction", "destination"]
        ].to_dict(orient="records")
        rows.append(
            {
                "vehicles": vehicles,
                "is_conflict": str(grp["is_conflict"].iloc[0]).strip().lower(),
            }
        )
    rng.shuffle(rows)
    yes = [r for r in rows if r["is_conflict"] == "yes"]
    no = [r for r in rows if r["is_conflict"] == "no"]
    half = min(len(yes), len(no), n // 2)
    return yes[:half] + no[:half]


# ── Step 1: Prepare eval data ───────────────────────────────────────────────────


@step
def prepare_eval_data(
    n_scenarios: int = 50,
    eval_seed: int = EVAL_SEED,
    use_csv: str = "",
) -> Annotated[dict, "eval_datasets"]:
    """
    Prepare evaluation datasets — two modes:

    Mode A (use_csv != ""): Load from existing CSV generated by generate_data.py.
      Uses the SAME Masri et al. conflict engine as training — correct ground truth.

    Mode B (use_csv == ""): Generate fresh scenarios via generate_data.py (seed=999).
      Leakage-free: seed=999 never used in training (seeds 42, 123).
    """
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
    except ImportError:
        pass

    assert (
        eval_seed not in TRAINING_SEEDS
    ), f"DATA LEAKAGE: eval_seed={eval_seed} is in training seeds {TRAINING_SEEDS}"
    logger.info(f"Leakage check passed: eval_seed={eval_seed} not in {TRAINING_SEEDS}")

    rng = random.Random(eval_seed)
    datasets = {}

    if use_csv and os.path.exists(use_csv):
        logger.info(f"Loading eval data from: {use_csv}")
        mixed_scenarios = _load_masri_csv_as_scenarios(use_csv, "mixed", n_scenarios, rng)
        mixed_keys = {json.dumps(s["vehicles"], sort_keys=True) for s in mixed_scenarios}

        for stype in SCENARIO_TYPES:
            if stype == "mixed":
                scenarios = mixed_scenarios
            else:
                scenarios = _load_masri_csv_as_scenarios(use_csv, stype, n_scenarios, rng)
                if len(scenarios) < n_scenarios:
                    needed = n_scenarios - len(scenarios)
                    logger.warning(
                        f"  {stype}: only {len(scenarios)} type-specific scenarios in CSV, "
                        f"supplementing with {needed} mixed scenarios to reach {n_scenarios}"
                    )
                    all_mixed = _load_masri_csv_as_scenarios(use_csv, "mixed", n_scenarios * 4, rng)
                    existing_keys = {json.dumps(s["vehicles"], sort_keys=True) for s in scenarios}
                    supplement = [
                        s
                        for s in all_mixed
                        if json.dumps(s["vehicles"], sort_keys=True) not in existing_keys
                        and json.dumps(s["vehicles"], sort_keys=True) not in mixed_keys
                    ]
                    rng.shuffle(supplement)
                    scenarios = scenarios + supplement[:needed]
            yes_n = sum(1 for s in scenarios if s["is_conflict"] == "yes")
            no_n = len(scenarios) - yes_n
            logger.info(
                f"  {stype}: {len(scenarios)} scenarios ({yes_n} conflict, {no_n} no-conflict)"
            )
            datasets[stype] = scenarios

    else:
        logger.info(f"Generating eval data (seed={eval_seed}) via generate_data.py...")
        n_records = n_scenarios * 40
        eval_csv = f"data/raw/eval_pipeline_{eval_seed}.csv"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "src" / "data" / "generate_data.py"),
                "--num-records",
                str(n_records),
                "--output",
                eval_csv,
                "--seed",
                str(eval_seed),
            ],
            check=True,
            cwd=str(ROOT),
        )
        logger.info(f"Generated {n_records} rows → {eval_csv}")
        target = n_scenarios // 2

        for stype in SCENARIO_TYPES:
            df = pd.read_csv(eval_csv)
            rows = []
            for sid, grp in df.groupby("scenario_id"):
                nv = len(grp)
                if stype == "4-vehicle" and nv != 4:
                    continue
                if stype == "8-vehicle" and nv != 8:
                    continue
                vehicles = grp[
                    [
                        "vehicle_id",
                        "lane",
                        "speed",
                        "distance_to_intersection",
                        "direction",
                        "destination",
                    ]
                ].to_dict(orient="records")
                rows.append(
                    {
                        "vehicles": vehicles,
                        "is_conflict": str(grp["is_conflict"].iloc[0]).strip().lower(),
                    }
                )
            rng.shuffle(rows)
            yes = [r for r in rows if r["is_conflict"] == "yes"]
            no = [r for r in rows if r["is_conflict"] == "no"]
            half = min(len(yes), len(no), target)
            datasets[stype] = yes[:half] + no[:half]
            logger.info(
                f"  {stype}: {len(datasets[stype])} scenarios "
                f"({half} conflict, {half} no-conflict)"
            )

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="prepare-eval-data"):
        mlflow.log_params(
            {
                "eval_seed": eval_seed,
                "use_csv": use_csv or "generated",
                "leakage_free": True,
            }
        )
        for stype, s in datasets.items():
            mlflow.log_metric(f"{stype.replace('-','_')}_n", len(s))

    return datasets


# ── Step 2: Evaluate all variants ──────────────────────────────────────────────


@step
def evaluate_all_variants(
    eval_datasets: dict,
) -> Annotated[list, "all_results"]:
    """
    Evaluate all 9 combinations (3 models x 3 scenario types).
    Logs per-step running metrics to MLflow → line charts in Model metrics tab.
    Tracks carbon emissions via CodeCarbon → emission_report.json.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
    except ImportError:
        pass

    from openai import OpenAI
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    OFFLINE_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Start CodeCarbon tracker ───────────────────────────────────────────────
    tracker = None
    try:
        from codecarbon import EmissionsTracker

        tracker = EmissionsTracker(
            project_name="traffic-intersection-evaluation",
            output_dir=str(OFFLINE_EVAL_DIR),
            log_level="error",
            save_to_file=False,
        )
        tracker.start()
        logger.info("[CodeCarbon] Emissions tracker started.")
    except ImportError:
        logger.info("[CodeCarbon] Not installed — emissions not tracked. pip install codecarbon")
    except Exception as exc:
        logger.warning(f"[CodeCarbon] Could not start tracker: {exc}")
        tracker = None

    all_results = []
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    for mode, model_id, label in VARIANTS:
        for stype in SCENARIO_TYPES:
            run_name = f"{label} {stype}"
            scenarios = eval_datasets[stype]
            y_true, y_pred = [], []

            logger.info(f"Evaluating: {run_name} ({len(scenarios)} scenarios)")

            with mlflow.start_run(run_name=run_name) as run:
                mlflow_run_id = run.info.run_id
                mlflow.log_params(
                    {
                        "model": model_id,
                        "mode": mode,
                        "scenario_type": stype,
                        "n_scenarios": len(scenarios),
                        "leakage_free": True,
                    }
                )

                for step_i, s in enumerate(scenarios, 1):
                    true_label = s["is_conflict"]
                    try:
                        pred_label = _predict(s["vehicles"], mode, model_id, client)
                    except Exception as e:
                        logger.warning(f"  Error step {step_i}: {e}")
                        pred_label = "no"

                    y_true.append(true_label)
                    y_pred.append(pred_label)

                    acc = accuracy_score(y_true, y_pred)
                    prec = precision_score(y_true, y_pred, pos_label="yes", zero_division=0)
                    rec = recall_score(y_true, y_pred, pos_label="yes", zero_division=0)
                    f1 = f1_score(y_true, y_pred, pos_label="yes", zero_division=0)

                    mlflow.log_metric("accuracy", round(acc, 4), step=step_i)
                    mlflow.log_metric("precision", round(prec, 4), step=step_i)
                    mlflow.log_metric("recall", round(rec, 4), step=step_i)
                    mlflow.log_metric("f1", round(f1, 4), step=step_i)
                    mlflow.log_metric("fnr", round(max(0, 1 - rec), 4), step=step_i)

                cm = confusion_matrix(y_true, y_pred, labels=["yes", "no"])
                tp, fn, fp, tn = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

                result = {
                    "run_name": run_name,
                    "mode": mode,
                    "label": label,
                    "scenario_type": stype,
                    "mlflow_run_id": mlflow_run_id,
                    "accuracy": accuracy_score(y_true, y_pred),
                    "precision": precision_score(y_true, y_pred, pos_label="yes", zero_division=0),
                    "recall": recall_score(y_true, y_pred, pos_label="yes", zero_division=0),
                    "f1": f1_score(y_true, y_pred, pos_label="yes", zero_division=0),
                    "fnr": 1 - recall_score(y_true, y_pred, pos_label="yes", zero_division=0),
                    "tp": int(tp),
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                }
                all_results.append(result)
                logger.info(f"  {run_name}: acc={result['accuracy']:.3f} f1={result['f1']:.3f}")

    # ── Stop CodeCarbon and save emission report ───────────────────────────────
    if tracker is not None:
        try:
            emissions_kg = tracker.stop()
            emissions_data = getattr(tracker, "final_emissions_data", None)
            emission_dict = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "project": "traffic-intersection-evaluation",
                "emissions_kg_co2": emissions_kg,
                "energy_consumed_kwh": getattr(emissions_data, "energy_consumed", None),
                "duration_s": getattr(emissions_data, "duration", None),
                "cpu_power_w": getattr(emissions_data, "cpu_power", None),
                "gpu_power_w": getattr(emissions_data, "gpu_power", None),
                "country_name": getattr(emissions_data, "country_name", None),
                "region": getattr(emissions_data, "region", None),
                "cloud_provider": getattr(emissions_data, "cloud_provider", None),
                "n_scenarios_evaluated": sum(len(eval_datasets[s]) for s in eval_datasets),
                "models_evaluated": [v[2] for v in VARIANTS],
            }
            emission_path = OFFLINE_EVAL_DIR / "emission_report.json"
            emission_path.write_text(json.dumps(emission_dict, indent=2, default=str))
            logger.info(f"[CodeCarbon] Emission report → {emission_path}")
        except Exception as exc:
            logger.warning(f"[CodeCarbon] Could not save emission report: {exc}")

    # ── Save raw evaluation data ───────────────────────────────────────────────
    eval_data_path = OFFLINE_EVAL_DIR / "evaluation_data.json"
    eval_data_path.write_text(json.dumps(all_results, indent=2, default=str))
    logger.info(f"Evaluation data → {eval_data_path}")

    return all_results


# ── Chart generation (standalone, also called by the ZenML step) ───────────────


def _make_charts(all_results: list, figures_dir: Path) -> dict:
    """
    Generate accuracy comparison and confusion matrix figures.
    Returns dict of {name: saved_path}.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Accuracy comparison: vertical grouped bar chart ───────────────────────
    # Groups: mixed | 8-vehicle | 4-vehicle (left to right)
    # Within each group: Fine-Tuned | Few-Shot | Zero-Shot
    scenario_order = ["mixed", "8-vehicle", "4-vehicle"]
    variant_configs = [
        ("fine-tuned", "Fine-Tuned", "#1976d2"),
        ("few-shot", "Few-Shot", "#ff7043"),
        ("zero-shot", "Zero-Shot", "#43a047"),
    ]

    x = np.arange(len(scenario_order))
    n_variants = len(variant_configs)
    width = 0.22

    fig, ax = plt.subplots(figsize=(13, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")

    for i, (mode, label, color) in enumerate(variant_configs):
        offset = (i - (n_variants - 1) / 2) * width
        values = []
        for stype in scenario_order:
            r = next(
                (r for r in all_results if r["mode"] == mode and r["scenario_type"] == stype),
                None,
            )
            values.append(r["accuracy"] * 100 if r else 0)
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=1.5,
            zorder=3,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.8,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color=color,
            )

    ax.set_xlabel("Scenario Type", fontsize=13, labelpad=10)
    ax.set_ylabel("Accuracy (%)", fontsize=13, labelpad=10)
    ax.set_title(
        "Model Accuracy Comparison by Scenario Type",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["Mixed", "8-Vehicle", "4-Vehicle"], fontsize=12)
    ax.set_ylim(0, 118)
    ax.yaxis.grid(True, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=11, loc="upper right", framealpha=0.9, edgecolor="#cccccc")

    # Formula annotation bottom-left
    ax.text(
        0.01,
        0.01,
        r"$\mathrm{Accuracy} = \dfrac{TP + TN}{TP + TN + FP + FN}$",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#fffde7",
            alpha=0.9,
            edgecolor="#f0c030",
            linewidth=1.2,
        ),
    )

    plt.tight_layout()
    acc_path = figures_dir / "accuracy_comparaison.png"
    fig.savefig(acc_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Accuracy comparison saved: {acc_path}")

    # ── Confusion matrices (fine-tuned only) ──────────────────────────────────
    ft_results = [r for r in all_results if r["mode"] == "fine-tuned"]
    _order = {"8-vehicle": 0, "mixed": 1, "4-vehicle": 2}
    ft_results = sorted(ft_results, key=lambda r: _order.get(r["scenario_type"], 99))

    cm_path = None
    if ft_results:
        n = len(ft_results)
        fig, axes = plt.subplots(1, n, figsize=(5 * n + 1, 5))
        fig.patch.set_facecolor("white")
        if n == 1:
            axes = [axes]

        _title_map = {
            "4-vehicle": "4-Vehicle\nFine-Tuned",
            "8-vehicle": "8-Vehicle\nFine-Tuned",
            "mixed": "Mixed\nFine-Tuned",
        }

        for ax, r in zip(axes, ft_results):
            tp, tn, fp, fn = r["tp"], r["tn"], r["fp"], r["fn"]
            yes_total = (tp + fn) or 1
            no_total = (fp + tn) or 1

            # Row-normalised: [[TP, FN], [FP, TN]] / row totals
            z = [
                [tp / yes_total, fn / yes_total],
                [fp / no_total, tn / no_total],
            ]
            counts = [[tp, fn], [fp, tn]]
            cell_labels = [["TP", "FN"], ["FP", "TN"]]

            ax.imshow(z, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Predicted Yes", "Predicted No"], fontsize=10)
            ax.set_yticklabels(["True Yes", "True No"], fontsize=10)

            for i in range(2):
                for j in range(2):
                    text_color = "white" if z[i][j] > 0.6 else "#222222"
                    ax.text(
                        j,
                        i,
                        f"{cell_labels[i][j]}\n{counts[i][j]}",
                        ha="center",
                        va="center",
                        fontsize=14,
                        fontweight="bold",
                        color=text_color,
                    )

            ax.set_title(
                _title_map.get(r["scenario_type"], r["scenario_type"]),
                fontsize=12,
                fontweight="bold",
                pad=10,
            )
            ax.set_xlabel("Predicted Label", fontsize=10)
            if ax is axes[0]:
                ax.set_ylabel("True Label", fontsize=10)

        fig.suptitle(
            "Confusion Matrices — Fine-Tuned GPT-4o-mini",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        cm_path = figures_dir / "confusion_matrices.png"
        fig.savefig(cm_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"Confusion matrices saved: {cm_path}")
    else:
        logger.warning("No fine-tuned results — confusion matrices skipped")

    return {
        "accuracy_comparaison": str(acc_path),
        "confusion_matrices": str(cm_path) if cm_path else None,
    }


# ── Step 3: Generate charts and write reports ─────────────────────────────────


@step
def generate_and_log_charts(
    all_results: list,
) -> None:
    """Generate figures and report files; log all artifacts to MLflow."""
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
    except ImportError:
        pass

    OFFLINE_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = OFFLINE_EVAL_DIR / "figures"

    paths = _make_charts(all_results, figures_dir)

    # ── Build report.json with MLflow experiment/run links ────────────────────
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    exp = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
    experiment_id = exp.experiment_id if exp else None

    def _run_url(run_id: str) -> str:
        if experiment_id and run_id:
            return f"{MLFLOW_URI}/#/experiments/{experiment_id}/runs/{run_id}"
        return f"{MLFLOW_URI}/#/runs/{run_id}" if run_id else None

    report = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "mlflow": {
            "tracking_uri": MLFLOW_URI,
            "experiment": MLFLOW_EXPERIMENT,
            "experiment_id": experiment_id,
            "experiment_url": (
                f"{MLFLOW_URI}/#/experiments/{experiment_id}" if experiment_id else None
            ),
        },
        "summary": {},
        "runs": [],
    }

    for r in all_results:
        run_id = r.get("mlflow_run_id", "")
        report["runs"].append(
            {
                "run_name": r["run_name"],
                "mode": r["mode"],
                "scenario_type": r["scenario_type"],
                "mlflow_run_id": run_id,
                "mlflow_url": _run_url(run_id),
                "metrics": {
                    "accuracy": r["accuracy"],
                    "precision": r["precision"],
                    "recall": r["recall"],
                    "f1": r["f1"],
                    "fnr": r["fnr"],
                },
                "confusion_matrix": {
                    "tp": r["tp"],
                    "tn": r["tn"],
                    "fp": r["fp"],
                    "fn": r["fn"],
                },
            }
        )

    for mode, _, label in VARIANTS:
        mode_results = [r for r in all_results if r["mode"] == mode]
        if mode_results:
            report["summary"][mode] = {
                "label": label,
                "avg_accuracy": sum(r["accuracy"] for r in mode_results) / len(mode_results),
                "avg_f1": sum(r["f1"] for r in mode_results) / len(mode_results),
                "avg_fnr": sum(r["fnr"] for r in mode_results) / len(mode_results),
            }

    report_path = OFFLINE_EVAL_DIR / "report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info(f"Report → {report_path}")

    # ── MLflow: log final aggregate metrics + all artifacts ───────────────────
    with mlflow.start_run(run_name="evaluation-charts"):
        for r in all_results:
            pfx = f"{r['mode'].replace('-','_')}_{r['scenario_type'].replace('-','_')}"
            mlflow.log_metrics(
                {
                    f"{pfx}_accuracy": r["accuracy"],
                    f"{pfx}_precision": r["precision"],
                    f"{pfx}_recall": r["recall"],
                    f"{pfx}_f1": r["f1"],
                    f"{pfx}_fnr": r["fnr"],
                }
            )
        for path in paths.values():
            if path and os.path.exists(path):
                mlflow.log_artifact(path, artifact_path="figures")
        mlflow.log_artifact(str(report_path))
        for fname in ("evaluation_data.json", "emission_report.json"):
            p = OFFLINE_EVAL_DIR / fname
            if p.exists():
                mlflow.log_artifact(str(p))

    # ── Print results table ────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("EVALUATION RESULTS (Masri et al. Table 4)")
    print(f"{'='*72}")
    print(f"{'Model':<14} {'Scenario':<12} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'FNR':>6}")
    print("─" * 72)
    for r in all_results:
        print(
            f"{r['label']:<14} {r['scenario_type']:<12} "
            f"{r['accuracy']*100:>5.1f}% "
            f"{r['precision']:>6.3f} {r['recall']:>6.3f} "
            f"{r['f1']:>6.3f} {r['fnr']:>6.3f}"
        )
    print(f"{'='*72}")

    print(f"\nOutputs → {OFFLINE_EVAL_DIR}")
    for name, rel in [
        ("Accuracy figure", "figures/accuracy_comparaison.png"),
        ("Confusion matrices", "figures/confusion_matrices.png"),
        ("Report (MLflow links)", "report.json"),
        ("Emission report", "emission_report.json"),
        ("Evaluation data", "evaluation_data.json"),
    ]:
        mark = "✓" if (OFFLINE_EVAL_DIR / rel).exists() else "✗"
        print(f"  {mark} {name:<28} {rel}")

    logger.info(f"All outputs saved to {OFFLINE_EVAL_DIR}/ and logged to MLflow")


# ── Pipeline ───────────────────────────────────────────────────────────────────


@pipeline(name="masri_evaluation_pipeline", enable_cache=False)
def evaluation_pipeline(
    n_scenarios: int = 50,
    eval_seed: int = EVAL_SEED,
    use_csv: str = "",
):
    """
    Masri et al. evaluation pipeline.

    prepare_eval_data -> evaluate_all_variants -> generate_and_log_charts
    """
    datasets = prepare_eval_data(n_scenarios=n_scenarios, eval_seed=eval_seed, use_csv=use_csv)
    all_results = evaluate_all_variants(eval_datasets=datasets)
    generate_and_log_charts(all_results=all_results)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Masri et al. evaluation pipeline")
    parser.add_argument("--n-scenarios", type=int, default=50)
    parser.add_argument("--eval-seed", type=int, default=EVAL_SEED)
    parser.add_argument("--use-csv", default="", help="Path to existing eval CSV")
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Regenerate figures from evaluation_data.json without re-running model calls.",
    )
    args = parser.parse_args()

    if args.figures_only:
        data_path = OFFLINE_EVAL_DIR / "evaluation_data.json"
        if not data_path.exists():
            data_path = ROOT / "reports" / "evaluation_summary.json"
        if not data_path.exists():
            print("ERROR: No evaluation data found. Run a full evaluation first.")
            sys.exit(1)
        with open(data_path) as f:
            data = json.load(f)
        results = data if isinstance(data, list) else data.get("results", [])
        print("Regenerating figures from saved data ...")
        paths = _make_charts(results, OFFLINE_EVAL_DIR / "figures")
        print("Done:")
        for v in paths.values():
            if v:
                print(f"  {v}")
        sys.exit(0)

    if args.eval_seed in TRAINING_SEEDS:
        print(f"ERROR: eval_seed={args.eval_seed} was used in training. Choose different seed.")
        sys.exit(1)

    print("Masri et al. Evaluation Pipeline")
    print(f"  Scenarios/type: {args.n_scenarios}")
    print(f"  Eval seed:      {args.eval_seed} (leakage-free)")
    print(f"  Use CSV:        {args.use_csv or '(generate fresh)'}")
    print(f"  Fine-tuned:     {os.environ.get('FINE_TUNED_MODEL_ID', '(not set)')}")
    print()

    evaluation_pipeline(
        n_scenarios=args.n_scenarios,
        eval_seed=args.eval_seed,
        use_csv=args.use_csv,
    )

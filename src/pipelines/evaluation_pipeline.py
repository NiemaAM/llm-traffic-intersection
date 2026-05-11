"""
src/pipelines/evaluation_pipeline.py
--------------------------------------
ZenML evaluation pipeline matching Masri et al. (2025) methodology.
Separated from training_pipeline.py — runs independently after training.

Steps
-----
  1. prepare_eval_data       Generate leakage-free eval dataset using src/data/generate_data.py
                             (seed=999, never used in training) OR load existing CSV
  2. evaluate_all_variants   3 models x 3 scenario types; logs per-step line charts to MLflow
  3. compute_rouge_scores    ROUGE-L scores for fine-tuned model (Figure 8)
  4. generate_and_log_charts Figure 6, 7, 8 as Plotly PNG → reports/figures/ + MLflow artifacts

Models:    zero-shot | few-shot | fine-tuned (DX7kzKtB)
Scenarios: 4-vehicle | 8-vehicle | mixed (2-8 vehicles)

Usage:
  export $(grep -v '^#' .env | grep -v '^$' | xargs)

  # Use existing eval CSV (recommended — same engine as training)
  MLFLOW_TRACKING_URI=http://localhost:5000 \
  PYTHONPATH=. python src/pipelines/evaluation_pipeline.py \
    --use-csv data/masri_finetune/eval_only_masri.csv

  # Generate fresh eval data (seed=999)
  MLFLOW_TRACKING_URI=http://localhost:5000 \
  PYTHONPATH=. python src/pipelines/evaluation_pipeline.py --n-scenarios 50

  # Quick test
  PYTHONPATH=. python src/pipelines/evaluation_pipeline.py \
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


def _lcs_length(x: list, y: list) -> int:
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = (
                dp[i - 1][j - 1] + 1 if x[i - 1] == y[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
            )
    return dp[m][n]


def _rouge_l(hypothesis: str, reference: str) -> float:
    h, r = hypothesis.lower().split(), reference.lower().split()
    if not h or not r:
        return 0.0
    lcs = _lcs_length(h, r)
    prec = lcs / len(h)
    rec = lcs / len(r)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def _load_masri_csv_as_scenarios(csv_path: str, stype: str, n: int, rng: random.Random) -> list:
    """
    Load scenarios from a masri-format CSV (scenario JSON column).
    Groups by vehicle count to match scenario type.
    """
    df = pd.read_csv(csv_path)

    # If CSV has 'scenario' column (masri format) parse it
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
            # mixed: accept all
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

    # Per-vehicle CSV format — group by scenario_id
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
      Recommended: use data/masri_finetune/eval_only_masri.csv or
                       data/raw/eval_only_balanced.csv

    Mode B (use_csv == ""): Generate fresh scenarios via generate_data.py (seed=999).
      Leakage-free: seed=999 never used in training (seeds 42, 123).

    Both modes balance yes/no per scenario type.
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
        # ── Mode A: Load from existing CSV ─────────────────────────────────────
        logger.info(f"Loading eval data from: {use_csv}")

        # Load the canonical mixed set FIRST so we can exclude its scenarios from
        # any padding — otherwise the same scenario could appear in both the
        # mixed eval set and the padded 4-vehicle/8-vehicle sets, inflating scores.
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
                    # Exclude scenarios already in this type-set AND in the mixed eval set
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
        # ── Mode B: Generate fresh using generate_data.py ──────────────────────
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
    all_results = []

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    for mode, model_id, label in VARIANTS:
        for stype in SCENARIO_TYPES:
            run_name = f"{label} {stype}"
            scenarios = eval_datasets[stype]
            y_true, y_pred = [], []

            logger.info(f"Evaluating: {run_name} ({len(scenarios)} scenarios)")

            with mlflow.start_run(run_name=run_name):
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
                # labels=["yes","no"] → rows/cols in [yes, no] order
                # ravel() gives [TP, FN, FP, TN]
                tp, fn, fp, tn = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

                result = {
                    "run_name": run_name,
                    "mode": mode,
                    "label": label,
                    "scenario_type": stype,
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

    return all_results


# ── Step 3: ROUGE-L scores ──────────────────────────────────────────────────────


@step
def compute_rouge_scores(
    eval_datasets: dict,
) -> Annotated[dict, "rouge_scores"]:
    """Compute ROUGE-L scores for fine-tuned model (Figure 8)."""
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
    except ImportError:
        pass

    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    # Separate system prompt for ROUGE: ask for structured detail, not just yes/no
    ROUGE_SYSTEM = (
        "You are an Urban Intersection Traffic Conflict Detector. "
        "When given a traffic scenario, respond with a structured report containing:\n"
        "Conflict Status: yes or no\n"
        "Conflicts Overview: describe which vehicles conflict and why\n"
        "Actions & Decisions: state what each vehicle should do (yield or proceed)\n"
        "Priority Assignment: rank vehicles by crossing priority\n"
        "Vehicle Waiting Times: estimate waiting time in seconds for each vehicle"
    )

    # ROUGE uses BASE_MODEL, not FT_MODEL.
    # The fine-tuned model outputs only "yes"/"no" regardless of prompt (by design),
    # so it can never generate the structured text ROUGE-L needs to score
    # decisions, priority, and waiting. The base model produces detailed descriptions.
    def _score_scenarios(scenarios: list) -> dict:
        scores = {"conflicts": [], "decisions": [], "priority": [], "waiting": []}
        for s in scenarios:
            text = _vehicles_to_text(s["vehicles"])
            try:
                resp = client.chat.completions.create(
                    model=BASE_MODEL,  # base model generates structured text
                    messages=[
                        {"role": "system", "content": ROUGE_SYSTEM},
                        {
                            "role": "user",
                            "content": (
                                "Analyze the following traffic scenario and provide a "
                                "structured report with Conflict Status, Conflicts Overview, "
                                "Actions & Decisions, Priority Assignment, and Vehicle "
                                "Waiting Times:\n\n" + text
                            ),
                        },
                    ],
                    max_tokens=300,
                    temperature=0,
                )
                output = resp.choices[0].message.content.strip()
            except Exception:
                output = ""

            is_conf = s["is_conflict"]
            # Reference strings use the same section headings the prompt requests
            scores["conflicts"].append(
                _rouge_l(
                    output,
                    f"conflict status {is_conf} conflicts overview vehicles conflict intersection",
                )
            )
            scores["decisions"].append(
                _rouge_l(output, "actions decisions vehicle yield proceed stop wait continue")
            )
            scores["priority"].append(
                _rouge_l(
                    output, "priority assignment vehicle priority first second third crossing order"
                )
            )
            scores["waiting"].append(
                _rouge_l(output, "vehicle waiting times seconds wait estimated time intersection")
            )
        return {k: round(sum(v) / len(v), 4) if v else 0.0 for k, v in scores.items()}

    logger.info(
        "Computing ROUGE-L scores (base model — evaluates language quality of conflict descriptions)..."
    )
    rouge_4v = _score_scenarios(eval_datasets.get("4-vehicle", []))
    rouge_mixed = _score_scenarios(eval_datasets.get("mixed", []))
    logger.info(f"  ROUGE 4-vehicle: {rouge_4v}")
    logger.info(f"  ROUGE mixed:     {rouge_mixed}")
    return {"4-vehicle": rouge_4v, "mixed": rouge_mixed}


# ── Chart generation (standalone, also called by the ZenML step) ───────────────


def _make_charts(
    all_results: list,
    rouge_scores: dict,
    figures_dir: str = "reports/figures",
) -> dict:
    """
    Generate Figures 6, 7, 8 matching Masri et al. paper style.
    Returns dict of saved paths.  Called by the ZenML step AND by --figures-only.
    """
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots

    os.makedirs(figures_dir, exist_ok=True)

    # ── Figure 6: Horizontal accuracy bar chart ────────────────────────────────
    labels = [f"{r['label']} {r['scenario_type']}" for r in all_results]
    accuracies = [round(r["accuracy"] * 100, 1) for r in all_results]
    sorted_data = sorted(zip(labels, accuracies), key=lambda x: x[1])
    labels_s, accs_s = zip(*sorted_data) if sorted_data else ([], [])

    # Enhanced color palette - gradient from light to dark blue
    colors = [
        "#e3f2fd",
        "#bbdefb",
        "#90caf9",
        "#64b5f6",
        "#42a5f5",
        "#2196f3",
        "#1e88e5",
        "#1976d2",
        "#1565c0",
    ]
    bar_colors = [colors[i % len(colors)] for i in range(len(accs_s))]

    fig6 = go.Figure(
        go.Bar(
            x=list(accs_s),
            y=list(labels_s),
            orientation="h",
            marker_color=bar_colors,
            marker_line_color="#1976d2",
            marker_line_width=1.5,
            hovertemplate="<b>%{y}</b><br>Accuracy: %{x:.1f}%<extra></extra>",
        )
    )
    fig6.update_layout(
        title=dict(
            text="<b>Model Accuracy Comparison</b>",
            font=dict(size=18, family="Arial, sans-serif", color="#111111"),
            x=0.02,
            xanchor="left",
            y=0.97,
            yanchor="top",
        ),
        xaxis=dict(
            title=dict(text="Accuracy (%)", font=dict(size=14, family="Arial, sans-serif")),
            range=[0, 100],
            tickvals=list(range(0, 101, 10)),
            tickfont=dict(size=12),
            gridcolor="#e0e0e0",
            gridwidth=1,
            showgrid=True,
        ),
        yaxis=dict(
            tickfont=dict(size=12, family="Arial, sans-serif"),
            autorange="reversed",  # Keep highest at top
        ),
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        margin=dict(l=340, r=40, t=90, b=40),
        height=550,
        width=1000,
    )
    # Add subtle background stripes
    for i in range(len(labels_s)):
        fig6.add_shape(
            type="rect",
            x0=0,
            x1=100,
            y0=i - 0.5,
            y1=i + 0.5,
            fillcolor="#f8f9fa" if i % 2 == 0 else "white",
            line=dict(width=0),
            layer="below",
        )
    fig6_path = os.path.join(figures_dir, "fig6_accuracy_comparison.png")
    pio.write_image(fig6, fig6_path, width=1000, height=550, scale=2)
    logger.info(f"Figure 6 saved: {fig6_path}")

    # ── Figure 7: Confusion matrices (fine-tuned only) ─────────────────────────
    ft_results = [r for r in all_results if r["mode"] == "fine-tuned"]
    # Match paper column order: 8-vehicle | mixed | 4-vehicle
    _order = {"8-vehicle": 0, "mixed": 1, "4-vehicle": 2}
    ft_results = sorted(ft_results, key=lambda r: _order.get(r["scenario_type"], 99))
    n = len(ft_results)

    if n > 0:
        _title_map = {
            "4-vehicle": "4-Vehicle Fine-Tuned",
            "8-vehicle": "8-Vehicle Fine-Tuned",
            "mixed": "Mixed Fine-Tuned",
        }
        fig7 = make_subplots(
            rows=1,
            cols=n,
            subplot_titles=[
                _title_map.get(r["scenario_type"], r["scenario_type"]) for r in ft_results
            ],
        )
        for col_idx, r in enumerate(ft_results, 1):
            tp, tn, fp, fn = r["tp"], r["tn"], r["fp"], r["fn"]
            yes_total = (tp + fn) or 1
            no_total = (fp + tn) or 1
            # Row-normalised for coloring: diagonal → 1.0 (darkest), errors → 0 (lightest)
            # Plotly renders y[0] at bottom → y=["No","Yes"] puts Yes at top (correct)
            z_vals = [
                [fp / no_total, tn / no_total],  # No  row (bottom)
                [tp / yes_total, fn / yes_total],
            ]  # Yes row (top)
            counts = [[fp, tn], [tp, fn]]

            # Enhanced blue color scheme for confusion matrices
            fig7.add_trace(
                go.Heatmap(
                    z=z_vals,
                    colorscale=[
                        [0.0, "#e8f4fb"],  # Very light blue for low values
                        [0.5, "#7fb7dc"],  # Medium blue
                        [1.0, "#1f4f8b"],  # Deep blue for high values
                    ],
                    showscale=False,
                    x=["Yes", "No"],
                    y=["No", "Yes"],
                    xgap=3,
                    ygap=3,
                    zmin=0,
                    zmax=1,
                    hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{customdata}<br>Proportion: %{z:.3f}<extra></extra>",
                    customdata=counts,
                ),
                row=1,
                col=col_idx,
            )
            # Adaptive text colour: dark on light cells, white on dark cells
            xref = "x" if col_idx == 1 else f"x{col_idx}"
            yref = "y" if col_idx == 1 else f"y{col_idx}"
            for row_i, (row_z, row_c) in enumerate(zip(z_vals, counts)):
                for col_i, (zval, count) in enumerate(zip(row_z, row_c)):
                    text_color = "#ffffff" if zval > 0.7 else "#333333"
                    fig7.add_annotation(
                        x=["Yes", "No"][col_i],
                        y=["No", "Yes"][row_i],
                        text=f"<b>{count}</b>",
                        showarrow=False,
                        font=dict(size=20, color=text_color, family="Arial, sans-serif"),
                        xref=xref,
                        yref=yref,
                    )
            fig7.update_xaxes(
                title_text="Predicted Label",
                title_font=dict(size=14, family="Arial, sans-serif"),
                tickfont=dict(size=12),
                row=1,
                col=col_idx,
            )
            if col_idx == 1:
                fig7.update_yaxes(
                    title_text="True Label",
                    title_font=dict(size=14, family="Arial, sans-serif"),
                    tickfont=dict(size=12),
                    row=1,
                    col=col_idx,
                )

        fig7.update_layout(
            title=dict(
                text="<b>Confusion Matrices for Fine-Tuned GPT-4o-mini</b>",
                font=dict(size=18, family="Arial, sans-serif", color="#111111"),
                x=0.5,
                xanchor="center",
                y=0.98,
                yanchor="top",
            ),
            plot_bgcolor="#fafafa",
            paper_bgcolor="white",
            height=520,
            width=360 * n + 100,
            margin=dict(l=100, r=50, t=120, b=40),
        )
        fig7_path = os.path.join(figures_dir, "fig7_confusion_matrices.png")
        pio.write_image(fig7, fig7_path, width=360 * n + 100, height=520, scale=2)
        logger.info(f"Figure 7 saved: {fig7_path}")
    else:
        fig7_path = None
        logger.warning("No fine-tuned results — Figure 7 skipped")

    fig8_path = None
    if rouge_scores and any(rouge_scores.get(stype) for stype in ["4-vehicle", "mixed"]):
        components = ["Conflicts Overview", "Decisions", "Priority Assignment", "Waiting Times"]
        keys = ["conflicts", "decisions", "priority", "waiting"]
        rouge_4v_d = rouge_scores.get("4-vehicle", {})
        rouge_mix_d = rouge_scores.get("mixed", {})
        scores_4v = [rouge_4v_d.get(k, 0) for k in keys]
        scores_mixed = [rouge_mix_d.get(k, 0) for k in keys]

        fig8 = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["<b>4-Vehicle Scenarios</b>", "<b>Mixed Vehicle Scenarios</b>"],
            horizontal_spacing=0.15,
        )
        fig8.add_trace(
            go.Bar(
                x=components,
                y=scores_4v,
                marker_color="#1976d2",  # Professional blue
                name="4-Vehicle",
                showlegend=False,
                text=[f"{v:.3f}" for v in scores_4v],
                textposition="outside",
                textfont=dict(size=11, color="#1976d2", family="Arial, sans-serif"),
                hovertemplate="<b>%{x}</b><br>ROUGE-L: %{y:.3f}<extra></extra>",
                marker_line_color="#1565c0",
                marker_line_width=1.5,
            ),
            row=1,
            col=1,
        )
        fig8.add_trace(
            go.Bar(
                x=components,
                y=scores_mixed,
                marker_color="#d32f2f",  # Professional red
                name="Mixed",
                showlegend=False,
                text=[f"{v:.3f}" for v in scores_mixed],
                textposition="outside",
                textfont=dict(size=11, color="#d32f2f", family="Arial, sans-serif"),
                hovertemplate="<b>%{x}</b><br>ROUGE-L: %{y:.3f}<extra></extra>",
                marker_line_color="#b71c1c",
                marker_line_width=1.5,
            ),
            row=1,
            col=2,
        )
        # Fixed [0, 1.0] range matching paper; x-axis title matches paper labels
        fig8.update_yaxes(
            range=[0, 1.0],
            title_text="<b>ROUGE-L Score</b>",
            title_font=dict(size=14, family="Arial, sans-serif"),
            tickfont=dict(size=12),
            gridcolor="#e0e0e0",
            showgrid=True,
            row=1,
            col=1,
        )
        fig8.update_yaxes(
            range=[0, 1.0],
            title_text="<b>ROUGE-L Score</b>",
            title_font=dict(size=14, family="Arial, sans-serif"),
            tickfont=dict(size=12),
            gridcolor="#e0e0e0",
            showgrid=True,
            row=1,
            col=2,
        )
        fig8.update_xaxes(
            tickangle=0, tickfont=dict(size=11, family="Arial, sans-serif"), row=1, col=1
        )
        fig8.update_xaxes(
            tickangle=0, tickfont=dict(size=11, family="Arial, sans-serif"), row=1, col=2
        )
        fig8.update_layout(
            title=dict(
                text="<b>ROUGE-L Score Comparison for 4-Vehicle and Mixed Scenarios</b>",
                font=dict(size=18, family="Arial, sans-serif", color="#111111"),
                x=0.5,
                y=0.97,
            ),
            plot_bgcolor="#fafafa",
            paper_bgcolor="white",
            height=520,
            width=1100,
            margin=dict(l=80, r=80, t=120, b=40),
        )
        fig8_path = os.path.join(figures_dir, "fig8_rouge_scores.png")
        pio.write_image(fig8, fig8_path, width=1100, height=520, scale=2)
        logger.info(f"Figure 8 saved: {fig8_path}")
    else:
        fig8_path = None
        logger.info("ROUGE-L skipped — Figure 8 not generated")

    return {"fig6": fig6_path, "fig7": fig7_path, "fig8": fig8_path}


# ── Step 4: Generate charts with Plotly ────────────────────────────────────────


@step
def generate_and_log_charts(
    all_results: list,
    rouge_scores: dict,
) -> None:
    """Generate Figures 6, 7, 8; save to reports/figures/ and log to MLflow."""
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
    except ImportError:
        pass

    figures_dir = "reports/figures"
    summary_path = "reports/evaluation_summary.json"
    os.makedirs(figures_dir, exist_ok=True)

    paths = _make_charts(all_results, rouge_scores, figures_dir)

    # Save summary JSON
    with open(summary_path, "w") as f:
        json.dump({"results": all_results, "rouge": rouge_scores}, f, indent=2)

    # Log to MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
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
        for stype, sc in rouge_scores.items():
            for k, v in sc.items():
                mlflow.log_metric(f"rouge_{stype.replace('-','_')}_{k}", v)
        mlflow.log_artifact(paths["fig6"], artifact_path="figures")
        if paths["fig7"]:
            mlflow.log_artifact(paths["fig7"], artifact_path="figures")
        mlflow.log_artifact(paths["fig8"], artifact_path="figures")
        mlflow.log_artifact(summary_path)

    # Print Table 4 style summary
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
    logger.info(f"Charts saved to {figures_dir}/ and logged to MLflow")


# ── Pipeline ───────────────────────────────────────────────────────────────────


@pipeline(name="masri_evaluation_pipeline", enable_cache=False)
def evaluation_pipeline(
    n_scenarios: int = 50,
    eval_seed: int = EVAL_SEED,
    use_csv: str = "",
    skip_rouge_l: bool = False,
):
    """
    Masri et al. evaluation pipeline.

    prepare_eval_data -> evaluate_all_variants + compute_rouge_scores
                      -> generate_and_log_charts (Figure 6, 7, 8)
    """
    datasets = prepare_eval_data(n_scenarios=n_scenarios, eval_seed=eval_seed, use_csv=use_csv)
    all_results = evaluate_all_variants(eval_datasets=datasets)
    if skip_rouge_l:
        logger.info("Skipping ROUGE-L scoring per configuration")
        rouge_scores = {"4-vehicle": {}, "mixed": {}}
    else:
        rouge_scores = compute_rouge_scores(eval_datasets=datasets)
    generate_and_log_charts(all_results=all_results, rouge_scores=rouge_scores)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Masri et al. evaluation pipeline")
    parser.add_argument("--n-scenarios", type=int, default=50)
    parser.add_argument("--eval-seed", type=int, default=EVAL_SEED)
    parser.add_argument("--use-csv", default="", help="Path to existing eval CSV")
    parser.add_argument(
        "--skip-rouge-l",
        action="store_true",
        help="Skip ROUGE-L scoring and do not generate Figure 8",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Regenerate figures from the last saved evaluation_summary.json "
        "without re-running any model calls.",
    )
    args = parser.parse_args()

    # ── figures-only shortcut ──────────────────────────────────────────────────
    if args.figures_only:
        summary_path = "reports/evaluation_summary.json"
        if not os.path.exists(summary_path):
            print(f"ERROR: {summary_path} not found. Run a full evaluation first.")
            sys.exit(1)
        with open(summary_path) as f:
            summary = json.load(f)
        print("Regenerating figures from saved summary …")
        paths = _make_charts(summary["results"], summary["rouge"])
        print("Done:")
        for v in paths.values():
            if v:
                print(f"  {v}")
        sys.exit(0)

    # ── full pipeline ──────────────────────────────────────────────────────────
    if args.eval_seed in TRAINING_SEEDS:
        print(f"ERROR: eval_seed={args.eval_seed} was used in training. Choose different seed.")
        sys.exit(1)

    print("Masri et al. Evaluation Pipeline")
    print(f"  Scenarios/type: {args.n_scenarios}")
    print(f"  Eval seed:      {args.eval_seed} (leakage-free)")
    print(f"  Use CSV:        {args.use_csv or '(generate fresh)'}")
    print(f"  Skip ROUGE-L:   {'Yes' if args.skip_rouge_l else 'No'}")
    print(f"  Fine-tuned:     {os.environ.get('FINE_TUNED_MODEL_ID', '(not set)')}")
    print()

    evaluation_pipeline(
        n_scenarios=args.n_scenarios,
        eval_seed=args.eval_seed,
        use_csv=args.use_csv,
        skip_rouge_l=args.skip_rouge_l,
    )

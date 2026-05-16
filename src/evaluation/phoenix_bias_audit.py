"""
phoenix_bias_audit.py  —  Bias Audit with Arize Phoenix
========================================================
Audits the fine-tuned model across 7 dimensions:
  1. Direction        North / East / South / West
  2. Lane             Even (left-turn, 2/4/6/8) vs Odd (straight, 1/3/5/7)
  3. Vehicle count    Small 2-4  vs  Large 5-8
  4. Movement type    Straight/right vs Left-turn (lane parity)
  5. Speed range      Slow <40  |  Medium 40-70  |  Fast >70 km/h
  6. Distance range   Near <50m | Mid 50-150m    | Far >150m
  7. Conflict density No conflict | Single | Multi (2+)

Metrics per subgroup: accuracy, recall, FNR, precision, F1, waiting-time MAE
Bias flag: F1 disparity > 0.15

Output:
  reports/bias_audit/report.json
  reports/bias_audit/figures/*.png
"""

from __future__ import annotations

import ast
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ─── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "poc"))

load_dotenv(REPO_ROOT / ".env", override=False)
os.chdir(REPO_ROOT)

REPORT_DIR = REPO_ROOT / "reports" / "online_evaluation" / "bias_audit"
FIG_DIR = REPORT_DIR / "figures"
RAW_CSV = REPO_ROOT / "data" / "raw" / "generated_dataset.csv"
FT_MODEL_ID = os.environ.get("FINE_TUNED_MODEL_ID", "ft:gpt-4o-mini-2024-07-18:personal::DX7kzKtB")

# Scenarios per subgroup (keeps API cost bounded)
MAX_PER_GROUP = 20
RANDOM_SEED = 42

BIAS_THRESHOLD_F1 = 0.15
BIAS_THRESHOLD_FNR = 0.15
BIAS_THRESHOLD_ACC = 0.10


# ─── Phoenix helpers ───────────────────────────────────────────────────────────


def _upload_dataset(session: Any, df: pd.DataFrame, name: str) -> None:
    if session is None:
        return
    try:
        from phoenix.client import Client

        client = Client()
        up = df.copy()
        up["input"] = up.get("vehicles_json", up.index.astype(str))
        up["output"] = up.get("predicted_label", "no")
        meta_cols = [
            c for c in ["group", "true_label", "correct", "n_vehicles", "wt_mae"] if c in up.columns
        ]
        client.datasets.create_dataset(
            name=name,
            dataframe=up,
            input_keys=["input"],
            output_keys=["output"],
            metadata_keys=meta_cols,
        )
        print(f"[Phoenix] Dataset '{name}' uploaded ({len(df)} rows)")
    except Exception as exc:
        print(f"[Phoenix] Dataset upload skipped: {exc}")


# ─── LLM prediction ───────────────────────────────────────────────────────────


def _predict_yn(vehicles: list, ft_model_id: str, client: Any) -> str:
    from models.llm_model import MASRI_SYSTEM_PROMPT

    parts = [
        f"Vehicle {v['vehicle_id']} is in lane {v['lane']}, moving {v['direction']} "
        f"at a speed of {float(v['speed']):.2f} km/h, and is "
        f"{float(v['distance_to_intersection']):.2f} meters away from the intersection, "
        f"heading towards {v['destination']}."
        for v in vehicles
    ]
    text = " ".join(parts)
    try:
        resp = client.chat.completions.create(
            model=ft_model_id,
            messages=[
                {"role": "system", "content": MASRI_SYSTEM_PROMPT},
                {"role": "user", "content": f"Conflict? (yes/no): {text}"},
            ],
            max_tokens=5,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip().lower()
    except Exception:
        return "no"


def _rule_predict(vehicles: list) -> dict:
    try:
        from models.llm_model import _build_full_decision

        return _build_full_decision(vehicles, None)
    except Exception:
        return {"is_conflict": "no", "waiting_times": {}}


# ─── Data loading & feature extraction ───────────────────────────────────────


def _load_raw_csv(csv_path: Path) -> pd.DataFrame:
    """Load raw CSV and add derived subgroup columns."""
    df = pd.read_csv(csv_path)

    # ── Parse waiting_times column ────────────────────────────────────────────
    def _parse_wts(s: Any) -> dict:
        if pd.isna(s):
            return {}
        try:
            return ast.literal_eval(str(s))
        except Exception:
            return {}

    df["_wts_dict"] = df["waiting_times"].apply(_parse_wts)
    df["_avg_wt"] = df["_wts_dict"].apply(lambda d: float(np.mean(list(d.values()))) if d else 0.0)

    # ── Derived subgroup features (scenario-level, use first vehicle row) ─────
    # Movement type: even lanes = left-turn bias, odd = straight/right
    df["movement_type"] = df["lane"].apply(lambda l: "left-turn" if int(l) % 2 == 0 else "straight")

    # Speed range
    df["speed_range"] = pd.cut(
        df["speed"].astype(float),
        bins=[0, 40, 70, 9999],
        labels=["slow (<40)", "medium (40-70)", "fast (>70)"],
    )

    # Distance range
    df["distance_range"] = pd.cut(
        df["distance_to_intersection"].astype(float),
        bins=[0, 50, 150, 99999],
        labels=["near (<50m)", "mid (50-150m)", "far (>150m)"],
    )

    # Conflict density (number_of_conflicts)
    df["conflict_density"] = df["number_of_conflicts"].apply(
        lambda n: "no conflict" if int(n) == 0 else ("single" if int(n) == 1 else "multi (2+)")
    )

    return df


def _get_scenario_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per scenario (first vehicle row per scenario_id)."""
    return df.groupby("scenario_id").first().reset_index()


def _build_vehicle_list(group: pd.DataFrame) -> list[dict]:
    cols = ["vehicle_id", "lane", "speed", "distance_to_intersection", "direction", "destination"]
    return group[cols].to_dict(orient="records")


# ─── Subgroup evaluation ──────────────────────────────────────────────────────


def _eval_subgroup(
    df_raw: pd.DataFrame,
    scenario_ids: list,
    ft_model_id: str,
    client: Any,
    group_label: str,
) -> tuple[dict, list[dict]]:
    """
    Evaluate model on a set of scenario_ids.
    Returns (metrics_dict, records_list).
    """
    records: list[dict] = []
    y_true_int: list[int] = []
    y_pred_int: list[int] = []
    wt_maes: list[float] = []

    for sc_id in scenario_ids:
        group = df_raw[df_raw["scenario_id"] == sc_id]
        if len(group) == 0:
            continue
        vehicles = _build_vehicle_list(group)
        first = group.iloc[0]
        true_label = str(first["is_conflict"]).strip().lower()
        true_int = 1 if true_label == "yes" else 0

        # LLM prediction
        pred_label = _predict_yn(vehicles, ft_model_id, client)
        pred_int = 1 if "yes" in pred_label else 0

        # Rule-based reference for waiting times
        ref = _rule_predict(vehicles)
        ref_wts = ref.get("waiting_times", {})
        avg_ref_wt = float(np.mean(list(ref_wts.values()))) if ref_wts else 0.0

        # Waiting-time MAE: compare pred_int==1 (yes) vs ref avg wt
        wt_pred = avg_ref_wt if pred_int == 1 else 0.0
        wt_mae = abs(wt_pred - avg_ref_wt)
        wt_maes.append(wt_mae)

        y_true_int.append(true_int)
        y_pred_int.append(pred_int)

        records.append(
            {
                "scenario_id": sc_id,
                "group": group_label,
                "true_label": true_label,
                "predicted_label": pred_label,
                "correct": pred_label == true_label,
                "n_vehicles": len(vehicles),
                "wt_mae": wt_mae,
                "vehicles_json": json.dumps({"vehicles": vehicles}),
            }
        )

    if not y_true_int:
        return {}, records

    acc = accuracy_score(y_true_int, y_pred_int)
    rec = recall_score(y_true_int, y_pred_int, zero_division=0)
    fn = sum(1 for t, p in zip(y_true_int, y_pred_int) if t == 1 and p == 0)
    tp = sum(1 for t, p in zip(y_true_int, y_pred_int) if t == 1 and p == 1)
    fnr = fn / (fn + tp + 1e-9)
    prec = precision_score(y_true_int, y_pred_int, zero_division=0)
    f1 = f1_score(y_true_int, y_pred_int, zero_division=0)

    metrics = {
        "n": len(y_true_int),
        "accuracy": round(acc, 4),
        "recall": round(rec, 4),
        "precision": round(round(prec, 4), 4),
        "f1": round(f1, 4),
        "fnr": round(fnr, 4),
        "wt_mae_s": round(float(np.mean(wt_maes)), 4),
        "conflict_rate_true": round(float(np.mean(y_true_int)), 4),
        "conflict_rate_pred": round(float(np.mean(y_pred_int)), 4),
    }
    return metrics, records


def _disparity(metrics_dict: dict, metric: str) -> float:
    vals = [v[metric] for v in metrics_dict.values() if v and metric in v]
    return round(max(vals) - min(vals), 4) if len(vals) >= 2 else 0.0


def _bias_flag(disparity_val: float, threshold: float = BIAS_THRESHOLD_F1) -> bool:
    return disparity_val > threshold


# ─── Core audit loop ──────────────────────────────────────────────────────────


def run_bias_audit(session: Any = None) -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    df_raw = _load_raw_csv(RAW_CSV)
    sc_df = _get_scenario_rows(df_raw)
    print(f"\nLoaded {len(sc_df)} scenarios from {RAW_CSV.name}")

    all_records: list[dict] = []

    # ─── Dimensions config ────────────────────────────────────────────────────
    DIMENSIONS: list[tuple[str, str]] = [
        ("direction", "direction"),
        ("lane_parity", "movement_type"),
        ("vehicle_count", None),  # special handling
        ("movement_type", "movement_type"),
        ("speed_range", "speed_range"),
        ("distance_range", "distance_range"),
        ("conflict_density", "conflict_density"),
    ]

    audit_results: dict[str, dict] = {}

    for dim_name, col in DIMENSIONS:
        print(f"\n--- Dimension: {dim_name} ---")
        dim_metrics: dict[str, dict] = {}

        if dim_name == "vehicle_count":
            # Count vehicles per scenario
            vc_df = df_raw.groupby("scenario_id").size().reset_index(name="n_vehicles")
            sc_aug = sc_df.merge(vc_df, on="scenario_id")
            sc_aug["vehicle_count_group"] = sc_aug["n_vehicles"].apply(
                lambda n: "small (2-4)" if n <= 4 else "large (5-8)"
            )
            groups = sc_aug.groupby("vehicle_count_group")["scenario_id"].apply(list)
        elif col is not None:
            groups = sc_df.groupby(col, observed=True)["scenario_id"].apply(list)
        else:
            continue

        for group_val, sc_ids in groups.items():
            sampled = list(
                pd.Series(sc_ids).sample(min(MAX_PER_GROUP, len(sc_ids)), random_state=RANDOM_SEED)
            )
            print(f"  {group_val}: {len(sampled)} scenarios", end="  ", flush=True)
            t0 = time.perf_counter()
            metrics, records = _eval_subgroup(
                df_raw, sampled, FT_MODEL_ID, client, group_label=str(group_val)
            )
            elapsed = time.perf_counter() - t0
            print(
                f"acc={metrics.get('accuracy','?'):.3f}  "
                f"fnr={metrics.get('fnr','?'):.3f}  "
                f"f1={metrics.get('f1','?'):.3f}  "
                f"({elapsed:.1f}s)"
            )
            if metrics:
                dim_metrics[str(group_val)] = metrics
            all_records.extend(records)

        f1_disp = _disparity(dim_metrics, "f1")
        fnr_disp = _disparity(dim_metrics, "fnr")
        acc_disp = _disparity(dim_metrics, "accuracy")
        audit_results[dim_name] = {
            "group_metrics": dim_metrics,
            "f1_disparity": f1_disp,
            "fnr_disparity": fnr_disp,
            "acc_disparity": acc_disp,
            "bias_detected": _bias_flag(f1_disp),
            "safety_bias_detected": _bias_flag(fnr_disp, BIAS_THRESHOLD_FNR),
        }

    # ── Overall summary ────────────────────────────────────────────────────────
    biased_dims = [d for d, r in audit_results.items() if r["bias_detected"]]
    safety_biased = [d for d, r in audit_results.items() if r["safety_bias_detected"]]

    report = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "model": FT_MODEL_ID,
        "max_per_group": MAX_PER_GROUP,
        "bias_threshold_f1": BIAS_THRESHOLD_F1,
        "bias_threshold_fnr": BIAS_THRESHOLD_FNR,
        "audit": audit_results,
        "biased_dimensions": biased_dims,
        "safety_biased_dimensions": safety_biased,
        "overall_bias_detected": len(biased_dims) > 0,
        "figures": sorted(p.name for p in FIG_DIR.glob("*.png")),
    }

    (REPORT_DIR / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\n[Bias] report.json → {REPORT_DIR / 'report.json'}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    _generate_bias_plots(audit_results)

    # Upload to Phoenix
    df_all = pd.DataFrame(all_records)
    _upload_dataset(session, df_all, "bias_audit_predictions")

    return report


# ─── Plots ────────────────────────────────────────────────────────────────────

plt_cfg = {
    "font.size": 11,
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
}

DIR_COLORS = {"north": "#4E79A7", "south": "#F28E2B", "east": "#59A14F", "west": "#E15759"}


def _generate_bias_plots(audit: dict) -> None:
    plt.rcParams.update(plt_cfg)

    # ── 1. FNR by direction (Safety fairness) ─────────────────────────────────
    if "direction" in audit and audit["direction"]["group_metrics"]:
        gm = audit["direction"]["group_metrics"]
        dirs = list(gm.keys())
        fnrs = [gm[d]["fnr"] for d in dirs]
        colors = [DIR_COLORS.get(d.lower(), "#999") for d in dirs]

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(dirs, fnrs, color=colors, alpha=0.85, width=0.5)
        for bar, v in zip(bars, fnrs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.3f}",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )
        ax.axhline(
            BIAS_THRESHOLD_FNR,
            color="red",
            ls="--",
            lw=1.5,
            label=f"Bias threshold (FNR={BIAS_THRESHOLD_FNR})",
        )
        disp = audit["direction"]["fnr_disparity"]
        ax.set(
            ylabel="False Negative Rate",
            title="Bias Audit — FNR by Direction (Safety Fairness)",
            ylim=(0, max(fnrs + [0.05]) * 1.5),
        )
        ax.legend()
        ax.annotate(
            f"Disparity: {disp:.3f}",
            xy=(0.98, 0.95),
            xycoords="axes fraction",
            ha="right",
            fontsize=9,
            color="red" if audit["direction"]["safety_bias_detected"] else "green",
            fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fnr_by_direction.png")
        plt.close(fig)

    # ── 2. Accuracy by vehicle count (complexity robustness) ─────────────────
    if "vehicle_count" in audit and audit["vehicle_count"]["group_metrics"]:
        gm = audit["vehicle_count"]["group_metrics"]
        groups = list(gm.keys())
        accs = [gm[g]["accuracy"] for g in groups]
        f1s = [gm[g]["f1"] for g in groups]
        x = np.arange(len(groups))
        w = 0.35

        fig, ax = plt.subplots(figsize=(7, 4))
        bars_a = ax.bar(x - w / 2, accs, w, label="Accuracy", color="#4E79A7", alpha=0.85)
        bars_f = ax.bar(x + w / 2, f1s, w, label="F1", color="#F28E2B", alpha=0.85)
        for bar, v in [*zip(bars_a, accs), *zip(bars_f, f1s)]:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.2f}",
                ha="center",
                fontsize=9,
            )
        ax.set(
            xticks=x,
            xticklabels=groups,
            ylabel="Score",
            title="Bias Audit — Accuracy & F1 by Vehicle Count",
            ylim=(0, 1.15),
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "accuracy_by_vehicle_count.png")
        plt.close(fig)

    # ── 3. Waiting-time MAE histogram across all dimensions ───────────────────
    all_maes: list[float] = []
    labels_for_mae: list[str] = []
    for dim_name, result in audit.items():
        for grp, m in result["group_metrics"].items():
            if m:
                all_maes.append(m["wt_mae_s"])
                labels_for_mae.append(f"{dim_name}:{grp}")

    if all_maes:
        fig, ax = plt.subplots(figsize=(9, 4))
        x_pos = np.arange(len(all_maes))
        colors_mae = plt.cm.tab20(np.linspace(0, 1, len(all_maes)))
        bars = ax.bar(x_pos, all_maes, color=colors_mae, alpha=0.85, width=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels_for_mae, rotation=45, ha="right", fontsize=7)
        ax.axhline(
            float(np.mean(all_maes)),
            color="red",
            ls="--",
            lw=1.5,
            label=f"Mean MAE = {np.mean(all_maes):.2f}s",
        )
        ax.set(
            ylabel="Waiting-Time MAE (s)", title="Bias Audit — Waiting-Time Error Across Subgroups"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "waiting_time_error_by_group.png")
        plt.close(fig)

    # ── 4. Priority / conflict-prediction confusion matrix (direction) ─────────
    if "direction" in audit and audit["direction"]["group_metrics"]:
        gm = audit["direction"]["group_metrics"]
        dirs = list(gm.keys())
        conf_rates = [
            [gm[d].get("conflict_rate_true", 0), gm[d].get("conflict_rate_pred", 0)] for d in dirs
        ]
        matrix = np.array(conf_rates)

        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(dirs))
        w = 0.35
        b_true = ax.bar(
            x - w / 2, matrix[:, 0], w, label="True conflict rate", color="#4E79A7", alpha=0.85
        )
        b_pred = ax.bar(
            x + w / 2, matrix[:, 1], w, label="Pred conflict rate", color="#E15759", alpha=0.85
        )
        ax.set(
            xticks=x,
            xticklabels=dirs,
            ylabel="Conflict rate",
            title="Bias Audit — Priority Assignment: True vs Predicted Conflict Rate by Direction",
            ylim=(0, 1.15),
        )
        ax.legend()
        for bar, v in [*zip(b_true, matrix[:, 0]), *zip(b_pred, matrix[:, 1])]:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.2f}",
                ha="center",
                fontsize=8,
            )
        fig.tight_layout()
        fig.savefig(FIG_DIR / "priority_conflict_rate_by_direction.png")
        plt.close(fig)

    # ── 5. Multi-dimension F1 disparity overview ──────────────────────────────
    dims = list(audit.keys())
    f1_disps = [audit[d]["f1_disparity"] for d in dims]
    fnr_disps = [audit[d]["fnr_disparity"] for d in dims]

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(dims))
    w = 0.35
    b1 = ax.bar(x - w / 2, f1_disps, w, label="F1 disparity", color="#4E79A7", alpha=0.85)
    b2 = ax.bar(x + w / 2, fnr_disps, w, label="FNR disparity", color="#E15759", alpha=0.85)
    ax.axhline(
        BIAS_THRESHOLD_F1,
        color="#4E79A7",
        ls="--",
        lw=1.2,
        alpha=0.7,
        label=f"F1 bias threshold ({BIAS_THRESHOLD_F1})",
    )
    ax.axhline(
        BIAS_THRESHOLD_FNR,
        color="#E15759",
        ls="--",
        lw=1.2,
        alpha=0.7,
        label=f"FNR bias threshold ({BIAS_THRESHOLD_FNR})",
    )
    ax.set(
        xticks=x,
        xticklabels=[d.replace("_", "\n") for d in dims],
        ylabel="Disparity (max − min)",
        title="Bias Audit — Disparity Overview Across All Dimensions",
        ylim=(0, max(f1_disps + fnr_disps + [0.05]) * 1.4),
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "disparity_overview.png")
    plt.close(fig)

    print(f"[Bias] 5 figures saved → {FIG_DIR}")


# ─── Entrypoint ───────────────────────────────────────────────────────────────


def main(session: Any = None) -> dict:
    report = run_bias_audit(session)

    print("\n" + "=" * 70)
    print("BIAS AUDIT SUMMARY")
    print("=" * 70)
    for dim, result in report["audit"].items():
        bias_flag = "⚠  BIAS" if result["bias_detected"] else "   OK"
        safe_flag = "⚠  SAFETY" if result["safety_bias_detected"] else "      "
        print(
            f"  {dim:<20} F1-disp={result['f1_disparity']:.3f}  "
            f"FNR-disp={result['fnr_disparity']:.3f}  {bias_flag}  {safe_flag}"
        )
        for grp, m in result["group_metrics"].items():
            if m:
                print(
                    f"    {grp:<20} n={m['n']:>3}  acc={m['accuracy']:.3f}  "
                    f"fnr={m['fnr']:.3f}  f1={m['f1']:.3f}  wt_mae={m['wt_mae_s']:.2f}s"
                )
    print("-" * 70)
    if report["biased_dimensions"]:
        print(f"  Bias detected in: {', '.join(report['biased_dimensions'])}")
    else:
        print("  No F1 bias detected across any dimension.")
    if report["safety_biased_dimensions"]:
        print(f"  Safety bias (FNR) in: {', '.join(report['safety_biased_dimensions'])}")
    print("=" * 70)
    return report


if __name__ == "__main__":
    main()

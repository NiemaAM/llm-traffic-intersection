"""
phoenix_robustness.py  —  Robustness / Perturbation Testing with Arize Phoenix
================================================================================
Systematically perturbs each input feature, measures prediction stability,
and logs results to Phoenix.

Features tested:
  speed        — Main traffic factor (arrival-time determinant)
  distance     — Arrival-time estimation
  lane         — Path conflict geometry
  destination  — Turn-vs-straight behaviour
  vehicle_order — Prompt ordering sensitivity

Output:
  reports/robustness/report.json
  reports/robustness/figures/*.png
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ─── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "poc"))

load_dotenv(REPO_ROOT / ".env", override=False)
os.chdir(REPO_ROOT)

REPORT_DIR = REPO_ROOT / "reports" / "online_evaluation" / "robustness"
FIG_DIR = REPORT_DIR / "figures"
TEST_CSV = REPO_ROOT / "data" / "masri_finetune" / "eval_only_masri.csv"
FT_MODEL_ID = os.environ.get("FINE_TUNED_MODEL_ID", "ft:gpt-4o-mini-2024-07-18:personal::DX7kzKtB")

N_SCENARIOS = 25  # scenarios per feature test (controls API cost)
RANDOM_SEED = 7

# Perturbation levels per feature
SPEED_DELTAS = [-40, -20, -10, -5, -2, 2, 5, 10, 20, 40]  # km/h
DISTANCE_DELTAS = [-100, -50, -25, -10, -5, 5, 10, 25, 50, 100]  # metres
LANE_SWAPS = "adjacent"  # swap within direction's lanes
DEST_VARIANTS = 3  # alternative destinations per vehicle
ORDER_VARIANTS = ["reverse", "shuffle", "shuffle", "shuffle"]  # reorderings

FEAT_COLORS = {
    "speed": "#E15759",
    "distance": "#4E79A7",
    "lane": "#76B7B2",
    "destination": "#59A14F",
    "vehicle_order": "#F28E2B",
}

LANE_MAP = {"north": [1, 2], "east": [3, 4], "south": [5, 6], "west": [7, 8]}
DEST_BY_DIR = {
    "north": ["F", "H", "E", "D", "C"],
    "east": ["H", "B", "G", "E", "F"],
    "south": ["B", "D", "A", "G", "H"],
    "west": ["D", "F", "B", "C", "A"],
}


# ─── Phoenix / dataset upload ─────────────────────────────────────────────────


def _upload_dataset(session: Any, df: pd.DataFrame, name: str) -> None:
    if session is None:
        return
    try:
        from phoenix.client import Client

        client = Client()
        up = df.copy()
        up["input"] = up.get("vehicles_json", up.index.astype(str))
        up["output"] = up.get("perturbed_pred", up.get("base_pred", "no"))
        meta_cols = [
            c for c in ["feature", "delta", "changed", "base_pred", "n_vehicles"] if c in up.columns
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


# ─── Fast yes/no prediction ───────────────────────────────────────────────────


def _predict_yn(vehicles: list, ft_model_id: str, client: Any) -> str:
    """Call fine-tuned model → 'yes' or 'no'."""
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


# ─── Perturbation generators ──────────────────────────────────────────────────


def _perturb_speed(vehicles: list, delta: float) -> list[dict]:
    return [{**dict(v), "speed": max(5.0, float(v["speed"]) + delta)} for v in vehicles]


def _perturb_distance(vehicles: list, delta: float) -> list[dict]:
    return [
        {
            **dict(v),
            "distance_to_intersection": max(5.0, float(v["distance_to_intersection"]) + delta),
        }
        for v in vehicles
    ]


def _perturb_lane(vehicles: list, v_idx: int) -> list[dict] | None:
    """Swap vehicle at v_idx to the other lane in its direction (or None if only one lane)."""
    direction = vehicles[v_idx].get("direction", "north")
    valid = LANE_MAP.get(direction, [1, 2])
    current = vehicles[v_idx].get("lane", valid[0])
    alts = [l for l in valid if l != current]
    if not alts:
        return None
    perturbed = [dict(v) for v in vehicles]
    perturbed[v_idx]["lane"] = alts[0]
    return perturbed


def _perturb_destination(vehicles: list, v_idx: int, alt_num: int) -> list[dict] | None:
    direction = vehicles[v_idx].get("direction", "north")
    current_dest = vehicles[v_idx].get("destination", "A")
    alts = [d for d in DEST_BY_DIR.get(direction, list("ABCDEFGH")) if d != current_dest]
    if alt_num >= len(alts):
        return None
    perturbed = [dict(v) for v in vehicles]
    perturbed[v_idx]["destination"] = alts[alt_num]
    return perturbed


def _perturb_order(vehicles: list, variant: str) -> list[dict]:
    if variant == "reverse":
        return list(reversed([dict(v) for v in vehicles]))
    else:  # shuffle
        perturbed = [dict(v) for v in vehicles]
        random.shuffle(perturbed)
        return perturbed


# ─── Data loading ─────────────────────────────────────────────────────────────


def _load_scenarios(csv_path: Path, n: int, seed: int) -> list[dict]:
    df = pd.read_csv(csv_path)
    df = df.sample(min(n, len(df)), random_state=seed).reset_index(drop=True)
    out = []
    for _, row in df.iterrows():
        try:
            s = json.loads(row["scenario"])
            vehicles = s.get("vehicles_scenario", s.get("vehicles", []))
            for i, v in enumerate(vehicles):
                v.setdefault("vehicle_id", f"V{i + 1}")
            out.append(
                {
                    "vehicles": vehicles,
                    "true_label": str(row["is_conflict"]).strip().lower(),
                }
            )
        except Exception:
            pass
    return out


# ─── Core evaluation ──────────────────────────────────────────────────────────


def run_robustness(session: Any = None) -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    random.seed(RANDOM_SEED)

    scenarios = _load_scenarios(TEST_CSV, N_SCENARIOS, RANDOM_SEED)
    print(f"\n--- Robustness testing ({len(scenarios)} scenarios × 5 features) ---")

    all_records: list[dict] = []

    # ════════════════════════════════════════════════════════════════
    # SPEED perturbation
    # ════════════════════════════════════════════════════════════════
    print("\n[speed]")
    speed_results: list[dict] = []
    for sc_idx, sc in enumerate(scenarios):
        vehicles = sc["vehicles"]
        base = _predict_yn(vehicles, FT_MODEL_ID, client)
        for delta in SPEED_DELTAS:
            perturbed = _perturb_speed(vehicles, delta)
            pred = _predict_yn(perturbed, FT_MODEL_ID, client)
            changed = pred != base
            speed_results.append(
                {
                    "feature": "speed",
                    "delta": delta,
                    "base_pred": base,
                    "perturbed_pred": pred,
                    "changed": changed,
                    "scenario_idx": sc_idx,
                    "n_vehicles": len(vehicles),
                    "vehicles_json": json.dumps({"vehicles": vehicles}),
                }
            )
        print(
            f"  [{sc_idx+1:>3}/{len(scenarios)}]  base={base}  "
            f"changes={sum(r['changed'] for r in speed_results if r['scenario_idx'] == sc_idx)}/{len(SPEED_DELTAS)}"
        )
    all_records.extend(speed_results)

    # ════════════════════════════════════════════════════════════════
    # DISTANCE perturbation
    # ════════════════════════════════════════════════════════════════
    print("\n[distance]")
    distance_results: list[dict] = []
    for sc_idx, sc in enumerate(scenarios):
        vehicles = sc["vehicles"]
        base = _predict_yn(vehicles, FT_MODEL_ID, client)
        for delta in DISTANCE_DELTAS:
            perturbed = _perturb_distance(vehicles, delta)
            pred = _predict_yn(perturbed, FT_MODEL_ID, client)
            changed = pred != base
            distance_results.append(
                {
                    "feature": "distance",
                    "delta": delta,
                    "base_pred": base,
                    "perturbed_pred": pred,
                    "changed": changed,
                    "scenario_idx": sc_idx,
                    "n_vehicles": len(vehicles),
                    "vehicles_json": json.dumps({"vehicles": vehicles}),
                }
            )
        print(
            f"  [{sc_idx+1:>3}/{len(scenarios)}]  base={base}  "
            f"changes={sum(r['changed'] for r in distance_results if r['scenario_idx'] == sc_idx)}/{len(DISTANCE_DELTAS)}"
        )
    all_records.extend(distance_results)

    # ════════════════════════════════════════════════════════════════
    # LANE perturbation
    # ════════════════════════════════════════════════════════════════
    print("\n[lane]")
    lane_results: list[dict] = []
    for sc_idx, sc in enumerate(scenarios):
        vehicles = sc["vehicles"]
        base = _predict_yn(vehicles, FT_MODEL_ID, client)
        for v_idx in range(len(vehicles)):
            perturbed = _perturb_lane(vehicles, v_idx)
            if perturbed is None:
                continue
            pred = _predict_yn(perturbed, FT_MODEL_ID, client)
            changed = pred != base
            lane_results.append(
                {
                    "feature": "lane",
                    "delta": v_idx,
                    "base_pred": base,
                    "perturbed_pred": pred,
                    "changed": changed,
                    "scenario_idx": sc_idx,
                    "n_vehicles": len(vehicles),
                    "vehicles_json": json.dumps({"vehicles": vehicles}),
                }
            )
        if lane_results:
            print(
                f"  [{sc_idx+1:>3}/{len(scenarios)}]  base={base}  "
                f"changes={sum(r['changed'] for r in lane_results if r['scenario_idx'] == sc_idx)}"
            )
    all_records.extend(lane_results)

    # ════════════════════════════════════════════════════════════════
    # DESTINATION perturbation
    # ════════════════════════════════════════════════════════════════
    print("\n[destination]")
    dest_results: list[dict] = []
    for sc_idx, sc in enumerate(scenarios):
        vehicles = sc["vehicles"]
        base = _predict_yn(vehicles, FT_MODEL_ID, client)
        for v_idx in range(len(vehicles)):
            for alt in range(DEST_VARIANTS):
                perturbed = _perturb_destination(vehicles, v_idx, alt)
                if perturbed is None:
                    continue
                pred = _predict_yn(perturbed, FT_MODEL_ID, client)
                changed = pred != base
                dest_results.append(
                    {
                        "feature": "destination",
                        "delta": alt,
                        "base_pred": base,
                        "perturbed_pred": pred,
                        "changed": changed,
                        "scenario_idx": sc_idx,
                        "n_vehicles": len(vehicles),
                        "vehicles_json": json.dumps({"vehicles": vehicles}),
                    }
                )
        if dest_results:
            print(
                f"  [{sc_idx+1:>3}/{len(scenarios)}]  base={base}  "
                f"changes={sum(r['changed'] for r in dest_results if r['scenario_idx'] == sc_idx)}"
            )
    all_records.extend(dest_results)

    # ════════════════════════════════════════════════════════════════
    # VEHICLE ORDER perturbation
    # ════════════════════════════════════════════════════════════════
    print("\n[vehicle_order]")
    order_results: list[dict] = []
    for sc_idx, sc in enumerate(scenarios):
        vehicles = sc["vehicles"]
        if len(vehicles) < 2:
            continue
        base = _predict_yn(vehicles, FT_MODEL_ID, client)
        for variant in ORDER_VARIANTS:
            perturbed = _perturb_order(vehicles, variant)
            if perturbed == vehicles:
                continue
            pred = _predict_yn(perturbed, FT_MODEL_ID, client)
            changed = pred != base
            order_results.append(
                {
                    "feature": "vehicle_order",
                    "delta": variant,
                    "base_pred": base,
                    "perturbed_pred": pred,
                    "changed": changed,
                    "scenario_idx": sc_idx,
                    "n_vehicles": len(vehicles),
                    "vehicles_json": json.dumps({"vehicles": vehicles}),
                }
            )
        print(
            f"  [{sc_idx+1:>3}/{len(scenarios)}]  base={base}  "
            f"changes={sum(r['changed'] for r in order_results if r['scenario_idx'] == sc_idx)}"
        )
    all_records.extend(order_results)

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    df_all = pd.DataFrame(all_records)

    def _feat_summary(df: pd.DataFrame, feature: str) -> dict:
        sub = df[df["feature"] == feature]
        if len(sub) == 0:
            return {}
        total_tests = len(sub)
        flip_count = int(sub["changed"].sum())
        flip_rate = flip_count / total_tests

        # Per-delta flip rate
        delta_rates: dict = {}
        for delta, grp in sub.groupby("delta"):
            delta_rates[str(delta)] = round(float(grp["changed"].mean()), 4)

        return {
            "total_tests": total_tests,
            "flip_count": flip_count,
            "flip_rate": round(flip_rate, 4),
            "delta_flip_rates": delta_rates,
        }

    feature_summaries = {
        feat: _feat_summary(df_all, feat)
        for feat in ["speed", "distance", "lane", "destination", "vehicle_order"]
    }

    # Overall stability = mean (1 - flip_rate) across all features
    stabilities = [1 - v["flip_rate"] for v in feature_summaries.values() if v]
    overall_stability = round(float(np.mean(stabilities)), 4) if stabilities else 1.0

    # Table example (speed: 40→42 type rows)
    example_table: list[dict] = []
    speed_df = df_all[df_all["feature"] == "speed"]
    for delta in [2, 5, 10, 20, 40]:
        sub = speed_df[speed_df["delta"] == delta]
        if len(sub):
            flip_rate = float(sub["changed"].mean())
            example_table.append(
                {
                    "feature": "speed",
                    "delta": f"+{delta} km/h",
                    "flip_rate": round(flip_rate, 4),
                    "example": f"40 km/h → {40+delta} km/h",
                    "decision_changed": "Yes (flipped)" if flip_rate > 0.5 else "No (stable)",
                }
            )

    # ── Plots ─────────────────────────────────────────────────────────────────
    _generate_robustness_plots(df_all, feature_summaries)

    # ── Report ─────────────────────────────────────────────────────────────────
    report = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "model": FT_MODEL_ID,
        "n_scenarios": len(scenarios),
        "overall_stability": overall_stability,
        "feature_summaries": feature_summaries,
        "feature_ranking_by_sensitivity": sorted(
            feature_summaries.keys(),
            key=lambda f: feature_summaries[f].get("flip_rate", 0),
            reverse=True,
        ),
        "example_speed_table": example_table,
        "figures": sorted(p.name for p in FIG_DIR.glob("*.png")),
    }

    (REPORT_DIR / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\n[Robustness] report.json → {REPORT_DIR / 'report.json'}")

    # Upload to Phoenix
    _upload_dataset(session, df_all, "robustness_perturbations")

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


def _generate_robustness_plots(df_all: pd.DataFrame, feat_summaries: dict) -> None:
    plt.rcParams.update(plt_cfg)

    # ── 1. Sensitivity by feature (overall flip rates) ────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    feats = sorted(
        feat_summaries, key=lambda f: feat_summaries[f].get("flip_rate", 0), reverse=True
    )
    flip_rates = [feat_summaries[f].get("flip_rate", 0) for f in feats]
    colors = [FEAT_COLORS.get(f, "#999") for f in feats]
    bars = ax.barh(feats[::-1], flip_rates[::-1], color=colors[::-1], alpha=0.85)
    for bar, v in zip(bars, flip_rates[::-1]):
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontsize=9)
    ax.axvline(0.1, color="orange", ls="--", lw=1, label="10% flip threshold")
    ax.axvline(0.3, color="red", ls="--", lw=1, label="30% flip threshold")
    ax.set(
        xlabel="Flip rate (fraction of predictions changed)",
        title="Robustness — Sensitivity by Feature",
        xlim=(0, max(flip_rates + [0.01]) * 1.4),
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sensitivity_by_feature.png")
    plt.close(fig)

    # ── 2. Perturbation heatmap (speed × distance: magnitude vs flip rate) ─────
    speed_df = df_all[df_all["feature"] == "speed"]
    dist_df = df_all[df_all["feature"] == "distance"]

    speed_deltas = sorted(speed_df["delta"].unique()) if len(speed_df) else []
    dist_deltas = sorted(dist_df["delta"].unique()) if len(dist_df) else []

    if speed_deltas and dist_deltas:
        s_rates = [float(speed_df[speed_df["delta"] == d]["changed"].mean()) for d in speed_deltas]
        d_rates = [float(dist_df[dist_df["delta"] == d]["changed"].mean()) for d in dist_deltas]
        matrix = np.outer(np.array(s_rates), np.array(d_rates))

        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
        ax.set_xticks(range(len(dist_deltas)))
        ax.set_xticklabels([f"{int(d):+d}m" for d in dist_deltas], fontsize=8)
        ax.set_yticks(range(len(speed_deltas)))
        ax.set_yticklabels([f"{int(d):+d}km/h" for d in speed_deltas], fontsize=8)
        ax.set(
            xlabel="Distance delta",
            ylabel="Speed delta",
            title="Perturbation Heatmap — Speed × Distance flip-rate product",
        )
        plt.colorbar(im, ax=ax, label="Flip-rate product")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "perturbation_heatmap.png")
        plt.close(fig)

    # ── 3. Stability curve — flip rate vs perturbation magnitude (speed) ──────
    if speed_deltas:
        abs_deltas = [abs(d) for d in speed_deltas]
        # Average flip rate for magnitude |delta|
        mag_to_rate: dict[int, list] = {}
        for _, row in speed_df.iterrows():
            mag = abs(row["delta"])
            mag_to_rate.setdefault(mag, []).append(int(row["changed"]))
        mags = sorted(mag_to_rate)
        rates = [np.mean(mag_to_rate[m]) for m in mags]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(mags, rates, marker="o", color=FEAT_COLORS["speed"], lw=2)
        ax.fill_between(mags, rates, alpha=0.15, color=FEAT_COLORS["speed"])
        ax.axhline(0.5, color="red", ls="--", lw=1, label="50% flip line")
        ax.set(
            xlabel="|Speed delta| (km/h)",
            ylabel="Flip rate",
            title="Stability Curve — Speed Perturbation Magnitude vs Flip Rate",
            ylim=(0, 1.05),
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "stability_curve_speed.png")
        plt.close(fig)

    # ── 4. Per-feature flip rate comparison (grouped bars) ───────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    feat_list = list(feat_summaries.keys())
    x = np.arange(len(feat_list))
    flips = [feat_summaries[f].get("flip_rate", 0) for f in feat_list]
    stab = [1 - r for r in flips]
    bars_f = ax.bar(
        x - 0.2,
        flips,
        width=0.35,
        label="Flip rate",
        alpha=0.85,
        color=[FEAT_COLORS.get(f, "#999") for f in feat_list],
    )
    bars_s = ax.bar(
        x + 0.2,
        stab,
        width=0.35,
        label="Stability",
        alpha=0.5,
        color=[FEAT_COLORS.get(f, "#999") for f in feat_list],
        hatch="///",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_", "\n") for f in feat_list], fontsize=9)
    ax.set(ylabel="Rate", title="Flip Rate vs Stability per Feature", ylim=(0, 1.1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "flip_vs_stability.png")
    plt.close(fig)

    print(f"[Robustness] 4 figures saved → {FIG_DIR}")


# ─── Entrypoint ───────────────────────────────────────────────────────────────


def main(session: Any = None) -> dict:
    report = run_robustness(session)

    print("\n" + "=" * 60)
    print("ROBUSTNESS SUMMARY")
    print("=" * 60)
    print(f"  Overall stability : {report['overall_stability']:.4f}")
    print(f"  Sensitivity rank  : {' > '.join(report['feature_ranking_by_sensitivity'])}")
    print("\n  Per-feature flip rates:")
    for feat, s in report["feature_summaries"].items():
        if s:
            bar = "█" * max(1, int(s["flip_rate"] * 20))
            print(
                f"    {feat:<18} {bar} {s['flip_rate']:.4f}  ({s['flip_count']}/{s['total_tests']} tests)"
            )
    if report["example_speed_table"]:
        print("\n  Speed perturbation examples:")
        print(f"  {'Delta':<12} {'Example':<20} {'Flip rate':<12} {'Stable?'}")
        print("  " + "-" * 55)
        for row in report["example_speed_table"]:
            stable = "stable" if row["flip_rate"] < 0.5 else "UNSTABLE"
            print(f"  {row['delta']:<12} {row['example']:<20} {row['flip_rate']:<12.4f} {stable}")
    print("=" * 60)
    return report


if __name__ == "__main__":
    main()

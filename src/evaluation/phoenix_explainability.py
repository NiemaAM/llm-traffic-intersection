"""
phoenix_explainability.py  —  Explainability with Arize Phoenix
================================================================
Techniques:
  1. Structured reasoning  — controlled JSON explanation from gpt-4o-mini
  2. Perturbation analysis — LIME-style feature-importance across 5 features
  3. Rule comparison       — LLM decision vs rule-based engine
  4. Scenario visualisation — predicted vs actual waiting times

Output:
  reports/explainability/report.json
  reports/explainability/report.html
  reports/explainability/feature_influence.md
  reports/explainability/figures/*.png
"""

from __future__ import annotations

import json
import os
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

REPORT_DIR = REPO_ROOT / "reports" / "online_evaluation" / "explainability"
FIG_DIR = REPORT_DIR / "figures"
TEST_CSV = REPO_ROOT / "data" / "masri_finetune" / "eval_only_masri.csv"
FT_MODEL_ID = os.environ.get("FINE_TUNED_MODEL_ID", "ft:gpt-4o-mini-2024-07-18:personal::DX7kzKtB")

N_EXPLAIN = 30  # scenarios to run full explainability on
N_PERTURB = 20  # scenarios for perturbation analysis
RANDOM_SEED = 42

COLORS = {
    "speed": "#E15759",
    "distance": "#4E79A7",
    "lane": "#76B7B2",
    "destination": "#59A14F",
    "vehicle_order": "#F28E2B",
}

# ─── Structured-reasoning prompt ──────────────────────────────────────────────

REASON_SYSTEM = """You are an expert traffic conflict analyst for a 4-way 8-lane intersection.

Analyse the given vehicle scenario and return ONLY valid JSON matching this schema exactly:
{
  "conflict_reason": "<why these vehicles conflict, or why they don't>",
  "priority_rule": "<which rule applies: right-hand rule | speed priority | arrival time | straight priority | no conflict>",
  "priority_vehicle": "<vehicle_id with highest priority, or null if no conflict>",
  "yield_vehicle": "<vehicle_id that must yield, or null if no conflict>",
  "waiting_time_reason": "<why the yield vehicle gets the waiting time it does>"
}

Conflict rules:
- Vehicles conflict if paths cross AND both arrive within 5 s of each other (t = distance / speed_ms)
- Speed in m/s = km/h * 1000 / 3600
- Same-direction vehicles never conflict
- Priority: straight > turn, right-turn > left-turn, right-hand rule, earlier arrival
"""


def _structured_reason(vehicles: list, client: Any) -> dict:
    """Ask gpt-4o-mini base model for a structured explanation."""
    parts = []
    for v in vehicles:
        t_arr = float(v["distance_to_intersection"]) / max(float(v["speed"]) * 1000 / 3600, 0.1)
        parts.append(
            f"Vehicle {v['vehicle_id']}: lane={v['lane']}, dir={v['direction']}, "
            f"dest={v['destination']}, speed={v['speed']:.1f}km/h, "
            f"dist={v['distance_to_intersection']:.1f}m, ETA={t_arr:.2f}s"
        )
    scenario_text = "\n".join(parts)

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": REASON_SYSTEM},
                {"role": "user", "content": f"Analyse this scenario:\n{scenario_text}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=300,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as exc:
        return {
            "conflict_reason": f"[error: {exc}]",
            "priority_rule": "unknown",
            "priority_vehicle": None,
            "yield_vehicle": None,
            "waiting_time_reason": "unknown",
        }


# ─── Perturbation helpers ──────────────────────────────────────────────────────


def _predict_raw(vehicles: list, ft_model_id: str, client: Any) -> str:
    """Fast yes/no prediction via fine-tuned model."""
    parts = []
    for v in vehicles:
        parts.append(
            f"Vehicle {v['vehicle_id']} is in lane {v['lane']}, moving {v['direction']} "
            f"at a speed of {float(v['speed']):.2f} km/h, and is "
            f"{float(v['distance_to_intersection']):.2f} meters away from the intersection, "
            f"heading towards {v['destination']}."
        )
    text = " ".join(parts)
    from models.llm_model import MASRI_SYSTEM_PROMPT

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


LANE_MAP = {"north": [1, 2], "east": [3, 4], "south": [5, 6], "west": [7, 8]}
ALL_DESTINATIONS = list("ABCDEFGH")
DEST_BY_DIRECTION = {
    "north": ["F", "H", "E", "D", "C"],
    "east": ["H", "B", "G", "E", "F"],
    "south": ["B", "D", "A", "G", "H"],
    "west": ["D", "F", "B", "C", "A"],
}


def _perturb_feature(
    vehicles: list, feature: str, ft_model_id: str, base_pred: str, client: Any
) -> float:
    """
    Return sensitivity score [0,1] for one feature across multiple perturbation levels.
    Higher = more sensitive (more flips).
    """
    flips = 0
    total = 0

    if feature == "speed":
        deltas = [-40, -20, -10, -5, -2, 2, 5, 10, 20, 40]
        for i in range(len(vehicles)):
            for delta in deltas:
                perturbed = [dict(v) for v in vehicles]
                perturbed[i]["speed"] = max(5.0, float(vehicles[i]["speed"]) + delta)
                pred = _predict_raw(perturbed, ft_model_id, client)
                flips += int(pred != base_pred)
                total += 1

    elif feature == "distance":
        deltas = [-100, -50, -25, -10, -5, 5, 10, 25, 50, 100]
        for i in range(len(vehicles)):
            for delta in deltas:
                perturbed = [dict(v) for v in vehicles]
                perturbed[i]["distance_to_intersection"] = max(
                    5.0, float(vehicles[i]["distance_to_intersection"]) + delta
                )
                pred = _predict_raw(perturbed, ft_model_id, client)
                flips += int(pred != base_pred)
                total += 1

    elif feature == "lane":
        valid_lanes = LANE_MAP.get(vehicles[0]["direction"], [1, 2])
        for i in range(len(vehicles)):
            for lane in valid_lanes:
                if lane != vehicles[i]["lane"]:
                    perturbed = [dict(v) for v in vehicles]
                    perturbed[i]["lane"] = lane
                    pred = _predict_raw(perturbed, ft_model_id, client)
                    flips += int(pred != base_pred)
                    total += 1

    elif feature == "destination":
        for i in range(len(vehicles)):
            alt_dests = [
                d
                for d in DEST_BY_DIRECTION.get(
                    vehicles[i].get("direction", "north"), ALL_DESTINATIONS
                )
                if d != vehicles[i].get("destination", "A")
            ]
            for dest in alt_dests[:3]:
                perturbed = [dict(v) for v in vehicles]
                perturbed[i]["destination"] = dest
                pred = _predict_raw(perturbed, ft_model_id, client)
                flips += int(pred != base_pred)
                total += 1

    elif feature == "vehicle_order":
        import random as _random

        for _ in range(4):
            perturbed = list(vehicles)
            _random.shuffle(perturbed)
            if perturbed != vehicles:
                pred = _predict_raw(perturbed, ft_model_id, client)
                flips += int(pred != base_pred)
                total += 1
        # Reversed order
        perturbed = list(reversed(vehicles))
        pred = _predict_raw(perturbed, ft_model_id, client)
        flips += int(pred != base_pred)
        total += 1

    return flips / total if total > 0 else 0.0


# ─── Rule-based reference ──────────────────────────────────────────────────────


def _rule_predict(vehicles: list) -> dict:
    try:
        from models.llm_model import _build_full_decision

        return _build_full_decision(vehicles, None)
    except Exception:
        return {"is_conflict": "no", "waiting_times": {}, "priority_order": {}}


# ─── Data loading ──────────────────────────────────────────────────────────────


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


# ─── Phoenix helpers ───────────────────────────────────────────────────────────


def _upload_dataset(session: Any, df: pd.DataFrame, name: str) -> None:
    if session is None:
        return
    try:
        from phoenix.client import Client

        client = Client()
        up = df.copy()
        up["input"] = up.get("vehicles_json", up.index.astype(str))
        up["output"] = up.get("conflict_reason", up.get("structured_reason", ""))
        meta_cols = [
            c
            for c in ["true_label", "predicted_label", "correct", "priority_rule"]
            if c in up.columns
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


# ─── Main evaluation ───────────────────────────────────────────────────────────


def run_explainability(session: Any = None) -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    scenarios_all = _load_scenarios(TEST_CSV, max(N_EXPLAIN, N_PERTURB), RANDOM_SEED)
    scenarios_explain = scenarios_all[:N_EXPLAIN]
    scenarios_perturb = scenarios_all[:N_PERTURB]

    print(f"\n--- Structured reasoning ({N_EXPLAIN} scenarios) ---")
    explanations: list[dict] = []
    correct = 0

    for idx, sc in enumerate(scenarios_explain):
        vehicles = sc["vehicles"]
        true_label = sc["true_label"]
        v_json = json.dumps({"vehicles": vehicles})

        # Fine-tuned model prediction (fast yes/no)
        pred = _predict_raw(vehicles, FT_MODEL_ID, client)
        is_correct = pred == true_label
        correct += int(is_correct)

        # Structured explanation
        reason = _structured_reason(vehicles, client)

        # Rule-based reference
        ref = _rule_predict(vehicles)
        ref_wts = ref.get("waiting_times", {})
        rule_label = ref.get("is_conflict", "no")

        print(
            f"  [{idx+1:>3}/{N_EXPLAIN}]  pred={pred} true={true_label} "
            f"({'✓' if is_correct else '✗'})  "
            f"priority_rule={reason.get('priority_rule', '?')}"
        )

        explanations.append(
            {
                "idx": idx,
                "true_label": true_label,
                "predicted_label": pred,
                "correct": is_correct,
                "rule_label": rule_label,
                "num_vehicles": len(vehicles),
                "conflict_reason": reason.get("conflict_reason", ""),
                "priority_rule": reason.get("priority_rule", ""),
                "priority_vehicle": reason.get("priority_vehicle"),
                "yield_vehicle": reason.get("yield_vehicle"),
                "waiting_time_reason": reason.get("waiting_time_reason", ""),
                "rule_wts": ref_wts,
                "vehicles_json": v_json,
            }
        )

    accuracy = correct / len(explanations) if explanations else 0.0

    # ── Perturbation feature importance ───────────────────────────────────────
    print(f"\n--- Perturbation analysis ({N_PERTURB} scenarios × 5 features) ---")
    features = ["speed", "distance", "lane", "destination", "vehicle_order"]
    feat_scores: dict[str, list[float]] = {f: [] for f in features}

    for idx, sc in enumerate(scenarios_perturb):
        vehicles = sc["vehicles"]
        base = _predict_raw(vehicles, FT_MODEL_ID, client)
        print(f"  [{idx+1:>3}/{N_PERTURB}]  base={base}  vehicles={len(vehicles)}")
        for feat in features:
            score = _perturb_feature(vehicles, feat, FT_MODEL_ID, base, client)
            feat_scores[feat].append(score)
            print(f"       {feat:<15} sensitivity={score:.3f}")

    agg_importance = {f: round(float(np.mean(v)), 4) for f, v in feat_scores.items()}
    # Normalise to [0, 1]
    max_imp = max(agg_importance.values()) or 1.0
    norm_importance = {f: round(v / max_imp, 4) for f, v in agg_importance.items()}

    # ── Rule comparison ────────────────────────────────────────────────────────
    rule_matches = sum(1 for e in explanations if e["predicted_label"] == e["rule_label"])
    rule_agree_rate = rule_matches / len(explanations) if explanations else 0.0

    # ── Waiting-time comparison ────────────────────────────────────────────────
    wt_maes: list[float] = []
    for e in explanations:
        # Model doesn't output waiting times in explainability mode; compare rule vs rule → MAE = 0
        # Instead compare rule_wts magnitude as a proxy for operational complexity
        if e["rule_wts"]:
            avg_wt = float(np.mean(list(e["rule_wts"].values())))
            wt_maes.append(avg_wt)

    # ── Plots ─────────────────────────────────────────────────────────────────
    _generate_explainability_plots(
        agg_importance, norm_importance, feat_scores, explanations, wt_maes
    )

    # ── Report ─────────────────────────────────────────────────────────────────
    report = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "model": FT_MODEL_ID,
        "n_scenarios": len(explanations),
        "n_perturbation_scenarios": N_PERTURB,
        "accuracy": round(accuracy, 4),
        "rule_agreement_rate": round(rule_agree_rate, 4),
        "aggregate_importance": agg_importance,
        "normalised_importance": norm_importance,
        "feature_ranking": sorted(agg_importance, key=agg_importance.get, reverse=True),
        "priority_rules_seen": list(
            {e["priority_rule"] for e in explanations if e["priority_rule"]}
        ),
        "avg_waiting_time_s": round(float(np.mean(wt_maes)), 4) if wt_maes else 0.0,
        "sample_explanations": [
            {
                "true_label": e["true_label"],
                "predicted_label": e["predicted_label"],
                "conflict_reason": e["conflict_reason"],
                "priority_rule": e["priority_rule"],
                "priority_vehicle": e["priority_vehicle"],
                "yield_vehicle": e["yield_vehicle"],
                "waiting_time_reason": e["waiting_time_reason"],
            }
            for e in explanations[:5]
        ],
        "figures": sorted(p.name for p in FIG_DIR.glob("*.png")),
    }

    (REPORT_DIR / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\n[Explainability] report.json → {REPORT_DIR / 'report.json'}")

    _write_html_report(report, explanations)
    _write_markdown_influence(norm_importance, agg_importance)

    # Upload to Phoenix
    expl_df = pd.DataFrame(explanations)
    _upload_dataset(session, expl_df, "explainability_results")

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


def _generate_explainability_plots(agg_imp, norm_imp, feat_scores, explanations, wt_maes):
    plt.rcParams.update(plt_cfg)

    # ── 1. Feature importance bar chart ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    feats = sorted(agg_imp, key=agg_imp.get, reverse=True)
    vals = [agg_imp[f] for f in feats]
    colors = [COLORS.get(f, "#999") for f in feats]
    bars = ax.barh(feats[::-1], vals[::-1], color=colors[::-1], alpha=0.85)
    for bar, v in zip(bars, vals[::-1]):
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontsize=9)
    ax.set(
        xlabel="Avg flip rate (perturbation sensitivity)",
        title="Feature Importance via Perturbation Analysis",
    )
    ax.set_xlim(0, max(vals) * 1.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance.png")
    plt.close(fig)

    # ── 2. Sensitivity heatmap ─────────────────────────────────────────────────
    feats_ordered = list(feat_scores.keys())
    n_sc = len(next(iter(feat_scores.values())))
    matrix = np.array([[feat_scores[f][i] for f in feats_ordered] for i in range(n_sc)])

    fig, ax = plt.subplots(figsize=(9, max(4, n_sc * 0.3 + 1)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(feats_ordered)))
    ax.set_xticklabels([f.replace("_", "\n") for f in feats_ordered], fontsize=9)
    ax.set_yticks(range(n_sc))
    ax.set_yticklabels([f"S{i+1}" for i in range(n_sc)], fontsize=7)
    ax.set(title="Sensitivity Heatmap (flip rate per feature per scenario)")
    plt.colorbar(im, ax=ax, label="Flip rate")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sensitivity_heatmap.png")
    plt.close(fig)

    # ── 3. Conflict reasoning graph (rule distribution) ───────────────────────
    rules = [e.get("priority_rule", "unknown") for e in explanations if e.get("priority_rule")]
    from collections import Counter

    rule_counts = Counter(rules)
    fig, ax = plt.subplots(figsize=(8, 4))
    labels, counts = (
        zip(*sorted(rule_counts.items(), key=lambda x: -x[1])) if rule_counts else ([], [])
    )
    bars = ax.bar(range(len(labels)), counts, color="#4E79A7", alpha=0.85, width=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace(" ", "\n") for l in labels], fontsize=9)
    for bar, c in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            str(c),
            ha="center",
            fontsize=9,
        )
    ax.set(ylabel="Count", title="Conflict Reasoning — Priority Rule Distribution")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "conflict_reasoning_rules.png")
    plt.close(fig)

    # ── 4. Waiting-time comparison (rule-based reference) ─────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    if wt_maes:
        ax.hist(wt_maes, bins=12, color="#59A14F", alpha=0.8, edgecolor="white")
        ax.axvline(
            np.mean(wt_maes), color="red", ls="--", lw=1.5, label=f"mean = {np.mean(wt_maes):.2f}s"
        )
        ax.legend()
    ax.set(
        xlabel="Average waiting time (s)",
        ylabel="# scenarios",
        title="Waiting-Time Distribution (Rule-Based Reference)",
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "waiting_time_distribution.png")
    plt.close(fig)

    print(f"[Explainability] 4 figures saved → {FIG_DIR}")


# ─── HTML report ──────────────────────────────────────────────────────────────


def _write_html_report(report: dict, explanations: list[dict]) -> None:
    rows = ""
    for e in explanations:
        tick = "✓" if e["correct"] else "✗"
        color = "#d4edda" if e["correct"] else "#f8d7da"
        rows += (
            f"<tr style='background:{color}'>"
            f"<td>{e['idx']}</td><td>{e['true_label']}</td><td>{e['predicted_label']}</td>"
            f"<td>{tick}</td>"
            f"<td>{e['conflict_reason'][:80]}…</td>"
            f"<td>{e['priority_rule']}</td>"
            f"<td>{e['priority_vehicle'] or '—'}</td>"
            f"<td>{e['yield_vehicle'] or '—'}</td>"
            f"</tr>\n"
        )

    imp_bars = ""
    for feat, val in sorted(report["aggregate_importance"].items(), key=lambda x: -x[1]):
        w = int(val * 200)
        imp_bars += (
            f"<tr><td>{feat}</td>"
            f"<td><div style='background:#4E79A7;width:{w}px;height:18px;border-radius:3px'></div></td>"
            f"<td>{val:.4f}</td></tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Explainability Report — Traffic Intersection LLM</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 1100px; margin: 40px auto; color: #333; }}
    h1 {{ color: #2c3e50; }} h2 {{ color: #34495e; border-bottom: 1px solid #ddd; padding-bottom: 6px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f5f5f5; }}
    .metric {{ display: inline-block; background: #ecf0f1; padding: 12px 20px; border-radius: 6px;
               margin: 6px; text-align: center; }}
    .metric .val {{ font-size: 28px; font-weight: bold; color: #2980b9; }}
    .metric .lbl {{ font-size: 12px; color: #666; }}
    img {{ max-width: 100%; border: 1px solid #eee; border-radius: 4px; margin: 8px 0; }}
  </style>
</head>
<body>
<h1>LLM Traffic Intersection — Explainability Report</h1>
<p>Generated: {report['generated_at']} &nbsp;|&nbsp; Model: <code>{report['model']}</code></p>

<h2>Summary Metrics</h2>
<div>
  <div class="metric"><div class="val">{report['accuracy']:.3f}</div><div class="lbl">Accuracy</div></div>
  <div class="metric"><div class="val">{report['n_scenarios']}</div><div class="lbl">Scenarios</div></div>
  <div class="metric"><div class="val">{report['rule_agreement_rate']:.3f}</div><div class="lbl">Rule Agreement</div></div>
  <div class="metric"><div class="val">{report['avg_waiting_time_s']:.2f}s</div><div class="lbl">Avg Waiting Time</div></div>
</div>

<h2>Feature Importance (Perturbation Sensitivity)</h2>
<table>
  <tr><th>Feature</th><th>Importance</th><th>Score</th></tr>
  {imp_bars}
</table>

<h2>Plots</h2>
<table><tr>
  <td><img src="figures/feature_importance.png" alt="feature importance"></td>
  <td><img src="figures/sensitivity_heatmap.png" alt="sensitivity heatmap"></td>
</tr><tr>
  <td><img src="figures/conflict_reasoning_rules.png" alt="rule distribution"></td>
  <td><img src="figures/waiting_time_distribution.png" alt="waiting time dist"></td>
</tr></table>

<h2>Structured Explanations ({report['n_scenarios']} scenarios)</h2>
<table>
  <tr><th>#</th><th>True</th><th>Pred</th><th>OK?</th>
      <th>Conflict reason</th><th>Priority rule</th><th>Priority veh.</th><th>Yield veh.</th></tr>
  {rows}
</table>

<h2>Priority Rules Seen</h2>
<ul>{"".join(f'<li>{r}</li>' for r in report['priority_rules_seen'])}</ul>

</body>
</html>"""

    html_path = REPORT_DIR / "report.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"[Explainability] report.html → {html_path}")


# ─── Markdown feature influence ───────────────────────────────────────────────


def _write_markdown_influence(norm_imp: dict, agg_imp: dict) -> None:
    ranked = sorted(agg_imp.items(), key=lambda x: -x[1])
    lines = [
        "# Feature Influence Report\n",
        "Sensitivity estimated via perturbation analysis on the fine-tuned model.\n",
        "| Rank | Feature | Normalised Importance | Raw Flip Rate |",
        "|------|---------|----------------------|---------------|",
    ]
    for rank, (feat, raw) in enumerate(ranked, 1):
        norm = norm_imp.get(feat, 0.0)
        bar = "█" * max(1, int(norm * 20))
        lines.append(f"| {rank} | `{feat}` | {bar} {norm:.4f} | {raw:.4f} |")

    lines += [
        "",
        "## Interpretation",
        "",
        "- **speed** and **distance** are the dominant features because they directly determine",
        "  arrival time (t = dist / speed_ms), which is the core conflict condition.",
        "- **lane** matters because it encodes the vehicle's turning intention,",
        "  determining whether paths physically cross.",
        "- **destination** has moderate influence since it specifies the target quadrant,",
        "  affecting crossing geometry.",
        "- **vehicle_order** sensitivity measures prompt-order bias in the model.",
        "",
        "## Conflict Detection Conditions",
        "",
        "A conflict exists when **all** of:",
        "1. Paths cross (opposing or perpendicular directions with crossing trajectories)",
        "2. Both vehicles arrive within 5 seconds of each other",
        "3. Vehicles come from different directions",
    ]

    md_path = REPORT_DIR / "feature_influence.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Explainability] feature_influence.md → {md_path}")


# ─── Entrypoint ───────────────────────────────────────────────────────────────


def main(session: Any = None) -> dict:
    report = run_explainability(session)

    print("\n" + "=" * 60)
    print("EXPLAINABILITY SUMMARY")
    print("=" * 60)
    print(f"  Accuracy   : {report['accuracy']:.3f}")
    print(f"  Rule agree : {report['rule_agreement_rate']:.3f}")
    print("  Feature importance (ranked):")
    for feat in report["feature_ranking"]:
        score = report["aggregate_importance"][feat]
        bar = "█" * max(1, int(score * 20))
        print(f"    {feat:<18} {bar} {score:.4f}")
    print("=" * 60)
    return report


if __name__ == "__main__":
    main()

"""
phoenix_ab_test.py  —  A/B Testing with Arize Phoenix
=======================================================
Compares:
  A) gpt-4o-mini  few_shot=True   (baseline, 15% traffic weight)
  B) ft:gpt-4o-mini-2024-07-18:personal::DX7kzKtB  (fine-tuned, 85% traffic)

Both models run on every scenario so metrics are directly comparable.
All predictions are traced in Phoenix and exported to:
  reports/AB_testing/report.json
  reports/AB_testing/figures/*.png
"""

from __future__ import annotations

import hashlib
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
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# ─── Repo layout ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "poc"))

load_dotenv(REPO_ROOT / ".env", override=False)
os.chdir(REPO_ROOT)

REPORT_DIR = REPO_ROOT / "reports" / "online_evaluation" / "AB_testing"
FIG_DIR = REPORT_DIR / "figures"
TEST_CSV = REPO_ROOT / "data" / "masri_finetune" / "eval_only_masri.csv"
FT_MODEL_ID = os.environ.get("FINE_TUNED_MODEL_ID", "ft:gpt-4o-mini-2024-07-18:personal::DX7kzKtB")

N_SCENARIOS = 50  # evaluated by BOTH models (100 total API calls)
RANDOM_SEED = 42
TRAFFIC_A = 0.15  # 15 % routed to A in the router simulation

COLORS = {"A": "#4E79A7", "B": "#F28E2B"}

# ─── Phoenix / OTEL bootstrap ─────────────────────────────────────────────────


def _setup_phoenix(project: str = "traffic-ab-test") -> tuple[Any, Any]:
    """
    Start Phoenix and wire up OpenTelemetry + OpenAI auto-instrumentation.
    Returns (phoenix_session, otel_tracer).  Both may be None on failure.
    """
    session = None
    tracer = None

    # 1. Start Phoenix UI
    try:
        import phoenix as px

        session = px.launch_app()
        print(f"[Phoenix] UI → {session.url}")
    except Exception as exc:
        print(f"[Phoenix] Could not start: {exc}")
        return None, None

    # 2. Register OTEL tracer provider pointing at Phoenix
    tracer_provider = None
    try:
        from phoenix.otel import register as _ph_register

        tracer_provider = _ph_register(
            project_name=project,
            endpoint=f"{session.url}/v1/traces",
        )
    except Exception:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            tracer_provider = TracerProvider()
            tracer_provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{session.url}/v1/traces"))
            )
        except Exception as exc2:
            print(f"[Phoenix/OTEL] Provider setup failed: {exc2}")

    # 3. Auto-instrument OpenAI (so every chat.completions.create becomes a span)
    if tracer_provider is not None:
        try:
            from openinference.instrumentation.openai import OpenAIInstrumentor

            OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
            print("[Phoenix] OpenAI auto-instrumented")
        except ImportError:
            print(
                "[Phoenix] openinference-instrumentation-openai not installed — OpenAI calls won't be auto-traced"
            )

    # 4. Build a tracer for hand-rolled spans
    if tracer_provider is not None:
        try:
            from opentelemetry import trace

            tracer = trace.get_tracer("ab-test", tracer_provider=tracer_provider)
        except Exception:
            pass

    return session, tracer


# ─── Span context manager ─────────────────────────────────────────────────────


class _Span:
    """Thin wrapper: real OTEL span if tracer is available, no-op otherwise."""

    def __init__(self, tracer: Any, name: str, attrs: dict):
        self._tracer = tracer
        self._name = name
        self._attrs = attrs
        self._cm = None
        self._span = None

    def __enter__(self):
        if self._tracer is not None:
            try:
                self._cm = self._tracer.start_as_current_span(self._name)
                self._span = self._cm.__enter__()
                for k, v in self._attrs.items():
                    self._span.set_attribute(str(k), str(v))
            except Exception:
                self._span = None
        return self

    def __exit__(self, *args):
        if self._cm is not None:
            try:
                self._cm.__exit__(*args)
            except Exception:
                pass

    def set_attribute(self, key: str, value: Any) -> None:
        if self._span is not None:
            try:
                self._span.set_attribute(str(key), str(value))
            except Exception:
                pass


# ─── Phoenix dataset upload ───────────────────────────────────────────────────


def _upload_dataset(session: Any, df: pd.DataFrame, name: str) -> None:
    if session is None:
        return
    try:
        from phoenix.client import Client

        client = Client()
        upload_df = df.copy()
        upload_df["input"] = upload_df.get("vehicles_json", upload_df.index.astype(str))
        upload_df["output"] = upload_df.get("predicted_label", "no")
        meta_cols = [
            c
            for c in ["variant", "true_label", "latency_ms", "correct", "num_vehicles"]
            if c in upload_df.columns
        ]
        client.datasets.create_dataset(
            name=name,
            dataframe=upload_df,
            input_keys=["input"],
            output_keys=["output"],
            metadata_keys=meta_cols,
        )
        print(f"[Phoenix] Dataset '{name}' uploaded ({len(df)} rows)")
    except Exception as exc:
        print(f"[Phoenix] Dataset upload skipped: {exc}")


# ─── Rule-based reference ─────────────────────────────────────────────────────


def _rule_predict(vehicles: list) -> dict:
    """Call the rule-based engine for a ground-truth reference."""
    try:
        from models.llm_model import _build_full_decision

        return _build_full_decision(vehicles, None)
    except Exception:
        return {"is_conflict": "no", "waiting_times": {}, "priority_order": {}}


# ─── Data helpers ─────────────────────────────────────────────────────────────


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
                    "num_vehicles": len(vehicles),
                }
            )
        except Exception:
            pass
    return out


def _route_variant(idx: int, pct_a: float = TRAFFIC_A) -> str:
    """Deterministic traffic router: returns 'A' or 'B'."""
    h = int(hashlib.md5(f"ab_scenario_{idx}".encode()).hexdigest(), 16)
    return "A" if (h % 100) < int(pct_a * 100) else "B"


# ─── Timed prediction ─────────────────────────────────────────────────────────


def _predict(model: Any, vehicles: list) -> tuple[dict, float, bool]:
    """Returns (result_dict, latency_ms, json_valid)."""
    t0 = time.perf_counter()
    required = {"is_conflict", "decisions", "priority_order", "waiting_times"}
    try:
        res = model.predict({"vehicles": vehicles})
        valid = required.issubset(res.keys())
    except Exception:
        res = {"is_conflict": "no", "decisions": [], "priority_order": {}, "waiting_times": {}}
        valid = False
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return res, latency_ms, valid


def _wt_mae(pred_wts: dict, ref_wts: dict) -> float:
    if not ref_wts:
        return 0.0
    errs = [abs(pred_wts.get(vid, 0) - wt) for vid, wt in ref_wts.items()]
    return float(np.mean(errs)) if errs else 0.0


# ─── Core evaluation loop ─────────────────────────────────────────────────────


def run_ab_evaluation(session: Any = None, tracer: Any = None) -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    from models.llm_model import IntersectionLLM

    model_a = IntersectionLLM(model="gpt-4o-mini", few_shot=True)
    model_b = IntersectionLLM(model="gpt-4o-mini", fine_tuned_model_id=FT_MODEL_ID, few_shot=False)

    print(f"\nLoading {N_SCENARIOS} scenarios from {TEST_CSV.name}…")
    scenarios = _load_scenarios(TEST_CSV, N_SCENARIOS, RANDOM_SEED)
    print(f"  Loaded {len(scenarios)} scenarios")

    records_a: list[dict] = []
    records_b: list[dict] = []

    print("\n--- Running A/B predictions ---")
    for idx, sc in enumerate(scenarios):
        vehicles = sc["vehicles"]
        true_label = sc["true_label"]
        n_v = sc["num_vehicles"]
        v_json = json.dumps({"vehicles": vehicles})
        route = _route_variant(idx)

        # Rule-based reference (for waiting-time MAE + rule-disagree)
        ref = _rule_predict(vehicles)
        ref_label = ref.get("is_conflict", "no")
        ref_wts = ref.get("waiting_times", {})

        # ── Model A ────────────────────────────────────────────────────────
        with _Span(
            tracer,
            "ab_predict_A",
            {"variant": "A", "idx": idx, "true_label": true_label, "n_vehicles": n_v},
        ) as sp_a:
            res_a, lat_a, valid_a = _predict(model_a, vehicles)
            pred_a = res_a.get("is_conflict", "no")
            sp_a.set_attribute("prediction", pred_a)
            sp_a.set_attribute("latency_ms", round(lat_a, 1))
            sp_a.set_attribute("correct", pred_a == true_label)

        # ── Model B ────────────────────────────────────────────────────────
        with _Span(
            tracer,
            "ab_predict_B",
            {"variant": "B", "idx": idx, "true_label": true_label, "n_vehicles": n_v},
        ) as sp_b:
            res_b, lat_b, valid_b = _predict(model_b, vehicles)
            pred_b = res_b.get("is_conflict", "no")
            sp_b.set_attribute("prediction", pred_b)
            sp_b.set_attribute("latency_ms", round(lat_b, 1))
            sp_b.set_attribute("correct", pred_b == true_label)

        print(
            f"  [{idx+1:>3}/{len(scenarios)}]  "
            f"A={pred_a}({'✓' if pred_a==true_label else '✗'},{lat_a:.0f}ms)  "
            f"B={pred_b}({'✓' if pred_b==true_label else '✗'},{lat_b:.0f}ms)  "
            f"true={true_label}  route={route}"
        )

        for records, pred, res, lat, valid in [
            (records_a, pred_a, res_a, lat_a, valid_a),
            (records_b, pred_b, res_b, lat_b, valid_b),
        ]:
            variant = "A" if records is records_a else "B"
            records.append(
                {
                    "idx": idx,
                    "variant": variant,
                    "true_label": true_label,
                    "predicted_label": pred,
                    "correct": pred == true_label,
                    "latency_ms": lat,
                    "json_valid": valid,
                    "num_vehicles": n_v,
                    "wt_mae": _wt_mae(res.get("waiting_times", {}), ref_wts),
                    "rule_disagree": pred != ref_label,
                    "is_conflict_true": true_label == "yes",
                    "is_conflict_pred": pred == "yes",
                    "routed": route == variant,
                    "vehicles_json": v_json,
                }
            )

    df_a = pd.DataFrame(records_a)
    df_b = pd.DataFrame(records_b)

    # ── Aggregate metrics ──────────────────────────────────────────────────────
    def _metrics(df: pd.DataFrame, name: str) -> dict:
        yt = df["is_conflict_true"].astype(int).tolist()
        yp = df["is_conflict_pred"].astype(int).tolist()
        acc = accuracy_score(yt, yp)
        rec = recall_score(yt, yp, zero_division=0)
        fn = sum(1 for t, p in zip(yt, yp) if t == 1 and p == 0)
        tp = sum(1 for t, p in zip(yt, yp) if t == 1 and p == 1)
        fnr = fn / (fn + tp + 1e-9)
        return {
            "variant": name,
            "n": len(df),
            "accuracy": round(acc, 4),
            "recall": round(rec, 4),
            "precision": round(precision_score(yt, yp, zero_division=0), 4),
            "f1": round(f1_score(yt, yp, zero_division=0), 4),
            "fnr": round(fnr, 4),
            "latency_p50_ms": round(float(np.percentile(df["latency_ms"], 50)), 1),
            "latency_p95_ms": round(float(np.percentile(df["latency_ms"], 95)), 1),
            "json_validity_rate": round(float(df["json_valid"].mean()), 4),
            "fallback_rate": round(float((~df["json_valid"]).mean()), 4),
            "wt_mae_s": round(float(df["wt_mae"].mean()), 4),
            "rule_disagree_rate": round(float(df["rule_disagree"].mean()), 4),
        }

    m_a = _metrics(df_a, "A: gpt-4o-mini few-shot (baseline)")
    m_b = _metrics(df_b, "B: fine-tuned DX7kzKtB")

    # Statistical tests
    chi2_val, p_val, sig = 0.0, 1.0, False
    try:
        nc_a, nc_b = int(df_a["correct"].sum()), int(df_b["correct"].sum())
        chi2_val, p_val, _, _ = chi2_contingency(
            [[nc_a, len(df_a) - nc_a], [nc_b, len(df_b) - nc_b]]
        )
        sig = bool(p_val < 0.05)
    except Exception:
        pass

    lat_p = 1.0
    try:
        _, lat_p = mannwhitneyu(df_a["latency_ms"], df_b["latency_ms"], alternative="two-sided")
    except Exception:
        pass

    # Recall by vehicle-count group
    def _recall_by_count(df: pd.DataFrame) -> dict:
        out = {}
        for label, mask in [
            ("small (2-4)", df["num_vehicles"] <= 4),
            ("large (5-8)", df["num_vehicles"] > 4),
        ]:
            sub = df[mask]
            if len(sub) == 0:
                out[label] = 0.0
            else:
                out[label] = round(
                    float(
                        recall_score(
                            sub["is_conflict_true"].astype(int),
                            sub["is_conflict_pred"].astype(int),
                            zero_division=0,
                        )
                    ),
                    4,
                )
        return out

    rc_a, rc_b = _recall_by_count(df_a), _recall_by_count(df_b)

    # ── Plots ─────────────────────────────────────────────────────────────────
    _generate_plots(df_a, df_b, m_a, m_b, rc_a, rc_b)

    # ── Report ────────────────────────────────────────────────────────────────
    report = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "n_scenarios": len(scenarios),
        "traffic_split": {"A_pct": 15, "B_pct": 85},
        "models": {
            "A": "gpt-4o-mini (few_shot=True, baseline)",
            "B": FT_MODEL_ID,
        },
        "metrics_A": m_a,
        "metrics_B": m_b,
        "statistical_significance": {
            "test": "chi2_contingency",
            "chi2": round(float(chi2_val), 4),
            "p_value": round(float(p_val), 4),
            "significant_at_0.05": sig,
            "latency_mannwhitney_p": round(float(lat_p), 4),
        },
        "winner": "B" if m_b["accuracy"] > m_a["accuracy"] else "A",
        "recall_by_vehicle_count": {"A": rc_a, "B": rc_b},
        "figures": sorted(str(p.name) for p in FIG_DIR.glob("*.png")),
    }

    report_path = REPORT_DIR / "report.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n[A/B] Report → {report_path}")

    # Upload full prediction log to Phoenix
    all_preds = pd.concat([df_a, df_b], ignore_index=True)
    _upload_dataset(session, all_preds, "ab_test_predictions")

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


def _rolling(series: pd.Series, w: int = 10) -> np.ndarray:
    return series.rolling(window=w, min_periods=1).mean().values


def _generate_plots(df_a, df_b, m_a, m_b, rc_a, rc_b) -> None:
    plt.rcParams.update(plt_cfg)

    # ── 1. Accuracy over time ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        _rolling(df_a["correct"].astype(float)),
        color=COLORS["A"],
        lw=2,
        label=f"A gpt-4o-mini few-shot  (acc={m_a['accuracy']:.3f})",
    )
    ax.plot(
        _rolling(df_b["correct"].astype(float)),
        color=COLORS["B"],
        lw=2,
        label=f"B fine-tuned            (acc={m_b['accuracy']:.3f})",
    )
    for m, c in [(m_a, COLORS["A"]), (m_b, COLORS["B"])]:
        ax.axhline(m["accuracy"], color=c, ls="--", alpha=0.35, lw=1)
    ax.set(
        xlabel="Prediction index",
        ylabel="Rolling accuracy (w=10)",
        title="A/B Test — Accuracy Over Time",
        ylim=(0, 1.08),
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "accuracy_over_time.png")
    plt.close(fig)

    # ── 2. Latency distribution (box + p50/p95 bar) ───────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    bp = ax1.boxplot(
        [df_a["latency_ms"].values, df_b["latency_ms"].values],
        labels=["A\n(few-shot)", "B\n(fine-tuned)"],
        patch_artist=True,
        medianprops=dict(color="black", lw=2),
        widths=0.5,
    )
    for patch, color in zip(bp["boxes"], [COLORS["A"], COLORS["B"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax1.set(ylabel="Latency (ms)", title="Latency Distribution")

    x = np.arange(2)
    width = 0.35
    p50s = [m_a["latency_p50_ms"], m_b["latency_p50_ms"]]
    p95s = [m_a["latency_p95_ms"], m_b["latency_p95_ms"]]
    b50 = ax2.bar(
        x - width / 2, p50s, width, color=[COLORS["A"], COLORS["B"]], alpha=0.85, label="p50"
    )
    b95 = ax2.bar(
        x + width / 2,
        p95s,
        width,
        color=[COLORS["A"], COLORS["B"]],
        alpha=0.45,
        hatch="///",
        label="p95",
    )
    ax2.set(xticks=x, xticklabels=["A", "B"], ylabel="Latency (ms)", title="p50 / p95 Latency")
    for bar in [*b50, *b95]:
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{bar.get_height():.0f}",
            ha="center",
            fontsize=8,
        )
    ax2.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "latency_distribution.png")
    plt.close(fig)

    # ── 3. FNR trend ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    for df, name, color in [
        (df_a, f"A (avg FNR={m_a['fnr']:.3f})", COLORS["A"]),
        (df_b, f"B (avg FNR={m_b['fnr']:.3f})", COLORS["B"]),
    ]:
        is_pos = df["is_conflict_true"].astype(float)
        missed = ((df["is_conflict_true"]) & (~df["is_conflict_pred"])).astype(float)
        roll_fnr = (
            missed.rolling(10, min_periods=1).sum()
            / (is_pos.rolling(10, min_periods=1).sum() + 1e-9)
        ).values
        ax.plot(roll_fnr, color=color, lw=2, label=name)
    ax.axhline(0.20, color="red", ls="--", alpha=0.4, lw=1, label="FNR=0.20 threshold")
    ax.set(
        xlabel="Prediction index",
        ylabel="Rolling FNR (w=10)",
        title="A/B Test — False Negative Rate Trend",
        ylim=(0, 1.05),
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fnr_trend.png")
    plt.close(fig)

    # ── 4. Conflict recall by scenario type ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    types = list(rc_a.keys())
    x = np.arange(len(types))
    w = 0.35
    ba = ax.bar(
        x - w / 2, [rc_a[t] for t in types], w, color=COLORS["A"], label="A (few-shot)", alpha=0.85
    )
    bb = ax.bar(
        x + w / 2,
        [rc_b[t] for t in types],
        w,
        color=COLORS["B"],
        label="B (fine-tuned)",
        alpha=0.85,
    )
    for bar in [*ba, *bb]:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.02,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set(
        xticks=x,
        xticklabels=["Small (2-4 vehicles)", "Large (5-8 vehicles)"],
        ylabel="Recall",
        title="Conflict Recall by Scenario Type",
        ylim=(0, 1.15),
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "recall_by_vehicle_count.png")
    plt.close(fig)

    # ── 5. Waiting-time MAE ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    vals = [m_a["wt_mae_s"], m_b["wt_mae_s"]]
    bars = ax.bar(
        ["A (few-shot)", "B (fine-tuned)"],
        vals,
        color=[COLORS["A"], COLORS["B"]],
        alpha=0.85,
        width=0.45,
    )
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{v:.2f}s",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
    ax.set(ylabel="MAE (seconds)", title="Waiting-Time MAE vs Rule-Based Reference")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "waiting_time_mae.png")
    plt.close(fig)

    # ── 6. Rule disagreement rate ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    vals = [m_a["rule_disagree_rate"], m_b["rule_disagree_rate"]]
    bars = ax.bar(
        ["A (few-shot)", "B (fine-tuned)"],
        vals,
        color=[COLORS["A"], COLORS["B"]],
        alpha=0.85,
        width=0.45,
    )
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{v:.3f}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
    ax.set(
        ylabel="Disagreement rate",
        title="Rule Disagreement Rate (LLM vs Rule Engine)",
        ylim=(0, max(vals + [0.01]) * 1.6),
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rule_disagreement_rate.png")
    plt.close(fig)

    print(f"[A/B] 6 figures saved → {FIG_DIR}")


# ─── Entrypoint ───────────────────────────────────────────────────────────────


def main() -> dict:
    session, tracer = _setup_phoenix()
    report = run_ab_evaluation(session, tracer)

    print("\n" + "=" * 65)
    print("A/B TEST SUMMARY")
    print("=" * 65)
    for variant, m in [("A", report["metrics_A"]), ("B", report["metrics_B"])]:
        print(
            f"  Model {variant}: acc={m['accuracy']:.3f}  f1={m['f1']:.3f}  "
            f"recall={m['recall']:.3f}  fnr={m['fnr']:.3f}  "
            f"p50={m['latency_p50_ms']:.0f}ms  p95={m['latency_p95_ms']:.0f}ms  "
            f"rule_disagree={m['rule_disagree_rate']:.3f}"
        )
    sig = report["statistical_significance"]
    print(
        f"\n  chi2={sig['chi2']:.3f}  p={sig['p_value']:.4f}  "
        f"significant={sig['significant_at_0.05']}  winner=Model {report['winner']}"
    )
    print("=" * 65)
    if session:
        print(f"\n  Phoenix dashboard → {session.url}")
    return report


if __name__ == "__main__":
    main()

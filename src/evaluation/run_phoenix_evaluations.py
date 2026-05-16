"""
run_phoenix_evaluations.py  —  Master runner for all Phoenix evaluations
=========================================================================
Starts Arize Phoenix, then runs the four evaluation modules sequentially:
  1. A/B Test          (reports/AB_testing/)
  2. Explainability    (reports/explainability/)
  3. Robustness        (reports/robustness/)
  4. Bias Audit        (reports/bias_audit/)

Usage
-----
  export OPENAI_API_KEY=sk-...
  cd /path/to/llm-traffic-intersection
  python src/evaluation/run_phoenix_evaluations.py [--skip ab|explain|robust|bias]

Environment variables (all optional, fall back to defaults in .env):
  OPENAI_API_KEY          Required
  FINE_TUNED_MODEL_ID     Default: ft:gpt-4o-mini-2024-07-18:personal::DX7kzKtB
  MLFLOW_TRACKING_URI     Used only if MLflow logging is active
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

# ─── Bootstrap path ───────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "poc"))

load_dotenv(REPO_ROOT / ".env", override=False)
os.chdir(REPO_ROOT)


# ─── Phoenix bootstrap ────────────────────────────────────────────────────────


def _start_phoenix(project: str = "traffic-intersection-evals") -> tuple[Any, Any]:
    """
    Start Arize Phoenix (local UI at http://localhost:6006) and configure
    OpenTelemetry tracing so all OpenAI calls are auto-captured.

    Returns (session, otel_tracer).  Both may be None if Phoenix is unavailable.
    """
    session = None
    tracer = None

    try:
        import phoenix as px

        session = px.launch_app(use_temp_dir=False)
        print(f"\n[Phoenix] UI started → {session.url}")
        print("[Phoenix] Keep this script running to preserve the dashboard.")
    except Exception as exc:
        print(f"[Phoenix] Could not start: {exc}")
        print("[Phoenix] Evaluations will run without Phoenix dashboard.")
        return None, None

    # Register OTEL
    tracer_provider = None
    try:
        from phoenix.otel import register as _ph_register

        tracer_provider = _ph_register(
            project_name=project,
            endpoint=f"{session.url}/v1/traces",
        )
        print(f"[Phoenix] OTEL tracer registered → project '{project}'")
    except Exception:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            tracer_provider = TracerProvider()
            tracer_provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{session.url}/v1/traces"))
            )
        except Exception as exc2:
            print(f"[Phoenix/OTEL] TracerProvider setup failed: {exc2}")

    if tracer_provider is not None:
        try:
            from openinference.instrumentation.openai import OpenAIInstrumentor

            OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
            print("[Phoenix] OpenAI auto-instrumented — all LLM calls will be traced.")
        except ImportError:
            print(
                "[Phoenix] openinference-instrumentation-openai not installed.\n"
                "          Install with: pip install openinference-instrumentation-openai"
            )

        try:
            from opentelemetry import trace

            tracer = trace.get_tracer("traffic-evals", tracer_provider=tracer_provider)
        except Exception:
            pass

    return session, tracer


# ─── Individual runners ───────────────────────────────────────────────────────


def _run_ab_test(session: Any, tracer: Any) -> dict:
    from evaluation.phoenix_ab_test import run_ab_evaluation

    print("\n" + "━" * 65)
    print("  STEP 1 / 4 — A/B Testing")
    print("━" * 65)
    return run_ab_evaluation(session, tracer)


def _run_explainability(session: Any) -> dict:
    from evaluation.phoenix_explainability import run_explainability

    print("\n" + "━" * 65)
    print("  STEP 2 / 4 — Explainability")
    print("━" * 65)
    return run_explainability(session)


def _run_robustness(session: Any) -> dict:
    from evaluation.phoenix_robustness import run_robustness

    print("\n" + "━" * 65)
    print("  STEP 3 / 4 — Robustness / Perturbation")
    print("━" * 65)
    return run_robustness(session)


def _run_bias_audit(session: Any) -> dict:
    from evaluation.phoenix_bias_audit import run_bias_audit

    print("\n" + "━" * 65)
    print("  STEP 4 / 4 — Bias Audit")
    print("━" * 65)
    return run_bias_audit(session)


# ─── Summary ──────────────────────────────────────────────────────────────────


def _print_summary(results: dict, session: Any, total_time_s: float) -> None:
    print("\n" + "═" * 70)
    print("  PHOENIX EVALUATION SUITE — FINAL SUMMARY")
    print("═" * 70)
    print(f"  Total runtime: {total_time_s:.0f}s")
    print()

    ab = results.get("ab_test")
    if ab:
        print("  [A/B Test]")
        ma, mb = ab.get("metrics_A", {}), ab.get("metrics_B", {})

        def _fmt(d, key, fmt=".3f"):
            v = d.get(key, None)
            return format(v, fmt) if isinstance(v, (int, float)) else str(v or "?")

        print(
            f"    Model A (few-shot) : acc={_fmt(ma,'accuracy')}  fnr={_fmt(ma,'fnr')}  "
            f"f1={_fmt(ma,'f1')}  p50={_fmt(ma,'latency_p50_ms','.0f')}ms"
        )
        print(
            f"    Model B (fine-tun) : acc={_fmt(mb,'accuracy')}  fnr={_fmt(mb,'fnr')}  "
            f"f1={_fmt(mb,'f1')}  p50={_fmt(mb,'latency_p50_ms','.0f')}ms"
        )
        sig = ab.get("statistical_significance", {})
        print(
            f"    chi2={_fmt(sig,'chi2')}  p={_fmt(sig,'p_value','.4f')}  "
            f"significant={sig.get('significant_at_0.05','?')}  "
            f"winner=Model {ab.get('winner','?')}"
        )
        print("    Outputs → reports/online_evaluation/AB_testing/")
    print()

    ex = results.get("explainability")
    if ex:
        print("  [Explainability]")

        def _fmt(d, key, fmt=".3f"):
            v = d.get(key, None)
            return format(v, fmt) if isinstance(v, (int, float)) else str(v or "?")

        print(f"    Accuracy        : {_fmt(ex,'accuracy')}")
        print(f"    Rule agreement  : {_fmt(ex,'rule_agreement_rate')}")
        ranked = ex.get("feature_ranking", [])
        imp = ex.get("aggregate_importance", {})
        if ranked:
            print(f"    Feature ranking : {' > '.join(ranked)}")
            print(f"    Top importance  : {ranked[0]} ({imp.get(ranked[0], 0):.4f})")
        print("    Outputs → reports/online_evaluation/explainability/")
    print()

    rb = results.get("robustness")
    if rb:
        print("  [Robustness]")

        def _fmt(d, key, fmt=".4f"):
            v = d.get(key, None)
            return format(v, fmt) if isinstance(v, (int, float)) else str(v or "?")

        print(f"    Overall stability : {_fmt(rb,'overall_stability')}")
        print(f"    Sensitivity rank  : {' > '.join(rb.get('feature_ranking_by_sensitivity', []))}")
        feats = rb.get("feature_summaries", {})
        for feat, s in feats.items():
            if s:
                bar = "█" * max(1, int(s["flip_rate"] * 15))
                print(f"    {feat:<18} {bar} {s['flip_rate']:.4f}")
        print("    Outputs → reports/online_evaluation/robustness/")
    print()

    ba = results.get("bias_audit")
    if ba:
        print("  [Bias Audit]")
        biased = ba.get("biased_dimensions", [])
        safe_bias = ba.get("safety_biased_dimensions", [])
        print(f"    Overall bias detected : {ba.get('overall_bias_detected', '?')}")
        if biased:
            print(f"    Biased dims (F1)      : {', '.join(biased)}")
        else:
            print("    No F1 bias across any dimension.")
        if safe_bias:
            print(f"    Safety bias (FNR)     : {', '.join(safe_bias)}")
        for dim, r in ba.get("audit", {}).items():
            flag = "⚠" if r.get("bias_detected") else " "
            print(
                f"    {flag} {dim:<22} F1-disp={r['f1_disparity']:.3f}  "
                f"FNR-disp={r['fnr_disparity']:.3f}"
            )
        print("    Outputs → reports/online_evaluation/bias_audit/")

    print()
    print("  Reports index:")
    for name, path in [
        ("A/B Testing", "reports/online_evaluation/AB_testing/report.json"),
        ("Explainability", "reports/online_evaluation/explainability/report.json"),
        ("Explainability HTML", "reports/online_evaluation/explainability/report.html"),
        ("Feature influence", "reports/online_evaluation/explainability/feature_influence.md"),
        ("Robustness", "reports/online_evaluation/robustness/report.json"),
        ("Bias Audit", "reports/online_evaluation/bias_audit/report.json"),
    ]:
        p = REPO_ROOT / path
        exists = "✓" if p.exists() else "✗"
        print(f"    {exists} {name:<22} {path}")

    if session:
        print(f"\n  Phoenix dashboard → {session.url}")
        print("  (Keep this process alive to browse the dashboard)")
    print("═" * 70)


# ─── Entrypoint ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all Phoenix LLM evaluation modules for traffic intersection"
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        choices=["ab", "explain", "robust", "bias"],
        default=[],
        help="Skip specific evaluation modules",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        choices=["ab", "explain", "robust", "bias"],
        default=None,
        help="Run only specific evaluation modules",
    )
    args = parser.parse_args()

    # Determine which modules to run
    all_modules = ["ab", "explain", "robust", "bias"]
    if args.only:
        to_run = [m for m in all_modules if m in args.only]
    else:
        to_run = [m for m in all_modules if m not in args.skip]

    print("=" * 65)
    print("  LLM Traffic Intersection — Phoenix Evaluation Suite")
    print("=" * 65)
    print(f"  Modules to run: {', '.join(to_run)}")
    print(
        f"  Fine-tuned model: {os.environ.get('FINE_TUNED_MODEL_ID', 'ft:gpt-4o-mini-2024-07-18:personal::DX7kzKtB')}"
    )

    # ── Start Phoenix ──────────────────────────────────────────────────────────
    session, tracer = _start_phoenix()

    results: dict = {}
    t_start = time.perf_counter()

    # ── Run evaluations ────────────────────────────────────────────────────────
    if "ab" in to_run:
        try:
            results["ab_test"] = _run_ab_test(session, tracer)
        except Exception:
            print("[A/B] FAILED:")
            traceback.print_exc()

    if "explain" in to_run:
        try:
            results["explainability"] = _run_explainability(session)
        except Exception:
            print("[Explainability] FAILED:")
            traceback.print_exc()

    if "robust" in to_run:
        try:
            results["robustness"] = _run_robustness(session)
        except Exception:
            print("[Robustness] FAILED:")
            traceback.print_exc()

    if "bias" in to_run:
        try:
            results["bias_audit"] = _run_bias_audit(session)
        except Exception:
            print("[Bias Audit] FAILED:")
            traceback.print_exc()

    total_time = time.perf_counter() - t_start

    # ── Save combined results ──────────────────────────────────────────────────
    combined_path = REPO_ROOT / "reports" / "online_evaluation" / "phoenix_evaluation_summary.json"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined_path.write_text(
        json.dumps(
            {
                "generated_at": pd.Timestamp.now().isoformat(),
                "total_runtime_s": round(total_time, 1),
                "modules_run": to_run,
                "results": {
                    k: {
                        key: val
                        for key, val in v.items()
                        if key not in ("sample_explanations",)  # skip large nested data
                    }
                    for k, v in results.items()
                },
            },
            indent=2,
            default=str,
        )
    )
    print(f"\n[Summary] Combined report → {combined_path}")

    _print_summary(results, session, total_time)

    # ── Keep process alive so Phoenix dashboard stays up ─────────────────────
    if session:
        print("\nPress Ctrl+C to stop Phoenix and exit.")
        try:
            import time as _time

            while True:
                _time.sleep(60)
        except KeyboardInterrupt:
            print("\n[Phoenix] Shutting down.")


if __name__ == "__main__":
    main()

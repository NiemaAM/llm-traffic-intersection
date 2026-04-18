"""
run_all_pipelines.py
--------------------
Master script that runs all ZenML pipelines in sequence.
Inspired by PseudoCodeRAG-Translator milestone structure.

Usage
-----
    # Run all pipelines
    python scripts/run_all_pipelines.py

    # Run specific milestone(s)
    python scripts/run_all_pipelines.py --milestone 3
    python scripts/run_all_pipelines.py --milestone 4
    python scripts/run_all_pipelines.py --milestone 5
    python scripts/run_all_pipelines.py --milestone 6
    python scripts/run_all_pipelines.py --milestone 3 4

    # Enable fine-tuning in M4 (costs ~$1-2, takes 10-30 min)
    python scripts/run_all_pipelines.py --milestone 4 --train

    # Quick mode (fewer scenarios, faster and cheaper)
    python scripts/run_all_pipelines.py --quick

Pipeline map
------------
    M3  data_pipeline      ingest → validate → features → DVC → Feast
    M4  training_pipeline  prepare → zero-shot → few-shot → train → eval → compare → energy
    M5  serving_pipeline   load_model → validate_api → tests → register
    M6  monitoring_pipeline drift → eval → robustness → bias → retrain_decision → log
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# ── Project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv
load_dotenv()


# ─── Colors for terminal output ───────────────────────────────────────────────

class C:
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    BLUE   = "\033[94m"
    RESET  = "\033[0m"
    CYAN   = "\033[96m"


def banner(text: str, color: str = C.BLUE) -> None:
    width = 60
    print(f"\n{color}{C.BOLD}{'─' * width}")
    print(f"  {text}")
    print(f"{'─' * width}{C.RESET}\n")


def ok(text: str)   -> None: print(f"  {C.GREEN}✅ {text}{C.RESET}")
def warn(text: str) -> None: print(f"  {C.YELLOW}⚠️  {text}{C.RESET}")
def err(text: str)  -> None: print(f"  {C.RED}❌ {text}{C.RESET}")
def info(text: str) -> None: print(f"  {C.CYAN}ℹ️  {text}{C.RESET}")


# ─── Pre-flight checks ────────────────────────────────────────────────────────

def preflight_checks(args) -> bool:
    banner("Pre-flight Checks", C.CYAN)
    all_ok = True

    # Python version
    import sys as _sys
    v = _sys.version_info
    if v >= (3, 11):
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        warn(f"Python {v.major}.{v.minor} — recommend 3.11+")

    # .env file
    env_path = ROOT / ".env"
    if env_path.exists():
        ok(".env file found")
    else:
        warn(".env not found — copy .env.example to .env and set OPENAI_API_KEY")

    # API key
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key and api_key.startswith("sk-"):
        ok("OPENAI_API_KEY set")
    else:
        if any(m in args.milestones for m in [4, 5, 6]):
            err("OPENAI_API_KEY not set — M4/M5/M6 evaluation steps will fail")
            all_ok = False
        else:
            warn("OPENAI_API_KEY not set (not needed for M3 data pipeline)")

    # MLflow
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    try:
        import requests
        r = requests.get(mlflow_uri, timeout=3)
        ok(f"MLflow reachable at {mlflow_uri}")
    except Exception:
        warn(f"MLflow not reachable at {mlflow_uri}")
        info("Start it: mlflow ui --port 5000")
        info("Or use Docker stack: cd deployment/docker && docker compose up -d")

    # ZenML
    try:
        import zenml
        ok(f"ZenML {zenml.__version__}")
    except ImportError:
        err("ZenML not installed — pip install 'zenml[server]'")
        all_ok = False

    # Raw data
    raw_csv = ROOT / "data" / "raw" / "generated_dataset.csv"
    if raw_csv.exists():
        import pandas as pd
        df = pd.read_csv(raw_csv)
        ok(f"Raw dataset exists ({len(df)} rows)")
    else:
        warn("Raw dataset not found — M3 pipeline will generate it")

    print()
    return all_ok


# ─── Individual pipeline runners ──────────────────────────────────────────────

def run_m3(args) -> bool:
    banner("Milestone 3 — Data Pipeline", C.GREEN)
    info("Steps: ingest → validate → engineer → DVC version → Feast feature store")
    print()

    try:
        from src.pipelines.data_pipeline import data_pipeline

        data_pipeline(
            num_records=args.num_records,
            seed=42,
        )
        ok("M3 Data Pipeline completed")
        info("Check artifacts: data/raw/generated_dataset.csv")
        info("                 data/processed/features.csv")
        info("                 models/scaler.joblib")
        return True

    except Exception as exc:
        err(f"M3 failed: {exc}")
        import traceback; traceback.print_exc()
        return False


def run_m4(args) -> bool:
    banner("Milestone 4 — Training Pipeline", C.YELLOW)
    info("Steps: prepare_data → zero-shot → few-shot → train → eval → compare → energy")
    if args.train:
        warn("Fine-tuning ENABLED — this will cost ~$1-2 and take 10-30 minutes")
    else:
        info("Fine-tuning SKIPPED (pass --train to enable)")
    print()

    try:
        from src.pipelines.training_pipeline import training_pipeline

        training_pipeline(
            model_name=os.environ.get("MODEL_NAME", "gpt-4o-mini"),
            max_scenarios=args.max_scenarios,
            skip_training=not args.train,
            max_examples=args.max_examples,
            n_epochs=3,
        )
        ok("M4 Training Pipeline completed")
        info(f"View results: {os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')}")
        info("Check: reports/model_comparison.json")
        return True

    except Exception as exc:
        err(f"M4 failed: {exc}")
        import traceback; traceback.print_exc()
        return False


def run_m5(args) -> bool:
    banner("Milestone 5 — Serving Pipeline", C.CYAN)
    api_url = os.environ.get("API_URL", "http://localhost:8000")
    info(f"Steps: load_model → validate_api → integration_tests → register_serving")
    info(f"API URL: {api_url}")
    print()

    # Check API is running
    try:
        import requests
        r = requests.get(f"{api_url}/health", timeout=5)
        if r.status_code != 200:
            warn(f"API returned {r.status_code} — is the FastAPI server running?")
            info("Start it: uvicorn src.api.app:app --port 8000")
            info("Or via Docker: cd deployment/docker && docker compose up -d")
    except Exception:
        warn(f"API not reachable at {api_url}")
        info("Start it: PYTHONPATH=. uvicorn src.api.app:app --reload --port 8000")
        info("Or via Docker: cd deployment/docker && docker compose up -d")

    try:
        from src.pipelines.serving_pipeline import serving_pipeline

        serving_pipeline(api_url=api_url)
        ok("M5 Serving Pipeline completed")
        info(f"Swagger UI: {api_url}/docs")
        return True

    except Exception as exc:
        err(f"M5 failed: {exc}")
        import traceback; traceback.print_exc()
        return False


def run_m6(args) -> bool:
    banner("Milestone 6 — Monitoring Pipeline", C.BLUE)
    info("Steps: load_data → drift → eval → robustness → bias → retrain_decision → log")
    print()

    try:
        from src.pipelines.monitoring_pipeline import monitoring_pipeline

        monitoring_pipeline(
            model_name=os.environ.get("MODEL_NAME", "gpt-4o-mini"),
            max_scenarios=args.max_scenarios,
        )
        ok("M6 Monitoring Pipeline completed")
        info("Check: reports/drift_report.html")
        info(f"MLflow: {os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')}")
        return True

    except Exception as exc:
        err(f"M6 failed: {exc}")
        import traceback; traceback.print_exc()
        return False


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run LLM Traffic Intersection MLOps Pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_all_pipelines.py                        # run all pipelines
  python scripts/run_all_pipelines.py --milestone 3          # M3 only
  python scripts/run_all_pipelines.py --milestone 4 --train  # M4 with fine-tuning
  python scripts/run_all_pipelines.py --milestone 4 6        # M4 and M6
  python scripts/run_all_pipelines.py --quick                # all, fewer scenarios
        """
    )
    parser.add_argument("--milestone",    nargs="+", type=int, default=[3,4,5,6],
                        help="Which milestones to run (default: 3 4 5 6)")
    parser.add_argument("--train",        action="store_true",
                        help="Enable fine-tuning in M4 (costs ~$1-2)")
    parser.add_argument("--quick",        action="store_true",
                        help="Quick mode: fewer scenarios (faster and cheaper)")
    parser.add_argument("--num-records",  type=int, default=1000,
                        help="Records to generate in M3 (default: 1000)")
    parser.add_argument("--max-scenarios",type=int, default=None,
                        help="Evaluation scenarios per run (default: 20, quick: 5)")
    parser.add_argument("--max-examples", type=int, default=500,
                        help="Max fine-tuning examples (default: 500)")
    parser.add_argument("--skip-preflight", action="store_true")

    args = parser.parse_args()
    args.milestones = args.milestone

    # Apply quick mode defaults
    if args.max_scenarios is None:
        args.max_scenarios = 5 if args.quick else 20
    if args.quick:
        args.num_records = 200

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"\n{C.BOLD}{C.BLUE}")
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║   🚦 LLM Traffic Intersection — MLOps Pipeline Runner   ║")
    print("  ║      CSC5382 – AI for Digital Transformation             ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print(f"{C.RESET}")
    print(f"  Milestones to run : {args.milestones}")
    print(f"  Max scenarios     : {args.max_scenarios}")
    print(f"  Fine-tuning       : {'Enabled' if args.train else 'Skipped'}")
    print(f"  Quick mode        : {'Yes' if args.quick else 'No'}")

    # ── Pre-flight ────────────────────────────────────────────────────────────
    if not args.skip_preflight:
        if not preflight_checks(args):
            print(f"\n{C.RED}Pre-flight checks failed. Fix the issues above and retry.{C.RESET}\n")
            sys.exit(1)

    # ── Run pipelines ─────────────────────────────────────────────────────────
    RUNNERS = {3: run_m3, 4: run_m4, 5: run_m5, 6: run_m6}
    NAMES   = {3: "M3 Data Pipeline", 4: "M4 Training Pipeline",
               5: "M5 Serving Pipeline", 6: "M6 Monitoring Pipeline"}

    results = {}
    t_start = time.time()

    for milestone in sorted(args.milestones):
        if milestone not in RUNNERS:
            warn(f"Unknown milestone {milestone} — skipping")
            continue

        t0 = time.time()
        success = RUNNERS[milestone](args)
        elapsed = time.time() - t0

        results[milestone] = {"success": success, "elapsed": elapsed}
        status = f"{C.GREEN}PASSED{C.RESET}" if success else f"{C.RED}FAILED{C.RESET}"
        print(f"\n  {NAMES[milestone]}: {status}  ({elapsed:.1f}s)\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.time() - t_start
    banner("Pipeline Summary", C.BOLD)

    all_passed = True
    for milestone, result in results.items():
        icon   = "✅" if result["success"] else "❌"
        status = "PASSED" if result["success"] else "FAILED"
        color  = C.GREEN if result["success"] else C.RED
        print(f"  {icon}  {NAMES[milestone]:<30} {color}{status}{C.RESET}  ({result['elapsed']:.1f}s)")
        if not result["success"]:
            all_passed = False

    print(f"\n  Total time: {total_elapsed:.1f}s")

    if all_passed:
        print(f"\n{C.GREEN}{C.BOLD}  🎉 All pipelines completed successfully!{C.RESET}\n")
    else:
        failed = [NAMES[m] for m, r in results.items() if not r["success"]]
        print(f"\n{C.YELLOW}  ⚠️  Some pipelines failed: {', '.join(failed)}{C.RESET}")
        print(f"  Check the output above for details.\n")

    print(f"  📊 MLflow dashboard : {os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')}")
    print(f"  🔍 ZenML dashboard  : http://127.0.0.1:8237")
    print()


if __name__ == "__main__":
    main()

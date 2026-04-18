#!/usr/bin/env python3
"""
run_pipelines.py
----------------
Convenience script to run all ZenML pipelines in order.
Usage: python scripts/run_pipelines.py [--milestone 3|4|6|all]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_data_pipeline():
    print("\n🔵 Running Milestone 3: Data Pipeline...")
    from src.pipelines.data_pipeline import data_pipeline
    data_pipeline(num_records=1000, seed=42)


def run_training_pipeline():
    print("\n🟡 Running Milestone 4: Training Pipeline...")
    from src.pipelines.training_pipeline import training_pipeline
    training_pipeline(model_name="gpt-4o-mini", max_scenarios=30)


def run_monitoring_pipeline():
    print("\n🟢 Running Milestone 6: Monitoring Pipeline...")
    from src.pipelines.monitoring_pipeline import monitoring_pipeline
    monitoring_pipeline(model_name="gpt-4o-mini", max_scenarios=20)


def main():
    parser = argparse.ArgumentParser(description="Run ML pipelines")
    parser.add_argument(
        "--milestone",
        choices=["3", "4", "6", "all"],
        default="all",
        help="Which pipeline(s) to run",
    )
    args = parser.parse_args()

    if args.milestone in ("3", "all"):
        run_data_pipeline()
    if args.milestone in ("4", "all"):
        run_training_pipeline()
    if args.milestone in ("6", "all"):
        run_monitoring_pipeline()

    print("\n✅ All requested pipelines completed.")


if __name__ == "__main__":
    main()

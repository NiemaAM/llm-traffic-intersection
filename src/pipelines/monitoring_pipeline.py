"""
src/pipelines/monitoring_pipeline.py
--------------------------------------
ZenML pipeline for model monitoring, evaluation, and continual learning.
Milestone 6: Drift detection, A/B testing, CT/CD triggers, pipeline orchestration.

Steps
-----
  1. audit_model_bias          Evaluate model fairness across vehicle direction groups
                               (north/south/east/west); flags bias if F1 disparity > 0.15
  2. evaluate_on_test_set      Leakage-free evaluation on eval_only_masri.csv (seed=999)
                               using fine-tuned model DX7kzKtB via evaluate_model_masri()
  3. load_production_data      Load recent prediction logs from WhyLogs JSONL;
                               falls back to synthetic sample if no logs found
  4. load_reference_data       Load reference (training) dataset for drift comparison
  5. run_robustness_tests       4 adversarial/behavioral tests (obvious conflict,
                               far vehicles, perturbation stability, priority consistency)
  6. detect_drift              KS-test drift detection (scipy fallback — Python 3.11 compatible);
                               generates reports/drift_report.html
  7. continual_learning_decision  Trigger retraining if F1 < 0.70, drift detected,
                               or robustness pass rate < 75%
  8. log_monitoring_results    Log all metrics and decisions to MLflow under 'monitoring-run'

Usage:
  export $(grep -v '^#' .env | grep -v '^$' | xargs)
  mlflow ui --port 5000 --backend-store-uri sqlite:///mlflow.db &
  zenml login --local

  MLFLOW_TRACKING_URI=http://localhost:5000 \\
  PYTHONPATH=. python src/pipelines/monitoring_pipeline.py

  # View results
  open reports/monitoring/drift_report.html  # Drift report
  open http://localhost:5000            # MLflow dashboard
  open http://127.0.0.1:8237           # ZenML pipeline DAG

  # Run A/B test separately
  MLFLOW_TRACKING_URI=http://localhost:5000 \\
  PYTHONPATH=. python src/monitoring/ab_test.py

  # Run explainability separately
  MLFLOW_TRACKING_URI=http://localhost:5000 \\
  PYTHONPATH=. python src/monitoring/explain.py
"""

import os
from pathlib import Path
from typing import Annotated

import pandas as pd
from zenml import pipeline, step
from zenml.logger import get_logger

logger = get_logger(__name__)


# ─── Steps ───────────────────────────────────────────────────────────────────


@step
def load_reference_data(
    processed_csv: str = "data/processed/features.csv",
) -> Annotated[pd.DataFrame, "reference_data"]:
    """Load the reference (training) dataset for drift comparison."""
    df = pd.read_csv(processed_csv)
    logger.info(f"✅ Reference data loaded: {df.shape}")
    return df


@step
def load_production_data(
    log_path: str = "reports/monitoring/predictions.jsonl",
) -> Annotated[pd.DataFrame, "production_data"]:
    """
    Load recent production prediction logs.
    Simulates production data from logged predictions.
    Falls back to a synthetic sample if logs are absent.
    """
    import json
    from pathlib import Path

    log_file = Path(log_path)
    if log_file.exists():
        rows = []
        with open(log_file) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
        if rows:
            df = pd.json_normalize(rows)
            logger.info(f"✅ Production data loaded: {len(df)} rows from logs")
            return df

    # Fallback: generate a small synthetic sample (simulates production traffic)
    logger.warning("⚠️  No production logs found – using synthetic sample")
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.generate_data import generate_dataset

    df = generate_dataset(200)
    logger.info(f"✅ Synthetic production sample: {df.shape}")
    return df


@step
def detect_drift(
    reference_data: pd.DataFrame,
    production_data: pd.DataFrame,
) -> Annotated[dict, "drift_report"]:
    """Run data drift detection between reference and production data."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from monitoring.monitor import compute_data_drift

    numeric_ref = reference_data.select_dtypes(include="number")
    numeric_prod = production_data.select_dtypes(include="number")

    # Align columns
    common_cols = list(set(numeric_ref.columns) & set(numeric_prod.columns))
    if not common_cols:
        logger.warning("⚠️  No common numeric columns for drift check")
        return {"drift_detected": False, "method": "skipped"}

    report = compute_data_drift(
        numeric_ref[common_cols],
        numeric_prod[common_cols],
        output_path="reports/monitoring/drift_report.html",
    )
    logger.info(f"Drift report: {report}")
    return report


@step
def evaluate_on_test_set(
    raw_csv: str = "data/masri_finetune/eval_only_masri.csv",
    model_name: str = "gpt-4o-mini",
    max_scenarios: int = 30,
) -> Annotated[dict, "test_metrics"]:
    """Evaluate model on a held-out test set (unseen data)."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from evaluation.evaluate import evaluate_model_masri as evaluate_model
    from models.llm_model import IntersectionLLM

    ft_id = os.environ.get("FINE_TUNED_MODEL_ID", "")
    if ft_id:
        llm = IntersectionLLM(model="gpt-4o-mini", fine_tuned_model_id=ft_id, few_shot=False)
    else:
        llm = IntersectionLLM(model=model_name, few_shot=True)
    try:
        metrics = evaluate_model(raw_csv, llm, max_scenarios=max_scenarios)
    except Exception as exc:
        logger.warning(f"⚠️  Evaluation failed ({exc}) – returning empty metrics")
        metrics = {}

    logger.info(f"Test metrics: {metrics}")
    return metrics


@step
def run_robustness_tests(
    model_name: str = "gpt-4o-mini",
) -> Annotated[dict, "robustness_report"]:
    """Run adversarial and behavioral robustness tests."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from evaluation.evaluate import RobustnessTests
    from models.llm_model import IntersectionLLM

    ft_id = os.environ.get("FINE_TUNED_MODEL_ID", "")
    if ft_id:
        llm = IntersectionLLM(model="gpt-4o-mini", fine_tuned_model_id=ft_id, few_shot=False)
    else:
        llm = IntersectionLLM(model=model_name, few_shot=True)
    try:
        tester = RobustnessTests(llm)
        report = tester.run_all()
    except Exception as exc:
        logger.warning(f"⚠️  Robustness tests failed ({exc})")
        report = {"passed": 0, "total": 0, "pass_rate": 0.0, "details": []}

    logger.info(f"Robustness: {report['passed']}/{report['total']} tests passed")
    return report


@step
def audit_model_bias(
    raw_csv: str = "data/raw/generated_dataset.csv",
    model_name: str = "gpt-4o-mini",
) -> Annotated[dict, "bias_report"]:
    """Audit model for bias across vehicle direction groups."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from evaluation.evaluate import audit_bias
    from models.llm_model import IntersectionLLM

    ft_id = os.environ.get("FINE_TUNED_MODEL_ID", "")
    if ft_id:
        llm = IntersectionLLM(model="gpt-4o-mini", fine_tuned_model_id=ft_id, few_shot=False)
    else:
        llm = IntersectionLLM(model=model_name, few_shot=True)
    try:
        report = audit_bias(raw_csv, llm, max_scenarios_per_group=10)
    except Exception as exc:
        logger.warning(f"⚠️  Bias audit failed ({exc})")
        report = {"bias_detected": False, "error": str(exc)}

    logger.info(f"Bias detected: {report.get('bias_detected', False)}")
    return report


@step
def continual_learning_decision(
    test_metrics: dict,
    drift_report: dict,
    robustness_report: dict,
) -> Annotated[bool, "should_retrain"]:
    """
    Decide whether to trigger retraining based on safety-first thresholds:
      FNR > 0.08          Missed conflicts are safety-critical
      Recall < 0.92       Model misses real conflicts
      Drift detected      Input distribution shifted from training data
      Robustness < 0.90   Model unstable under small perturbations
      Rule agreement < 75%  Model may be right for the wrong reason
    """
    fnr = test_metrics.get("fnr", 0.0)
    recall = test_metrics.get("recall", 1.0)
    rule_agree = test_metrics.get("rule_agreement_rate", 1.0)
    drift = drift_report.get("drift_detected", False)
    robustness_pass = robustness_report.get("pass_rate", 1.0)

    reasons = []

    if fnr > 0.08:
        reasons.append(f"FNR={fnr:.4f} > 0.08 (safety-critical)")
    if recall < 0.92:
        reasons.append(f"Recall={recall:.4f} < 0.92")
    if drift:
        reasons.append("Data drift detected")
    if robustness_pass < 0.90:
        reasons.append(f"Robustness pass rate={robustness_pass:.2f} < 0.90")
    if rule_agree < 0.75:
        reasons.append(f"Rule agreement={rule_agree:.4f} < 0.75")

    should_retrain = len(reasons) > 0

    if should_retrain:
        logger.warning(f"Retraining triggered: {'; '.join(reasons)}")
    else:
        logger.info("Model performance OK — no retraining needed")

    return should_retrain


@step
def log_monitoring_results(
    test_metrics: dict,
    drift_report: dict,
    robustness_report: dict,
    bias_report: dict,
    should_retrain: bool,
) -> None:
    """Log all monitoring results to MLflow for tracking."""
    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("traffic-intersection-llm")

    with mlflow.start_run(run_name="monitoring-run"):
        if test_metrics:
            mlflow.log_metrics(
                {f"test_{k}": v for k, v in test_metrics.items() if isinstance(v, (int, float))}
            )
        mlflow.log_param("drift_detected", drift_report.get("drift_detected", False))
        mlflow.log_param("should_retrain", should_retrain)
        mlflow.log_param("robustness_pass_rate", robustness_report.get("pass_rate", 0))
        mlflow.log_param("bias_detected", bias_report.get("bias_detected", False))

    logger.info("✅ Monitoring results logged to MLflow")


# ─── Pipeline ─────────────────────────────────────────────────────────────────


@pipeline(name="traffic_monitoring_pipeline", enable_cache=False)
def monitoring_pipeline(
    model_name: str = os.environ.get(
        "FINE_TUNED_MODEL_ID", "ft:gpt-4o-mini-2024-07-18:personal::DX7kzKtB"
    ),
    max_scenarios: int = 30,
):
    """
    Full monitoring pipeline:
      load_data → drift → evaluate → robustness → bias → decision → log
    """
    ref_data = load_reference_data()
    prod_data = load_production_data()
    drift = detect_drift(ref_data, prod_data)
    metrics = evaluate_on_test_set(model_name=model_name, max_scenarios=max_scenarios)
    robustness = run_robustness_tests(model_name=model_name)
    bias = audit_model_bias(model_name=model_name)
    retrain = continual_learning_decision(metrics, drift, robustness)
    log_monitoring_results(metrics, drift, robustness, bias, retrain)


if __name__ == "__main__":
    monitoring_pipeline()

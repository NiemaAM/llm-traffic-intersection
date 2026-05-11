"""
src/monitoring/ab_test.py
-------------------------
Online A/B test: base GPT-4o-mini (model A) vs fine-tuned DX7kzKtB (model B).
Uses scipy chi2_contingency for statistical significance.
Results logged to MLflow.

Usage:
  export OPENAI_API_KEY=sk-...
  MLFLOW_TRACKING_URI=http://localhost:5000 \
  PYTHONPATH=. python src/monitoring/ab_test.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import chi2_contingency
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))  # Add project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))  # Add src directory

# Change to project root directory
os.chdir(Path(__file__).parent.parent.parent)

load_dotenv(Path(__file__).parent.parent / ".env", override=False)

from models.llm_model import IntersectionLLM  # noqa: E402
from monitoring.monitor import ABTestRouter  # noqa: E402


def main():
    # ── Config ─────────────────────────────────────────────────────────────────────

    MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT = "traffic-intersection-llm"
    TEST_CSV = "data/masri_finetune/eval_only_masri.csv"
    N_SCENARIOS = 30  # scenarios per variant
    RANDOM_STATE = 7

    FT_MODEL_ID = os.environ.get(
        "FINE_TUNED_MODEL_ID",
        "ft:gpt-4o-mini-2024-07-18:personal::DX7kzKtB",
    )

    # ── Models ─────────────────────────────────────────────────────────────────────

    model_a = IntersectionLLM(model="gpt-4o-mini", few_shot=True)  # baseline
    model_b = IntersectionLLM(  # fine-tuned
        model="gpt-4o-mini",
        fine_tuned_model_id=FT_MODEL_ID,
        few_shot=False,
    )

    # ── Load test scenarios ─────────────────────────────────────────────────────────

    df = pd.read_csv(TEST_CSV).sample(N_SCENARIOS * 2, random_state=RANDOM_STATE)
    half = N_SCENARIOS
    df_a = df.iloc[:half]
    df_b = df.iloc[half:]

    # ── ABTestRouter ────────────────────────────────────────────────────────────────

    router = ABTestRouter(
        model_a_config={"name": "gpt-4o-mini-base", "model": "gpt-4o-mini"},
        model_b_config={"name": f"ft-{FT_MODEL_ID.split('::')[-1]}", "model": FT_MODEL_ID},
        traffic_split=0.5,
    )

    # ── Evaluate each variant ───────────────────────────────────────────────────────

    def evaluate_variant(model, df_subset, variant_name):
        y_true, y_pred = [], []
        for _, row in df_subset.iterrows():
            try:
                scenario = json.loads(row["scenario"])
                vehicles = scenario.get("vehicles_scenario", [])
                true_label = str(row["is_conflict"]).strip().lower()
                result = model.predict({"vehicles": vehicles})
                pred_label = result.get("is_conflict", "no")
                y_true.append(true_label)
                y_pred.append(pred_label)
                correct = pred_label == true_label
                router.record_outcome(f"{variant_name}-{len(y_true)}", correct)
                print(
                    f"  [{variant_name}] [{len(y_true)}/{len(df_subset)}] "
                    f"true={true_label} pred={pred_label} {'✓' if correct else '✗'}"
                )
            except Exception as e:
                print(f"  [{variant_name}] Error: {e}")

        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, pos_label="yes", zero_division=0)
        wins = sum(p == t for p, t in zip(y_pred, y_true))
        return {
            "accuracy": accuracy,
            "f1": f1,
            "correct": wins,
            "total": len(y_true),
            "y_true": y_true,
            "y_pred": y_pred,
        }

    print("\n[A] Baseline: gpt-4o-mini (few-shot)")
    res_a = evaluate_variant(model_a, df_a, "A")

    print(f"\n[B] Fine-tuned: {FT_MODEL_ID.split('::')[-1]}")
    res_b = evaluate_variant(model_b, df_b, "B")

    # ── Statistical significance (chi2) ─────────────────────────────────────────────

    contingency = [
        [res_a["correct"], res_a["total"] - res_a["correct"]],
        [res_b["correct"], res_b["total"] - res_b["correct"]],
    ]
    chi2, p_value, dof, _ = chi2_contingency(contingency)
    significant = p_value < 0.05
    winner = "B (fine-tuned)" if res_b["accuracy"] > res_a["accuracy"] else "A (baseline)"

    # ── Print summary ────────────────────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("A/B TEST RESULTS")
    print("=" * 60)
    print(
        f"Model A (baseline):   accuracy={res_a['accuracy']:.4f}  F1={res_a['f1']:.4f}  ({res_a['correct']}/{res_a['total']})"
    )
    print(
        f"Model B (fine-tuned): accuracy={res_b['accuracy']:.4f}  F1={res_b['f1']:.4f}  ({res_b['correct']}/{res_b['total']})"
    )
    print(f"Chi2={chi2:.4f}  p-value={p_value:.4f}  dof={dof}")
    print(f"Statistically significant: {significant}")
    print(f"Winner: {winner}")
    print("=" * 60)

    router_summary = router.summary()
    print(f"\nABTestRouter summary: {router_summary}")

    # ── Log to MLflow ────────────────────────────────────────────────────────────────

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="ab-test"):
        mlflow.log_params(
            {
                "model_a": "gpt-4o-mini-base",
                "model_b": FT_MODEL_ID,
                "n_scenarios_per_variant": N_SCENARIOS,
                "test_data": TEST_CSV,
                "statistical_test": "chi2_contingency",
            }
        )
        mlflow.log_metrics(
            {
                "model_a_accuracy": res_a["accuracy"],
                "model_a_f1": res_a["f1"],
                "model_b_accuracy": res_b["accuracy"],
                "model_b_f1": res_b["f1"],
                "chi2_statistic": chi2,
                "p_value": p_value,
                "significant": float(significant),
                "accuracy_lift": res_b["accuracy"] - res_a["accuracy"],
                "f1_lift": res_b["f1"] - res_a["f1"],
            }
        )
        mlflow.log_param("winner", winner)
        print(f"\n✅ Logged to MLflow: {MLFLOW_URI}")


if __name__ == "__main__":
    main()

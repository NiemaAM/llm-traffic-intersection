"""
data_pipeline.py
----------------
ZenML pipeline for data ingestion, validation, and feature engineering.
Milestone 3: Full data pipeline with Great Expectations + DVC + Feast.
"""

from pathlib import Path
from typing import Annotated

import pandas as pd
from zenml import pipeline, step
from zenml.logger import get_logger

logger = get_logger(__name__)

RAW_DATA_PATH = "data/raw/generated_dataset.csv"
PROCESSED_DATA_PATH = "data/processed/features.csv"
SCALER_PATH = "models/scaler.joblib"


# ─── Steps ───────────────────────────────────────────────────────────────────


@step
def ingest_data(
    num_records: int = 1000,
    seed: int = 42,
) -> Annotated[pd.DataFrame, "raw_data"]:
    """
    Step 1 – Data Ingestion.
    Generates synthetic intersection data and writes it to data/raw/.
    In production this step would pull from a data lake / database.
    """
    import random

    random.seed(seed)

    # Import generator from src
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.generate_data import _save_layout, generate_dataset

    _save_layout()
    df = generate_dataset(num_records)

    out = Path(RAW_DATA_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info(f"✅ Ingested {len(df)} records → {out}")
    return df


@step
def validate_data(
    df: pd.DataFrame,
) -> Annotated[pd.DataFrame, "validated_data"]:
    """
    Step 2 – Data Validation.
    Runs schema checks and Great Expectations suite.
    Raises on validation failure.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.validate_data import validate_schema

    result = validate_schema(df)
    logger.info(result.report())

    if not result.success:
        raise ValueError(f"Data validation failed:\n{result.report()}")

    logger.info("✅ Data validation passed")
    return df


@step
def engineer_features(
    df: pd.DataFrame,
) -> Annotated[pd.DataFrame, "feature_data"]:
    """
    Step 3 – Feature Engineering.
    Runs the scikit-learn feature pipeline.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    import joblib
    from sklearn.preprocessing import StandardScaler

    from features.preprocess import NUMERIC_FEATURES, build_feature_pipeline

    pipeline = build_feature_pipeline()
    df_feat = pipeline.fit_transform(df)

    feat_cols = [c for c in NUMERIC_FEATURES if c in df_feat.columns]
    scaler = StandardScaler()
    df_feat[feat_cols] = scaler.fit_transform(df_feat[feat_cols])

    # Save artefacts
    Path(SCALER_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)

    out = Path(PROCESSED_DATA_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(out, index=False)

    logger.info(f"✅ Features saved → {out}  shape={df_feat.shape}")
    return df_feat


@step
def version_data(
    df: pd.DataFrame,
) -> Annotated[pd.DataFrame, "versioned_data"]:
    """
    Step 4 – Data Versioning with DVC.
    Commits the processed dataset to DVC tracking.
    Falls back gracefully if DVC is not initialized.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["dvc", "add", PROCESSED_DATA_PATH], capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            logger.info(f"✅ DVC versioned: {PROCESSED_DATA_PATH}")
        else:
            logger.warning(f"⚠️  DVC add failed (non-fatal): {result.stderr[:200]}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.warning("⚠️  DVC not available – skipping versioning step")

    return df


@step
def push_to_feature_store(
    df: pd.DataFrame,
) -> None:
    """
    Step 5 – Feature Store (Feast).
    Materializes features to the offline Feast feature store.
    Falls back if Feast is not configured.
    """
    try:
        from feast import FeatureStore

        store = FeatureStore(repo_path="data/feature_store")
        store.materialize_incremental(end_date=pd.Timestamp.now(tz="UTC"))
        logger.info("✅ Features materialized to Feast feature store")
    except Exception as exc:
        logger.warning(f"⚠️  Feast not configured ({exc}) – skipping feature store push")


# ─── Pipeline ─────────────────────────────────────────────────────────────────


@pipeline(name="traffic_data_pipeline", enable_cache=True)
def data_pipeline(num_records: int = 1000, seed: int = 42):
    """
    Full data pipeline:
      ingest → validate → engineer → version → feature_store
    """
    raw = ingest_data(num_records=num_records, seed=seed)
    valid = validate_data(df=raw)
    features = engineer_features(df=valid)
    versioned = version_data(df=features)
    push_to_feature_store(df=versioned)


if __name__ == "__main__":
    data_pipeline()

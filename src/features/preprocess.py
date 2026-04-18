"""
preprocess.py
-------------
Data preprocessing and feature engineering using scikit-learn Pipelines.
Produces clean features ready for both the feature store and model training.
"""

import ast
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ─── Custom transformers ──────────────────────────────────────────────────────

class DirectionEncoder(BaseEstimator, TransformerMixin):
    """Encode cardinal directions to integers (N=0, E=1, S=2, W=3)."""
    _map = {"north": 0, "east": 1, "south": 2, "west": 3}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["direction_enc"] = df["direction"].map(self._map).fillna(-1).astype(int)
        return df


class ConflictFlagEncoder(BaseEstimator, TransformerMixin):
    """Convert is_conflict yes/no to binary 1/0."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["conflict_label"] = (df["is_conflict"] == "yes").astype(int)
        return df


class WaitingTimeExtractor(BaseEstimator, TransformerMixin):
    """Extract numeric waiting time for the current vehicle from the dict string."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        waits = []
        for _, row in df.iterrows():
            try:
                d = ast.literal_eval(row["waiting_times"])
                waits.append(d.get(row["vehicle_id"], 0) or 0)
            except Exception:
                waits.append(0)
        df["vehicle_waiting_time"] = waits
        return df


class PriorityExtractor(BaseEstimator, TransformerMixin):
    """Extract numeric priority rank for the current vehicle."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        prios = []
        for _, row in df.iterrows():
            try:
                d = ast.literal_eval(row["priority_order"])
                prios.append(d.get(row["vehicle_id"]) or 99)
            except Exception:
                prios.append(99)
        df["vehicle_priority"] = prios
        return df


class ScenarioAggFeatures(BaseEstimator, TransformerMixin):
    """
    Add scenario-level aggregated features:
    - avg_speed_in_scenario
    - max_speed_in_scenario
    - num_vehicles_in_scenario
    - avg_distance_in_scenario
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        agg = df.groupby("scenario_id").agg(
            avg_speed_in_scenario=("speed", "mean"),
            max_speed_in_scenario=("speed", "max"),
            num_vehicles_in_scenario=("vehicle_id", "count"),
            avg_distance_in_scenario=("distance_to_intersection", "mean"),
        ).reset_index()
        df = df.merge(agg, on="scenario_id", how="left")
        return df


class DropRawColumns(BaseEstimator, TransformerMixin):
    """Drop columns that are no longer needed after feature extraction."""

    def __init__(self, cols=None):
        self.cols = cols or [
            "vehicle_id", "destination", "decisions",
            "places_of_conflicts", "conflict_vehicles",
            "priority_order", "waiting_times", "scenario_id",
            "is_conflict",
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=[c for c in self.cols if c in X.columns])


# ─── Pipeline factory ─────────────────────────────────────────────────────────

def build_feature_pipeline() -> Pipeline:
    """Build the full feature engineering pipeline."""
    return Pipeline([
        ("direction_enc",     DirectionEncoder()),
        ("conflict_flag",     ConflictFlagEncoder()),
        ("waiting_time",      WaitingTimeExtractor()),
        ("priority",          PriorityExtractor()),
        ("scenario_agg",      ScenarioAggFeatures()),
        ("drop_raw",          DropRawColumns()),
    ])


NUMERIC_FEATURES = [
    "lane", "speed", "distance_to_intersection", "direction_enc",
    "vehicle_waiting_time", "vehicle_priority",
    "avg_speed_in_scenario", "max_speed_in_scenario",
    "num_vehicles_in_scenario", "avg_distance_in_scenario",
    "number_of_conflicts",
]
TARGET_COLUMN = "conflict_label"


def preprocess(
    input_path: str | Path,
    output_path: str | Path,
    scaler_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Full preprocessing: load CSV → feature pipeline → optional scaling → save.

    Returns the processed DataFrame.
    """
    import joblib

    df = pd.read_csv(input_path)
    print(f"📥 Loaded {len(df)} rows from {input_path}")

    pipeline = build_feature_pipeline()
    df_feat = pipeline.fit_transform(df)

    # Scale numeric features
    feat_cols = [c for c in NUMERIC_FEATURES if c in df_feat.columns]
    scaler = StandardScaler()
    df_feat[feat_cols] = scaler.fit_transform(df_feat[feat_cols])

    if scaler_path:
        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"💾 Scaler saved → {scaler_path}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(out, index=False)
    print(f"✅ Processed data saved → {out}  ({len(df_feat)} rows, {len(df_feat.columns)} cols)")
    return df_feat


if __name__ == "__main__":
    preprocess(
        "data/raw/generated_dataset.csv",
        "data/processed/features.csv",
        "models/scaler.joblib",
    )

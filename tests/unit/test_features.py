"""
test_features.py
Unit tests for feature engineering pipeline.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.generate_data import generate_dataset
from src.features.preprocess import (
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    ConflictFlagEncoder,
    DirectionEncoder,
    PriorityExtractor,
    ScenarioAggFeatures,
    WaitingTimeExtractor,
    build_feature_pipeline,
)


@pytest.fixture
def sample_df():
    return generate_dataset(num_records=100)


class TestDirectionEncoder:
    def test_encodes_all_directions(self, sample_df):
        enc = DirectionEncoder()
        result = enc.fit_transform(sample_df)
        assert "direction_enc" in result.columns
        assert set(result["direction_enc"].unique()).issubset({0, 1, 2, 3})

    def test_no_negative_encodings(self, sample_df):
        enc = DirectionEncoder()
        result = enc.fit_transform(sample_df)
        assert (result["direction_enc"] >= 0).all()


class TestConflictFlagEncoder:
    def test_binary_output(self, sample_df):
        enc = ConflictFlagEncoder()
        result = enc.fit_transform(sample_df)
        assert "conflict_label" in result.columns
        assert set(result["conflict_label"].unique()).issubset({0, 1})

    def test_yes_maps_to_one(self, sample_df):
        enc = ConflictFlagEncoder()
        result = enc.fit_transform(sample_df)
        yes_rows = result[sample_df["is_conflict"] == "yes"]
        assert (yes_rows["conflict_label"] == 1).all()


class TestWaitingTimeExtractor:
    def test_column_created(self, sample_df):
        ext = WaitingTimeExtractor()
        result = ext.fit_transform(sample_df)
        assert "vehicle_waiting_time" in result.columns

    def test_non_negative(self, sample_df):
        ext = WaitingTimeExtractor()
        result = ext.fit_transform(sample_df)
        assert (result["vehicle_waiting_time"] >= 0).all()


class TestPriorityExtractor:
    def test_column_created(self, sample_df):
        ext = PriorityExtractor()
        result = ext.fit_transform(sample_df)
        assert "vehicle_priority" in result.columns

    def test_priority_positive(self, sample_df):
        ext = PriorityExtractor()
        result = ext.fit_transform(sample_df)
        assert (result["vehicle_priority"] > 0).all()


class TestScenarioAggFeatures:
    def test_agg_columns_created(self, sample_df):
        agg = ScenarioAggFeatures()
        result = agg.fit_transform(sample_df)
        for col in [
            "avg_speed_in_scenario",
            "max_speed_in_scenario",
            "num_vehicles_in_scenario",
            "avg_distance_in_scenario",
        ]:
            assert col in result.columns

    def test_num_vehicles_positive(self, sample_df):
        agg = ScenarioAggFeatures()
        result = agg.fit_transform(sample_df)
        assert (result["num_vehicles_in_scenario"] > 0).all()


class TestFullPipeline:
    def test_pipeline_runs(self, sample_df):
        pipeline = build_feature_pipeline()
        result = pipeline.fit_transform(sample_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_df)

    def test_target_column_present(self, sample_df):
        pipeline = build_feature_pipeline()
        result = pipeline.fit_transform(sample_df)
        assert TARGET_COLUMN in result.columns

    def test_no_raw_text_columns(self, sample_df):
        pipeline = build_feature_pipeline()
        result = pipeline.fit_transform(sample_df)
        # Raw columns that should have been dropped
        for col in ["decisions", "priority_order", "waiting_times", "conflict_vehicles"]:
            assert col not in result.columns

    def test_numeric_features_present(self, sample_df):
        pipeline = build_feature_pipeline()
        result = pipeline.fit_transform(sample_df)
        for col in NUMERIC_FEATURES:
            assert col in result.columns, f"Missing feature: {col}"

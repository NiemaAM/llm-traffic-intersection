"""
test_data.py
Unit tests for data generation and validation.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.generate_data import generate_scenario, generate_dataset, INTERSECTION_LAYOUT
from src.data.validate_data import validate_schema, REQUIRED_COLUMNS


class TestDataGeneration:

    def test_scenario_has_vehicles(self):
        records = generate_scenario(num_vehicles=3)
        assert len(records) == 3

    def test_scenario_fields_present(self):
        records = generate_scenario(num_vehicles=2)
        for r in records:
            for col in REQUIRED_COLUMNS:
                assert col in r, f"Missing column: {col}"

    def test_direction_valid(self):
        records = generate_scenario(num_vehicles=4)
        valid = set(INTERSECTION_LAYOUT["incoming_directions"])
        for r in records:
            assert r["direction"] in valid

    def test_is_conflict_valid(self):
        records = generate_scenario(num_vehicles=4)
        for r in records:
            assert r["is_conflict"] in ("yes", "no")

    def test_dataset_row_count(self):
        df = generate_dataset(num_records=50)
        assert len(df) >= 50

    def test_dataset_columns(self):
        df = generate_dataset(num_records=20)
        for col in REQUIRED_COLUMNS:
            assert col in df.columns

    def test_speed_in_range(self):
        df = generate_dataset(num_records=100)
        assert df["speed"].min() >= 0
        assert df["speed"].max() <= 200

    def test_distance_in_range(self):
        df = generate_dataset(num_records=100)
        assert df["distance_to_intersection"].min() >= 0

    def test_lane_in_range(self):
        df = generate_dataset(num_records=100)
        assert df["lane"].min() >= 1


class TestDataValidation:

    def setup_method(self):
        self.df = generate_dataset(num_records=100)

    def test_valid_dataset_passes(self):
        result = validate_schema(self.df)
        assert result.success, f"Validation failed: {result.failed}"

    def test_missing_column_fails(self):
        bad_df = self.df.drop(columns=["vehicle_id"])
        result = validate_schema(bad_df)
        assert not result.success

    def test_invalid_direction_fails(self):
        bad_df = self.df.copy()
        bad_df.loc[0, "direction"] = "diagonal"
        result = validate_schema(bad_df)
        assert not result.success

    def test_invalid_conflict_flag_fails(self):
        bad_df = self.df.copy()
        bad_df.loc[0, "is_conflict"] = "maybe"
        result = validate_schema(bad_df)
        assert not result.success

    def test_empty_dataframe_fails(self):
        result = validate_schema(pd.DataFrame())
        assert not result.success

    def test_null_vehicle_id_fails(self):
        bad_df = self.df.copy()
        bad_df.loc[0, "vehicle_id"] = None
        result = validate_schema(bad_df)
        assert not result.success

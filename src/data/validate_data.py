"""
validate_data.py
----------------
Data validation and schema verification using Great Expectations.
Validates raw ingested data before it enters the ML pipeline.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Any

try:
    import great_expectations as gx
    from great_expectations.core import ExpectationSuite
    GX_AVAILABLE = True
except ImportError:
    GX_AVAILABLE = False


# ─── Schema definition ───────────────────────────────────────────────────────

SCHEMA: dict[str, dict[str, Any]] = {
    "vehicle_id":                {"dtype": "string",  "nullable": False, "unique": False},
    "lane":                      {"dtype": "int",     "nullable": False, "min": 1, "max": 10},
    "speed":                     {"dtype": "float",   "nullable": False, "min": 0.0, "max": 200.0},
    "distance_to_intersection":  {"dtype": "float",   "nullable": False, "min": 0.0, "max": 2000.0},
    "direction":                 {"dtype": "string",  "nullable": False,
                                  "allowed": ["north", "south", "east", "west"]},
    "destination":               {"dtype": "string",  "nullable": False},
    "is_conflict":               {"dtype": "string",  "nullable": False, "allowed": ["yes", "no"]},
    "number_of_conflicts":       {"dtype": "int",     "nullable": False, "min": 0},
    "places_of_conflicts":       {"dtype": "string",  "nullable": False},
    "conflict_vehicles":         {"dtype": "string",  "nullable": False},
    "decisions":                 {"dtype": "string",  "nullable": False},
    "priority_order":            {"dtype": "string",  "nullable": False},
    "waiting_times":             {"dtype": "string",  "nullable": False},
    "scenario_id":               {"dtype": "string",  "nullable": False},
}

REQUIRED_COLUMNS = list(SCHEMA.keys())


# ─── Pandas-based validation (always available) ──────────────────────────────

class ValidationResult:
    def __init__(self):
        self.passed: list[str] = []
        self.failed: list[str] = []

    @property
    def success(self) -> bool:
        return len(self.failed) == 0

    def report(self) -> str:
        lines = ["=== Validation Report ==="]
        lines.append(f"✅ Passed: {len(self.passed)}")
        lines.append(f"❌ Failed: {len(self.failed)}")
        if self.failed:
            lines.append("\nFailed checks:")
            for f in self.failed:
                lines.append(f"  - {f}")
        return "\n".join(lines)


def validate_schema(df: pd.DataFrame) -> ValidationResult:
    result = ValidationResult()

    # 1. Required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        result.failed.append(f"Missing columns: {missing}")
    else:
        result.passed.append("All required columns present")

    if missing:
        return result  # can't continue without required cols

    # 2. Null checks
    for col, spec in SCHEMA.items():
        if not spec.get("nullable", True):
            n_null = df[col].isna().sum()
            if n_null > 0:
                result.failed.append(f"Column '{col}' has {n_null} null values")
            else:
                result.passed.append(f"'{col}' no nulls ✓")

    # 3. Allowed values
    for col, spec in SCHEMA.items():
        if "allowed" in spec and col in df.columns:
            bad = df[~df[col].isin(spec["allowed"])][col].unique()
            if len(bad) > 0:
                result.failed.append(f"Column '{col}' has unexpected values: {bad[:5]}")
            else:
                result.passed.append(f"'{col}' values in allowed set ✓")

    # 4. Numeric range checks
    for col, spec in SCHEMA.items():
        if col not in df.columns:
            continue
        if "min" in spec:
            try:
                lo = pd.to_numeric(df[col], errors="coerce")
                bad = (lo < spec["min"]).sum()
                if bad:
                    result.failed.append(f"'{col}' has {bad} values below min={spec['min']}")
                else:
                    result.passed.append(f"'{col}' min range ✓")
            except Exception:
                pass
        if "max" in spec:
            try:
                hi = pd.to_numeric(df[col], errors="coerce")
                bad = (hi > spec["max"]).sum()
                if bad:
                    result.failed.append(f"'{col}' has {bad} values above max={spec['max']}")
                else:
                    result.passed.append(f"'{col}' max range ✓")
            except Exception:
                pass

    # 5. Row count
    if len(df) == 0:
        result.failed.append("DataFrame is empty")
    else:
        result.passed.append(f"Row count: {len(df)} ✓")

    return result


# ─── Great Expectations suite (optional) ─────────────────────────────────────

def build_gx_suite(df: pd.DataFrame, suite_name: str = "traffic_data_suite") -> dict:
    """
    Build and run a Great Expectations validation suite.
    Returns the validation result as a dict.
    Falls back gracefully if GX is not installed.
    """
    if not GX_AVAILABLE:
        print("⚠️  great_expectations not installed – skipping GX validation")
        return {"success": True, "skipped": True}

    context = gx.get_context()
    ds = context.sources.add_or_update_pandas("traffic_ds")
    da = ds.add_dataframe_asset("traffic_asset")
    batch_request = da.build_batch_request(dataframe=df)

    suite = context.add_or_update_expectation_suite(suite_name)
    validator = context.get_validator(batch_request=batch_request,
                                      expectation_suite=suite)

    validator.expect_table_columns_to_match_ordered_list(REQUIRED_COLUMNS)
    validator.expect_column_values_to_not_be_null("vehicle_id")
    validator.expect_column_values_to_not_be_null("is_conflict")
    validator.expect_column_values_to_be_in_set("is_conflict", ["yes", "no"])
    validator.expect_column_values_to_be_in_set(
        "direction", ["north", "south", "east", "west"])
    validator.expect_column_values_to_be_between("speed", min_value=0, max_value=200)
    validator.expect_column_values_to_be_between(
        "distance_to_intersection", min_value=0, max_value=2000)
    validator.expect_column_values_to_be_between("lane", min_value=1, max_value=10)

    results = validator.validate()
    context.build_data_docs()

    return results.to_json_dict()


def validate_file(csv_path: str | Path) -> ValidationResult:
    """Validate a CSV file. Runs both schema checks and GX if available."""
    df = pd.read_csv(csv_path)
    result = validate_schema(df)
    print(result.report())

    if GX_AVAILABLE:
        gx_result = build_gx_suite(df)
        gx_success = gx_result.get("success", False)
        print(f"\n📊 Great Expectations: {'PASSED' if gx_success else 'FAILED'}")

    return result


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/generated_dataset.csv"
    validate_file(path)

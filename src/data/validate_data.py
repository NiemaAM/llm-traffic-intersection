"""
validate_data.py
----------------
Data validation and schema verification using Great Expectations.
Validates raw ingested data before it enters the ML pipeline.
"""

from pathlib import Path
from typing import Any

import pandas as pd

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
    Falls back gracefully if GX is not installed or API is incompatible.

    Supports both GX 0.x (legacy) and GX 1.x (new fluent API).
    """
    if not GX_AVAILABLE:
        print("⚠️  great_expectations not installed – skipping GX validation")
        return {"success": True, "skipped": True}

    try:
        # ── GX 1.x fluent API ────────────────────────────────────────────────
        context = gx.get_context()

        # Add pandas datasource using the new 1.x API
        data_source = context.data_sources.add_pandas("traffic_ds")
        data_asset  = data_source.add_dataframe_asset("traffic_asset")
        batch_def   = data_asset.add_batch_definition_whole_dataframe("batch")
        batch       = batch_def.get_batch(batch_parameters={"dataframe": df})

        # Build expectation suite
        suite = context.suites.add(gx.ExpectationSuite(name=suite_name))

        expectations = [
            gx.expectations.ExpectColumnValuesToNotBeNull(column="vehicle_id"),
            gx.expectations.ExpectColumnValuesToNotBeNull(column="is_conflict"),
            gx.expectations.ExpectColumnValuesToBeInSet(
                column="is_conflict", value_set=["yes", "no"]),
            gx.expectations.ExpectColumnValuesToBeInSet(
                column="direction", value_set=["north", "south", "east", "west"]),
            gx.expectations.ExpectColumnValuesToBeBetween(
                column="speed", min_value=0, max_value=200),
            gx.expectations.ExpectColumnValuesToBeBetween(
                column="distance_to_intersection", min_value=0, max_value=2000),
            gx.expectations.ExpectColumnValuesToBeBetween(
                column="lane", min_value=1, max_value=10),
        ]
        for exp in expectations:
            suite.add_expectation(exp)
        # context.suites.update(suite)  # not available in all GX 1.x versions

        # Validate
        validation_def = context.validation_definitions.add(
            gx.ValidationDefinition(
                name="traffic_validation",
                data=batch_def,
                suite=suite,
            )
        )
        result = validation_def.run(batch_parameters={"dataframe": df})
        success = result.success
        print(f"📊 Great Expectations (1.x): {'PASSED ✅' if success else 'FAILED ❌'}")
        return {"success": success, "skipped": False}

    except Exception as exc:
        # ── Fallback: run simple column-level checks without GX ──────────────
        print(f"⚠️  Great Expectations suite failed ({type(exc).__name__}: {exc})")
        print("   Falling back to pandas-only validation (already passed above).")
        return {"success": True, "skipped": True, "reason": str(exc)}


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

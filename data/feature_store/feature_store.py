"""
feature_store.py
----------------
Feast feature store definition for vehicle intersection features.
Milestone 3: Feature store for preprocessing and feature engineering.
"""

from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureService, FeatureView, Field, FileSource
from feast.types import Float32, Int32, String


# ─── Data source ─────────────────────────────────────────────────────────────

vehicle_source = FileSource(
    path=str(Path(__file__).parent.parent / "data" / "processed" / "features.csv"),
    timestamp_field="event_timestamp",  # add this column during preprocessing
    created_timestamp_column="created",
)

# ─── Entity ───────────────────────────────────────────────────────────────────

vehicle_entity = Entity(
    name="vehicle",
    join_keys=["vehicle_id"],
    description="A vehicle approaching an intersection",
)

# ─── Feature views ────────────────────────────────────────────────────────────

vehicle_features_view = FeatureView(
    name="vehicle_intersection_features",
    entities=[vehicle_entity],
    ttl=timedelta(days=7),
    schema=[
        Field(name="lane",                     dtype=Int32),
        Field(name="speed",                    dtype=Float32),
        Field(name="distance_to_intersection", dtype=Float32),
        Field(name="direction_enc",            dtype=Int32),
        Field(name="vehicle_waiting_time",     dtype=Float32),
        Field(name="vehicle_priority",         dtype=Int32),
        Field(name="number_of_conflicts",      dtype=Int32),
    ],
    source=vehicle_source,
    tags={"team": "traffic-ml", "milestone": "3"},
)

scenario_features_view = FeatureView(
    name="scenario_aggregate_features",
    entities=[vehicle_entity],
    ttl=timedelta(days=7),
    schema=[
        Field(name="avg_speed_in_scenario",      dtype=Float32),
        Field(name="max_speed_in_scenario",      dtype=Float32),
        Field(name="num_vehicles_in_scenario",   dtype=Int32),
        Field(name="avg_distance_in_scenario",   dtype=Float32),
    ],
    source=vehicle_source,
    tags={"team": "traffic-ml", "milestone": "3"},
)

# ─── Feature service ──────────────────────────────────────────────────────────

conflict_detection_service = FeatureService(
    name="conflict_detection_service",
    features=[vehicle_features_view, scenario_features_view],
    description="All features required for LLM conflict detection input preparation",
)

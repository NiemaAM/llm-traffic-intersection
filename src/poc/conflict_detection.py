"""
conflict_detection.py
---------------------
Core rule-based conflict detection engine.
Implements the intersection conflict logic from Masri et al. (2025) as the
deterministic baseline. This module is used by:
  - The PoC Streamlit app (Milestone 2)
  - The LLM evaluation harness (Milestone 4) for ground-truth labelling
  - Unit tests

Reference:
  Masri et al. (2025) – "Large Language Models (LLMs) as Traffic Control
  Systems at Urban Intersections: A New Paradigm"
  arXiv: https://arxiv.org/abs/2411.10869
  Code:  https://github.com/sarimasri3/Intersection-Conflict-Detection
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

# ─── Intersection layout ─────────────────────────────────────────────────────

LAYOUT_PATH = Path(__file__).parent.parent.parent / "data" / "external" / "intersection_layout.json"

# Pairs of directions whose through-movements physically cross
CONFLICT_DIRECTION_PAIRS: set[frozenset] = {
    frozenset({"north", "south"}),
    frozenset({"east", "west"}),
    frozenset({"north", "east"}),
    frozenset({"south", "west"}),
    frozenset({"north", "west"}),
    frozenset({"south", "east"}),
}

# Distance threshold (metres): both vehicles must be within this to be "near"
NEAR_INTERSECTION_THRESHOLD = 80.0


# ─── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class Vehicle:
    vehicle_id: str
    lane: int
    speed: float  # km/h
    distance_to_intersection: float  # metres
    direction: str  # north | south | east | west
    destination: str

    @property
    def time_to_intersection(self) -> float:
        """Estimated seconds to reach the intersection (v in km/h → m/s)."""
        speed_ms = self.speed / 3.6
        if speed_ms <= 0:
            return float("inf")
        return self.distance_to_intersection / speed_ms

    @property
    def is_near(self) -> bool:
        return self.distance_to_intersection <= NEAR_INTERSECTION_THRESHOLD


@dataclass
class ConflictPair:
    vehicle1: Vehicle
    vehicle2: Vehicle
    location: str = "intersection"

    @property
    def vehicle1_id(self) -> str:
        return self.vehicle1.vehicle_id

    @property
    def vehicle2_id(self) -> str:
        return self.vehicle2.vehicle_id


@dataclass
class IntersectionDecision:
    is_conflict: bool
    conflict_pairs: list[ConflictPair] = field(default_factory=list)
    priority_order: dict[str, int | None] = field(default_factory=dict)
    waiting_times: dict[str, int] = field(default_factory=dict)
    decisions: list[str] = field(default_factory=list)

    @property
    def number_of_conflicts(self) -> int:
        return len(self.conflict_pairs)

    @property
    def places_of_conflicts(self) -> list[str]:
        return list({p.location for p in self.conflict_pairs})

    def to_dict(self) -> dict:
        return {
            "is_conflict": "yes" if self.is_conflict else "no",
            "number_of_conflicts": self.number_of_conflicts,
            "places_of_conflicts": self.places_of_conflicts,
            "conflict_vehicles": [
                {"vehicle1_id": p.vehicle1_id, "vehicle2_id": p.vehicle2_id}
                for p in self.conflict_pairs
            ],
            "decisions": self.decisions,
            "priority_order": self.priority_order,
            "waiting_times": self.waiting_times,
        }


# ─── Parsing helpers ──────────────────────────────────────────────────────────


def parse_intersection_layout(data: dict) -> dict:
    """Return the intersection layout dict (pass-through, validates keys)."""
    required = {"incoming_directions", "lanes_per_direction", "destinations"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Intersection layout missing keys: {missing}")
    return data


def load_intersection_layout(path: Path | str | None = None) -> dict:
    p = Path(path) if path else LAYOUT_PATH
    with open(p) as f:
        return json.load(f)


def parse_vehicles(scenario_data: dict, layout: dict) -> list[Vehicle]:
    """
    Parse a scenario dict (with a 'vehicles_scenario' or 'vehicles' key)
    into a list of Vehicle objects. Validates direction against layout.
    """
    raw = scenario_data.get("vehicles_scenario") or scenario_data.get("vehicles", [])
    valid_dirs = set(layout["incoming_directions"])
    vehicles = []
    for v in raw:
        direction = v["direction"].strip().lower()
        if direction not in valid_dirs:
            raise ValueError(
                f"Vehicle {v['vehicle_id']}: unknown direction '{direction}'. "
                f"Valid: {valid_dirs}"
            )
        vehicles.append(
            Vehicle(
                vehicle_id=str(v["vehicle_id"]),
                lane=int(v["lane"]),
                speed=float(v["speed"]),
                distance_to_intersection=float(v["distance_to_intersection"]),
                direction=direction,
                destination=str(v["destination"]),
            )
        )
    return vehicles


# ─── Conflict detection ───────────────────────────────────────────────────────


def _directions_conflict(d1: str, d2: str) -> bool:
    return frozenset({d1, d2}) in CONFLICT_DIRECTION_PAIRS


def detect_conflicts(vehicles: list[Vehicle]) -> list[ConflictPair]:
    """
    Detect all conflicting vehicle pairs.
    Two vehicles conflict when:
      1. They approach from directions whose paths physically cross.
      2. Both are within NEAR_INTERSECTION_THRESHOLD metres of the intersection.
    """
    conflicts = []
    for i, v1 in enumerate(vehicles):
        for v2 in vehicles[i + 1 :]:
            if _directions_conflict(v1.direction, v2.direction):
                if v1.is_near and v2.is_near:
                    conflicts.append(ConflictPair(v1, v2))
    return conflicts


# ─── Priority & decisions ─────────────────────────────────────────────────────


def assign_priorities(
    vehicles: list[Vehicle],
    conflicts: list[ConflictPair],
) -> IntersectionDecision:
    """
    Assign priority ranks and compute waiting times.

    Priority rule (from Masri et al.):
      - Among conflicting vehicles, the one with the higher speed gets right-of-way
        (lower rank number = higher priority).
      - Non-conflicting vehicles are ranked after conflicting winners.
    """
    priority: dict[str, int | None] = {v.vehicle_id: None for v in vehicles}
    waiting: dict[str, int] = {v.vehicle_id: 0 for v in vehicles}
    decisions: list[str] = []
    yielding_ids: set[str] = set()
    rank = 1

    for cp in conflicts:
        v1, v2 = cp.vehicle1, cp.vehicle2
        if v1.speed >= v2.speed:
            winner, loser = v1, v2
        else:
            winner, loser = v2, v1

        if priority[winner.vehicle_id] is None:
            priority[winner.vehicle_id] = rank
            rank += 1

        if loser.vehicle_id not in yielding_ids:
            yielding_ids.add(loser.vehicle_id)
            # Waiting time proportional to the winner's time-to-intersection
            wait = max(2, math.ceil(winner.time_to_intersection) + 1)
            waiting[loser.vehicle_id] = wait
            decisions.append(
                f"Potential conflict: Vehicle {loser.vehicle_id} "
                f"must yield to Vehicle {winner.vehicle_id}"
            )

    # Assign ranks to vehicles not involved in yielding
    for v in vehicles:
        if priority[v.vehicle_id] is None and v.vehicle_id not in yielding_ids:
            priority[v.vehicle_id] = rank
            rank += 1

    # Assign ranks to yielding vehicles (they go last)
    for v in vehicles:
        if v.vehicle_id in yielding_ids and priority[v.vehicle_id] is None:
            priority[v.vehicle_id] = rank
            rank += 1

    return IntersectionDecision(
        is_conflict=len(conflicts) > 0,
        conflict_pairs=conflicts,
        priority_order=priority,
        waiting_times=waiting,
        decisions=decisions,
    )


# ─── High-level convenience function ─────────────────────────────────────────


def analyze_intersection(
    vehicles: list[Vehicle] | list[dict],
    layout: dict | None = None,
) -> IntersectionDecision:
    """
    Full pipeline: vehicle list → conflict detection → priority assignment.

    Accepts either Vehicle objects or raw dicts (with the same fields).
    Layout is loaded from disk if not provided.
    """
    if layout is None:
        try:
            layout = load_intersection_layout()
        except FileNotFoundError:
            from src.data.generate_data import INTERSECTION_LAYOUT

            layout = INTERSECTION_LAYOUT

    if vehicles and isinstance(vehicles[0], dict):
        scenario = {"vehicles": vehicles}
        vehicles = parse_vehicles(scenario, layout)

    conflicts = detect_conflicts(vehicles)
    return assign_priorities(vehicles, conflicts)

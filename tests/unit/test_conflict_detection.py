"""
test_conflict_detection.py
Unit tests for the rule-based conflict detection engine (Milestone 2).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.poc.conflict_detection import (
    NEAR_INTERSECTION_THRESHOLD,
    Vehicle,
    analyze_intersection,
    assign_priorities,
    detect_conflicts,
    parse_vehicles,
)

LAYOUT = {
    "incoming_directions": ["north", "south", "east", "west"],
    "lanes_per_direction": 2,
    "destinations": {
        "north": ["A", "B"], "south": ["C", "D"],
        "east":  ["E", "F"], "west":  ["G", "H"],
    },
}


def make_vehicle(vid, speed, distance, direction, lane=1, dest="A"):
    return Vehicle(
        vehicle_id=vid, lane=lane, speed=speed,
        distance_to_intersection=distance,
        direction=direction, destination=dest,
    )


# ─── Vehicle model ────────────────────────────────────────────────────────────

class TestVehicleModel:
    def test_near_when_within_threshold(self):
        v = make_vehicle("V1", 50, NEAR_INTERSECTION_THRESHOLD - 1, "north")
        assert v.is_near is True

    def test_not_near_when_far(self):
        v = make_vehicle("V1", 50, NEAR_INTERSECTION_THRESHOLD + 1, "north")
        assert v.is_near is False

    def test_time_to_intersection_positive(self):
        v = make_vehicle("V1", 60, 100, "north")
        assert v.time_to_intersection > 0

    def test_time_to_intersection_zero_speed(self):
        v = make_vehicle("V1", 0, 100, "north")
        import math
        assert math.isinf(v.time_to_intersection)


# ─── Conflict detection ───────────────────────────────────────────────────────

class TestDetectConflicts:
    def test_north_south_near_is_conflict(self):
        vs = [
            make_vehicle("V1", 60, 50, "north"),
            make_vehicle("V2", 50, 55, "south", dest="C"),
        ]
        conflicts = detect_conflicts(vs)
        assert len(conflicts) == 1

    def test_east_west_near_is_conflict(self):
        vs = [
            make_vehicle("V1", 60, 30, "east",  dest="E"),
            make_vehicle("V2", 55, 70, "west",  dest="G"),
        ]
        conflicts = detect_conflicts(vs)
        assert len(conflicts) == 1

    def test_north_east_near_is_conflict(self):
        vs = [
            make_vehicle("V1", 60, 50, "north"),
            make_vehicle("V2", 50, 60, "east", dest="E"),
        ]
        conflicts = detect_conflicts(vs)
        assert len(conflicts) == 1

    def test_same_direction_no_conflict(self):
        vs = [
            make_vehicle("V1", 60, 40, "north"),
            make_vehicle("V2", 50, 50, "north", lane=2),
        ]
        conflicts = detect_conflicts(vs)
        assert len(conflicts) == 0

    def test_far_vehicles_no_conflict(self):
        vs = [
            make_vehicle("V1", 50, 400, "north"),
            make_vehicle("V2", 50, 500, "south", dest="C"),
        ]
        conflicts = detect_conflicts(vs)
        assert len(conflicts) == 0

    def test_one_near_one_far_no_conflict(self):
        vs = [
            make_vehicle("V1", 50, 30,  "north"),
            make_vehicle("V2", 50, 500, "south", dest="C"),
        ]
        conflicts = detect_conflicts(vs)
        assert len(conflicts) == 0

    def test_three_vehicle_multi_conflict(self):
        vs = [
            make_vehicle("V1", 60, 40, "north"),
            make_vehicle("V2", 55, 50, "south", dest="C"),
            make_vehicle("V3", 60, 45, "east",  dest="E"),
        ]
        conflicts = detect_conflicts(vs)
        # N↔S and N↔E are both conflicts; S↔E is also a conflict pair
        assert len(conflicts) >= 2


# ─── Priority assignment ──────────────────────────────────────────────────────

class TestPriorityAssignment:
    def test_faster_gets_priority_1(self):
        vs = [
            make_vehicle("FAST", 80, 40, "north"),
            make_vehicle("SLOW", 30, 50, "south", dest="C"),
        ]
        conflicts = detect_conflicts(vs)
        decision = assign_priorities(vs, conflicts)
        assert decision.priority_order["FAST"] == 1
        assert decision.priority_order["SLOW"] > 1

    def test_winner_waits_zero(self):
        vs = [
            make_vehicle("FAST", 80, 40, "north"),
            make_vehicle("SLOW", 30, 50, "south", dest="C"),
        ]
        conflicts = detect_conflicts(vs)
        decision = assign_priorities(vs, conflicts)
        assert decision.waiting_times["FAST"] == 0

    def test_loser_waits_nonzero(self):
        vs = [
            make_vehicle("FAST", 80, 40, "north"),
            make_vehicle("SLOW", 30, 50, "south", dest="C"),
        ]
        conflicts = detect_conflicts(vs)
        decision = assign_priorities(vs, conflicts)
        assert decision.waiting_times["SLOW"] > 0

    def test_no_conflict_no_waiting(self):
        vs = [
            make_vehicle("V1", 50, 400, "north"),
            make_vehicle("V2", 50, 500, "south", dest="C"),
        ]
        conflicts = detect_conflicts(vs)
        decision = assign_priorities(vs, conflicts)
        assert decision.is_conflict is False
        assert all(w == 0 for w in decision.waiting_times.values())

    def test_decisions_list_populated_on_conflict(self):
        vs = [
            make_vehicle("V1", 80, 40, "north"),
            make_vehicle("V2", 30, 50, "south", dest="C"),
        ]
        conflicts = detect_conflicts(vs)
        decision = assign_priorities(vs, conflicts)
        assert len(decision.decisions) > 0
        assert "yield" in decision.decisions[0].lower()


# ─── Full pipeline ────────────────────────────────────────────────────────────

class TestAnalyzeIntersection:
    def test_dict_input_works(self):
        vehicles = [
            {"vehicle_id": "D001", "lane": 1, "speed": 60,
             "distance_to_intersection": 50, "direction": "north", "destination": "A"},
            {"vehicle_id": "D002", "lane": 1, "speed": 55,
             "distance_to_intersection": 55, "direction": "south", "destination": "C"},
        ]
        decision = analyze_intersection(vehicles, layout=LAYOUT)
        assert decision.is_conflict is True

    def test_to_dict_has_all_keys(self):
        vs = [make_vehicle("V1", 60, 50, "north"), make_vehicle("V2", 55, 55, "south", dest="C")]
        d = analyze_intersection(vs, layout=LAYOUT).to_dict()
        for key in ["is_conflict", "number_of_conflicts", "places_of_conflicts",
                    "conflict_vehicles", "decisions", "priority_order", "waiting_times"]:
            assert key in d

    def test_is_conflict_string_yes_or_no(self):
        vs = [make_vehicle("V1", 60, 50, "north"), make_vehicle("V2", 55, 55, "south", dest="C")]
        d = analyze_intersection(vs, layout=LAYOUT).to_dict()
        assert d["is_conflict"] in ("yes", "no")


# ─── Parse vehicles ───────────────────────────────────────────────────────────

class TestParseVehicles:
    def test_valid_scenario_parses(self):
        scenario = {"vehicles": [
            {"vehicle_id": "V1", "lane": 1, "speed": 50,
             "distance_to_intersection": 100, "direction": "north", "destination": "A"},
        ]}
        vs = parse_vehicles(scenario, LAYOUT)
        assert len(vs) == 1
        assert vs[0].vehicle_id == "V1"

    def test_invalid_direction_raises(self):
        scenario = {"vehicles": [
            {"vehicle_id": "V1", "lane": 1, "speed": 50,
             "distance_to_intersection": 100, "direction": "diagonal", "destination": "A"},
        ]}
        with pytest.raises(ValueError, match="unknown direction"):
            parse_vehicles(scenario, LAYOUT)

    def test_vehicles_scenario_key_works(self):
        scenario = {"vehicles_scenario": [
            {"vehicle_id": "V1", "lane": 1, "speed": 50,
             "distance_to_intersection": 100, "direction": "east", "destination": "E"},
        ]}
        vs = parse_vehicles(scenario, LAYOUT)
        assert vs[0].direction == "east"

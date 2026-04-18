"""
generate_data.py
----------------
Synthetic intersection scenario generator.
Produces vehicle-level records with annotated conflict labels and decisions.
"""

import json
import random
import string
import argparse
import pandas as pd
from pathlib import Path


# ─── Intersection layout ────────────────────────────────────────────────────

INTERSECTION_LAYOUT = {
    "incoming_directions": ["north", "south", "east", "west"],
    "lanes_per_direction": 2,
    "destinations": {
        "north": ["A", "B"],
        "south": ["C", "D"],
        "east":  ["E", "F"],
        "west":  ["G", "H"],
    },
    "conflict_pairs": [
        ("north", "south"),
        ("east",  "west"),
        ("north", "east"),
        ("south", "west"),
        ("north", "west"),
        ("south", "east"),
    ],
}

LAYOUT_PATH = Path(__file__).parent.parent.parent / "data" / "external" / "intersection_layout.json"


def _save_layout() -> None:
    LAYOUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LAYOUT_PATH, "w") as f:
        json.dump(INTERSECTION_LAYOUT, f, indent=2)


def _random_vehicle_id() -> str:
    return "V" + "".join(random.choices(string.digits, k=4))


def _detect_conflicts(vehicles: list[dict]) -> dict:
    """Return conflict metadata for a list of vehicle dicts."""
    conflict_pairs = []
    conflict_places = []
    directions = {v["vehicle_id"]: v["direction"] for v in vehicles}
    close_threshold = 80  # metres

    pairs = [(v1, v2) for i, v1 in enumerate(vehicles) for v2 in vehicles[i + 1:]]
    for v1, v2 in pairs:
        d1, d2 = v1["direction"], v2["direction"]
        near1 = v1["distance_to_intersection"] < close_threshold
        near2 = v2["distance_to_intersection"] < close_threshold
        if (d1, d2) in [(a, b) for a, b in INTERSECTION_LAYOUT["conflict_pairs"]] or \
           (d2, d1) in [(a, b) for a, b in INTERSECTION_LAYOUT["conflict_pairs"]]:
            if near1 and near2:
                conflict_pairs.append({"vehicle1_id": v1["vehicle_id"], "vehicle2_id": v2["vehicle_id"]})
                conflict_places.append("intersection")

    is_conflict = len(conflict_pairs) > 0
    return {
        "is_conflict": "yes" if is_conflict else "no",
        "number_of_conflicts": len(conflict_pairs),
        "places_of_conflicts": str(list(set(conflict_places))) if conflict_places else "[]",
        "conflict_vehicles": str(conflict_pairs),
    }


def _assign_decisions(vehicles: list[dict], conflict_meta: dict) -> tuple[str, str, str]:
    """Assign priority order, waiting times, and decisions based on conflicts."""
    conflict_pairs = eval(conflict_meta["conflict_vehicles"])  # noqa: S307
    priorities = {v["vehicle_id"]: None for v in vehicles}
    waiting = {v["vehicle_id"]: 0 for v in vehicles}
    decisions = []

    priority_rank = 1
    yielding = set()

    for pair in conflict_pairs:
        v1_id, v2_id = pair["vehicle1_id"], pair["vehicle2_id"]
        v1 = next(v for v in vehicles if v["vehicle_id"] == v1_id)
        v2 = next(v for v in vehicles if v["vehicle_id"] == v2_id)
        # Priority: higher speed gets right-of-way first
        if v1["speed"] >= v2["speed"]:
            winner, loser = v1_id, v2_id
        else:
            winner, loser = v2_id, v1_id

        if priorities[winner] is None:
            priorities[winner] = priority_rank
            priority_rank += 1
        if loser not in yielding:
            yielding.add(loser)
            waiting[loser] = random.randint(2, 6)
            decisions.append(f"Potential conflict: Vehicle {loser} must yield to Vehicle {winner}")

    for vid in priorities:
        if priorities[vid] is None and vid not in yielding:
            priorities[vid] = priority_rank
            priority_rank += 1

    return str(decisions), str(priorities), str(waiting)


def generate_scenario(num_vehicles: int | None = None) -> list[dict]:
    """Generate one intersection scenario with 2–8 vehicles."""
    if num_vehicles is None:
        num_vehicles = random.randint(2, 8)

    vehicles = []
    layout = INTERSECTION_LAYOUT

    for _ in range(num_vehicles):
        direction = random.choice(layout["incoming_directions"])
        lane = random.randint(1, layout["lanes_per_direction"])
        destination = random.choice(layout["destinations"][direction])
        vehicles.append({
            "vehicle_id": _random_vehicle_id(),
            "lane": lane,
            "speed": round(random.uniform(10, 90), 2),
            "distance_to_intersection": round(random.uniform(10, 500), 2),
            "direction": direction,
            "destination": destination,
        })

    conflict_meta = _detect_conflicts(vehicles)
    decisions, priorities, waiting = _assign_decisions(vehicles, conflict_meta)

    # Generate ONE scenario_id shared by all vehicles in this scenario
    scenario_id = "S" + "".join(random.choices(string.digits, k=6))

    records = []
    for v in vehicles:
        record = {**v, **conflict_meta,
                  "decisions": decisions,
                  "priority_order": priorities,
                  "waiting_times": waiting,
                  "scenario_id": scenario_id}
        records.append(record)
    return records


def generate_dataset(num_records: int = 1000, num_vehicles: int | None = None) -> pd.DataFrame:
    """Generate a full dataset of `num_records` vehicle rows across multiple scenarios."""
    rows = []
    while len(rows) < num_records:
        rows.extend(generate_scenario(num_vehicles))
    df = pd.DataFrame(rows[:num_records])
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic intersection dataset")
    parser.add_argument("--num-records", type=int, default=1000)
    parser.add_argument("--num-vehicles", type=int, default=None,
                        help="Fixed vehicles per scenario (random 2-8 if omitted)")
    parser.add_argument("--output", type=str,
                        default="data/raw/generated_dataset.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    _save_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_dataset(args.num_records, args.num_vehicles)
    df.to_csv(out_path, index=False)
    print(f"✅ Dataset saved → {out_path}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
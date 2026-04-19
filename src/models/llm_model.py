"""
llm_model.py
------------
LLM-based intersection conflict resolver.
Uses prompt engineering (zero-shot + few-shot) with an OpenAI-compatible API.
Inspired by the PseudoCodeRAG-Translator pattern of structured prompt training.
"""

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

# ─── Prompt templates ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an intelligent traffic intersection controller.
Your task is to analyze vehicle scenarios at a 4-way 8-lane intersection,
detect conflicts between vehicles, and output structured control decisions.

Always respond in valid JSON with this exact schema:
{
  "is_conflict": "yes" | "no",
  "number_of_conflicts": <int>,
  "conflict_vehicles": [{"vehicle1_id": "...", "vehicle2_id": "..."}],
  "decisions": ["..."],
  "priority_order": {"<vehicle_id>": <rank_int>},
  "waiting_times": {"<vehicle_id>": <seconds_int>}
}

Intersection layout (8 lanes, 4 directions):
- North: Lane 1 (right/straight → F,H), Lane 2 (left → E,D,C)
- East:  Lane 3 (right/straight → H,B), Lane 4 (left → G,E,F)
- South: Lane 5 (right/straight → B,D), Lane 6 (left → A,G,H)
- West:  Lane 7 (right/straight → D,F), Lane 8 (left → B,C,A)

Conflict detection rules:
1. Two vehicles CONFLICT if ALL of the following are true:
   a) Their paths physically cross (opposing or perpendicular directions with crossing trajectories)
   b) Both arrive within 5 seconds of each other (time = distance / speed_in_m_s)
   c) Speed in m/s = speed_km_h * 1000 / 3600
2. Same direction vehicles NEVER conflict.
3. Right turns rarely conflict with other right turns.

Priority rules (apply when conflict exists):
1. Straight-going vehicle has priority over turning vehicle.
2. Right-turning vehicle has priority over left-turning vehicle.
3. Right-hand rule: vehicle coming from the right has priority.
4. If arriving more than 1 second earlier: earlier-arriving vehicle has priority.
5. Priority rank 1 = highest priority (does not wait). Rank 2+ must yield.

Decision format: "Potential conflict: Vehicle X must yield to Vehicle Y"
"""

FEW_SHOT_EXAMPLES = [
    # ── Example 1: Classic N↔S conflict, close arrival ────────────────────────
    {
        "role": "user",
        "content": json.dumps({
            "vehicles": [
                {"vehicle_id": "V1001", "lane": 1, "speed": 60, "distance_to_intersection": 50,
                 "direction": "north", "destination": "F"},
                {"vehicle_id": "V1002", "lane": 5, "speed": 45, "distance_to_intersection": 55,
                 "direction": "south", "destination": "D"},
            ]
        })
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "is_conflict": "yes",
            "number_of_conflicts": 1,
            "conflict_vehicles": [{"vehicle1_id": "V1001", "vehicle2_id": "V1002"}],
            "decisions": ["Potential conflict: Vehicle V1002 must yield to Vehicle V1001"],
            "priority_order": {"V1001": 1, "V1002": 2},
            "waiting_times": {"V1001": 0, "V1002": 3}
        })
    },
    # ── Example 2: No conflict — vehicles too far apart in arrival time ────────
    {
        "role": "user",
        "content": json.dumps({
            "vehicles": [
                {"vehicle_id": "V2001", "lane": 1, "speed": 70, "distance_to_intersection": 40,
                 "direction": "north", "destination": "H"},
                {"vehicle_id": "V2002", "lane": 3, "speed": 25, "distance_to_intersection": 380,
                 "direction": "east", "destination": "B"},
            ]
        })
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "is_conflict": "no",
            "number_of_conflicts": 0,
            "conflict_vehicles": [],
            "decisions": [],
            "priority_order": {"V2001": 1, "V2002": 2},
            "waiting_times": {"V2001": 0, "V2002": 0}
        })
    },
    # ── Example 3: N↔E conflict, straight vs right turn ──────────────────────
    {
        "role": "user",
        "content": json.dumps({
            "vehicles": [
                {"vehicle_id": "V3001", "lane": 1, "speed": 55, "distance_to_intersection": 70,
                 "direction": "north", "destination": "H"},
                {"vehicle_id": "V3002", "lane": 3, "speed": 50, "distance_to_intersection": 75,
                 "direction": "east", "destination": "B"},
            ]
        })
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "is_conflict": "yes",
            "number_of_conflicts": 1,
            "conflict_vehicles": [{"vehicle1_id": "V3001", "vehicle2_id": "V3002"}],
            "decisions": ["Potential conflict: Vehicle V3002 must yield to Vehicle V3001"],
            "priority_order": {"V3001": 1, "V3002": 2},
            "waiting_times": {"V3001": 0, "V3002": 3}
        })
    },
    # ── Example 4: 3 vehicles, 2 conflicts ────────────────────────────────────
    {
        "role": "user",
        "content": json.dumps({
            "vehicles": [
                {"vehicle_id": "V4001", "lane": 1, "speed": 60, "distance_to_intersection": 65,
                 "direction": "north", "destination": "F"},
                {"vehicle_id": "V4002", "lane": 5, "speed": 55, "distance_to_intersection": 70,
                 "direction": "south", "destination": "D"},
                {"vehicle_id": "V4003", "lane": 3, "speed": 50, "distance_to_intersection": 72,
                 "direction": "east", "destination": "B"},
            ]
        })
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "is_conflict": "yes",
            "number_of_conflicts": 2,
            "conflict_vehicles": [
                {"vehicle1_id": "V4001", "vehicle2_id": "V4002"},
                {"vehicle1_id": "V4001", "vehicle2_id": "V4003"},
            ],
            "decisions": [
                "Potential conflict: Vehicle V4002 must yield to Vehicle V4001",
                "Potential conflict: Vehicle V4003 must yield to Vehicle V4001",
            ],
            "priority_order": {"V4001": 1, "V4002": 2, "V4003": 2},
            "waiting_times": {"V4001": 0, "V4002": 3, "V4003": 3}
        })
    },
    # ── Example 5: No conflict — same direction ────────────────────────────────
    {
        "role": "user",
        "content": json.dumps({
            "vehicles": [
                {"vehicle_id": "V5001", "lane": 1, "speed": 50, "distance_to_intersection": 100,
                 "direction": "north", "destination": "F"},
                {"vehicle_id": "V5002", "lane": 2, "speed": 45, "distance_to_intersection": 120,
                 "direction": "north", "destination": "E"},
            ]
        })
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "is_conflict": "no",
            "number_of_conflicts": 0,
            "conflict_vehicles": [],
            "decisions": [],
            "priority_order": {"V5001": 1, "V5002": 2},
            "waiting_times": {"V5001": 0, "V5002": 0}
        })
    },
]


# ─── LLM Client ──────────────────────────────────────────────────────────────

# ─── Masri et al. fine-tuning system prompt ──────────────────────────────────
# Used when calling a fine-tuned model — matches the training format exactly

MASRI_SYSTEM_PROMPT = """You are an Urban Intersection Traffic Conflict Detector, responsible for monitoring a four-way intersection with traffic coming from the north, east, south, and west. Each direction has two lanes guiding vehicles to different destinations:
- North: Lane 1 directs vehicles to F and H, Lane 2 directs vehicles to E, D, and C.
- East: Lane 3 leads to H and B, Lane 4 leads to G, E, and F.
- South: Lane 5 directs vehicles to B and D, Lane 6 directs vehicles to A, G, and H.
- West: Lane 7 directs vehicles to D and F, Lane 8 directs vehicles to B, C, and A.
Analyze the traffic data from all directions and lanes, and determine if there is a potential conflict between vehicles at the intersection. Respond only with yes or no."""


def _vehicles_to_text(vehicles: list) -> str:
    """Convert vehicles list to natural language — matches Masri et al. training format."""
    parts = []
    for v in vehicles:
        parts.append(
            f"Vehicle {v['vehicle_id']} is in lane {v['lane']}, moving {v['direction']} "
            f"at a speed of {float(v['speed']):.2f} km/h, and is "
            f"{float(v['distance_to_intersection']):.2f} meters away from the intersection, "
            f"heading towards {v['destination']}."
        )
    return " ".join(parts)


def _build_full_decision(vehicles: list, is_conflict: bool) -> dict:
    """
    Build a full structured decision from a yes/no conflict flag.
    Uses the rule-based engine to compute priorities and waiting times.
    """
    import sys as _sys
    from pathlib import Path as _Path

    # Try to use the original rule-based engine for detailed decisions
    poc_path = str(_Path(__file__).parent.parent / "poc")
    if poc_path not in _sys.path:
        _sys.path.insert(0, poc_path)

    try:
        from conflict_detection_orig import (
            detect_conflicts,
            parse_intersection_layout,
            parse_vehicles,
        )

        LAYOUT_DATA = {
            "intersection_layout": {
                "north": {"1": ["F", "H"], "2": ["E", "D", "C"]},
                "east": {"3": ["H", "B"], "4": ["G", "E", "F"]},
                "south": {"5": ["B", "D"], "6": ["A", "G", "H"]},
                "west": {"7": ["D", "F"], "8": ["B", "C", "A"]},
            }
        }
        layout = parse_intersection_layout(LAYOUT_DATA)
        vobjs = parse_vehicles({"vehicles_scenario": vehicles}, layout)
        conflicts = detect_conflicts(vobjs)

        # Aggregate across conflicts — take max waiting time per vehicle
        # (a vehicle may appear in multiple conflicts; it should wait the longest)
        priority_order: dict = {}
        waiting_times: dict = {}
        for c in conflicts:
            for vid, rank in c["priority_order"].items():
                if vid not in priority_order:
                    priority_order[vid] = rank
            for vid, wt in c["waiting_times"].items():
                waiting_times[vid] = max(waiting_times.get(vid, 0), int(wt))

        return {
            "is_conflict": "yes" if conflicts else "no",
            "number_of_conflicts": len(conflicts),
            "conflict_vehicles": [
                {"vehicle1_id": c["vehicle1_id"], "vehicle2_id": c["vehicle2_id"]}
                for c in conflicts
            ],
            "decisions": [c["decision"] for c in conflicts],
            "priority_order": priority_order,
            "waiting_times": waiting_times,
        }
    except Exception:
        # Fallback: return minimal result based on LLM yes/no
        vids = [v["vehicle_id"] for v in vehicles]
        return {
            "is_conflict": "yes" if is_conflict else "no",
            "number_of_conflicts": 1 if is_conflict else 0,
            "conflict_vehicles": (
                [{"vehicle1_id": vids[0], "vehicle2_id": vids[1]}]
                if is_conflict and len(vids) >= 2
                else []
            ),
            "decisions": (
                [f"Potential conflict: Vehicle {vids[1]} must yield to Vehicle {vids[0]}"]
                if is_conflict and len(vids) >= 2
                else []
            ),
            "priority_order": {},
            "waiting_times": {},
        }


class IntersectionLLM:
    """
    Wrapper around an OpenAI-compatible LLM for intersection conflict resolution.

    Supports:
    - Zero-shot inference
    - Few-shot prompting
    - Fine-tuned model (pass fine_tuned_model_id)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        few_shot: bool = True,
        temperature: float = 0.0,
        fine_tuned_model_id: str | None = None,
    ):
        self.model = fine_tuned_model_id or model
        self.few_shot = few_shot
        self.temperature = temperature
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        # Fine-tuned models use Masri et al. format (natural language → yes/no)
        self._is_finetuned = bool(fine_tuned_model_id)

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key)
            self._available = True
        except ImportError:
            self._client = None
            self._available = False

    def _build_messages(self, scenario: dict) -> list[dict]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if self.few_shot:
            messages.extend(FEW_SHOT_EXAMPLES)
        messages.append({"role": "user", "content": json.dumps(scenario)})
        return messages

    def predict(self, scenario: dict) -> dict:
        """
        Run inference on a single scenario dict with a 'vehicles' list.

        Fine-tuned model: uses Masri et al. natural language format → yes/no,
        then enriches with rule-based engine for full structured output.

        Base model: uses JSON system prompt → full structured JSON output.
        """
        if not self._available:
            raise RuntimeError("openai package not installed. Run: pip install openai")
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set.")

        vehicles = scenario.get("vehicles", [])

        if self._is_finetuned:
            # ── Fine-tuned model: Masri et al. format ────────────────────────
            text = _vehicles_to_text(vehicles)
            messages = [
                {"role": "system", "content": MASRI_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Conflict? (yes/no): {text}",
                },
            ]
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=5,
            )
            answer = response.choices[0].message.content.strip().lower()
            is_conflict = "yes" in answer
            # Enrich with rule-based engine for full structured output
            return _build_full_decision(vehicles, is_conflict)

        else:
            # ── Base model: full JSON format ──────────────────────────────────
            messages = self._build_messages(scenario)
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            return json.loads(content)

    def predict_batch(self, scenarios: list[dict]) -> list[dict]:
        """Run inference on a list of scenario dicts."""
        return [self.predict(s) for s in scenarios]

    def predict_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Given a DataFrame (one row per vehicle), group by scenario_id,
        build scenarios, run inference, and return a results DataFrame.
        """
        results = []
        for scenario_id, group in df.groupby("scenario_id"):
            vehicles = group[[
                "vehicle_id", "lane", "speed",
                "distance_to_intersection", "direction", "destination"
            ]].to_dict(orient="records")
            scenario = {"vehicles": vehicles}
            try:
                pred = self.predict(scenario)
                pred["scenario_id"] = scenario_id
                results.append(pred)
            except Exception as exc:
                results.append({
                    "scenario_id": scenario_id,
                    "error": str(exc),
                })
        return pd.DataFrame(results)


# ─── Fine-tuning helper ───────────────────────────────────────────────────────

def build_finetune_example(row: pd.Series) -> dict:
    """
    Convert a raw dataset row into an OpenAI fine-tuning JSONL record.
    """
    import ast

    vehicles = [{
        "vehicle_id": row["vehicle_id"],
        "lane": int(row["lane"]),
        "speed": float(row["speed"]),
        "distance_to_intersection": float(row["distance_to_intersection"]),
        "direction": row["direction"],
        "destination": row["destination"],
    }]

    try:
        priority_order = ast.literal_eval(row["priority_order"])
        waiting_times  = ast.literal_eval(row["waiting_times"])
        conflict_vehicles = ast.literal_eval(row["conflict_vehicles"])
        decisions      = ast.literal_eval(row["decisions"])
    except Exception:
        priority_order = {}
        waiting_times  = {}
        conflict_vehicles = []
        decisions      = []

    assistant_response = {
        "is_conflict":          row["is_conflict"],
        "number_of_conflicts":  int(row["number_of_conflicts"]),
        "conflict_vehicles":    conflict_vehicles,
        "decisions":            decisions,
        "priority_order":       priority_order,
        "waiting_times":        waiting_times,
    }

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": json.dumps({"vehicles": vehicles})},
            {"role": "assistant", "content": json.dumps(assistant_response)},
        ]
    }


def prepare_finetune_dataset(
    csv_path: str | Path,
    output_path: str | Path,
    max_examples: int = 500,
) -> Path:
    """
    Convert the raw CSV dataset into an OpenAI fine-tuning JSONL file.
    Groups by scenario so each example covers one scenario.
    """
    df = pd.read_csv(csv_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        count = 0
        for scenario_id, group in df.groupby("scenario_id"):
            if count >= max_examples:
                break
            # Use first row for labels (all rows in scenario share same labels)
            row = group.iloc[0]

            import ast
            try:
                conflict_vehicles = ast.literal_eval(row["conflict_vehicles"])
                decisions = ast.literal_eval(row["decisions"])
                priority_order = ast.literal_eval(row["priority_order"])
                waiting_times = ast.literal_eval(row["waiting_times"])
            except Exception:
                conflict_vehicles = []
                decisions = []
                priority_order = {}
                waiting_times = {}

            vehicles = group[[
                "vehicle_id", "lane", "speed",
                "distance_to_intersection", "direction", "destination"
            ]].to_dict(orient="records")

            assistant_response = {
                "is_conflict": row["is_conflict"],
                "number_of_conflicts": int(row["number_of_conflicts"]),
                "conflict_vehicles": conflict_vehicles,
                "decisions": decisions,
                "priority_order": priority_order,
                "waiting_times": waiting_times,
            }

            example = {
                "messages": [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": json.dumps({"vehicles": vehicles})},
                    {"role": "assistant", "content": json.dumps(assistant_response)},
                ]
            }
            f.write(json.dumps(example) + "\n")
            count += 1

    print(f"✅ Fine-tuning dataset saved → {out}  ({count} examples)")
    return out

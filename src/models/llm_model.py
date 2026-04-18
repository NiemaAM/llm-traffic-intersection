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
        Returns parsed JSON response.
        """
        if not self._available:
            raise RuntimeError(
                "openai package not installed. Run: pip install openai"
            )
        if not self._api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable not set."
            )

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
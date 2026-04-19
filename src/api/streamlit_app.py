"""
streamlit_app.py
----------------
Milestone 5 – Production Streamlit Front-End.
Same look and feel as the Milestone 2 PoC with Plotly animations.
Dual-mode: calls FastAPI when available, direct LLM otherwise (HF Space).
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# ── Auto-load .env if present ─────────────────────────────────────────────────
try:
    from dotenv import load_dotenv

    _env = Path(__file__).parent.parent.parent / ".env"
    if _env.exists():
        load_dotenv(dotenv_path=_env, override=False)
except ImportError:
    pass

# ── Config ────────────────────────────────────────────────────────────────────

API_URL = os.environ.get("API_URL", "http://localhost:8000")

LAYOUT_DATA = {
    "north": {"1": ["F", "H"], "2": ["E", "D", "C"]},
    "east": {"3": ["H", "B"], "4": ["G", "E", "F"]},
    "south": {"5": ["B", "D"], "6": ["A", "G", "H"]},
    "west": {"7": ["D", "F"], "8": ["B", "C", "A"]},
}

LANE_DIRECTION = {
    "1": "north",
    "2": "north",
    "3": "east",
    "4": "east",
    "5": "south",
    "6": "south",
    "7": "west",
    "8": "west",
}

DEFAULT_SCENARIO = [
    {
        "vehicle_id": "V001",
        "lane": 1,
        "speed": 50,
        "distance_to_intersection": 100,
        "direction": "north",
        "destination": "F",
    },
    {
        "vehicle_id": "V002",
        "lane": 3,
        "speed": 50,
        "distance_to_intersection": 100,
        "direction": "east",
        "destination": "B",
    },
]

PRESET_SCENARIOS = {
    "⚠️ Classic Conflict (N↔E)": [
        {
            "vehicle_id": "V001",
            "lane": 1,
            "speed": 50,
            "distance_to_intersection": 100,
            "direction": "north",
            "destination": "F",
        },
        {
            "vehicle_id": "V002",
            "lane": 3,
            "speed": 50,
            "distance_to_intersection": 100,
            "direction": "east",
            "destination": "B",
        },
    ],
    "🚨 Head-on (N↔S)": [
        {
            "vehicle_id": "V003",
            "lane": 1,
            "speed": 70,
            "distance_to_intersection": 110,
            "direction": "north",
            "destination": "H",
        },
        {
            "vehicle_id": "V004",
            "lane": 5,
            "speed": 65,
            "distance_to_intersection": 105,
            "direction": "south",
            "destination": "D",
        },
    ],
    "🟢 No Conflict (far arrival gap)": [
        {
            "vehicle_id": "V005",
            "lane": 1,
            "speed": 50,
            "distance_to_intersection": 50,
            "direction": "north",
            "destination": "F",
        },
        {
            "vehicle_id": "V006",
            "lane": 3,
            "speed": 20,
            "distance_to_intersection": 500,
            "direction": "east",
            "destination": "B",
        },
    ],
    "🟠 Multi-vehicle (4 cars)": [
        {
            "vehicle_id": "V007",
            "lane": 1,
            "speed": 60,
            "distance_to_intersection": 100,
            "direction": "north",
            "destination": "F",
        },
        {
            "vehicle_id": "V008",
            "lane": 5,
            "speed": 55,
            "distance_to_intersection": 110,
            "direction": "south",
            "destination": "D",
        },
        {
            "vehicle_id": "V009",
            "lane": 3,
            "speed": 65,
            "distance_to_intersection": 95,
            "direction": "east",
            "destination": "B",
        },
        {
            "vehicle_id": "V010",
            "lane": 7,
            "speed": 50,
            "distance_to_intersection": 200,
            "direction": "west",
            "destination": "F",
        },
    ],
}

# ── Helpers ───────────────────────────────────────────────────────────────────


def _api_reachable(url: str) -> bool:
    try:
        return requests.get(f"{url}/health", timeout=2).status_code == 200
    except Exception:
        return False


def _setup_import_paths() -> None:
    for p in [
        str(Path(__file__).parent.parent),
        str(Path(__file__).parent.parent.parent / "src"),
        "/app/src",
        "/app",
    ]:
        if p not in sys.path:
            sys.path.insert(0, p)


def _llm_predict_direct(vehicles_raw: list) -> dict:
    """Call LLM directly — standalone mode (no FastAPI)."""
    _setup_import_paths()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    fine_tuned = os.environ.get("FINE_TUNED_MODEL_ID", "").strip()
    few_shot = os.environ.get("FEW_SHOT", "true").lower() == "true"

    if not api_key:
        return {"error": "OPENAI_API_KEY not set"}

    try:
        from models.llm_model import IntersectionLLM
    except ImportError:
        return {"error": "Cannot import IntersectionLLM"}

    llm = IntersectionLLM(
        model=model_name,
        api_key=api_key,
        few_shot=few_shot,
        fine_tuned_model_id=fine_tuned or None,
    )
    import time as _t

    t0 = _t.time()
    result = llm.predict({"vehicles": vehicles_raw})
    result["latency_ms"] = round((_t.time() - t0) * 1000)
    return result


def _load_visualization():
    """Load the original visualization module."""
    _setup_import_paths()
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "poc"))
        from visualization_orig import visualize_intersection, visualize_solution

        return visualize_intersection, visualize_solution
    except ImportError:
        return None, None


def _parse_vehicles_orig(vehicles_raw: list):
    """Parse vehicles using original engine for visualization."""
    _setup_import_paths()
    sys.path.insert(0, str(Path(__file__).parent.parent / "poc"))
    try:
        from conflict_detection_orig import (
            detect_conflicts,
            parse_intersection_layout,
            parse_vehicles,
        )

        LAYOUT_DATA_ORIG = {
            "intersection_layout": {
                "north": {"1": ["F", "H"], "2": ["E", "D", "C"]},
                "east": {"3": ["H", "B"], "4": ["G", "E", "F"]},
                "south": {"5": ["B", "D"], "6": ["A", "G", "H"]},
                "west": {"7": ["D", "F"], "8": ["B", "C", "A"]},
            }
        }
        layout = parse_intersection_layout(LAYOUT_DATA_ORIG)
        vehicles_obj = parse_vehicles({"vehicles_scenario": vehicles_raw}, layout)
        conflicts = detect_conflicts(vehicles_obj)
        return vehicles_obj, conflicts, layout
    except Exception:
        return None, [], None


# ── JSON helpers ──────────────────────────────────────────────────────────────


def to_json(vehicles: list) -> str:
    return json.dumps({"vehicles_scenario": vehicles}, indent=2)


def from_json(text: str) -> tuple[list | None, str | None]:
    try:
        data = json.loads(text)
        if "vehicles_scenario" not in data:
            return None, "JSON must have a 'vehicles_scenario' key."
        if not isinstance(data["vehicles_scenario"], list):
            return None, "'vehicles_scenario' must be a list."
        return data["vehicles_scenario"], None
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"


def sync_json() -> None:
    st.session_state.json_text = to_json(st.session_state.vehicles)


def on_json_edit() -> None:
    vehicles, err = from_json(st.session_state.json_text)
    if vehicles is not None:
        st.session_state.vehicles = vehicles
        st.session_state._json_error = None
    else:
        st.session_state._json_error = err


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🚦 Intersection Conflict Resolver",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ─────────────────────────────────────────────────────────────

if "vehicles" not in st.session_state:
    st.session_state.vehicles = list(DEFAULT_SCENARIO)
if "json_text" not in st.session_state:
    st.session_state.json_text = to_json(DEFAULT_SCENARIO)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Configuration")

    # Model status
    st.subheader("Model Status")
    fine_tuned_id = os.environ.get("FINE_TUNED_MODEL_ID", "").strip()
    model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY", "")

    if _api_reachable(API_URL):
        st.success("✅ FastAPI connected")
        try:
            info = requests.get(f"{API_URL}/health", timeout=3).json()
            st.caption(f"Model: `{info.get('model', 'unknown')}`")
        except Exception:
            pass
    else:
        st.info("🤖 Standalone mode")
        if api_key:
            st.success("✅ API key loaded")
        else:
            st.error("❌ OPENAI_API_KEY not set")

    if fine_tuned_id:
        st.success("🏆 Fine-tuned model active")
        st.caption("Accuracy: 78.3% | F1: 0.78")
    else:
        st.caption(f"Base model: `{model_name}`")

    st.divider()

    # Animation controls
    st.subheader("Animation")
    anim_steps = st.slider("Frames", 20, 80, 40, 5)
    anim_interval = st.slider("ms per frame", 40, 200, 80, 10)

    st.divider()

    # Layout reference
    st.subheader("Layout Reference")
    for direction, lanes in LAYOUT_DATA.items():
        lane_str = "  |  ".join(f"Lane {ln} → {'/'.join(dests)}" for ln, dests in lanes.items())
        st.caption(f"**{direction.capitalize()}:** {lane_str}")

    st.divider()
    st.markdown(
        "**Milestone 5 – Production Serving**\n\n"
        "Baseline: [Masri et al., 2025](https://arxiv.org/abs/2411.10869)"
    )

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🚦 LLM-Driven Intersection Conflict Resolver")
st.markdown(
    "Production serving of the **GPT-4o-mini (fine-tuned)** conflict detection agent. "
    "Analyzes vehicle scenarios at a 4-way 8-lane intersection and issues structured "
    "control decisions in real time."
)
st.divider()

# ── Scenario section ──────────────────────────────────────────────────────────

st.header("🚗 Define Intersection Scenario")

c1, c2, c3, c4, c5 = st.columns([3, 1, 1, 1, 1])
with c1:
    preset = st.selectbox(
        "Preset", ["— Custom —"] + list(PRESET_SCENARIOS), label_visibility="collapsed"
    )
with c2:
    if st.button("📂 Load", use_container_width=True) and preset != "— Custom —":
        st.session_state.vehicles = list(PRESET_SCENARIOS[preset])
        sync_json()
        st.rerun()
with c3:
    if st.button("🎲 Random", use_container_width=True):
        used, vlist = set(), []
        for _ in range(random.randint(2, 4)):
            lane = random.choice([ln for ln in range(1, 9) if ln not in used])
            used.add(lane)
            direction = LANE_DIRECTION[str(lane)]
            dests = LAYOUT_DATA[direction][str(lane)]
            vlist.append(
                {
                    "vehicle_id": f"V{random.randint(100, 999)}",
                    "lane": lane,
                    "speed": round(random.uniform(20, 80), 1),
                    "distance_to_intersection": round(random.uniform(50, 400), 1),
                    "direction": direction,
                    "destination": random.choice(dests),
                }
            )
        st.session_state.vehicles = vlist
        sync_json()
        st.rerun()
with c4:
    if st.button("➕ Add", use_container_width=True):
        st.session_state.vehicles.append(
            {
                "vehicle_id": f"V{random.randint(100, 999)}",
                "lane": 1,
                "speed": 50.0,
                "distance_to_intersection": 100.0,
                "direction": "north",
                "destination": "F",
            }
        )
        sync_json()
        st.rerun()
with c5:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.vehicles = []
        sync_json()
        st.rerun()

# ── Vehicle editor + live JSON ────────────────────────────────────────────────

editor_col, json_col = st.columns([1, 1], gap="large")

with editor_col:
    st.markdown("#### 🗂 Vehicle Editor")
    st.caption("Direction is automatically determined by lane number.")

    if st.session_state.vehicles:
        hdr = st.columns([2, 1, 2, 2, 2, 2, 1])
        for lbl, col in zip(
            ["Vehicle ID", "Lane", "Speed (km/h)", "Distance (m)", "Direction", "Destination", ""],
            hdr,
        ):
            col.markdown(f"**{lbl}**")

        for i, v in enumerate(st.session_state.vehicles):
            cols = st.columns([2, 1, 2, 2, 2, 2, 1])
            v["vehicle_id"] = cols[0].text_input(
                "id", str(v["vehicle_id"]), key=f"id_{i}", label_visibility="collapsed"
            )
            v["lane"] = cols[1].number_input(
                "ln", 1, 8, int(v["lane"]), key=f"ln_{i}", label_visibility="collapsed"
            )
            v["speed"] = cols[2].number_input(
                "sp",
                0.0,
                200.0,
                float(v["speed"]),
                step=1.0,
                key=f"sp_{i}",
                label_visibility="collapsed",
            )
            v["distance_to_intersection"] = cols[3].number_input(
                "di",
                0.0,
                2000.0,
                float(v["distance_to_intersection"]),
                step=10.0,
                key=f"di_{i}",
                label_visibility="collapsed",
            )
            auto_dir = LANE_DIRECTION[str(v["lane"])]
            v["direction"] = auto_dir
            cols[4].text_input(
                "dr", auto_dir, key=f"dr_{i}", disabled=True, label_visibility="collapsed"
            )
            dests = LAYOUT_DATA[auto_dir][str(v["lane"])]
            cur = v["destination"] if v["destination"] in dests else dests[0]
            v["destination"] = cols[5].selectbox(
                "dt",
                dests,
                index=dests.index(cur),
                key=f"dt_{i}",
                label_visibility="collapsed",
            )
            if cols[6].button("❌", key=f"del_{i}"):
                st.session_state.vehicles.pop(i)
                sync_json()
                st.rerun()

        sync_json()
    else:
        st.info("ℹ️ Add at least 2 vehicles to run the analysis.")

with json_col:
    st.markdown("#### 📋 Live JSON Scenario")
    st.caption("Updates in real time. Edit directly — valid changes sync back.")
    st.text_area(
        label="json",
        key="json_text",
        height=350,
        on_change=on_json_edit,
        label_visibility="collapsed",
    )
    if st.session_state.get("_json_error"):
        st.error(f"⚠ {st.session_state._json_error}")
    else:
        st.success("✅ JSON is valid and synced.")

# ── Analyze button ────────────────────────────────────────────────────────────

st.divider()

if len(st.session_state.vehicles) < 2:
    st.warning("Add at least **2 vehicles** to run the analysis.")
else:
    if st.button("🔍 Analyze Intersection", type="primary", use_container_width=True):
        vehicles_raw, json_err = from_json(st.session_state.json_text)
        if json_err:
            st.error(f"Cannot run: {json_err}")
            st.stop()

        use_api = _api_reachable(API_URL)
        payload = {"vehicles": vehicles_raw}

        with st.spinner(
            "Consulting the LLM" + (" via FastAPI…" if use_api else " directly (standalone mode)…")
        ):
            if use_api:
                try:
                    r = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
                    result = r.json() if r.status_code == 200 else None
                    if result is None:
                        st.error(f"API Error {r.status_code}: {r.text[:300]}")
                        st.stop()
                except Exception as exc:
                    st.error(f"Request failed: {exc}")
                    st.stop()
            else:
                result = _llm_predict_direct(vehicles_raw)
                if "error" in result:
                    st.error(f"❌ {result['error']}")
                    st.stop()

        # ── Results ───────────────────────────────────────────────────────────
        st.header("📊 Analysis Results")

        is_conflict = str(result.get("is_conflict", "no")).lower() == "yes"
        n_conflicts = int(result.get("number_of_conflicts", 0))

        if is_conflict:
            st.error(f"⚠️ **CONFLICT DETECTED** — {n_conflicts} conflict(s) at this intersection")
        else:
            st.success("✅ **No Conflicts Detected** — intersection is clear")

        m1, m2, m3 = st.columns(3)
        m1.metric("Conflicts", n_conflicts)
        m2.metric("Vehicles Analyzed", len(vehicles_raw))
        m3.metric("Latency", f"{result.get('latency_ms', 0):.0f} ms")

        decisions = result.get("decisions", [])
        if decisions:
            st.subheader("📋 Control Decisions")
            for d in decisions:
                st.warning(f"🔔 {d}")
        else:
            st.info("ℹ️ No control actions required.")

        priority = result.get("priority_order", {})
        waiting = result.get("waiting_times", {})
        if priority:
            st.subheader("🏆 Priority Order & Waiting Times")
            rows = sorted(
                [
                    {
                        "Vehicle": vid,
                        "Priority Rank": rank if rank is not None else "—",
                        "Wait (s)": waiting.get(vid, 0),
                    }
                    for vid, rank in priority.items()
                ],
                key=lambda row: (row["Priority Rank"] == "—", row["Priority Rank"]),
            )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        cv = result.get("conflict_vehicles", [])
        if cv and isinstance(cv[0], dict):
            st.subheader("🚨 Conflicting Vehicle Pairs")
            st.dataframe(pd.DataFrame(cv), use_container_width=True, hide_index=True)

        with st.expander("🔧 Full JSON Response"):
            st.json(result)

        # ── Animated visualizations ───────────────────────────────────────────
        vehicles_obj, rb_conflicts, layout = _parse_vehicles_orig(vehicles_raw)
        visualize_intersection, visualize_solution = _load_visualization()

        if vehicles_obj and visualize_intersection:
            st.divider()
            st.header("🎬 Animated Intersection Visualization")

            tab1, tab2 = st.tabs(
                [
                    "⚠️ Problem View (Conflicts Highlighted)",
                    "✅ Solution View (Wait Times Applied)",
                ]
            )
            with tab1:
                visualize_intersection(
                    layout=layout,
                    vehicles=vehicles_obj,
                    steps=anim_steps,
                    interval=anim_interval,
                )
            with tab2:
                # Use LLM conflicts (with priorities/waits) for solution view
                llm_conflicts = []
                for pair in cv:
                    if isinstance(pair, dict):
                        v1_id = pair.get("vehicle1_id", "")
                        v2_id = pair.get("vehicle2_id", "")
                        llm_conflicts.append(
                            {
                                "vehicle1_id": v1_id,
                                "vehicle2_id": v2_id,
                                "decision": next(
                                    (d for d in decisions if v1_id in d or v2_id in d), ""
                                ),
                                "priority_order": {
                                    v1_id: priority.get(v1_id, 1),
                                    v2_id: priority.get(v2_id, 2),
                                },
                                "waiting_times": {
                                    v1_id: waiting.get(v1_id, 0),
                                    v2_id: waiting.get(v2_id, 0),
                                },
                            }
                        )
                visualize_solution(
                    layout=layout,
                    vehicles=vehicles_obj,
                    conflicts=llm_conflicts if llm_conflicts else rb_conflicts,
                    steps=anim_steps,
                    interval=anim_interval,
                )

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "🚦 Milestone 5 – Production Serving  |  "
    "LLM-Driven Agents for Traffic Intersection Conflict Resolution  |  "
    "CSC5382 – AI for Digital Transformation"
)

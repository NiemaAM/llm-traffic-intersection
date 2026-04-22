"""
poc_app.py
----------
Milestone 2 – Proof of Concept Streamlit Application.

Uses the original conflict detection engine (Masri et al., 2025) and the
original animated Plotly visualization module. Features:
  - Default scenario pre-loaded with a clear N↔E conflict
  - Live bidirectional JSON panel
  - Animated Plotly visualizations (problem + solution views)
  - LLM engine option (GPT-4o-mini, optional)
  - Automated end-to-end test runner
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

# ── Auto-load .env from project root ─────────────────────────────────────────
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass  # dotenv not installed – rely on env vars being set externally

import pandas as pd
import streamlit as st

# ── Make project root importable ─────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "poc"))

# ── Import original engines ───────────────────────────────────────────────────
from conflict_detection_orig import (  # noqa: E402
    detect_conflicts,
    parse_intersection_layout,
    parse_vehicles,
)
from visualization_orig import (  # noqa: E402
    visualize_intersection,
    visualize_solution,
)

# ─── Intersection layout (original 8-lane layout) ────────────────────────────

LAYOUT_DATA = {
    "intersection_layout": {
        "north": {"1": ["F", "H"], "2": ["E", "D", "C"]},
        "east":  {"3": ["H", "B"], "4": ["G", "E", "F"]},
        "south": {"5": ["B", "D"], "6": ["A", "G", "H"]},
        "west":  {"7": ["D", "F"], "8": ["B", "C", "A"]},
    }
}
LAYOUT = parse_intersection_layout(LAYOUT_DATA)

LANE_DIRECTION = {
    "1": "north", "2": "north",
    "3": "east",  "4": "east",
    "5": "south", "6": "south",
    "7": "west",  "8": "west",
}

# ── Default scenario: classic N↔E conflict ────────────────────────────────────
DEFAULT_SCENARIO = [
    {"vehicle_id": "V001", "lane": 1, "speed": 50,
     "distance_to_intersection": 100, "direction": "north", "destination": "F"},
    {"vehicle_id": "V002", "lane": 3, "speed": 50,
     "distance_to_intersection": 100, "direction": "east",  "destination": "B"},
]

PRESET_SCENARIOS = {
    "⚠️ Classic Conflict (N↔E)": [
        {"vehicle_id": "V001", "lane": 1, "speed": 50,
         "distance_to_intersection": 100, "direction": "north", "destination": "F"},
        {"vehicle_id": "V002", "lane": 3, "speed": 50,
         "distance_to_intersection": 100, "direction": "east",  "destination": "B"},
    ],
    "🚨 Head-on (N↔S)": [
        {"vehicle_id": "V003", "lane": 1, "speed": 70,
         "distance_to_intersection": 110, "direction": "north", "destination": "H"},
        {"vehicle_id": "V004", "lane": 5, "speed": 65,
         "distance_to_intersection": 105, "direction": "south", "destination": "D"},
    ],
    "🟢 No Conflict (far arrival gap)": [
        {"vehicle_id": "V005", "lane": 1, "speed": 50,
         "distance_to_intersection": 50,  "direction": "north", "destination": "F"},
        {"vehicle_id": "V006", "lane": 3, "speed": 20,
         "distance_to_intersection": 500, "direction": "east",  "destination": "B"},
    ],
    "🟠 Multi-vehicle (4 cars)": [
        {"vehicle_id": "V007", "lane": 1, "speed": 60,
         "distance_to_intersection": 100, "direction": "north", "destination": "F"},
        {"vehicle_id": "V008", "lane": 5, "speed": 55,
         "distance_to_intersection": 110, "direction": "south", "destination": "D"},
        {"vehicle_id": "V009", "lane": 3, "speed": 65,
         "distance_to_intersection": 95,  "direction": "east",  "destination": "B"},
        {"vehicle_id": "V010", "lane": 7, "speed": 50,
         "distance_to_intersection": 200, "direction": "west",  "destination": "F"},
    ],
}


# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🚦 Intersection Conflict PoC",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── JSON helpers ─────────────────────────────────────────────────────────────

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


def sync_json():
    st.session_state.json_text = to_json(st.session_state.vehicles)


def on_json_edit():
    vehicles, err = from_json(st.session_state.json_text)
    if vehicles is not None:
        st.session_state.vehicles = vehicles
        st.session_state._json_error = None
    else:
        st.session_state._json_error = err


# ─── Session state ────────────────────────────────────────────────────────────

if "vehicles" not in st.session_state:
    st.session_state.vehicles = list(DEFAULT_SCENARIO)
if "json_text" not in st.session_state:
    st.session_state.json_text = to_json(DEFAULT_SCENARIO)
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Configuration")

    st.subheader("Engine")
    engine = st.radio(
        "Inference engine",
        ["🔧 Rule-Based (Baseline)", "🤖 LLM (GPT-4o-mini)"],
    )
    use_llm = "LLM" in engine

    if use_llm:
        # ── Read from st.secrets (Streamlit Cloud) or .env / environment ──
        def _get_secret(key: str, default: str = "") -> str:
            """Check st.secrets first (Streamlit Cloud), then environment."""
            try:
                return st.secrets[key]
            except (KeyError, FileNotFoundError):
                return os.environ.get(key, default)

        api_key    = _get_secret("OPENAI_API_KEY")
        model_name = _get_secret("MODEL_NAME", "gpt-4o-mini")
        fine_tuned = _get_secret("FINE_TUNED_MODEL_ID")
        few_shot   = _get_secret("FEW_SHOT", "true").lower() == "true"

        # Inject into environment so downstream code can use os.environ
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        # Show status
        if api_key:
            st.success("API key loaded")
            st.caption(f"Model: `{model_name}`")
            if fine_tuned:
                st.caption(f"Fine-tuned: `{fine_tuned}`")
        else:
            st.error("OPENAI_API_KEY not found")
            st.caption("Add it to your Streamlit secrets or .env file.")

        few_shot = st.toggle("Few-shot examples", value=few_shot)
    else:
        api_key, model_name, few_shot, fine_tuned = "", "gpt-4o-mini", True, ""

    st.divider()
    st.subheader("Animation")
    anim_steps    = st.slider("Frames",       20, 80,  40, 5)
    anim_interval = st.slider("ms per frame", 40, 200, 80, 10)

    st.divider()
    st.subheader("Layout Reference")
    for direction, lanes in LAYOUT_DATA["intersection_layout"].items():
        lane_str = "  |  ".join(
            f"Lane {l} → {'/'.join(d)}" for l, d in lanes.items()
        )
        st.caption(f"**{direction.capitalize()}:** {lane_str}")

    st.divider()
    st.markdown(
        "**Milestone 2 – Proof of Concept**\n\n"
        "Baseline: [Masri et al., 2025](https://arxiv.org/abs/2411.10869)"
    )


# ─── Header ───────────────────────────────────────────────────────────────────

st.title("🚦 Intersection Conflict Detector – Proof of Concept")
st.markdown(
    "Detects vehicle conflicts at a **4-way, 8-lane intersection** using the "
    "rule-based baseline (Masri et al., 2025) or GPT-4o-mini. "
    "Animated Plotly visualizations show both the conflict and its resolution."
)
st.divider()


# ─── Scenario section ─────────────────────────────────────────────────────────

st.header("🚗 Define Intersection Scenario")

# Preset / action buttons
c1, c2, c3, c4, c5 = st.columns([3, 1, 1, 1, 1])
with c1:
    preset = st.selectbox("Preset", ["— Custom —"] + list(PRESET_SCENARIOS),
                          label_visibility="collapsed")
with c2:
    if st.button("📂 Load", use_container_width=True) and preset != "— Custom —":
        st.session_state.vehicles = list(PRESET_SCENARIOS[preset])
        sync_json()
        st.rerun()
with c3:
    if st.button("🎲 Random", use_container_width=True):
        used, vlist = set(), []
        for _ in range(random.randint(2, 4)):
            lane = random.choice([l for l in range(1, 9) if l not in used])
            used.add(lane)
            direction = LANE_DIRECTION[str(lane)]
            dests = LAYOUT_DATA["intersection_layout"][direction][str(lane)]
            vlist.append({
                "vehicle_id": f"V{random.randint(100,999)}",
                "lane": lane,
                "speed": round(random.uniform(20, 80), 1),
                "distance_to_intersection": round(random.uniform(50, 400), 1),
                "direction": direction,
                "destination": random.choice(dests),
            })
        st.session_state.vehicles = vlist
        sync_json()
        st.rerun()
with c4:
    if st.button("➕ Add", use_container_width=True):
        st.session_state.vehicles.append({
            "vehicle_id": f"V{random.randint(100,999)}",
            "lane": 1, "speed": 50.0,
            "distance_to_intersection": 100.0,
            "direction": "north", "destination": "F",
        })
        sync_json()
        st.rerun()
with c5:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.vehicles = []
        sync_json()
        st.rerun()


# ── Two-column layout: vehicle editor | JSON ──────────────────────────────────

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
                "id", str(v["vehicle_id"]), key=f"id_{i}", label_visibility="collapsed")
            v["lane"] = cols[1].number_input(
                "ln", 1, 8, int(v["lane"]), key=f"ln_{i}", label_visibility="collapsed")
            v["speed"] = cols[2].number_input(
                "sp", 0.0, 200.0, float(v["speed"]), step=1.0, key=f"sp_{i}",
                label_visibility="collapsed")
            v["distance_to_intersection"] = cols[3].number_input(
                "di", 0.0, 2000.0, float(v["distance_to_intersection"]),
                step=10.0, key=f"di_{i}", label_visibility="collapsed")

            # Direction auto-derived from lane
            auto_dir = LANE_DIRECTION[str(v["lane"])]
            v["direction"] = auto_dir
            cols[4].text_input("dr", auto_dir, key=f"dr_{i}",
                               disabled=True, label_visibility="collapsed")

            dests = LAYOUT_DATA["intersection_layout"][auto_dir][str(v["lane"])]
            cur   = v["destination"] if v["destination"] in dests else dests[0]
            v["destination"] = cols[5].selectbox(
                "dt", dests, index=dests.index(cur), key=f"dt_{i}",
                label_visibility="collapsed")

            if cols[6].button("❌", key=f"del_{i}"):
                st.session_state.vehicles.pop(i)
                sync_json()
                st.rerun()

        sync_json()  # keep JSON in sync with any edits
    else:
        st.info("ℹ️ Add at least 2 vehicles to run the analysis.")

with json_col:
    st.markdown("#### 📋 Live JSON Scenario")
    st.caption(
        "This panel updates in real time as you edit the vehicle list. "
        "You can also edit it directly — valid changes sync back."
    )
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


# ─── Analyze button ───────────────────────────────────────────────────────────

st.divider()
if len(st.session_state.vehicles) < 2:
    st.warning("Add at least **2 vehicles** to run the analysis.")
else:
    if st.button("🔍 Analyze Intersection", type="primary", use_container_width=True):

        vehicles_raw, err = from_json(st.session_state.json_text)
        if err:
            st.error(f"Cannot run: {err}")
            st.stop()

        scenario = {"vehicles_scenario": vehicles_raw}

        with st.spinner("Analyzing…"):
            import warnings
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    vehicles_obj = parse_vehicles(scenario, LAYOUT)
                    conflicts    = detect_conflicts(vehicles_obj)
                except Exception as exc:
                    st.error(f"Engine error: {exc}")
                    st.stop()

        for w in caught:
            st.warning(str(w.message))

        # ── If LLM mode, also call LLM ────────────────────────────────────
        llm_result = None
        if use_llm:
            if not api_key:
                st.error("❌ OPENAI_API_KEY not set. Add it to your .env file and restart.")
                st.stop()
            os.environ["OPENAI_API_KEY"] = api_key
            try:
                from src.models.llm_model import IntersectionLLM
                llm = IntersectionLLM(model=model_name, api_key=api_key,
                                      few_shot=few_shot,
                                      fine_tuned_model_id=fine_tuned or None)
                llm_result = llm.predict({"vehicles": vehicles_raw})
            except Exception as exc:
                st.error(f"❌ LLM error: {exc}")
                st.stop()

        # ── Results ───────────────────────────────────────────────────────
        st.header("📊 Analysis Results")

        is_conflict = len(conflicts) > 0
        n_conflicts = len(conflicts)

        if is_conflict:
            st.error(f"⚠️ **CONFLICT DETECTED** — {n_conflicts} conflict(s) at this intersection")
        else:
            st.success("✅ **No Conflicts Detected** — intersection is clear")

        m1, m2, m3 = st.columns(3)
        m1.metric("Conflicts", n_conflicts)
        m2.metric("Vehicles",  len(vehicles_obj))
        m3.metric("Status",    "⚠️ Conflict" if is_conflict else "✅ Clear")

        # Decisions
        if conflicts:
            st.subheader("📋 Control Decisions")
            for c in conflicts:
                st.warning(f"🔔 {c['decision']}")

        # Priority + waiting table
        all_priorities = {}
        all_waiting    = {}
        for c in conflicts:
            all_priorities.update(c["priority_order"])
            all_waiting.update(c["waiting_times"])

        if all_priorities:
            st.subheader("🏆 Priority Order & Waiting Times")
            rows = sorted(
                [{"Vehicle": vid, "Priority Rank": rank, "Wait (s)": all_waiting.get(vid, 0)}
                 for vid, rank in all_priorities.items()],
                key=lambda r: r["Priority Rank"],
            )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # LLM comparison panel
        if llm_result:
            with st.expander("🤖 LLM Engine Response (GPT-4o-mini)"):
                st.json(llm_result)

        # ── Animated visualizations ────────────────────────────────────────
        st.divider()
        st.header("🎬 Animated Intersection Visualization")

        tab1, tab2 = st.tabs([
            "⚠️ Problem View (Conflicts Highlighted)",
            "✅ Solution View (Wait Times Applied)",
        ])
        with tab1:
            visualize_intersection(
                layout=LAYOUT,
                vehicles=vehicles_obj,
                steps=anim_steps,
                interval=anim_interval,
            )
        with tab2:
            visualize_solution(
                layout=LAYOUT,
                vehicles=vehicles_obj,
                conflicts=conflicts,
                steps=anim_steps,
                interval=anim_interval,
            )

        # Raw JSON
        with st.expander("🔧 Full Rule-Based JSON Result"):
            st.json({
                "is_conflict": "yes" if is_conflict else "no",
                "number_of_conflicts": n_conflicts,
                "conflict_vehicles": [
                    {"vehicle1_id": c["vehicle1_id"], "vehicle2_id": c["vehicle2_id"]}
                    for c in conflicts
                ],
                "decisions": [c["decision"] for c in conflicts],
                "priority_order": all_priorities,
                "waiting_times": all_waiting,
            })


# ─── End-to-end automated tests ───────────────────────────────────────────────

st.divider()
with st.expander("🧪 End-to-End Scenario Tests (Automated)"):

    TEST_CASES = [
        {
            "name": "Classic N↔E conflict (same speed & distance)",
            "vehicles": [
                {"vehicle_id": "T001", "lane": 1, "speed": 50,
                 "distance_to_intersection": 100, "direction": "north", "destination": "F"},
                {"vehicle_id": "T002", "lane": 3, "speed": 50,
                 "distance_to_intersection": 100, "direction": "east",  "destination": "B"},
            ],
            "expected_conflict": True,
        },
        {
            "name": "No conflict: arrival times far apart",
            "vehicles": [
                {"vehicle_id": "T003", "lane": 1, "speed": 50,
                 "distance_to_intersection": 50,  "direction": "north", "destination": "F"},
                {"vehicle_id": "T004", "lane": 3, "speed": 20,
                 "distance_to_intersection": 500, "direction": "east",  "destination": "B"},
            ],
            "expected_conflict": False,
        },
        {
            "name": "No conflict: same direction (N lane1 & N lane2)",
            "vehicles": [
                {"vehicle_id": "T005", "lane": 1, "speed": 50,
                 "distance_to_intersection": 100, "direction": "north", "destination": "F"},
                {"vehicle_id": "T006", "lane": 2, "speed": 50,
                 "distance_to_intersection": 100, "direction": "north", "destination": "E"},
            ],
            "expected_conflict": False,
        },
        {
            "name": "Conflict: S↔W close arrival",
            "vehicles": [
                {"vehicle_id": "T007", "lane": 5, "speed": 60,
                 "distance_to_intersection": 100, "direction": "south", "destination": "B"},
                {"vehicle_id": "T008", "lane": 7, "speed": 60,
                 "distance_to_intersection": 100, "direction": "west",  "destination": "D"},
            ],
            "expected_conflict": True,
        },
        {
            "name": "Three vehicles — at least one conflict expected",
            "vehicles": [
                {"vehicle_id": "T009", "lane": 1, "speed": 55,
                 "distance_to_intersection": 100, "direction": "north", "destination": "F"},
                {"vehicle_id": "T010", "lane": 3, "speed": 55,
                 "distance_to_intersection": 100, "direction": "east",  "destination": "B"},
                {"vehicle_id": "T011", "lane": 5, "speed": 55,
                 "distance_to_intersection": 100, "direction": "south", "destination": "D"},
            ],
            "expected_conflict": True,
        },
    ]

    # ── Engine selector for tests ─────────────────────────────────────────────
    test_engine = st.radio(
        "Run tests with:",
        ["🔧 Rule-Based", "🤖 LLM (GPT-4o-mini)", "🔀 Both (compare)"],
        horizontal=True,
        key="test_engine_radio",
    )

    if test_engine == "🤖 LLM (GPT-4o-mini)" or test_engine == "🔀 Both (compare)":
        _test_api_key = os.environ.get("OPENAI_API_KEY", "")
        _test_model   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
        if _test_api_key:
            st.caption(f"LLM: `{_test_model}` — key loaded from .env ✅")
        else:
            st.warning("⚠️ OPENAI_API_KEY not set — LLM tests will be skipped.")

    if test_engine == "🔧 Rule-Based":
        st.caption("Runs offline — no API key needed.")
    elif test_engine == "🔀 Both (compare)":
        st.caption("Runs both engines and shows whether they agree on each scenario.")

    def _run_rule_based(vehicles):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vobjs = parse_vehicles({"vehicles_scenario": vehicles}, LAYOUT)
            cfls  = detect_conflicts(vobjs)
        return len(cfls) > 0, cfls

    def _run_llm(vehicles):
        from src.models.llm_model import IntersectionLLM
        _key   = os.environ.get("OPENAI_API_KEY", "")
        _model = os.environ.get("MODEL_NAME", "gpt-4o-mini")
        _few   = os.environ.get("FEW_SHOT", "true").lower() == "true"
        if not _key:
            return None, "No API key"
        llm    = IntersectionLLM(model=_model, api_key=_key, few_shot=_few)
        result = llm.predict({"vehicles": vehicles})
        got    = str(result.get("is_conflict", "no")).lower() == "yes"
        return got, result

    if st.button("▶️ Run All Tests"):
        passed_rb, passed_llm, rows = 0, 0, []
        total = len(TEST_CASES)

        for tc in TEST_CASES:
            row = {
                "Test":     tc["name"],
                "Expected": "Yes" if tc["expected_conflict"] else "No",
            }

            # ── Rule-based ────────────────────────────────────────────────
            if test_engine in ("🔧 Rule-Based", "🔀 Both (compare)"):
                try:
                    got_rb, _ = _run_rule_based(tc["vehicles"])
                    ok_rb = got_rb == tc["expected_conflict"]
                    if ok_rb: passed_rb += 1
                    row["Rule-Based"]    = "Yes" if got_rb else "No"
                    row["RB Result"]     = "✅ Pass" if ok_rb else "❌ Fail"
                except Exception as exc:
                    row["Rule-Based"] = "ERROR"
                    row["RB Result"]  = f"❌ {exc}"

            # ── LLM ───────────────────────────────────────────────────────
            if test_engine in ("🤖 LLM (GPT-4o-mini)", "🔀 Both (compare)"):
                if not os.environ.get("OPENAI_API_KEY"):
                    row["LLM"]        = "SKIPPED"
                    row["LLM Result"] = "⚠️ No key"
                else:
                    try:
                        got_llm, _ = _run_llm(tc["vehicles"])
                        if got_llm is None:
                            row["LLM"] = "SKIPPED"
                            row["LLM Result"] = "⚠️ No key"
                        else:
                            ok_llm = got_llm == tc["expected_conflict"]
                            if ok_llm: passed_llm += 1
                            row["LLM"]        = "Yes" if got_llm else "No"
                            row["LLM Result"] = "✅ Pass" if ok_llm else "❌ Fail"
                    except Exception as exc:
                        row["LLM"]        = "ERROR"
                        row["LLM Result"] = f"❌ {exc}"

            # ── Agreement (both mode) ─────────────────────────────────────
            if test_engine == "🔀 Both (compare)":
                rb  = row.get("Rule-Based", "?")
                llm = row.get("LLM", "?")
                if rb not in ("ERROR", "SKIPPED") and llm not in ("ERROR", "SKIPPED"):
                    row["Agree"] = "✅ Yes" if rb == llm else "⚠️ No"
                else:
                    row["Agree"] = "—"

            rows.append(row)

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Summary
        if test_engine == "🔧 Rule-Based":
            if passed_rb == total:
                st.success(f"🎉 All {total}/{total} rule-based tests passed!")
            else:
                st.warning(f"⚠️ {passed_rb}/{total} rule-based tests passed.")

        elif test_engine == "🤖 LLM (GPT-4o-mini)":
            if passed_llm == total:
                st.success(f"🎉 All {total}/{total} LLM tests passed!")
            else:
                st.warning(f"⚠️ {passed_llm}/{total} LLM tests passed.")

        else:  # Both
            agree = sum(
                1 for r in rows
                if r.get("Rule-Based") not in ("ERROR","SKIPPED")
                and r.get("LLM") not in ("ERROR","SKIPPED")
                and r.get("Rule-Based") == r.get("LLM")
            )
            st.info(
                f"Rule-Based: **{passed_rb}/{total}** passed  |  "
                f"LLM: **{passed_llm}/{total}** passed  |  "
                f"Agreement: **{agree}/{total}** scenarios"
            )
            if agree == total:
                st.success("✅ Both engines agree on all scenarios!")
            else:
                st.warning(f"⚠️ Engines disagree on {total - agree} scenario(s) — review the table above.")


# ─── Footer ───────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "🚦 Milestone 2 – Proof of Concept  |  "
    "LLM-Driven Agents for Traffic Intersection Conflict Resolution  |  "
    "CSC5382 – AI for Digital Transformation"
)

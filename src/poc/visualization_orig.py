"""Visualization module for Intersection Conflict Detection.

Provides two animated Plotly figures rendered via Streamlit:
  - visualize_intersection : animates the problem (vehicles approach, conflict
    pairs highlighted in red).
  - visualize_solution     : animates the resolution (waiting times from
    detect_conflicts applied so conflicting vehicles do not overlap).

Data structures consumed (from conflict_detection.py)
------------------------------------------------------
Vehicle attributes
    vehicle_id, lane (str "1"–"8"), speed (km/h),
    distance_to_intersection (m), direction (north/east/south/west),
    destination (A–H), time_to_intersection (s),
    movement_type (straight / left / right / unknown)

detect_conflicts() return value
    list[dict] each dict has:
        vehicle1_id, vehicle2_id  – vehicle IDs
        decision                  – human-readable string
        place                     – 'intersection'
        priority_order            – {vehicle_id: 1 (higher) or 2 (lower)}
        waiting_times             – {vehicle_id: seconds (int)}

Intersection layout  (intersection_layout.json)
    north: lane 1 → [F,H],      lane 2 → [E,D,C]
    east:  lane 3 → [H,B],      lane 4 → [G,E,F]
    south: lane 5 → [B,D],      lane 6 → [A,G,H]
    west:  lane 7 → [D,F],      lane 8 → [B,C,A]

    Lanes 1,3,5,7 : index 0=right, 1=straight, 2=left
    Lanes 2,4,6,8 : left-turn dedicated

Coordinate system
    Centre of intersection = (0, 0).
    Each road arm has two lanes of width LANE_W each (total road = 2*LANE_W).
    BOX_HALF is the half-width of the central box (= LANE_W).
    Road arms extend ROAD_LEN units outward from the box.
"""

from __future__ import annotations

import math
from typing import Any

import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Geometry constants
# ---------------------------------------------------------------------------

LANE_W   = 1.5          # width of one lane (world units)
BOX_HALF = LANE_W       # half-size of central intersection box
ROAD_LEN = 12.0         # length of each arm outside the box

# ---------------------------------------------------------------------------
# Arm geometry
# Each direction entry defines:
#   ox, oy   – origin = centre of stop-line (box edge)
#   indx/y   – unit vector pointing INTO the intersection
#   px, py   – perpendicular unit vector (left of travel direction)
#
# Lane offsets (along perp from road centre-line):
#   Lanes 1,3,5,7 (right lane of the arm) → +LANE_W/2
#   Lanes 2,4,6,8 (left  lane of the arm) → -LANE_W/2
# ---------------------------------------------------------------------------

_ARMS: dict[str, dict] = {
    "north": dict(ox=0.0,      oy= BOX_HALF,  indx= 0.0, indy=-1.0, px=-1.0, py= 0.0),
    "south": dict(ox=0.0,      oy=-BOX_HALF,  indx= 0.0, indy= 1.0, px= 1.0, py= 0.0),
    "east":  dict(ox= BOX_HALF, oy=0.0,        indx=-1.0, indy= 0.0, px= 0.0, py=-1.0),
    "west":  dict(ox=-BOX_HALF, oy=0.0,        indx= 1.0, indy= 0.0, px= 0.0, py= 1.0),
}

_LANE_PERP_OFFSET: dict[str, float] = {
    "1": +LANE_W / 2, "2": -LANE_W / 2,
    "3": +LANE_W / 2, "4": -LANE_W / 2,
    "5": +LANE_W / 2, "6": -LANE_W / 2,
    "7": +LANE_W / 2, "8": -LANE_W / 2,
}

# Destination exit positions mapped from layout:
#   north exits (A,H) → y+
#   east  exits (B,G) → x+
#   south exits (C,D) → y-
#   west  exits (E,F) → x-
_EXIT_HALF = ROAD_LEN * 0.65
_DEST_POS: dict[str, tuple[float, float]] = {
    "A": (-LANE_W / 2,  BOX_HALF + _EXIT_HALF),  # north, left lane
    "H": ( LANE_W / 2,  BOX_HALF + _EXIT_HALF),  # north, right lane
    "B": ( BOX_HALF + _EXIT_HALF,  LANE_W / 2),  # east,  right lane
    "G": ( BOX_HALF + _EXIT_HALF, -LANE_W / 2),  # east,  left lane
    "C": ( LANE_W / 2, -BOX_HALF - _EXIT_HALF),  # south, right lane
    "D": (-LANE_W / 2, -BOX_HALF - _EXIT_HALF),  # south, left lane
    "E": (-BOX_HALF - _EXIT_HALF,  LANE_W / 2),  # west,  right lane
    "F": (-BOX_HALF - _EXIT_HALF, -LANE_W / 2),  # west,  left lane
}

# Plotly arrow angle (degrees, 0=up/north, clockwise) for approach direction
_APPROACH_ANGLE: dict[str, float] = {
    "north": 180.0,  # arrow points south (toward box)
    "south":   0.0,  # arrow points north (toward box)
    "east":  270.0,  # vehicle travels west  → arrow points left  (270)
    "west":   90.0,  # vehicle travels east  → arrow points right (90)
}

_COLORS = [
    "#00d4ff", "#ff6b35", "#7bc67e", "#ff4757",
    "#ffd32a", "#a29bfe", "#fd79a8", "#55efc4",
]


# ---------------------------------------------------------------------------
# Position helpers
# ---------------------------------------------------------------------------

def _stop_line_pos(vehicle: Any) -> tuple[float, float]:
    arm = _ARMS[str(vehicle.direction).lower()]
    off = _LANE_PERP_OFFSET[str(vehicle.lane)]
    return (arm["ox"] + arm["px"] * off,
            arm["oy"] + arm["py"] * off)


def _start_pos(vehicle: Any) -> tuple[float, float]:
    arm = _ARMS[str(vehicle.direction).lower()]
    off = _LANE_PERP_OFFSET[str(vehicle.lane)]
    return (arm["ox"] + arm["px"] * off - arm["indx"] * ROAD_LEN,
            arm["oy"] + arm["py"] * off - arm["indy"] * ROAD_LEN)


def _exit_pos(vehicle: Any) -> tuple[float, float]:
    dest = str(vehicle.destination).upper()
    return _DEST_POS.get(dest, (0.0, BOX_HALF + _EXIT_HALF))


def _lerp(a: tuple[float, float], b: tuple[float, float], t: float) -> tuple[float, float]:
    t = max(0.0, min(1.0, t))
    return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t


def _crossing_angle(vehicle: Any) -> float:
    """Arrow angle while crossing: point from stop-line toward exit."""
    sl = _stop_line_pos(vehicle)
    ep = _exit_pos(vehicle)
    dx, dy = ep[0] - sl[0], ep[1] - sl[1]
    # Plotly angle: 0=up (y+), clockwise → atan2(x, y)
    return math.degrees(math.atan2(dx, dy)) % 360


def _vehicle_pos_at_t(
    vehicle: Any,
    t: float,
    approach_end: float,
    wait_end: float,
) -> tuple[tuple[float, float], str]:
    """Return ((x,y), phase) at normalised time t ∈ [0,1].

    Timeline fractions are pre-computed by _build_figure from real physics:
      [0,            approach_end)   approach arm → stop-line  (speed-accurate)
      [approach_end, wait_end)       stationary at stop-line   (wait time)
      [wait_end,     1.0]            cross through centre → exit

    Parameters
    ----------
    approach_end : normalised t at which vehicle reaches the stop-line
                   derived from vehicle.time_to_intersection / total_sim_time
    wait_end     : normalised t at which vehicle starts crossing
                   approach_end + wait_seconds / total_sim_time
    """
    cross_dur = max(1.0 - wait_end, 0.02)

    start = _start_pos(vehicle)
    stop  = _stop_line_pos(vehicle)
    mid   = (0.0, 0.0)
    dest  = _exit_pos(vehicle)

    if t < approach_end:
        sub = t / max(approach_end, 1e-9)
        return _lerp(start, stop, sub), "approach"

    elif t < wait_end:
        return stop, "wait"

    else:
        sub = (t - wait_end) / cross_dur
        if sub < 0.5:
            return _lerp(stop, mid, sub * 2), "cross"
        else:
            return _lerp(mid, dest, (sub - 0.5) * 2), "cross"


# ---------------------------------------------------------------------------
# Road background traces
# ---------------------------------------------------------------------------

def _road_bg_traces() -> list[go.BaseTraceType]:
    traces: list[go.BaseTraceType] = []
    road_col = "#3d3d3d"
    edge_col = "#666666"

    # Central box
    b = BOX_HALF
    traces.append(go.Scatter(
        x=[-b, b, b, -b, -b], y=[-b, -b, b, b, -b],
        fill="toself", fillcolor=road_col,
        line=dict(color=edge_col, width=1),
        mode="lines", showlegend=False, hoverinfo="skip",
    ))

    for direction, arm in _ARMS.items():
        ox, oy     = arm["ox"], arm["oy"]
        indx, indy = arm["indx"], arm["indy"]
        px, py     = arm["px"], arm["py"]
        hw = LANE_W  # half-width of the whole road

        far_x = ox - indx * ROAD_LEN
        far_y = oy - indy * ROAD_LEN

        # Road arm rectangle
        corners = [
            (ox    + px * hw, oy    + py * hw),
            (ox    - px * hw, oy    - py * hw),
            (far_x - px * hw, far_y - py * hw),
            (far_x + px * hw, far_y + py * hw),
        ]
        xs = [c[0] for c in corners] + [corners[0][0]]
        ys = [c[1] for c in corners] + [corners[0][1]]
        traces.append(go.Scatter(
            x=xs, y=ys, fill="toself", fillcolor=road_col,
            line=dict(color=edge_col, width=1),
            mode="lines", showlegend=False, hoverinfo="skip",
        ))

        # Centre dashed lane divider
        traces.append(go.Scatter(
            x=[ox, far_x], y=[oy, far_y],
            mode="lines",
            line=dict(color="white", width=1, dash="dash"),
            showlegend=False, hoverinfo="skip",
        ))

        # Stop-line (thick white bar)
        traces.append(go.Scatter(
            x=[ox + px * hw, ox - px * hw],
            y=[oy + py * hw, oy - py * hw],
            mode="lines",
            line=dict(color="white", width=3),
            showlegend=False, hoverinfo="skip",
        ))

    # Destination labels
    for dest, (dx, dy) in _DEST_POS.items():
        traces.append(go.Scatter(
            x=[dx], y=[dy],
            mode="text", text=[f"<b>{dest}</b>"],
            textfont=dict(color="#ffd700", size=14),
            showlegend=False, hoverinfo="skip",
        ))

    # Compass labels
    label_offset = BOX_HALF + ROAD_LEN + 1.2
    for lbl, lx, ly in [("N", 0, label_offset), ("S", 0, -label_offset),
                         ("E", label_offset, 0), ("W", -label_offset, 0)]:
        traces.append(go.Scatter(
            x=[lx], y=[ly], mode="text", text=[lbl],
            textfont=dict(color="#aaaaaa", size=15),
            showlegend=False, hoverinfo="skip",
        ))

    # -----------------------------------------------------------------------
    # Lane number labels + direction arrows
    # Lanes are placed at mid-arm (ROAD_LEN * 0.5 from stop-line outward).
    # Each lane gets:
    #   - a circle badge with the lane number
    #   - an arrow marker showing the direction vehicles travel (toward the box)
    #
    # Lane map:
    #   north → lane 1 (right/east side of arm), lane 2 (left/west side)
    #   east  → lane 3 (right/south side),       lane 4 (left/north side)
    #   south → lane 5 (right/west side),         lane 6 (left/east side)
    #   west  → lane 7 (right/north side),        lane 8 (left/south side)
    # -----------------------------------------------------------------------
    # exit_lane=True  → traffic flows OUT of the intersection (arrow flipped 180°)
    # exit_lane=False → traffic flows INTO the intersection
    _LANE_INFO = [
        # (lane_id, direction, perp_offset, exit_lane)
        ("1", "north", +LANE_W / 2, False),
        ("2", "north", -LANE_W / 2, True),   # exit lane – away from box
        ("3", "east",  +LANE_W / 2, False),
        ("4", "east",  -LANE_W / 2, True),   # exit lane – away from box
        ("5", "south", +LANE_W / 2, False),
        ("6", "south", -LANE_W / 2, True),   # exit lane – away from box
        ("7", "west",  +LANE_W / 2, False),
        ("8", "west",  -LANE_W / 2, True),   # exit lane – away from box
    ]

    # Place label + arrow at 50% along the arm from the stop-line outward
    LABEL_DIST = ROAD_LEN * 0.50

    for lane_id, direction, perp_off, exit_lane in _LANE_INFO:
        arm = _ARMS[direction]
        ox, oy     = arm["ox"], arm["oy"]
        indx, indy = arm["indx"], arm["indy"]
        px, py     = arm["px"], arm["py"]

        # Mid-arm position (outward from stop-line)
        mx = ox + px * perp_off - indx * LABEL_DIST
        my = oy + py * perp_off - indy * LABEL_DIST

        # ---- lane number badge ----
        traces.append(go.Scatter(
            x=[mx], y=[my],
            mode="markers+text",
            marker=dict(
                size=22,
                color="rgba(20,20,60,0.82)",
                symbol="circle",
                line=dict(color="#88aaff", width=1.5),
            ),
            text=[f"<b>{lane_id}</b>"],
            textposition="middle center",
            textfont=dict(color="#88aaff", size=11, family="monospace"),
            showlegend=False,
            hovertemplate=f"<b>Lane {lane_id}</b><br>Direction: {direction}<br>{'Exit' if exit_lane else 'Entry'} lane<extra></extra>",
            name="",
        ))

        # ---- direction arrow ----
        # Entry lanes: arrow points toward the box (approach angle)
        # Exit  lanes: arrow points away from box  (approach angle + 180°)
        arrow_dist = ROAD_LEN * 0.28
        ax = ox + px * perp_off - indx * arrow_dist
        ay = oy + py * perp_off - indy * arrow_dist
        arrow_angle = (_APPROACH_ANGLE[direction] + (180.0 if exit_lane else 0.0)) % 360

        traces.append(go.Scatter(
            x=[ax], y=[ay],
            mode="markers",
            marker=dict(
                size=13,
                color="#88aaff",
                symbol="arrow",
                angle=arrow_angle,
                opacity=0.75,
            ),
            showlegend=False,
            hoverinfo="skip",
            name="",
        ))

    return traces


# ---------------------------------------------------------------------------
# Build animated figure
# ---------------------------------------------------------------------------

def _build_figure(
    vehicles: list[Any],
    steps: int,
    interval: int,
    title: str,
    waiting_times: dict[str, float],
    highlight_conflicts: list[dict] | None = None,
) -> go.Figure:
    road_bg = _road_bg_traces()

    # ── Speed-aware timeline ────────────────────────────────────────────────
    # Simulate real physics:
    #   - Each vehicle has time_to_intersection (s) = distance / speed
    #   - Crossing the box takes a fixed CROSS_TIME seconds
    #   - Waiting adds wait_seconds on top of that
    #
    # Total simulated duration = max over all vehicles of:
    #     TTA + wait_seconds + CROSS_TIME
    # Each vehicle's normalised timeline fractions are then:
    #     approach_end = TTA / total_sim_time
    #     wait_end     = (TTA + wait_seconds) / total_sim_time
    # (clamped so crossing always occupies at least MIN_CROSS_FRAC of the timeline)

    CROSS_TIME     = 4.0   # seconds a vehicle takes to traverse the box
    MIN_CROSS_FRAC = 0.10  # crossing gets at least 10% of the animation

    # Gather per-vehicle TTA (guard against inf for stopped vehicles)
    def _tta(v: Any) -> float:
        tta = getattr(v, "time_to_intersection", None)
        if tta is None or tta == float("inf"):
            speed_ms = (v.speed * 1000) / 3600
            return v.distance_to_intersection / speed_ms if speed_ms > 0 else 9999.0
        return float(tta)

    veh_tta   = {str(v.vehicle_id): _tta(v) for v in vehicles}
    veh_wait  = {str(v.vehicle_id): float(waiting_times.get(str(v.vehicle_id), 0.0))
                 for v in vehicles}

    total_sim_time = max(
        veh_tta[str(v.vehicle_id)] + veh_wait[str(v.vehicle_id)] + CROSS_TIME
        for v in vehicles
    ) if vehicles else 1.0

    # Per-vehicle normalised fractions
    veh_approach_end: dict[str, float] = {}
    veh_wait_end:     dict[str, float] = {}

    for v in vehicles:
        vid  = str(v.vehicle_id)
        tta  = veh_tta[vid]
        wait = veh_wait[vid]

        ae = tta / total_sim_time
        we = (tta + wait) / total_sim_time

        # Ensure crossing always has room
        ae = min(ae, 1.0 - MIN_CROSS_FRAC)
        we = min(we, 1.0 - MIN_CROSS_FRAC)
        we = max(we, ae)   # wait_end >= approach_end

        veh_approach_end[vid] = ae
        veh_wait_end[vid]     = we

    # Conflict overlay traces (problem view only)
    conflict_overlay: list[go.BaseTraceType] = []
    if highlight_conflicts:
        vid_map = {str(v.vehicle_id): v for v in vehicles}
        for c in highlight_conflicts:
            v1 = vid_map.get(str(c["vehicle1_id"]))
            v2 = vid_map.get(str(c["vehicle2_id"]))
            if v1 is None or v2 is None:
                continue
            sl1 = _stop_line_pos(v1)
            sl2 = _stop_line_pos(v2)
            conflict_overlay.append(go.Scatter(
                x=[sl1[0], 0.0, sl2[0]],
                y=[sl1[1], 0.0, sl2[1]],
                mode="lines",
                line=dict(color="#ff4444", width=2, dash="dot"),
                name=f"⚠ {c['vehicle1_id']} ↔ {c['vehicle2_id']}",
                hovertemplate=(
                    f"<b>Conflict</b><br>"
                    f"{c['decision']}<extra></extra>"
                ),
            ))

    def _v_traces(t_norm: float) -> list[go.BaseTraceType]:
        traces = []
        for idx, v in enumerate(vehicles):
            colour = _COLORS[idx % len(_COLORS)]
            vid    = str(v.vehicle_id)
            ae     = veh_approach_end[vid]
            we     = veh_wait_end[vid]
            wt     = waiting_times.get(vid, 0)

            (x, y), phase = _vehicle_pos_at_t(v, t_norm, ae, we)
            angle = (_APPROACH_ANGLE[str(v.direction).lower()]
                     if phase in ("approach", "wait")
                     else _crossing_angle(v))
            traces.append(go.Scatter(
                x=[x], y=[y],
                mode="markers+text",
                marker=dict(
                    size=18, color=colour,
                    symbol="arrow", angle=angle,
                    line=dict(width=1, color="white"),
                ),
                text=[str(v.vehicle_id)],
                textposition="top center",
                textfont=dict(color=colour, size=10, family="monospace"),
                name=str(v.vehicle_id),
                hovertemplate=(
                    f"<b>{v.vehicle_id}</b><br>"
                    f"Direction: {v.direction} → {v.destination}<br>"
                    f"Lane: {v.lane} | Speed: {v.speed} km/h<br>"
                    f"Movement: {v.movement_type}<br>"
                    f"TTA: {veh_tta[vid]:.1f}s | Wait: {wt}s<extra></extra>"
                ),
            ))
        return traces

    init_data = road_bg + conflict_overlay + _v_traces(0.0)

    frames = []
    for step in range(steps):
        t_norm = step / max(steps - 1, 1)
        frame_data = list(road_bg) + list(conflict_overlay) + _v_traces(t_norm)
        frames.append(go.Frame(data=frame_data, name=str(step)))

    axis_range = BOX_HALF + ROAD_LEN + 2.5
    fig = go.Figure(
        data=init_data,
        frames=frames,
        layout=go.Layout(
            title=dict(text=title, font=dict(color="white", size=15), x=0.5),
            paper_bgcolor="#1a1a2e",
            plot_bgcolor="#16213e",
            xaxis=dict(
                range=[-axis_range, axis_range],
                scaleanchor="y", scaleratio=1,
                showgrid=False, zeroline=False, showticklabels=False,
            ),
            yaxis=dict(
                range=[-axis_range, axis_range],
                showgrid=False, zeroline=False, showticklabels=False,
            ),
            legend=dict(
                font=dict(color="white", size=10),
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="#555555",
                borderwidth=1,
            ),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=-0.06, x=0.5, xanchor="center",
                direction="left",
                buttons=[
                    dict(
                        label="▶  Play",
                        method="animate",
                        args=[None, dict(
                            frame=dict(duration=interval, redraw=True),
                            fromcurrent=True, mode="immediate",
                        )],
                    ),
                    dict(
                        label="⏸  Pause",
                        method="animate",
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            mode="immediate",
                        )],
                    ),
                ],
                font=dict(color="#111111"),
                bgcolor="#dddddd",
            )],
            sliders=[dict(
                steps=[
                    dict(
                        method="animate",
                        args=[[str(s)], dict(
                            mode="immediate",
                            frame=dict(duration=interval, redraw=True),
                            transition=dict(duration=0),
                        )],
                        label=str(s),
                    )
                    for s in range(steps)
                ],
                transition=dict(duration=0),
                x=0.05, y=0.0, len=0.90,
                currentvalue=dict(
                    prefix="Frame: ",
                    font=dict(color="white", size=11),
                    visible=True,
                ),
                font=dict(color="white", size=9),
            )],
            height=580,
            margin=dict(l=10, r=10, t=55, b=110),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def visualize_intersection(
    layout: Any,
    vehicles: list[Any],
    steps: int = 40,
    interval: int = 80,
) -> None:
    """Render the **problem** animation in Streamlit.

    Vehicles approach at their actual speeds and distances with no waiting
    applied. Conflicting vehicle pairs are connected by a red dashed line
    through the intersection centre.

    Parameters
    ----------
    layout   : intersection layout dict (from parse_intersection_layout)
    vehicles : list of Vehicle objects (from parse_vehicles)
    steps    : number of animation frames (default 40)
    interval : milliseconds per frame (default 80)
    """
    if not vehicles:
        st.info("No vehicles to visualize.")
        return

    try:
        from conflict_detection_orig import detect_conflicts
        conflicts = detect_conflicts(vehicles)
    except Exception:
        conflicts = []

    fig = _build_figure(
        vehicles=vehicles,
        steps=steps,
        interval=interval,
        title="🚦 Intersection – Problem View (Conflicts Highlighted)",
        waiting_times={},
        highlight_conflicts=conflicts,
    )
    st.plotly_chart(fig, use_container_width=True)

    if conflicts:
        st.markdown("#### ⚠️ Detected Conflicts")
        for c in conflicts:
            st.markdown(f"- **{c['vehicle1_id']}** ↔ **{c['vehicle2_id']}** — {c['decision']}")
    else:
        st.success("✅ No conflicts detected between these vehicles.")


def visualize_solution(
    layout: Any,
    vehicles: list[Any],
    conflicts: list[dict],
    steps: int = 50,
    interval: int = 80,
) -> None:
    """Render the **solution** animation in Streamlit.

    Waiting times computed by ``detect_conflicts`` are applied: lower-priority
    vehicles pause at the stop-line for exactly the number of seconds assigned
    in ``conflict['waiting_times']`` before crossing.

    Parameters
    ----------
    layout    : intersection layout dict
    vehicles  : list of Vehicle objects
    conflicts : direct output of detect_conflicts() – list[dict] with keys:
                vehicle1_id, vehicle2_id, priority_order, waiting_times
    steps     : number of animation frames (default 50)
    interval  : milliseconds per frame (default 80)
    """
    if not vehicles:
        st.info("No vehicles to visualize.")
        return

    # Aggregate waiting times across all conflicts (take max per vehicle)
    waiting_times: dict[str, float] = {}
    if conflicts:
        for c in conflicts:
            for vid, wt in c.get("waiting_times", {}).items():
                waiting_times[str(vid)] = max(
                    waiting_times.get(str(vid), 0.0), float(wt)
                )

    fig = _build_figure(
        vehicles=vehicles,
        steps=steps,
        interval=interval,
        title="✅ Intersection – Solution View (Wait Times Applied)",
        waiting_times=waiting_times,
        highlight_conflicts=None,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Solution summary
    if conflicts:
        st.markdown("#### 🕐 Conflict Resolution Summary")
        rows = []
        for c in conflicts:
            p  = c["priority_order"]
            wt = c["waiting_times"]
            v1, v2 = str(c["vehicle1_id"]), str(c["vehicle2_id"])
            rows.append(
                f"| {v1} | {'🥇' if p.get(v1)==1 else '🔴 yield'} | {wt.get(v1, 0)}s "
                f"| {v2} | {'🥇' if p.get(v2)==1 else '🔴 yield'} | {wt.get(v2, 0)}s |"
            )
        header = (
            "| Vehicle A | Priority | Wait | Vehicle B | Priority | Wait |\n"
            "|-----------|----------|------|-----------|----------|------|\n"
        )
        st.markdown(header + "\n".join(rows))
    else:
        st.success("✅ No conflicts to resolve – all vehicles proceed without waiting.")

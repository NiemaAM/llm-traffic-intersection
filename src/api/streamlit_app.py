"""# v5.2 - embedded modules, all vehicles, waiting times fixed
streamlit_app.py  —  Milestone 5
---------------------------------
Production Streamlit front-end with:
 - Live Plotly intersection visualization (always visible)
 - Animated Problem + Solution tabs after analysis
 - Dual-mode: FastAPI when available, direct LLM otherwise (HF Space)
 - Delete bug fixed (uses stable vehicle_id keys)
All visualization and conflict detection code is embedded directly
to avoid import path issues on HF Space.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDED: conflict_detection_orig.py
# ═══════════════════════════════════════════════════════════════════════════════


# Mapping of opposite directions
OPPOSITE_DIRECTIONS = {"north": "south", "east": "west", "south": "north", "west": "east"}

# Enable or disable logging for debugging
log = False


def parse_intersection_layout(data):
    """
    Parses the intersection layout from the given data.

    Args:
        data (dict): Data containing the intersection layout.

    Returns:
        dict: Parsed intersection layout.
    """
    return data["intersection_layout"]


class Vehicle:
    """
    Represents a vehicle approaching an intersection.

    Attributes:
        vehicle_id (str): Unique identifier for the vehicle.
        lane (str): Lane number.
        speed (float): Speed in km/h.
        distance_to_intersection (float): Distance to intersection in meters.
        direction (str): Direction of approach ('north', 'east', 'south', 'west').
        destination (str): Destination road.
        time_to_intersection (float): Time to reach the intersection in seconds.
        movement_type (str): Type of movement ('straight', 'left', 'right', or 'unknown').
    """

    VALID_DIRECTIONS = ["north", "east", "south", "west"]

    def __init__(
        self,
        vehicle_id,
        lane,
        speed,
        distance_to_intersection,
        direction,
        destination,
        intersection_layout,
    ):
        """
        Initializes a Vehicle instance.

        Args:
            vehicle_id (str): Unique identifier for the vehicle.
            lane (str): Lane number.
            speed (float): Speed in km/h.
            distance_to_intersection (float): Distance to intersection in meters.
            direction (str): Direction of approach.
            destination (str): Destination road.
            intersection_layout (dict): Layout of the intersection.
        """
        self.vehicle_id = vehicle_id
        self.lane = str(lane)
        self.speed = speed  # in km/h
        self.distance_to_intersection = distance_to_intersection  # in meters
        self.direction = direction.lower()
        self.destination = destination
        self.validate_inputs()
        self.time_to_intersection = self.compute_time_to_intersection()
        self.movement_type = self.get_movement_type(intersection_layout)
        # If logging is enabled, print vehicle information after initialization
        if log:
            print(
                f"Initialized Vehicle {self.vehicle_id}: lane={self.lane}, speed={self.speed}, "
                f"distance_to_intersection={self.distance_to_intersection}, direction={self.direction}, "
                f"destination={self.destination}, movement_type={self.movement_type}, "
                f"time_to_intersection={self.time_to_intersection:.2f}s"
            )

    def validate_inputs(self):
        """
        Validates the inputs for the vehicle.
        """
        if self.speed < 0:
            raise ValueError(f"Vehicle {self.vehicle_id} has negative speed.")
        if self.distance_to_intersection < 0:
            raise ValueError(f"Vehicle {self.vehicle_id} has negative distance to intersection.")
        if self.direction not in self.VALID_DIRECTIONS:
            raise ValueError(f"Vehicle {self.vehicle_id} has invalid direction '{self.direction}'.")
        if not self.vehicle_id:
            raise ValueError("Vehicle ID cannot be empty.")

    def compute_time_to_intersection(self):
        """
        Computes the time for the vehicle to reach the intersection.

        Returns:
            float: Time to intersection in seconds.
        """
        speed_m_per_s = (self.speed * 1000) / 3600  # Convert km/h to m/s
        if speed_m_per_s == 0:
            return float("inf")  # Infinite time if speed is zero
        time = self.distance_to_intersection / speed_m_per_s
        return time

    def get_movement_type(self, intersection_layout):
        """
        Determines the movement type (straight, left, right) based on the intersection layout.

        Args:
            intersection_layout (dict): Layout of the intersection.

        Returns:
            str: Movement type ('straight', 'left', 'right', or 'unknown').
        """
        direction = self.direction
        lane = self.lane
        destination = self.destination

        lane_destinations = intersection_layout.get(direction, {}).get(lane, [])
        if not lane_destinations:
            warnings.warn(
                f"Vehicle {self.vehicle_id} is in an unknown lane '{lane}' for direction '{direction}'.",
                category=UserWarning,
            )
            return "unknown"
        if destination not in lane_destinations:
            warnings.warn(
                f"Destination '{destination}' not accessible from lane '{lane}' for direction '{direction}'.",
                category=UserWarning,
            )
            return "unknown"

        index = lane_destinations.index(destination)
        if lane in ["1", "3", "5", "7"]:
            if index == 0:
                movement_type = "right"
            elif index == 1:
                movement_type = "straight"
            elif index == 2:
                movement_type = "left"
            else:
                movement_type = "unknown"
        elif lane in ["2", "4", "6", "8"]:
            movement_type = "left"  # These lanes are dedicated left-turn lanes
        else:
            movement_type = "unknown"

        if movement_type == "unknown":
            warnings.warn(
                f"Vehicle {self.vehicle_id} has unknown movement type.", category=UserWarning
            )
        return movement_type


def parse_vehicles(data, intersection_layout):
    """
    Parses vehicle data from the given scenario.

    Args:
        data (dict): Vehicle scenario data.
        intersection_layout (dict): Layout of the intersection.

    Returns:
        list of Vehicle: List of Vehicle objects.
    """
    vehicles = []
    vehicle_ids = set()
    for vehicle_data in data["vehicles_scenario"]:
        vehicle_id = vehicle_data["vehicle_id"]
        if vehicle_id in vehicle_ids:
            raise ValueError(f"Duplicate vehicle ID detected: {vehicle_id}")
        vehicle_ids.add(vehicle_id)
        vehicle = Vehicle(
            vehicle_id=vehicle_id,
            lane=vehicle_data["lane"],
            speed=vehicle_data["speed"],
            distance_to_intersection=vehicle_data["distance_to_intersection"],
            direction=vehicle_data["direction"],
            destination=vehicle_data["destination"],
            intersection_layout=intersection_layout,
        )
        vehicles.append(vehicle)
    return vehicles


def paths_cross(vehicle1, vehicle2):
    """
    Determines if the paths of two vehicles cross.

    Args:
        vehicle1 (Vehicle): First vehicle.
        vehicle2 (Vehicle): Second vehicle.

    Returns:
        bool: True if paths cross, False otherwise.
    """
    if log:
        print(
            f"Checking if paths cross between Vehicle {vehicle1.vehicle_id} and Vehicle {vehicle2.vehicle_id}"
        )
    if "unknown" in [vehicle1.movement_type, vehicle2.movement_type]:
        if log:
            print(
                f"At least one vehicle has unknown movement type: {vehicle1.movement_type}, {vehicle2.movement_type}"
            )
        return False
    if vehicle1.vehicle_id == vehicle2.vehicle_id:
        if log:
            print("Comparing the same vehicle.")
        return False

    # Same direction
    if vehicle1.direction == vehicle2.direction:
        if log:
            print(f"Vehicles are coming from the same direction: {vehicle1.direction}")
        return False

    # Vehicles going straight from opposite directions do not conflict
    if (
        vehicle1.movement_type == "straight"
        and vehicle2.movement_type == "straight"
        and OPPOSITE_DIRECTIONS[vehicle1.direction] == vehicle2.direction
    ):
        if log:
            print(
                f"Vehicles are going straight from opposite directions: {vehicle1.direction} and {vehicle2.direction}"
            )
        return False

    # Opposite left turns do not conflict
    if (
        vehicle1.movement_type == "left"
        and vehicle2.movement_type == "left"
        and OPPOSITE_DIRECTIONS[vehicle1.direction] == vehicle2.direction
    ):
        if log:
            print(
                f"Vehicles are making left turns from opposite directions: {vehicle1.direction} and {vehicle2.direction}"
            )
        return False

    # Right turns from opposite directions do not conflict
    if (
        vehicle1.movement_type == "right"
        and vehicle2.movement_type == "right"
        and OPPOSITE_DIRECTIONS[vehicle1.direction] == vehicle2.direction
    ):
        if log:
            print(
                f"Vehicles are making right turns from opposite directions: {vehicle1.direction} and {vehicle2.direction}"
            )
        return False

    # Right turns from adjacent directions do not conflict
    if (
        vehicle1.movement_type == "right"
        and vehicle2.movement_type == "right"
        and vehicle1.direction != vehicle2.direction
        and OPPOSITE_DIRECTIONS[vehicle1.direction] != vehicle2.direction
    ):
        if log:
            print(
                f"Vehicles are making right turns from adjacent directions: {vehicle1.direction} and {vehicle2.direction}"
            )
        return False

    # Vehicles going straight from perpendicular directions conflict
    if (
        vehicle1.movement_type == "straight"
        and vehicle2.movement_type == "straight"
        and (vehicle1.direction != vehicle2.direction)
        and (OPPOSITE_DIRECTIONS[vehicle1.direction] != vehicle2.direction)
    ):
        if log:
            print(
                f"Vehicles are going straight from perpendicular directions: {vehicle1.direction} and {vehicle2.direction}"
            )
        return True

    # Left turn conflicts
    if vehicle1.movement_type == "left" or vehicle2.movement_type == "left":
        if log:
            print(
                f"At least one vehicle is turning left: {vehicle1.movement_type}, {vehicle2.movement_type}"
            )
        return True

    # Right turn vs straight from adjacent directions conflict
    if (
        vehicle1.movement_type == "right"
        and vehicle2.movement_type == "straight"
        and (vehicle1.direction != vehicle2.direction)
        and (OPPOSITE_DIRECTIONS[vehicle1.direction] != vehicle2.direction)
    ) or (
        vehicle2.movement_type == "right"
        and vehicle1.movement_type == "straight"
        and (vehicle1.direction != vehicle2.direction)
        and (OPPOSITE_DIRECTIONS[vehicle2.direction] != vehicle1.direction)
    ):
        if log:
            print(
                "One vehicle is turning right and the other is going straight from adjacent directions."
            )
        return True

    # For all other cases, assume paths do not cross
    if log:
        print(f"Vehicles do not conflict: {vehicle1.vehicle_id} and {vehicle2.vehicle_id}")
    return False


def arrival_time_close(vehicle1, vehicle2, threshold=4.0):
    """
    Checks if the arrival times of two vehicles are within a certain threshold.

    Args:
        vehicle1 (Vehicle): First vehicle.
        vehicle2 (Vehicle): Second vehicle.
        threshold (float): Time difference threshold in seconds.

    Returns:
        bool: True if arrival times are within the threshold, False otherwise.
    """
    if vehicle1.time_to_intersection == float("inf") or vehicle2.time_to_intersection == float(
        "inf"
    ):
        if log:
            print(
                f"At least one vehicle has infinite time to intersection: {vehicle1.time_to_intersection}, {vehicle2.time_to_intersection}"
            )
        return False
    time_diff = abs(vehicle1.time_to_intersection - vehicle2.time_to_intersection)
    if log:
        print(
            f"Time difference between Vehicle {vehicle1.vehicle_id} and Vehicle {vehicle2.vehicle_id}: {time_diff:.2f}s"
        )
    return time_diff <= threshold


def is_vehicle_on_right(vehicle1, vehicle2):
    """
    Determines if vehicle2 is on the right of vehicle1 based on their directions.

    Args:
        vehicle1 (Vehicle): First vehicle.
        vehicle2 (Vehicle): Second vehicle.

    Returns:
        bool: True if vehicle2 is on the right of vehicle1, False otherwise.
    """
    direction_order = ["north", "east", "south", "west"]
    idx1 = direction_order.index(vehicle1.direction)
    idx2 = direction_order.index(vehicle2.direction)
    result = (idx2 - idx1) % 4 == 1
    if log:
        print(
            f"Vehicle {vehicle2.vehicle_id} is {'on the right of' if result else 'not on the right of'} Vehicle {vehicle1.vehicle_id}"
        )
    return result


def apply_priority_rules(vehicle1, vehicle2):
    """
    Applies priority rules to determine which vehicle must yield.

    Args:
        vehicle1 (Vehicle): First vehicle.
        vehicle2 (Vehicle): Second vehicle.

    Returns:
        tuple: (decision message, vehicle priorities dictionary)
    """
    if log:
        print(
            f"Applying priority rules between Vehicle {vehicle1.vehicle_id} and Vehicle {vehicle2.vehicle_id}"
        )
    time_difference = abs(vehicle1.time_to_intersection - vehicle2.time_to_intersection)
    if log:
        print(f"Time difference: {time_difference:.2f}s")
    priority = {}
    if time_difference <= 1.0:
        if log:
            print("Vehicles arrive within 1 second of each other")
        # 1. Straight over turn
        if vehicle1.movement_type == "straight" and vehicle2.movement_type != "straight":
            if log:
                print(
                    f"Vehicle {vehicle1.vehicle_id} is going straight, Vehicle {vehicle2.vehicle_id} is turning"
                )
            decision = f"Potential conflict: Vehicle {vehicle2.vehicle_id} must yield to Vehicle {vehicle1.vehicle_id}"
            priority = {vehicle1.vehicle_id: 1, vehicle2.vehicle_id: 2}
        elif vehicle2.movement_type == "straight" and vehicle1.movement_type != "straight":
            if log:
                print(
                    f"Vehicle {vehicle2.vehicle_id} is going straight, Vehicle {vehicle1.vehicle_id} is turning"
                )
            decision = f"Potential conflict: Vehicle {vehicle1.vehicle_id} must yield to Vehicle {vehicle2.vehicle_id}"
            priority = {vehicle2.vehicle_id: 1, vehicle1.vehicle_id: 2}
        # 2. Right turn over left turn
        elif vehicle1.movement_type == "right" and vehicle2.movement_type == "left":
            if log:
                print(
                    f"Vehicle {vehicle1.vehicle_id} is turning right, Vehicle {vehicle2.vehicle_id} is turning left"
                )
            decision = f"Potential conflict: Vehicle {vehicle2.vehicle_id} must yield to Vehicle {vehicle1.vehicle_id}"
            priority = {vehicle1.vehicle_id: 1, vehicle2.vehicle_id: 2}
        elif vehicle2.movement_type == "right" and vehicle1.movement_type == "left":
            if log:
                print(
                    f"Vehicle {vehicle2.vehicle_id} is turning right, Vehicle {vehicle1.vehicle_id} is turning left"
                )
            decision = f"Potential conflict: Vehicle {vehicle1.vehicle_id} must yield to Vehicle {vehicle2.vehicle_id}"
            priority = {vehicle2.vehicle_id: 1, vehicle1.vehicle_id: 2}
        # 3. Right-hand rule
        else:
            if is_vehicle_on_right(vehicle1, vehicle2):
                if log:
                    print(
                        f"Vehicle {vehicle2.vehicle_id} is on the right of Vehicle {vehicle1.vehicle_id}"
                    )
                decision = f"Potential conflict: Vehicle {vehicle1.vehicle_id} must yield to Vehicle {vehicle2.vehicle_id}"
                priority = {vehicle2.vehicle_id: 1, vehicle1.vehicle_id: 2}
            else:
                if log:
                    print(
                        f"Vehicle {vehicle1.vehicle_id} is on the right of Vehicle {vehicle2.vehicle_id}"
                    )
                decision = f"Potential conflict: Vehicle {vehicle2.vehicle_id} must yield to Vehicle {vehicle1.vehicle_id}"
                priority = {vehicle1.vehicle_id: 1, vehicle2.vehicle_id: 2}
    else:
        # Vehicle that arrives later must yield
        if vehicle1.time_to_intersection > vehicle2.time_to_intersection:
            if log:
                print(
                    f"Vehicle {vehicle1.vehicle_id} arrives later than Vehicle {vehicle2.vehicle_id}"
                )
            decision = f"Potential conflict: Vehicle {vehicle1.vehicle_id} must yield to Vehicle {vehicle2.vehicle_id}"
            priority = {vehicle2.vehicle_id: 1, vehicle1.vehicle_id: 2}
        else:
            if log:
                print(
                    f"Vehicle {vehicle2.vehicle_id} arrives later than Vehicle {vehicle1.vehicle_id}"
                )
            decision = f"Potential conflict: Vehicle {vehicle2.vehicle_id} must yield to Vehicle {vehicle1.vehicle_id}"
            priority = {vehicle1.vehicle_id: 1, vehicle2.vehicle_id: 2}
    if log:
        print(f"Decision: {decision}")
    return decision, priority


def compute_waiting_times(vehicles, priorities):
    """
    Computes the waiting time for each vehicle based on priority and arrival times.

    Args:
        vehicles (list of Vehicle): List of Vehicle objects.
        priorities (dict): Dictionary mapping vehicle IDs to their priority levels.

    Returns:
        dict: Dictionary mapping vehicle IDs to their waiting times.
    """
    # The vehicle with higher priority (lower number) proceeds first.
    # Waiting time is the difference in arrival times if lower priority vehicle arrives earlier.
    waiting_times = {}
    for vehicle_id, priority in priorities.items():
        vehicle = next((v for v in vehicles if v.vehicle_id == vehicle_id), None)
        if vehicle is None:
            continue
        # Vehicles with priority 1 have zero waiting time
        if priority == 1:
            waiting_times[vehicle_id] = 0
        else:
            # Find the vehicle(s) with higher priority
            higher_priority_vehicles = [v_id for v_id, p in priorities.items() if p < priority]
            max_wait = 0
            for hp_vehicle_id in higher_priority_vehicles:
                hp_vehicle = next((v for v in vehicles if v.vehicle_id == hp_vehicle_id), None)
                if hp_vehicle:
                    # Calculate the additional waiting time needed
                    traversal_time = 2  # Assume it takes 2 seconds to clear the intersection
                    wait_time = max(
                        0,
                        (hp_vehicle.time_to_intersection + traversal_time)
                        - vehicle.time_to_intersection,
                    )
                    max_wait = max(max_wait, wait_time)
            waiting_times[vehicle_id] = math.ceil(max_wait)
    return waiting_times


def detect_conflicts(vehicles):
    """
    Detects conflicts between vehicles approaching an intersection.

    Args:
        vehicles (list of Vehicle): List of Vehicle objects.

    Returns:
        list of dict: List of conflicts detected. Each conflict is a dictionary containing:
            - 'vehicle1_id': ID of the first vehicle.
            - 'vehicle2_id': ID of the second vehicle.
            - 'decision': Conflict decision message.
            - 'place': Place of the conflict ('intersection').
            - 'priority_order': Dictionary of vehicle IDs to their priority levels.
            - 'waiting_times': Dictionary of vehicle IDs to their waiting times.
    """
    conflicts = []
    n = len(vehicles)
    for i in range(n):
        for j in range(i + 1, n):
            vehicle1 = vehicles[i]
            vehicle2 = vehicles[j]
            if log:
                print(f"\nEvaluating vehicles {vehicle1.vehicle_id} and {vehicle2.vehicle_id}")
            if paths_cross(vehicle1, vehicle2):
                if arrival_time_close(vehicle1, vehicle2):
                    decision, priority = apply_priority_rules(vehicle1, vehicle2)
                    waiting_times = compute_waiting_times([vehicle1, vehicle2], priority)
                    conflicts.append(
                        {
                            "vehicle1_id": vehicle1.vehicle_id,
                            "vehicle2_id": vehicle2.vehicle_id,
                            "decision": decision,
                            "place": "intersection",
                            "priority_order": priority,
                            "waiting_times": waiting_times,
                        }
                    )
                else:
                    if log:
                        print("Vehicles do not arrive close in time; no conflict.")
            else:
                if log:
                    print("Vehicles paths do not cross; no conflict.")

    return conflicts


def output_conflicts(conflicts):
    """
    Outputs the conflicts detected.

    Args:
        conflicts (list of dict): List of conflict dictionaries.
    """
    for conflict in conflicts:
        if log:
            print(conflict["decision"])
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDED: visualization_orig.py  (functions only — no st.set_page_config)
# ═══════════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Geometry constants
# ---------------------------------------------------------------------------

LANE_W = 1.5  # width of one lane (world units)
BOX_HALF = LANE_W  # half-size of central intersection box
ROAD_LEN = 12.0  # length of each arm outside the box

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

_ARMS: Dict[str, Dict] = {
    "north": dict(ox=0.0, oy=BOX_HALF, indx=0.0, indy=-1.0, px=-1.0, py=0.0),
    "south": dict(ox=0.0, oy=-BOX_HALF, indx=0.0, indy=1.0, px=1.0, py=0.0),
    "east": dict(ox=BOX_HALF, oy=0.0, indx=-1.0, indy=0.0, px=0.0, py=-1.0),
    "west": dict(ox=-BOX_HALF, oy=0.0, indx=1.0, indy=0.0, px=0.0, py=1.0),
}

_LANE_PERP_OFFSET: Dict[str, float] = {
    "1": +LANE_W / 2,
    "2": -LANE_W / 2,
    "3": +LANE_W / 2,
    "4": -LANE_W / 2,
    "5": +LANE_W / 2,
    "6": -LANE_W / 2,
    "7": +LANE_W / 2,
    "8": -LANE_W / 2,
}

# Destination exit positions mapped from layout:
#   north exits (A,H) → y+
#   east  exits (B,G) → x+
#   south exits (C,D) → y-
#   west  exits (E,F) → x-
_EXIT_HALF = ROAD_LEN * 0.65
_DEST_POS: Dict[str, Tuple[float, float]] = {
    "A": (-LANE_W / 2, BOX_HALF + _EXIT_HALF),  # north, left lane
    "H": (LANE_W / 2, BOX_HALF + _EXIT_HALF),  # north, right lane
    "B": (BOX_HALF + _EXIT_HALF, LANE_W / 2),  # east,  right lane
    "G": (BOX_HALF + _EXIT_HALF, -LANE_W / 2),  # east,  left lane
    "C": (LANE_W / 2, -BOX_HALF - _EXIT_HALF),  # south, right lane
    "D": (-LANE_W / 2, -BOX_HALF - _EXIT_HALF),  # south, left lane
    "E": (-BOX_HALF - _EXIT_HALF, LANE_W / 2),  # west,  right lane
    "F": (-BOX_HALF - _EXIT_HALF, -LANE_W / 2),  # west,  left lane
}

# Plotly arrow angle (degrees, 0=up/north, clockwise) for approach direction
_APPROACH_ANGLE: Dict[str, float] = {
    "north": 180.0,  # arrow points south (toward box)
    "south": 0.0,  # arrow points north (toward box)
    "east": 270.0,  # vehicle travels west  → arrow points left  (270)
    "west": 90.0,  # vehicle travels east  → arrow points right (90)
}

_COLORS = [
    "#00d4ff",
    "#ff6b35",
    "#7bc67e",
    "#ff4757",
    "#ffd32a",
    "#a29bfe",
    "#fd79a8",
    "#55efc4",
]


# ---------------------------------------------------------------------------
# Position helpers
# ---------------------------------------------------------------------------


def _stop_line_pos(vehicle: Any) -> Tuple[float, float]:
    arm = _ARMS[str(vehicle.direction).lower()]
    off = _LANE_PERP_OFFSET[str(vehicle.lane)]
    return (arm["ox"] + arm["px"] * off, arm["oy"] + arm["py"] * off)


def _start_pos(vehicle: Any) -> Tuple[float, float]:
    arm = _ARMS[str(vehicle.direction).lower()]
    off = _LANE_PERP_OFFSET[str(vehicle.lane)]
    return (
        arm["ox"] + arm["px"] * off - arm["indx"] * ROAD_LEN,
        arm["oy"] + arm["py"] * off - arm["indy"] * ROAD_LEN,
    )


def _exit_pos(vehicle: Any) -> Tuple[float, float]:
    dest = str(vehicle.destination).upper()
    return _DEST_POS.get(dest, (0.0, BOX_HALF + _EXIT_HALF))


def _lerp(a: Tuple[float, float], b: Tuple[float, float], t: float) -> Tuple[float, float]:
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
) -> Tuple[Tuple[float, float], str]:
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
    stop = _stop_line_pos(vehicle)
    mid = (0.0, 0.0)
    dest = _exit_pos(vehicle)

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


def _road_bg_traces() -> List[go.BaseTraceType]:
    traces: List[go.BaseTraceType] = []
    road_col = "#3d3d3d"
    edge_col = "#666666"

    # Central box
    b = BOX_HALF
    traces.append(
        go.Scatter(
            x=[-b, b, b, -b, -b],
            y=[-b, -b, b, b, -b],
            fill="toself",
            fillcolor=road_col,
            line=dict(color=edge_col, width=1),
            mode="lines",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    for direction, arm in _ARMS.items():
        ox, oy = arm["ox"], arm["oy"]
        indx, indy = arm["indx"], arm["indy"]
        px, py = arm["px"], arm["py"]
        hw = LANE_W  # half-width of the whole road

        far_x = ox - indx * ROAD_LEN
        far_y = oy - indy * ROAD_LEN

        # Road arm rectangle
        corners = [
            (ox + px * hw, oy + py * hw),
            (ox - px * hw, oy - py * hw),
            (far_x - px * hw, far_y - py * hw),
            (far_x + px * hw, far_y + py * hw),
        ]
        xs = [c[0] for c in corners] + [corners[0][0]]
        ys = [c[1] for c in corners] + [corners[0][1]]
        traces.append(
            go.Scatter(
                x=xs,
                y=ys,
                fill="toself",
                fillcolor=road_col,
                line=dict(color=edge_col, width=1),
                mode="lines",
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Centre dashed lane divider
        traces.append(
            go.Scatter(
                x=[ox, far_x],
                y=[oy, far_y],
                mode="lines",
                line=dict(color="white", width=1, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Stop-line (thick white bar)
        traces.append(
            go.Scatter(
                x=[ox + px * hw, ox - px * hw],
                y=[oy + py * hw, oy - py * hw],
                mode="lines",
                line=dict(color="white", width=3),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Destination labels
    for dest, (dx, dy) in _DEST_POS.items():
        traces.append(
            go.Scatter(
                x=[dx],
                y=[dy],
                mode="text",
                text=[f"<b>{dest}</b>"],
                textfont=dict(color="#ffd700", size=14),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Compass labels
    label_offset = BOX_HALF + ROAD_LEN + 1.2
    for lbl, lx, ly in [
        ("N", 0, label_offset),
        ("S", 0, -label_offset),
        ("E", label_offset, 0),
        ("W", -label_offset, 0),
    ]:
        traces.append(
            go.Scatter(
                x=[lx],
                y=[ly],
                mode="text",
                text=[lbl],
                textfont=dict(color="#aaaaaa", size=15),
                showlegend=False,
                hoverinfo="skip",
            )
        )

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
        ("2", "north", -LANE_W / 2, True),  # exit lane – away from box
        ("3", "east", +LANE_W / 2, False),
        ("4", "east", -LANE_W / 2, True),  # exit lane – away from box
        ("5", "south", +LANE_W / 2, False),
        ("6", "south", -LANE_W / 2, True),  # exit lane – away from box
        ("7", "west", +LANE_W / 2, False),
        ("8", "west", -LANE_W / 2, True),  # exit lane – away from box
    ]

    # Place label + arrow at 50% along the arm from the stop-line outward
    LABEL_DIST = ROAD_LEN * 0.50

    for lane_id, direction, perp_off, exit_lane in _LANE_INFO:
        arm = _ARMS[direction]
        ox, oy = arm["ox"], arm["oy"]
        indx, indy = arm["indx"], arm["indy"]
        px, py = arm["px"], arm["py"]

        # Mid-arm position (outward from stop-line)
        mx = ox + px * perp_off - indx * LABEL_DIST
        my = oy + py * perp_off - indy * LABEL_DIST

        # ---- lane number badge ----
        traces.append(
            go.Scatter(
                x=[mx],
                y=[my],
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
            )
        )

        # ---- direction arrow ----
        # Entry lanes: arrow points toward the box (approach angle)
        # Exit  lanes: arrow points away from box  (approach angle + 180°)
        arrow_dist = ROAD_LEN * 0.28
        ax = ox + px * perp_off - indx * arrow_dist
        ay = oy + py * perp_off - indy * arrow_dist
        arrow_angle = (_APPROACH_ANGLE[direction] + (180.0 if exit_lane else 0.0)) % 360

        traces.append(
            go.Scatter(
                x=[ax],
                y=[ay],
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
            )
        )

    return traces


# ---------------------------------------------------------------------------
# Build animated figure
# ---------------------------------------------------------------------------


def _build_figure(
    vehicles: List[Any],
    steps: int,
    interval: int,
    title: str,
    waiting_times: Dict[str, float],
    highlight_conflicts: Optional[List[Dict]] = None,
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

    CROSS_TIME = 6.0  # seconds a vehicle takes to traverse the box (increased for clarity)
    MIN_CROSS_FRAC = 0.15  # crossing gets at least 15% of the animation
    MIN_WAIT_FRAC = 0.12  # waiting gets at least 12% of the animation when wait > 0

    # Gather per-vehicle TTA (guard against inf for stopped vehicles)
    def _tta(v: Any) -> float:
        tta = getattr(v, "time_to_intersection", None)
        if tta is None or tta == float("inf"):
            speed_ms = (v.speed * 1000) / 3600
            return v.distance_to_intersection / speed_ms if speed_ms > 0 else 9999.0
        return float(tta)

    veh_tta = {str(v.vehicle_id): _tta(v) for v in vehicles}
    veh_wait = {
        str(v.vehicle_id): float(waiting_times.get(str(v.vehicle_id), 0.0)) for v in vehicles
    }

    # Boost small waiting times so they are clearly visible in the animation
    # Any vehicle with wait > 0 gets at least MIN_WAIT_SEC seconds of wait phase
    MIN_WAIT_SEC = 4.0
    veh_wait_boosted = {
        vid: max(wt, MIN_WAIT_SEC) if wt > 0 else 0.0 for vid, wt in veh_wait.items()
    }

    total_sim_time = (
        max(
            veh_tta[str(v.vehicle_id)] + veh_wait_boosted[str(v.vehicle_id)] + CROSS_TIME
            for v in vehicles
        )
        if vehicles
        else 1.0
    )

    # Per-vehicle normalised fractions
    veh_approach_end: Dict[str, float] = {}
    veh_wait_end: Dict[str, float] = {}

    for v in vehicles:
        vid = str(v.vehicle_id)
        tta = veh_tta[vid]
        wait = veh_wait_boosted[vid]

        ae = tta / total_sim_time
        we = (tta + wait) / total_sim_time

        # Ensure crossing always has room
        ae = min(ae, 1.0 - MIN_CROSS_FRAC)
        we = min(we, 1.0 - MIN_CROSS_FRAC)
        we = max(we, ae)  # wait_end >= approach_end

        # Enforce minimum wait fraction so waiting is clearly visible
        if veh_wait[vid] > 0 and (we - ae) < MIN_WAIT_FRAC:
            we = min(ae + MIN_WAIT_FRAC, 1.0 - MIN_CROSS_FRAC)

        veh_approach_end[vid] = ae
        veh_wait_end[vid] = we

    # Conflict overlay traces (problem view only)
    conflict_overlay: List[go.BaseTraceType] = []
    if highlight_conflicts:
        vid_map = {str(v.vehicle_id): v for v in vehicles}
        for c in highlight_conflicts:
            v1 = vid_map.get(str(c["vehicle1_id"]))
            v2 = vid_map.get(str(c["vehicle2_id"]))
            if v1 is None or v2 is None:
                continue
            sl1 = _stop_line_pos(v1)
            sl2 = _stop_line_pos(v2)
            conflict_overlay.append(
                go.Scatter(
                    x=[sl1[0], 0.0, sl2[0]],
                    y=[sl1[1], 0.0, sl2[1]],
                    mode="lines",
                    line=dict(color="#ff4444", width=2, dash="dot"),
                    name=f"⚠ {c['vehicle1_id']} ↔ {c['vehicle2_id']}",
                    hovertemplate=(f"<b>Conflict</b><br>" f"{c['decision']}<extra></extra>"),
                )
            )

    def _v_traces(t_norm: float) -> List[go.BaseTraceType]:
        traces = []
        for idx, v in enumerate(vehicles):
            colour = _COLORS[idx % len(_COLORS)]
            vid = str(v.vehicle_id)
            ae = veh_approach_end[vid]
            we = veh_wait_end[vid]
            wt = waiting_times.get(vid, 0)

            (x, y), phase = _vehicle_pos_at_t(v, t_norm, ae, we)
            angle = (
                _APPROACH_ANGLE[str(v.direction).lower()]
                if phase in ("approach", "wait")
                else _crossing_angle(v)
            )
            traces.append(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers+text",
                    marker=dict(
                        size=18,
                        color=colour,
                        symbol="arrow",
                        angle=angle,
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
                )
            )
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
                scaleanchor="y",
                scaleratio=1,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
            ),
            yaxis=dict(
                range=[-axis_range, axis_range],
                showgrid=False,
                zeroline=False,
                showticklabels=False,
            ),
            legend=dict(
                font=dict(color="white", size=10),
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="#555555",
                borderwidth=1,
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=-0.06,
                    x=0.5,
                    xanchor="center",
                    direction="left",
                    buttons=[
                        dict(
                            label="▶  Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=interval, redraw=True),
                                    fromcurrent=True,
                                    mode="immediate",
                                ),
                            ],
                        ),
                        dict(
                            label="⏸  Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    mode="immediate",
                                ),
                            ],
                        ),
                    ],
                    font=dict(color="#111111"),
                    bgcolor="#dddddd",
                )
            ],
            sliders=[
                dict(
                    steps=[
                        dict(
                            method="animate",
                            args=[
                                [str(s)],
                                dict(
                                    mode="immediate",
                                    frame=dict(duration=interval, redraw=True),
                                    transition=dict(duration=0),
                                ),
                            ],
                            label=str(s),
                        )
                        for s in range(steps)
                    ],
                    transition=dict(duration=0),
                    x=0.05,
                    y=0.0,
                    len=0.90,
                    currentvalue=dict(
                        prefix="Frame: ",
                        font=dict(color="white", size=11),
                        visible=True,
                    ),
                    font=dict(color="white", size=9),
                )
            ],
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
    vehicles: List[Any],
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
    vehicles: List[Any],
    conflicts: List[Dict],
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
    waiting_times: Dict[str, float] = {}
    if conflicts:
        for c in conflicts:
            for vid, wt in c.get("waiting_times", {}).items():
                waiting_times[str(vid)] = max(waiting_times.get(str(vid), 0.0), float(wt))

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
            p = c["priority_order"]
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


# ═══════════════════════════════════════════════════════════════════════════════
# APP CODE
# ═══════════════════════════════════════════════════════════════════════════════

# ── Auto-load .env ────────────────────────────────────────────────────────────
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
_LAYOUT_ORIG = parse_intersection_layout(
    {
        "intersection_layout": {
            "north": {"1": ["F", "H"], "2": ["E", "D", "C"]},
            "east": {"3": ["H", "B"], "4": ["G", "E", "F"]},
            "south": {"5": ["B", "D"], "6": ["A", "G", "H"]},
            "west": {"7": ["D", "F"], "8": ["B", "C", "A"]},
        }
    }
)

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
    "\u26a0\ufe0f Classic Conflict (N\u2194E)": [
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
    "\U0001f6a8 Head-on (N\u2194S)": [
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
    "\U0001f7e2 No Conflict (far gap)": [
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
    "\U0001f7e0 Multi-vehicle (4 cars)": [
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

# ── Visualization helpers ─────────────────────────────────────────────────────


def _setup_llm_paths() -> None:
    for p in [
        str(Path(__file__).parent),
        str(Path(__file__).parent.parent),
        str(Path(__file__).parent.parent.parent / "src"),
        "/app",
        "/app/src",
    ]:
        if p not in sys.path:
            sys.path.insert(0, p)


def _get_vehicles_obj(vehicles_raw):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vobjs = parse_vehicles({"vehicles_scenario": vehicles_raw}, _LAYOUT_ORIG)
        return vobjs
    except Exception:
        return None


def _make_problem_fig(vehicles_raw, steps, interval):
    vobjs = _get_vehicles_obj(vehicles_raw)
    if not vobjs:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            conflicts = detect_conflicts(vobjs)
        return _build_figure(
            vehicles=vobjs,
            steps=steps,
            interval=interval,
            title="\U0001f6a6 Intersection \u2013 Problem View",
            waiting_times={},
            highlight_conflicts=conflicts,
        )
    except Exception as e:
        return None


def _make_solution_fig(vehicles_raw, llm_waiting_times, steps, interval):
    vobjs = _get_vehicles_obj(vehicles_raw)
    if not vobjs:
        return None
    try:
        return _build_figure(
            vehicles=vobjs,
            steps=steps,
            interval=interval,
            title="\u2705 Intersection \u2013 Solution View",
            waiting_times=llm_waiting_times,
            highlight_conflicts=None,
        )
    except Exception:
        return None


# ── API / LLM helpers ─────────────────────────────────────────────────────────


def _api_reachable(url: str) -> bool:
    try:
        return requests.get(f"{url}/health", timeout=2).status_code == 200
    except Exception:
        return False


def _llm_predict_direct(vehicles_raw: list) -> dict:
    _setup_llm_paths()
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
    import time as _t

    llm = IntersectionLLM(
        model=model_name,
        api_key=api_key,
        few_shot=few_shot,
        fine_tuned_model_id=fine_tuned or None,
    )
    t0 = _t.time()
    result = llm.predict({"vehicles": vehicles_raw})
    result["latency_ms"] = round((_t.time() - t0) * 1000)
    return result


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
    page_title="\U0001f6a6 Intersection Conflict Resolver",
    page_icon="\U0001f6a6",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ─────────────────────────────────────────────────────────────

if "vehicles" not in st.session_state:
    st.session_state.vehicles = list(DEFAULT_SCENARIO)
if "json_text" not in st.session_state:
    st.session_state.json_text = to_json(DEFAULT_SCENARIO)
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Configuration")
    st.subheader("Model Status")
    fine_tuned_id = os.environ.get("FINE_TUNED_MODEL_ID", "").strip()
    model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY", "")

    if _api_reachable(API_URL):
        st.success("\u2705 FastAPI connected")
        try:
            info = requests.get(f"{API_URL}/health", timeout=3).json()
            st.caption(f"Model: `{info.get('model', 'unknown')}`")
        except Exception:
            pass
    else:
        st.info("Standalone mode (direct LLM)")
        if api_key:
            st.success("API key loaded")
        else:
            st.error("OPENAI_API_KEY not set")

    if fine_tuned_id:
        st.success("Fine-tuned model active")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Metric": "Accuracy", "Value": "78.33%"},
                    {"Metric": "Precision", "Value": "79.31%"},
                    {"Metric": "Recall", "Value": "76.67%"},
                    {"Metric": "F1 Score", "Value": "77.97%"},
                    {"Metric": "FNR", "Value": "23.33%"},
                    {"Metric": "Avg Latency", "Value": "1.63 s"},
                    {"Metric": "Error Rate", "Value": "0.00%"},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption(f"Base model: `{model_name}`")

    st.divider()
    st.subheader("Animation")
    anim_steps = st.slider("Frames", 20, 80, 40, 5)
    anim_interval = st.slider("ms per frame", 40, 200, 80, 10)

    st.divider()
    st.subheader("Layout Reference")
    for direction, lanes in LAYOUT_DATA.items():
        lane_str = "  |  ".join(f"Lane {ln} -> {'/'.join(dests)}" for ln, dests in lanes.items())
        st.caption(f"**{direction.capitalize()}:** {lane_str}")

    st.divider()
    st.markdown(
        "**Milestone 5 - Production Serving**\n\n"
        "Baseline: [Masri et al., 2025](https://arxiv.org/abs/2411.10869)\n\n"
        "Code: [github.com/NiemaAM/llm-traffic-intersection]"
        "(https://github.com/NiemaAM/llm-traffic-intersection)"
    )

# ── Header ────────────────────────────────────────────────────────────────────

st.title("LLM-Driven Intersection Conflict Resolver")
st.markdown(
    "Production serving of **GPT-4o-mini (fine-tuned)** conflict detection agent. "
    "Analyzes vehicle scenarios at a 4-way 8-lane intersection in real time."
)
st.divider()

# ── Scenario section ──────────────────────────────────────────────────────────

st.header("Define Intersection Scenario")

c1, c2, c3, c4, c5 = st.columns([3, 1, 1, 1, 1])
with c1:
    preset = st.selectbox(
        "Preset", ["\u2014 Custom \u2014"] + list(PRESET_SCENARIOS), label_visibility="collapsed"
    )
with c2:
    if st.button("\U0001f4c2 Load", use_container_width=True) and preset != "\u2014 Custom \u2014":
        st.session_state.vehicles = list(PRESET_SCENARIOS[preset])
        st.session_state.last_result = None
        sync_json()
        st.rerun()
with c3:
    if st.button("\U0001f3b2 Random", use_container_width=True):
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
        st.session_state.last_result = None
        sync_json()
        st.rerun()
with c4:
    if st.button("\u2795 Add", use_container_width=True):
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
        st.session_state.last_result = None
        sync_json()
        st.rerun()
with c5:
    if st.button("\U0001f5d1\ufe0f Clear", use_container_width=True):
        st.session_state.vehicles = []
        st.session_state.last_result = None
        sync_json()
        st.rerun()

# ── Vehicle editor + JSON ─────────────────────────────────────────────────────

editor_col, json_col = st.columns([1, 1], gap="large")

with editor_col:
    st.markdown("#### Vehicle Editor")
    st.caption("Direction is automatically determined by lane number.")

    if st.session_state.vehicles:
        hdr = st.columns([2, 1, 2, 2, 2, 2, 1])
        for lbl, col in zip(
            ["Vehicle ID", "Lane", "Speed (km/h)", "Distance (m)", "Direction", "Destination", ""],
            hdr,
        ):
            col.markdown(f"**{lbl}**")

        to_delete = None
        for v in list(st.session_state.vehicles):
            vid = v["vehicle_id"]
            cols = st.columns([2, 1, 2, 2, 2, 2, 1])
            v["vehicle_id"] = cols[0].text_input(
                "id", str(vid), key=f"id_{vid}", label_visibility="collapsed"
            )
            v["lane"] = cols[1].number_input(
                "ln", 1, 8, int(v["lane"]), key=f"ln_{vid}", label_visibility="collapsed"
            )
            v["speed"] = cols[2].number_input(
                "sp",
                0.0,
                200.0,
                float(v["speed"]),
                step=1.0,
                key=f"sp_{vid}",
                label_visibility="collapsed",
            )
            v["distance_to_intersection"] = cols[3].number_input(
                "di",
                0.0,
                2000.0,
                float(v["distance_to_intersection"]),
                step=10.0,
                key=f"di_{vid}",
                label_visibility="collapsed",
            )
            auto_dir = LANE_DIRECTION[str(v["lane"])]
            v["direction"] = auto_dir
            cols[4].text_input(
                "dr", auto_dir, key=f"dr_{vid}", disabled=True, label_visibility="collapsed"
            )
            dests = LAYOUT_DATA[auto_dir][str(v["lane"])]
            cur = v["destination"] if v["destination"] in dests else dests[0]
            v["destination"] = cols[5].selectbox(
                "dt", dests, index=dests.index(cur), key=f"dt_{vid}", label_visibility="collapsed"
            )
            if cols[6].button("🗑", key=f"del_{vid}"):
                to_delete = vid

        if to_delete is not None:
            st.session_state.vehicles = [
                v for v in st.session_state.vehicles if v["vehicle_id"] != to_delete
            ]
            st.session_state.last_result = None
            sync_json()
            st.rerun()

        sync_json()
    else:
        st.info("Add at least 2 vehicles to run the analysis.")

with json_col:
    st.markdown("#### Live JSON Scenario")
    st.caption("Updates in real time. Edit directly \u2014 valid changes sync back.")
    st.text_area(
        label="json",
        key="json_text",
        height=350,
        on_change=on_json_edit,
        label_visibility="collapsed",
    )
    if st.session_state.get("_json_error"):
        st.error(f"{st.session_state._json_error}")
    else:
        st.success("JSON is valid and synced.")

vehicles_raw = list(st.session_state.vehicles)
result = st.session_state.last_result

# ── Analyze button ────────────────────────────────────────────────────────────

st.divider()
if len(vehicles_raw) < 2:
    st.warning("Add at least **2 vehicles** to run the analysis.")
else:
    if st.button("Analyze Intersection", type="primary", use_container_width=True):
        v_raw, jerr = from_json(st.session_state.json_text)
        if jerr:
            st.error(f"Cannot run: {jerr}")
            st.stop()

        use_api = _api_reachable(API_URL)
        with st.spinner(
            "Consulting the LLM" + (" via FastAPI\u2026" if use_api else " directly\u2026")
        ):
            if use_api:
                try:
                    r = requests.post(f"{API_URL}/predict", json={"vehicles": v_raw}, timeout=30)
                    res = r.json() if r.status_code == 200 else None
                    if res is None:
                        st.error(f"API Error {r.status_code}: {r.text[:300]}")
                        st.stop()
                except Exception as exc:
                    st.error(f"Request failed: {exc}")
                    st.stop()
            else:
                res = _llm_predict_direct(v_raw)
                if "error" in res:
                    st.error(f"\u274c {res['error']}")
                    st.stop()

        # Save the exact vehicles that were analyzed (from JSON, not widget state)
        st.session_state.last_result = res
        st.session_state.last_vehicles_analyzed = v_raw
        st.rerun()

# ── Results ───────────────────────────────────────────────────────────────────

if result is not None:
    st.header("Analysis Results")
    is_conflict = str(result.get("is_conflict", "no")).lower() == "yes"
    n_conflicts = int(result.get("number_of_conflicts", 0))

    if is_conflict:
        st.error(f"**CONFLICT DETECTED** — {n_conflicts} conflict(s)")
    else:
        st.success("**No Conflicts Detected** — intersection is clear")

    m1, m2, m3 = st.columns(3)
    m1.metric("Conflicts", n_conflicts)
    m2.metric("Vehicles Analyzed", len(vehicles_raw))
    m3.metric("Latency", f"{result.get('latency_ms', 0):.0f} ms")

    decisions = result.get("decisions", [])
    if decisions:
        st.subheader("Control Decisions")
        for d in decisions:
            st.warning(f"{d}")
    else:
        st.info("No control actions required.")

    priority = result.get("priority_order", {})
    waiting = result.get("waiting_times", {})
    if priority:
        st.subheader("Priority Order & Waiting Times")
        rows = sorted(
            [
                {
                    "Vehicle": vid,
                    "Priority Rank": rank if rank is not None else "\u2014",
                    "Wait (s)": waiting.get(vid, 0),
                }
                for vid, rank in priority.items()
            ],
            key=lambda row: (row["Priority Rank"] == "\u2014", row["Priority Rank"]),
        )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    cv = result.get("conflict_vehicles", [])
    if cv and isinstance(cv[0], dict):
        st.subheader("Conflicting Vehicle Pairs")
        st.dataframe(pd.DataFrame(cv), use_container_width=True, hide_index=True)

    with st.expander("Full JSON Response"):
        st.json(result)

# ── Visualization (shown after analysis) ─────────────────────────────────────

# Use vehicles that were actually analyzed (saved at analysis time)
viz_vehicles = st.session_state.get("last_vehicles_analyzed", vehicles_raw)

if result is not None and len(viz_vehicles) >= 2:
    st.divider()
    st.markdown("#### Intersection Visualization")

    tab1, tab2 = st.tabs(
        [
            "Problem View (Conflicts Highlighted)",
            "Solution View (Wait Times Applied)",
        ]
    )
    with tab1:
        fig = _make_problem_fig(viz_vehicles, anim_steps, anim_interval)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Visualization unavailable.")
    with tab2:
        waiting: dict = {}
        for pair in result.get("conflict_vehicles", []):
            if isinstance(pair, dict):
                for vid in [pair.get("vehicle1_id", ""), pair.get("vehicle2_id", "")]:
                    wt = result.get("waiting_times", {}).get(vid, 0)
                    waiting[str(vid)] = float(wt)
        fig = _make_solution_fig(vehicles_raw, waiting, anim_steps, anim_interval)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Visualization unavailable.")

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Milestone 5 - Production Serving  |  "
    "LLM-Driven Agents for Traffic Intersection Conflict Resolution  |  "
    "CSC5382 - AI for Digital Transformation"
)

"""
simulation.py
-------------
Physics-lite simulation engine that moves ambulances along precomputed
routes and keeps real-time metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import itertools
import math
import random
import time

from graph_model import GraphModel, GridNode

Point = Tuple[float, float]
COLOR_CYCLE = ["#2b83ba", "#abdda4", "#fdae61", "#d7191c", "#ffffbf"]


@dataclass
class Ambulance:
    id: int
    position: Tuple[float, float]
    speed: float = 3.0  # grid units per second
    color: str = "#2b83ba"
    path: List[Point] = field(default_factory=list)
    target_index: int = 0
    assignment_id: Optional[int] = None

    def assign_path(self, path: List[Point], assignment_id: Optional[int]) -> None:
        self.path = path
        self.target_index = 0
        self.assignment_id = assignment_id

    def step(self, dt: float) -> bool:
        """Move along the planned path; return True if destination reached this tick."""
        if not self.path or self.target_index >= len(self.path):
            return False
        target = self.path[self.target_index]
        dx = target[0] - self.position[0]
        dy = target[1] - self.position[1]
        distance = math.hypot(dx, dy)
        if distance <= 1e-4:
            self.position = (float(target[0]), float(target[1]))
            self.target_index += 1
            if self.target_index >= len(self.path):
                return True
            return False
        travel = self.speed * dt
        if travel >= distance:
            self.position = (float(target[0]), float(target[1]))
            self.target_index += 1
            if self.target_index >= len(self.path):
                return True
        else:
            ratio = travel / distance
            self.position = (self.position[0] + dx * ratio, self.position[1] + dy * ratio)
        return False

    @property
    def idle(self) -> bool:
        return not self.path or self.target_index >= len(self.path)


@dataclass
class EmergencyEvent:
    id: int
    location: Point
    reported_at: float
    urgency: int
    resolved_at: Optional[float] = None


class SimulationEngine:
    """Coordinate ambulances moving in real time."""

    def __init__(self, model: GraphModel, algorithm: str = "dijkstra") -> None:
        self.model = model
        self.algorithm = algorithm
        self.ambulances: List[Ambulance] = []
        self.emergencies: Dict[int, EmergencyEvent] = {}
        self.metrics: Dict[str, float] = {
            "distance_travelled": 0.0,
            "responses": 0,
            "average_response_time": 0.0,  # minutes
        }
        self.response_history: List[float] = []  # minutes
        self._emergency_counter = itertools.count(1)
        self._ambulance_counter = itertools.count(1)
        self.random = random.Random(99)
        self.sim_time = 0.0  # seconds
        self._default_speed = 3.0 if self.model.grid_mode else 15.0  # ~54 km/h in OSM mode

    # ------------------------------------------------------------------ #
    # Setup helpers
    # ------------------------------------------------------------------ #
    def spawn_ambulances(self, count: int = 3, positions: Optional[List[GridNode]] = None) -> None:
        self.ambulances = []
        self._ambulance_counter = itertools.count(1)
        if positions is None:
            positions = self.model.generate_scenario(ambulances=count, emergencies=0).ambulances
        for idx, spot in enumerate(positions, start=1):
            self.ambulances.append(
                Ambulance(
                    id=next(self._ambulance_counter),
                    position=(float(spot[0]), float(spot[1])),
                    color=_color_for_index(idx),
                    speed=self._default_speed,
                )
            )

    def add_ambulance(self, position: Optional[GridNode] = None) -> Ambulance:
        position = position or self.model.random_node()
        idx = len(self.ambulances) + 1
        ambulance = Ambulance(
            id=next(self._ambulance_counter),
            position=(float(position[0]), float(position[1])),
            color=_color_for_index(idx),
            speed=self._default_speed,
        )
        self.ambulances.append(ambulance)
        return ambulance

    def create_emergency(self, location: Optional[GridNode] = None, urgency: Optional[int] = None) -> EmergencyEvent:
        location = location or self.model.random_node()
        urgency = urgency or self.random.randint(1, 5)
        emergency = EmergencyEvent(
            id=next(self._emergency_counter),
            location=location,
            reported_at=self.sim_time,
            urgency=urgency,
        )
        self.emergencies[emergency.id] = emergency
        return emergency

    # ------------------------------------------------------------------ #
    # Assignment + movement
    # ------------------------------------------------------------------ #
    def assign_path(self, ambulance_id: int, path: List[Point], emergency_id: int) -> None:
        ambulance = self._find_ambulance(ambulance_id)
        normalized = self._normalize_path(ambulance, path)
        ambulance.assign_path(normalized, emergency_id)

    def tick(self, dt: float = 0.2) -> List[int]:
        completed: List[int] = []
        self.sim_time += dt
        for ambulance in self.ambulances:
            old_position = ambulance.position
            finished = ambulance.step(dt)
            self.metrics["distance_travelled"] += math.dist(old_position, ambulance.position)
            if finished and ambulance.assignment_id is not None:
                event = self.emergencies.get(ambulance.assignment_id)
                if event and event.resolved_at is None:
                    event.resolved_at = self.sim_time
                    self.metrics["responses"] += 1
                    self._update_response_average(event.resolved_at - event.reported_at)
                completed.append(ambulance.id)
                ambulance.assignment_id = None
                ambulance.path = []
        return completed

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def snapshot(self) -> Dict:
        units = "blocks" if self.model.grid_mode else "km"
        ambulances_payload = []
        for amb in self.ambulances:
            remaining_raw = self._remaining_path_distance(amb)
            ambulances_payload.append(
                {
                    "id": amb.id,
                    "x": amb.position[0],
                    "y": amb.position[1],
                    "idle": amb.idle,
                    "assignment": amb.assignment_id,
                    "path": amb.path[amb.target_index :] if amb.path else [],
                    "remaining_distance": self._display_distance(remaining_raw),
                    "distance_units": units,
                    "eta_minutes": self._eta_minutes(remaining_raw, amb.speed),
                }
            )
        distance_units = "blocks" if self.model.grid_mode else "km"
        distance_value = self.metrics["distance_travelled"]
        display_distance = distance_value if self.model.grid_mode else distance_value / 1000.0
        return {
            "ambulances": ambulances_payload,
            "emergencies": [
                {
                    "id": event.id,
                    "x": event.location[0],
                    "y": event.location[1],
                    "urgency": event.urgency,
                    "resolved": event.resolved_at is not None,
                }
                for event in self.emergencies.values()
            ],
            "metrics": {
                **self.metrics,
                "distance_display": display_distance,
                "distance_units": distance_units,
                "sim_time_minutes": self.sim_time / 60.0,
            },
            "response_history": self.response_history[-200:],
        }

    def active_emergencies(self) -> int:
        return sum(1 for event in self.emergencies.values() if event.resolved_at is None)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _find_ambulance(self, ambulance_id: int) -> Ambulance:
        for ambulance in self.ambulances:
            if ambulance.id == ambulance_id:
                return ambulance
        raise ValueError(f"Ambulance {ambulance_id} not found")

    def _update_response_average(self, latest: float) -> None:
        responses = self.metrics["responses"]
        current_avg = self.metrics["average_response_time"]
        latest_minutes = latest / 60.0
        new_avg = ((responses - 1) * current_avg + latest_minutes) / responses if responses else latest_minutes
        self.metrics["average_response_time"] = new_avg
        self.response_history.append(latest_minutes)

    def _normalize_path(self, ambulance: Ambulance, path: List[Point]) -> List[Point]:
        if not path:
            return []
        if _points_close(path[0], ambulance.position):
            return path
        return [ambulance.position] + path

    def _edge_distance(self, start: Point, end: Point) -> float:
        data = self.model.graph.get_edge_data(start, end)
        if data and "distance" in data:
            return float(data["distance"])
        return math.dist(start, end)

    def _remaining_path_distance(self, ambulance: Ambulance) -> float:
        if not ambulance.path or ambulance.target_index >= len(ambulance.path):
            return 0.0
        idx = ambulance.target_index
        remaining = 0.0
        if idx == 0:
            # Haven't started moving; take full path length
            for i in range(len(ambulance.path) - 1):
                remaining += self._edge_distance(ambulance.path[i], ambulance.path[i + 1])
            return remaining
        prev_node = ambulance.path[idx - 1]
        target_node = ambulance.path[idx]
        segment_total = math.dist(prev_node, target_node)
        edge_length = self._edge_distance(prev_node, target_node)
        if segment_total <= 1e-8:
            residual = 0.0
        else:
            travelled = math.dist(prev_node, ambulance.position)
            progress = min(max(travelled / segment_total, 0.0), 1.0)
            residual = edge_length * (1 - progress)
        remaining += residual
        for i in range(idx, len(ambulance.path) - 1):
            remaining += self._edge_distance(ambulance.path[i], ambulance.path[i + 1])
        return remaining

    def _display_distance(self, raw: float) -> float:
        if raw <= 0:
            return 0.0
        if self.model.grid_mode:
            return raw
        return raw / 1000.0  # meters -> km

    def _eta_minutes(self, raw: float, speed: float) -> float:
        if raw <= 0 or speed <= 0:
            return 0.0
        travel_seconds = raw / speed
        return travel_seconds / 60.0


def _color_for_index(idx: int) -> str:
    return COLOR_CYCLE[(idx - 1) % len(COLOR_CYCLE)]


def _points_close(a: Point, b: Point, eps: float = 1e-4) -> bool:
    return math.dist(a, b) <= eps



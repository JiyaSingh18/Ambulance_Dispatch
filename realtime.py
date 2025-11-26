"""
realtime.py
-----------
Priority-queue based dispatcher that reacts to incoming emergencies in
real time. Combines greedy urgency handling with the global optimizer
to re-balance assignments when new requests arrive and all ambulances
are busy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import time
from typing import Dict, List, Optional, Tuple

from assignment import AssignmentResult, assign_ambulances

Point = Tuple[float, float]


@dataclass(order=True)
class Emergency:
    priority: int
    timestamp: float
    location: Point = field(compare=False)
    id: int = field(compare=False, default=0)
    description: str = field(compare=False, default="General Emergency")


@dataclass
class AmbulanceState:
    id: int
    position: Point
    busy_until: float = 0.0
    destination: Optional[Point] = None

    @property
    def available(self) -> bool:
        return time.time() >= self.busy_until


class RealtimeDispatcher:
    """Maintain live assignment decisions."""

    def __init__(
        self,
        graph,
        ambulances: List[AmbulanceState],
        algorithm: str = "dijkstra",
        method: str = "hungarian",
    ) -> None:
        self.graph = graph
        self.ambulances = ambulances
        self.algorithm = algorithm
        self.method = method
        self.emergency_queue: List[Emergency] = []
        self.active_assignments: Dict[int, int] = {}  # ambulance_id -> emergency_id
        self.assignment_paths: Dict[int, List[Point]] = {}
        self.future_plan: Dict[int, int] = {}  # ambulance_id -> emergency_id planned when everyone busy
        self.planned_paths: Dict[Tuple[int, int], List[Point]] = {}
        self.planned_costs: Dict[Tuple[int, int], float] = {}
        self.emergency_counter = 0
        self.history: List[AssignmentResult] = []

    def add_emergency(self, location: Point, urgency: int) -> Emergency:
        self.emergency_counter += 1
        emergency = Emergency(priority=-urgency, timestamp=time.time(), location=location, id=self.emergency_counter)
        heapq.heappush(self.emergency_queue, emergency)
        self.rebalance_assignments()
        return emergency

    def rebalance_assignments(self) -> None:
        """Assign available ambulances immediately, otherwise perform a global optimization."""
        available = [a for a in self.ambulances if a.available]
        pending = sorted(self.emergency_queue)
        if not pending:
            return
        if not available:
            self._plan_future_assignments(pending)
            return

        assigned_ids: List[int] = []
        # Honor any planned assignments first
        remaining_available = []
        for ambulance in available:
            planned_id = self.future_plan.get(ambulance.id)
            if not planned_id:
                remaining_available.append(ambulance)
                continue
            emergency = next((e for e in pending if e.id == planned_id), None)
            if emergency is None:
                remaining_available.append(ambulance)
                self.future_plan.pop(ambulance.id, None)
                continue
            assigned_ids.append(emergency.id)
            path = self.planned_paths.pop((ambulance.id, emergency.id), None)
            travel_time = self.planned_costs.pop((ambulance.id, emergency.id), None)
            self._commit_assignment(ambulance, emergency, path or [ambulance.position, emergency.location], travel_time)
            self.future_plan.pop(ambulance.id, None)

        pending = [e for e in pending if e.id not in assigned_ids]
        if not pending:
            self._remove_resolved_emergencies(assigned_ids)
            return

        emergency_positions = [e.location for e in pending]
        ambulance_positions = [a.position for a in remaining_available]
        if not ambulance_positions:
            # No one left to assign immediately
            self._remove_resolved_emergencies(assigned_ids)
            return
        result = assign_ambulances(
            self.graph,
            ambulance_positions,
            emergency_positions,
            self.algorithm,
            self.method,
            ambulance_labels=[amb.id for amb in remaining_available],
            emergency_labels=[e.id for e in pending],
        )
        self.history.append(result)
        for idx_in_available, ambulance in enumerate(remaining_available):
            if idx_in_available not in result.assignments:
                continue
            emergency_index = result.assignments[idx_in_available]
            emergency = pending[emergency_index]
            assigned_ids.append(emergency.id)
            path = result.path_lookup[(idx_in_available, emergency_index)]
            travel_time = result.cost_matrix[idx_in_available][emergency_index]
            self._commit_assignment(ambulance, emergency, path, travel_time)
            self.future_plan.pop(ambulance.id, None)
        if assigned_ids:
            self._remove_resolved_emergencies(assigned_ids)

    def release_ambulance(self, ambulance_id: int) -> None:
        """Mark an ambulance as free."""
        if ambulance_id in self.active_assignments:
            self.active_assignments.pop(ambulance_id)
        for ambulance in self.ambulances:
            if ambulance.id == ambulance_id:
                ambulance.busy_until = time.time()
                ambulance.destination = None
                break
        self.rebalance_assignments()

    def snapshot(self) -> Dict:
        """Return serializable state for UI refresh."""
        return {
            "ambulances": [
                {
                    "id": amb.id,
                    "position": amb.position,
                    "destination": amb.destination,
                    "busy_until": amb.busy_until,
                    "available": amb.available,
                }
                for amb in self.ambulances
            ],
            "pending_emergencies": [
                {"id": e.id, "location": e.location, "urgency": -e.priority} for e in sorted(self.emergency_queue)
            ],
            "history": [
                {
                    "method": h.method,
                    "total_cost": h.total_cost,
                    "assignments": h.assignments,
                    "cost_matrix": h.cost_matrix,
                    "paths": h.path_lookup,
                }
                for h in self.history[-5:]
            ],
            "future_plan": self.future_plan.copy(),
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _plan_future_assignments(self, pending: List[Emergency]) -> None:
        emergency_positions = [e.location for e in pending]
        ambulance_positions = [self._projected_position(a) for a in self.ambulances]
        result = assign_ambulances(
            self.graph,
            ambulance_positions,
            emergency_positions,
            self.algorithm,
            self.method,
            ambulance_labels=[amb.id for amb in self.ambulances],
            emergency_labels=[e.id for e in pending],
        )
        for amb_idx, emergency_idx in result.assignments.items():
            ambulance_id = self.ambulances[amb_idx].id
            emergency_id = pending[emergency_idx].id
            self.future_plan[ambulance_id] = emergency_id
            self.planned_paths[(ambulance_id, emergency_id)] = result.path_lookup[(amb_idx, emergency_idx)]
            self.planned_costs[(ambulance_id, emergency_id)] = result.cost_matrix[amb_idx][emergency_idx]
        if result.assignments:
            self.history.append(result)

    def _projected_position(self, ambulance: AmbulanceState) -> Point:
        return ambulance.destination or ambulance.position

    def _remove_resolved_emergencies(self, resolved_ids: List[int]) -> None:
        self.emergency_queue = [e for e in self.emergency_queue if e.id not in resolved_ids]
        heapq.heapify(self.emergency_queue)

    def _commit_assignment(
        self,
        ambulance: AmbulanceState,
        emergency: Emergency,
        path: List[Point],
        travel_time: Optional[float] = None,
    ) -> None:
        self.active_assignments[ambulance.id] = emergency.id
        self.assignment_paths[ambulance.id] = path
        ambulance.destination = emergency.location
        if travel_time is None:
            travel_time = max(len(path) - 1, 1.0)
        ambulance.busy_until = time.time() + travel_time


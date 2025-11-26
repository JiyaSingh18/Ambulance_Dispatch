"""
assignment.py
-------------
Implements ambulance-to-emergency assignment strategies. Includes a
random baseline and the optimal Hungarian Method leveraging SciPy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple
import random

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - SciPy should be installed but fall back gracefully
    linear_sum_assignment = None  # type: ignore

from routing import route_cost_matrix


Assignment = Dict[int, int]


@dataclass
class AssignmentResult:
    assignments: Assignment
    cost_matrix: List[List[float]]
    path_lookup: Dict[Tuple[int, int], List[Tuple[int, int]]]
    total_cost: float
    method: str
    ambulance_labels: List[Any] = field(default_factory=list)
    emergency_labels: List[Any] = field(default_factory=list)


def random_assignment(num_ambulances: int, num_emergencies: int) -> Assignment:
    """Assign emergencies randomly, limited by min(len(ambulances), len(emergencies))."""
    indices = list(range(num_emergencies))
    random.shuffle(indices)
    assignment: Assignment = {}
    for i in range(min(num_ambulances, num_emergencies)):
        assignment[i] = indices[i]
    return assignment


def hungarian_assignment(cost_matrix: Sequence[Sequence[float]]) -> Assignment:
    """Optimal assignment using SciPy's Hungarian algorithm."""
    if linear_sum_assignment is None:
        raise ImportError("SciPy is required for Hungarian assignment.")
    cost_np = np.array(cost_matrix)
    row_ind, col_ind = linear_sum_assignment(cost_np)
    return {int(r): int(c) for r, c in zip(row_ind, col_ind)}


def assign_ambulances(
    graph,
    ambulance_positions: Sequence[Tuple[int, int]],
    emergency_positions: Sequence[Tuple[int, int]],
    algorithm: str = "dijkstra",
    method: str = "hungarian",
    ambulance_labels: Sequence[Any] | None = None,
    emergency_labels: Sequence[Any] | None = None,
) -> AssignmentResult:
    """Orchestrate the cost matrix generation and assignment selection."""
    if not emergency_positions:
        return AssignmentResult({}, [], {}, 0.0, method, list(ambulance_labels or []), list(emergency_labels or []))
    cost_matrix, path_lookup = route_cost_matrix(
        graph, ambulance_positions, emergency_positions, algorithm
    )
    if method == "random":
        assignment = random_assignment(len(ambulance_positions), len(emergency_positions))
    else:
        assignment = hungarian_assignment(cost_matrix)
    total_cost = sum(cost_matrix[i][j] for i, j in assignment.items())
    amb_labels = list(ambulance_labels) if ambulance_labels is not None else list(range(len(ambulance_positions)))
    emergency_labels = list(emergency_labels) if emergency_labels is not None else list(range(len(emergency_positions)))
    return AssignmentResult(assignment, cost_matrix, path_lookup, total_cost, method, amb_labels, emergency_labels)


"""
routing.py
----------
Shortest-path utilities built on top of NetworkX. Supports both Dijkstra
and A* with a Manhattan heuristic, returning the concrete path plus an
aggregate travel time that incorporates traffic multipliers.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import networkx as nx

Point = Tuple[int, int]
PathIndex = Tuple[int, int]


def manhattan_distance(node_a: Point, node_b: Point) -> float:
    """A* heuristic tailored for the grid layout."""
    return abs(node_a[0] - node_b[0]) + abs(node_a[1] - node_b[1])


def compute_shortest_path(
    graph: nx.Graph,
    start: Point,
    end: Point,
    algorithm: str = "dijkstra",
) -> Tuple[List[Point], float]:
    """
    Calculate the shortest route between two points.

    Parameters
    ----------
    graph:
        City graph from graph_model.
    start, end:
        Grid coordinates.
    algorithm:
        One of {"dijkstra", "astar"}.
    """
    if algorithm not in {"dijkstra", "astar"}:
        raise ValueError("algorithm must be 'dijkstra' or 'astar'")

    if algorithm == "dijkstra":
        path = nx.dijkstra_path(graph, start, end, weight="weight")
        length = nx.dijkstra_path_length(graph, start, end, weight="weight")
    else:
        path = nx.astar_path(
            graph,
            start,
            end,
            heuristic=manhattan_distance,
            weight="weight",
        )
        length = nx.astar_path_length(
            graph,
            start,
            end,
            heuristic=manhattan_distance,
            weight="weight",
        )
    return list(path), float(length)


def route_cost_matrix(
    graph: nx.Graph,
    ambulance_positions: Sequence[Point],
    emergency_positions: Sequence[Point],
    algorithm: str = "dijkstra",
) -> Tuple[List[List[float]], Dict[PathIndex, List[Point]]]:
    """Build a cost matrix representing travel time from each ambulance to each emergency."""
    matrix: List[List[float]] = []
    path_map: Dict[PathIndex, List[Point]] = {}
    for amb_idx, amb in enumerate(ambulance_positions):
        row = []
        for emergency_idx, emergency in enumerate(emergency_positions):
            path, cost = compute_shortest_path(graph, amb, emergency, algorithm)
            row.append(cost)
            path_map[(amb_idx, emergency_idx)] = path
        matrix.append(row)
    return matrix, path_map


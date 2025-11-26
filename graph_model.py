"""
Graph construction and management utilities for the Emergency Ambulance Dispatch
Optimization project.

The `GraphModel` class builds either the default 10x10 city grid or a real-world graph
downloaded from OpenStreetMap (via OSMnx). Each edge stores a base distance multiplied
by a traffic multiplier in the [1, 5] range.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np

GridNode = Tuple[float, float]

try:  # Lazy optional dependency
    import osmnx as ox  # type: ignore
except ImportError:  # pragma: no cover
    ox = None  # type: ignore


@dataclass
class Scenario:
    """Generated ambulances and emergencies for simulations or tests."""

    ambulances: List[GridNode]
    emergencies: List[GridNode]
    urgency: List[int]


class GraphModel:
    """
    Encapsulates a road network with per-edge traffic multipliers.

    By default it generates a 10×10 grid, but `from_osm` can be used to build a graph from
    real-world data (e.g., Mumbai).
    """

    def __init__(
        self,
        size: int = 10,
        traffic_seed: int | None = None,
        base_graph: nx.Graph | None = None,
        positions: Dict[GridNode, Tuple[float, float]] | None = None,
    ) -> None:
        self.size = size
        self.random = random.Random(traffic_seed)
        self.traffic: Dict[Tuple[GridNode, GridNode], float] = {}
        self.metadata: Dict[str, str] = {"mode": "grid"}
        if base_graph is None:
            self.graph = nx.grid_graph(dim=[size, size])
            self.positions = {node: (float(node[0]), float(node[1])) for node in self.graph.nodes}
            self.grid_mode = True
        else:
            self.graph = base_graph
            self.positions = positions or {node: (_to_pair(node)) for node in self.graph.nodes}
            self.grid_mode = False
            self.metadata["mode"] = "osm"
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for u, v, data in self.graph.edges(data=True):
            distance = data.get("distance") or data.get("length") or _euclidean(self.positions[u], self.positions[v])
            traffic = data.get("traffic") or self.random.uniform(1.0, 5.0)
            weight = float(distance) * float(traffic)
            data["distance"] = float(distance)
            data["traffic"] = float(traffic)
            data["weight"] = weight
            self.traffic[(u, v)] = float(traffic)
            self.traffic[(v, u)] = float(traffic)

    @property
    def nodes(self) -> List[GridNode]:
        return list(self.graph.nodes)

    def random_node(self) -> GridNode:
        return self.random.choice(self.nodes)

    def closest_node(self, target: GridNode) -> GridNode:
        """Return the existing graph node closest to the provided coordinates."""
        tx, ty = float(target[0]), float(target[1])
        best_node = None
        best_distance = float("inf")
        for node in self.nodes:
            nx, ny = float(node[0]), float(node[1])
            dist = (nx - tx) ** 2 + (ny - ty) ** 2
            if dist < best_distance:
                best_distance = dist
                best_node = node
        if best_node is None:
            raise ValueError("Graph has no nodes to match against")
        return best_node

    def generate_scenario(
        self,
        ambulances: int = 3,
        emergencies: int = 5,
    ) -> Scenario:
        amb_positions = self._unique_random_nodes(ambulances)
        emergency_positions = self._unique_random_nodes(emergencies)
        urgency = [self.random.randint(1, 5) for _ in emergency_positions]
        return Scenario(amb_positions, emergency_positions, urgency)

    def _unique_random_nodes(self, count: int) -> List[GridNode]:
        choices = self.nodes[:]
        self.random.shuffle(choices)
        return choices[:count]

    def update_random_traffic(self, delta: float = 0.5) -> None:
        for u, v in self.graph.edges:
            change = self.random.uniform(-delta, delta)
            traffic = self.graph.edges[u, v]["traffic"] + change
            traffic = float(np.clip(traffic, 1.0, 5.0))
            self.graph.edges[u, v]["traffic"] = traffic
            self.graph.edges[u, v]["weight"] = traffic
            self.traffic[(u, v)] = traffic
            self.traffic[(v, u)] = traffic

    def weighted_shortest_path(
        self, source: GridNode, target: GridNode, algorithm: str = "dijkstra"
    ) -> Tuple[List[GridNode], float]:
        if algorithm not in {"dijkstra", "astar"}:
            raise ValueError("algorithm must be 'dijkstra' or 'astar'")
        if algorithm == "dijkstra":
            path = nx.shortest_path(
                self.graph, source=source, target=target, weight="weight"
            )
        else:
            heuristic = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])
            path = nx.astar_path(
                self.graph,
                source=source,
                target=target,
                heuristic=heuristic,
                weight="weight",
            )
        travel_time = sum(
            self.graph.edges[path[i], path[i + 1]]["weight"]
            for i in range(len(path) - 1)
        )
        return path, travel_time

    def to_heatmap(self, resolution: int = 20) -> np.ndarray:
        if self.grid_mode:
            heatmap = np.zeros((self.size, self.size))
            for (x, y) in self.graph.nodes:
                neighbors = list(self.graph.neighbors((x, y)))
                if not neighbors:
                    continue
                avg = np.mean([self.graph.edges[(x, y), nbr]["traffic"] for nbr in neighbors])
                heatmap[int(x), int(y)] = avg
            return heatmap

        coords = np.array(list(self.positions.values()))
        xs, ys = coords[:, 0], coords[:, 1]
        x_bins = np.linspace(xs.min(), xs.max(), resolution + 1)
        y_bins = np.linspace(ys.min(), ys.max(), resolution + 1)
        heatmap = np.zeros((resolution, resolution))
        counts = np.zeros((resolution, resolution))
        for (u, v), traffic in self.traffic.items():
            x = (self.positions[u][0] + self.positions[v][0]) / 2
            y = (self.positions[u][1] + self.positions[v][1]) / 2
            xi = np.searchsorted(x_bins, x, side="right") - 1
            yi = np.searchsorted(y_bins, y, side="right") - 1
            if 0 <= xi < resolution and 0 <= yi < resolution:
                heatmap[xi, yi] += traffic
                counts[xi, yi] += 1
        counts[counts == 0] = 1
        return heatmap / counts

    def edge_traffic_dict(self) -> Dict[str, float]:
        readable: Dict[str, float] = {}
        for (u, v), traffic in self.traffic.items():
            key = f"{u[0]}_{u[1]}->{v[0]}_{v[1]}"
            readable[key] = traffic
        return readable

    @classmethod
    def from_osm(
        cls,
        place: str,
        network_type: str = "drive",
        traffic_seed: int | None = None,
    ) -> GraphModel:
        ox_module = _require_osmnx()
        raw = ox_module.graph_from_place(place, network_type=network_type)
        undirected = _to_simple_undirected(raw)
        mapping = {
            node: (round(data.get("x", 0.0), 5), round(data.get("y", 0.0), 5))
            for node, data in raw.nodes(data=True)
        }
        relabeled = nx.relabel_nodes(undirected, mapping, copy=True)
        positions = {coord: coord for coord in mapping.values()}
        model = cls(
            size=max(int(math.sqrt(len(relabeled))) or 10, 10),
            traffic_seed=traffic_seed,
            base_graph=relabeled,
            positions=positions,
        )
        model.metadata["source"] = place
        model.metadata["network_type"] = network_type
        return model


def demo_graph(seed: int | None = 7) -> GraphModel:
    return GraphModel(size=10, traffic_seed=seed)


def _euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _to_pair(node: Iterable[float]) -> Tuple[float, float]:
    if isinstance(node, tuple) and len(node) == 2:
        return float(node[0]), float(node[1])
    return (float(node), float(node))


def _require_osmnx():
    if ox is None:  # pragma: no cover
        raise ImportError(
            "OSMnx is required for real-city graphs. Install it via `pip install osmnx`."
        )
    return ox


def _to_simple_undirected(graph: nx.Graph) -> nx.Graph:
    """
    Convert a possibly directed, multi-edge graph into a simple undirected graph
    keeping the shortest edge distance between any two nodes.
    """
    simple = nx.Graph()
    for u, v, data in graph.edges(data=True):
        distance = float(data.get("length") or data.get("distance") or 1.0)
        if simple.has_edge(u, v):
            if distance < simple[u][v]["distance"]:
                simple[u][v]["distance"] = distance
        else:
            simple.add_edge(u, v, distance=distance)
    # Carry over isolated nodes so we keep coordinates
    for node in graph.nodes:
        if node not in simple:
            simple.add_node(node)
    return simple


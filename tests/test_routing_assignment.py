import numpy as np

from assignment import assign_ambulances
from graph_model import GraphModel
from routing import compute_shortest_path, route_cost_matrix


def test_compute_shortest_path_matches_cost_matrix():
    model = GraphModel(traffic_seed=21)
    start, end = (0, 0), (3, 3)
    path, cost = compute_shortest_path(model.graph, start, end, "dijkstra")
    assert path[0] == start and path[-1] == end
    matrix, _ = route_cost_matrix(model.graph, [start], [end], "dijkstra")
    assert np.isclose(matrix[0][0], cost)


def test_assignment_returns_mapping():
    model = GraphModel(traffic_seed=11)
    scenario = model.generate_scenario(ambulances=3, emergencies=3)
    result = assign_ambulances(
        model.graph,
        scenario.ambulances,
        scenario.emergencies,
        algorithm="astar",
        method="hungarian",
    )
    assert len(result.assignments) == 3
    assert result.total_cost > 0
    assert (0, 0) in result.path_lookup


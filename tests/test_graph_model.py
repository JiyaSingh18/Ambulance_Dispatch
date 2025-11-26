import math

from graph_model import GraphModel


def test_graph_has_100_nodes():
    model = GraphModel(traffic_seed=1)
    assert len(model.graph.nodes) == 100
    assert len(model.graph.edges) > 0


def test_scenario_generation_counts():
    model = GraphModel(traffic_seed=2)
    scenario = model.generate_scenario(ambulances=3, emergencies=5)
    assert len(scenario.ambulances) == 3
    assert len(scenario.emergencies) == 5
    assert len(scenario.urgency) == 5


def test_weighted_shortest_path_cost_positive():
    model = GraphModel(traffic_seed=3)
    src, dst = (0, 0), (5, 5)
    path, travel_time = model.weighted_shortest_path(src, dst, "dijkstra")
    assert path[0] == src
    assert path[-1] == dst
    assert travel_time > 0
    # A* should yield the same nodes but potentially different expansion order
    astar_path, astar_time = model.weighted_shortest_path(src, dst, "astar")
    assert astar_path[0] == src
    assert astar_path[-1] == dst
    assert math.isclose(astar_time, travel_time, rel_tol=1e-6)


def test_heatmap_shape():
    model = GraphModel(traffic_seed=4)
    heatmap = model.to_heatmap()
    assert heatmap.shape == (model.size, model.size)


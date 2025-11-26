import time

from realtime import AmbulanceState, RealtimeDispatcher
from graph_model import GraphModel


def test_dispatcher_assigns_when_available():
    model = GraphModel(traffic_seed=5)
    ambulances = [AmbulanceState(id=1, position=(0, 0))]
    dispatcher = RealtimeDispatcher(model.graph, ambulances)
    dispatcher.add_emergency((1, 1), urgency=5)
    dispatcher.rebalance_assignments()
    snapshot = dispatcher.snapshot()
    assert snapshot["pending_emergencies"] == []
    assert dispatcher.active_assignments[1] >= 1


def test_future_plan_when_busy():
    model = GraphModel(traffic_seed=6)
    ambulance = AmbulanceState(id=1, position=(0, 0))
    ambulance.busy_until = time.time() + 10
    dispatcher = RealtimeDispatcher(model.graph, [ambulance])
    dispatcher.add_emergency((3, 3), urgency=4)
    dispatcher.rebalance_assignments()
    assert dispatcher.future_plan.get(1) is not None


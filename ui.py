"""
ui.py
-----
Streamlit dashboard that stitches together the graph model, routing,
assignment, analytics, and simulation layers into a cohesive demo.
"""
from __future__ import annotations

import json
import math
import random
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from graph_model import GraphModel, GridNode
from realtime import AmbulanceState, RealtimeDispatcher
from simulation import SimulationEngine

Point = Tuple[int, int]

st.set_page_config(
    page_title="Emergency Ambulance Dispatch Optimization",
    layout="wide",
    page_icon="🚑",
)

MAP_CHOICES = ["Grid (10x10)", "Mumbai (OSM)"]
DEFAULT_MAP = MAP_CHOICES[1]
MUMBAI_AMBULANCE_COUNT = 91


def _coerce_point(point: Any) -> Tuple[float, float]:
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        return float(point[0]), float(point[1])
    try:
        value = float(point)
    except (TypeError, ValueError):
        value = 0.0
    return value, value


def _coordinate_matrix(model: GraphModel) -> np.ndarray:
    positions = getattr(model, "positions", None) or {}
    if not positions:
        return np.array([[0.0, 0.0]], dtype=float)
    coords = np.array([_coerce_point(pos) for pos in positions.values()], dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 2)
    return coords


def _road_geometry(model: GraphModel) -> Tuple[List[float], List[float]]:
    lons: List[float] = []
    lats: List[float] = []
    positions = getattr(model, "positions", None) or {}
    if not positions or not model.graph.edges:
        return lons, lats
    for u, v in model.graph.edges:
        start = positions.get(u)
        end = positions.get(v)
        if start is None or end is None:
            continue
        sx, sy = _coerce_point(start)
        ex, ey = _coerce_point(end)
        lons.extend([sx, ex, None])
        lats.extend([sy, ey, None])
    return lons, lats


def _manual_emergency_node() -> GridNode | None:
    if not st.session_state.get("manual_emergency"):
        return None
    lat = st.session_state.get("manual_emergency_lat")
    lon = st.session_state.get("manual_emergency_lon")
    if lat is None or lon is None:
        return None
    model: GraphModel = st.session_state.model
    try:
        target = (float(lon), float(lat))
    except (TypeError, ValueError):
        st.warning("Invalid manual emergency coordinates; using random location instead.")
        return None
    return model.closest_node(target)


def _log_metric_history(sim_snapshot: Dict) -> None:
    metrics = sim_snapshot["metrics"]
    minute_mark = metrics.get("sim_time_minutes", 0.0)
    distance_history = st.session_state.setdefault("distance_history", [])
    distance_history.append(
        {"Minutes": minute_mark, "Distance": metrics.get("distance_display", 0.0)}
    )
    if len(distance_history) > 300:
        distance_history.pop(0)
    active_history = st.session_state.setdefault("active_emergencies_history", [])
    active_history.append({"Minutes": minute_mark, "Active": st.session_state.sim.active_emergencies()})
    if len(active_history) > 300:
        active_history.pop(0)


# ------------------------------------------------------------------ #
# Session bootstrap
# ------------------------------------------------------------------ #
def init_state() -> None:
    if st.session_state.get("initialized"):
        return
    st.session_state.map_source = st.session_state.get("map_source", DEFAULT_MAP)
    st.session_state.route_algo = "dijkstra"
    st.session_state.assignment_method = "hungarian"
    st.session_state.speed = "Real-time"
    st.session_state.running = False
    st.session_state.show_paths = True
    st.session_state.map_zoom = st.session_state.get("map_zoom")
    st.session_state.manual_emergency = st.session_state.get("manual_emergency", False)
    st.session_state.manual_emergency_lat = st.session_state.get("manual_emergency_lat", 19.076)
    st.session_state.manual_emergency_lon = st.session_state.get("manual_emergency_lon", 72.877)
    st.session_state.manual_emergency_urgency = st.session_state.get("manual_emergency_urgency", 3)
    st.session_state.emergency_lookup: Dict[int, Dict] = {}
    set_environment(st.session_state.map_source, st.session_state.route_algo, st.session_state.assignment_method)
    st.session_state.initialized = True


def set_environment(map_source: str, route_algo: str, assignment_method: str) -> None:
    with st.spinner(f"Loading {map_source} network..."):
        model = load_model(map_source)
    sim = SimulationEngine(model, algorithm=route_algo)
    ambulance_count = MUMBAI_AMBULANCE_COUNT if map_source == "Mumbai (OSM)" else 6
    scenario = model.generate_scenario(ambulances=ambulance_count)
    sim.spawn_ambulances(positions=scenario.ambulances)
    dispatcher = build_dispatcher(model, sim, route_algo, assignment_method)
    st.session_state.model = model
    st.session_state.sim = sim
    st.session_state.dispatcher = dispatcher
    st.session_state.emergency_lookup = {}
    st.session_state.running = False


def load_model(map_source: str) -> GraphModel:
    if map_source == "Mumbai (OSM)":
        try:
            return GraphModel.from_osm("Mumbai, India", traffic_seed=7)
        except Exception as exc:  # pragma: no cover - depends on network/env
            st.error(f"Failed to load Mumbai graph ({exc}). Falling back to grid.")
            return GraphModel(traffic_seed=7)
    return GraphModel(traffic_seed=7)


def serialize_snapshot(obj: Any) -> Any:
    """Convert tuples/keys into JSON-serializable primitives."""
    if isinstance(obj, dict):
        return {str(k): serialize_snapshot(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize_snapshot(item) for item in obj]
    return obj


def build_dispatcher(model: GraphModel, sim: SimulationEngine, algo: str, method: str) -> RealtimeDispatcher:
    states = [
        AmbulanceState(
            id=amb.id,
            position=(amb.position[0], amb.position[1]),
        )
        for amb in sim.ambulances
    ]
    dispatcher = RealtimeDispatcher(model.graph, states, algorithm=algo, method=method)
    return dispatcher


# ------------------------------------------------------------------ #
# Simulation helpers
# ------------------------------------------------------------------ #
def sync_dispatcher_with_sim() -> None:
    sim: SimulationEngine = st.session_state.sim
    dispatcher: RealtimeDispatcher = st.session_state.dispatcher
    for state in dispatcher.ambulances:
        sim_amb = next(a for a in sim.ambulances if a.id == state.id)
        state.position = (sim_amb.position[0], sim_amb.position[1])
        if sim_amb.assignment_id is None:
            state.destination = None
            state.busy_until = time.time()


def apply_dispatcher_paths_to_sim() -> None:
    dispatcher: RealtimeDispatcher = st.session_state.dispatcher
    sim: SimulationEngine = st.session_state.sim
    lookup = st.session_state.emergency_lookup
    for amb_id, emergency_id in dispatcher.active_assignments.items():
        target = lookup.get(emergency_id)
        path = dispatcher.assignment_paths.get(amb_id)
        if not target or not path:
            continue
        sim_event_id = target["sim_id"]
        sim_amb = next(a for a in sim.ambulances if a.id == amb_id)
        if sim_amb.assignment_id == sim_event_id:
            continue
        sim.assign_path(amb_id, path, sim_event_id)


def add_emergency(location: GridNode | None = None, urgency: int | None = None) -> None:
    model: GraphModel = st.session_state.model
    sim: SimulationEngine = st.session_state.sim
    dispatcher: RealtimeDispatcher = st.session_state.dispatcher
    urgency = urgency or random.randint(1, 5)
    location = location or model.random_node()
    emergency = dispatcher.add_emergency(location, urgency)
    sim_event = sim.create_emergency(location=location, urgency=urgency)
    st.session_state.emergency_lookup[emergency.id] = {
        "sim_id": sim_event.id,
        "location": location,
        "urgency": urgency,
    }


def cleanup_resolved_emergencies() -> None:
    sim: SimulationEngine = st.session_state.sim
    resolved_sim_ids = {
        event.id for event in sim.emergencies.values() if event.resolved_at is not None
    }
    for dispatch_id, info in list(st.session_state.emergency_lookup.items()):
        if info["sim_id"] in resolved_sim_ids:
            st.session_state.emergency_lookup.pop(dispatch_id)


def run_steps(iterations: int = 15) -> None:
    sim: SimulationEngine = st.session_state.sim
    dispatcher: RealtimeDispatcher = st.session_state.dispatcher
    dt = 0.25 if st.session_state.speed == "Real-time" else 0.4
    sleep_time = 0.04 if st.session_state.speed == "Real-time" else 0.005
    for _ in range(iterations):
        sync_dispatcher_with_sim()
        dispatcher.rebalance_assignments()
        apply_dispatcher_paths_to_sim()
        completed = sim.tick(dt)
        for amb_id in completed:
            dispatcher.release_ambulance(amb_id)
        cleanup_resolved_emergencies()
        time.sleep(sleep_time)


def add_ambulance_unit() -> None:
    sim: SimulationEngine = st.session_state.sim
    dispatcher: RealtimeDispatcher = st.session_state.dispatcher
    ambulance = sim.add_ambulance()
    dispatcher.ambulances.append(
        AmbulanceState(
            id=ambulance.id,
            position=(ambulance.position[0], ambulance.position[1]),
        )
    )
    st.toast(f"Ambulance {ambulance.id} added near {tuple(round(v, 2) for v in ambulance.position)}", icon="🚑")


# ------------------------------------------------------------------ #
# Rendering helpers
# ------------------------------------------------------------------ #
def render_grid(snapshot: Dict, show_paths: bool) -> go.Figure:
    model: GraphModel = st.session_state.model
    coords = _coordinate_matrix(model)
    node_xs = coords[:, 0]
    node_ys = coords[:, 1]
    has_nodes = node_xs.size > 0

    if not model.grid_mode:
        center_lat = float(node_ys.mean()) if has_nodes else 19.076
        center_lon = float(node_xs.mean()) if has_nodes else 72.877
        fig = go.Figure()
        road_lons, road_lats = _road_geometry(model)
        if road_lons and road_lats:
            fig.add_trace(
                go.Scattermapbox(
                    lon=road_lons,
                    lat=road_lats,
                    mode="lines",
                    line=dict(color="#1f2937", width=1),
                    name="Road Network",
                    hoverinfo="skip",
                    opacity=0.55,
                )
            )
        fig.add_trace(
            go.Scattermapbox(
                lon=node_xs,
                lat=node_ys,
                mode="markers",
                marker=dict(size=3, color="#4b5563"),
                name="Intersections",
                hoverinfo="skip",
            )
        )
        for amb in snapshot["ambulances"]:
            hover = (
                f"Ambulance {amb['id']}<br>"
                f"{'Idle' if amb['idle'] else 'Responding'}<br>"
                f"Remaining: {amb['remaining_distance']:.2f} {amb['distance_units']}<br>"
                f"ETA: {amb['eta_minutes']:.1f} min"
            )
            fig.add_trace(
                go.Scattermapbox(
                    lon=[amb["x"]],
                    lat=[amb["y"]],
                    mode="markers+text",
                    marker=dict(size=18, color="#2b83ba"),
                    text=[f"🚑 {amb['id']}"],
                    textposition="top center",
                    hovertext=hover,
                    hoverinfo="text",
                    name=f"Ambulance {amb['id']}",
                )
            )
            if show_paths and amb["path"]:
                lons = [amb["x"]] + [node[0] for node in amb["path"]]
                lats = [amb["y"]] + [node[1] for node in amb["path"]]
                fig.add_trace(
                    go.Scattermapbox(
                        lon=lons,
                        lat=lats,
                        mode="lines",
                        line=dict(color="#1f78b4", width=4),
                        name=f"Route {amb['id']}",
                        hoverinfo="text",
                        hovertext=[
                            (
                                f"Ambulance {amb['id']} route<br>"
                                f"Remaining: {amb['remaining_distance']:.2f} {amb['distance_units']}<br>"
                                f"ETA: {amb['eta_minutes']:.1f} min"
                            )
                        ]
                        * len(lons),
                        showlegend=False,
                    )
                )
        for emergency in snapshot["emergencies"]:
            color = "#d7191c" if not emergency["resolved"] else "#1b9e77"
            fig.add_trace(
                go.Scattermapbox(
                    lon=[emergency["x"]],
                    lat=[emergency["y"]],
                    mode="markers+text",
                    marker=dict(
                        size=16,
                        color=color,
                        symbol="circle",
                        opacity=0.95,
                    ),
                    text=[f"⚠️{emergency['urgency']}"],
                    textposition="bottom center",
                    name=f"Emergency {emergency['id']}",
                )
            )
        lat_span = float(node_ys.max() - node_ys.min()) if has_nodes else 0.02
        lon_span = float(node_xs.max() - node_xs.min()) if has_nodes else 0.02
        zoom_level = st.session_state.map_zoom if st.session_state.get("map_zoom") is not None else _estimate_zoom(lat_span, lon_span)
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=zoom_level,
            ),
            height=540,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#0f172a",
            font=dict(color="#f8fafc"),
        )
        return fig

    padding_x = (node_xs.max() - node_xs.min()) * 0.05 if has_nodes else 1
    padding_y = (node_ys.max() - node_ys.min()) * 0.05 if has_nodes else 1
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=node_xs,
            y=node_ys,
            mode="markers",
            marker=dict(size=4, color="#7f8c8d"),
            name="Intersections",
        )
    )
    for amb in snapshot["ambulances"]:
        hover = (
            f"Ambulance {amb['id']}<br>"
            f"{'Idle' if amb['idle'] else 'Responding'}<br>"
            f"Remaining: {amb['remaining_distance']:.2f} {amb['distance_units']}<br>"
            f"ETA: {amb['eta_minutes']:.1f} min"
        )
        fig.add_trace(
            go.Scatter(
                x=[amb["x"]],
                y=[amb["y"]],
                mode="markers+text",
                marker=dict(size=18, color="#2b83ba"),
                text=[f"🚑 {amb['id']}"],
                textposition="top center",
                name=f"Ambulance {amb['id']}",
                hovertext=hover,
                hoverinfo="text",
            )
        )
        if show_paths and amb["path"]:
            path_xs = [amb["x"]] + [node[0] for node in amb["path"]]
            path_ys = [amb["y"]] + [node[1] for node in amb["path"]]
            fig.add_trace(
                go.Scatter(
                    x=path_xs,
                    y=path_ys,
                    mode="lines",
                    line=dict(color="#1f78b4", width=3),
                    showlegend=False,
                )
            )
    for emergency in snapshot["emergencies"]:
        color = "#d7191c" if not emergency["resolved"] else "#1b9e77"
        fig.add_trace(
            go.Scatter(
                x=[emergency["x"]],
                y=[emergency["y"]],
                mode="markers+text",
                marker=dict(
                    size=18,
                    color=color,
                    symbol="circle",
                    line=dict(color="#ffffff", width=1),
                ),
                text=[f"⚠️{emergency['urgency']}"],
                textposition="bottom center",
                name=f"Emergency {emergency['id']}",
            )
        )
    fig.update_layout(
        xaxis=dict(
            range=[node_xs.min() - padding_x, node_xs.max() + padding_x] if has_nodes else [-1, 10],
            showgrid=False,
            zeroline=False,
            title="East-West (km)",
        ),
        yaxis=dict(
            range=[node_ys.min() - padding_y, node_ys.max() + padding_y] if has_nodes else [-1, 10],
            showgrid=False,
            zeroline=False,
            title="North-South (km)",
        ),
        height=540,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="#f8fafc"),
    )
    return fig


def _estimate_zoom(lat_span: float, lon_span: float) -> float:
    span = max(lat_span, lon_span, 0.01)
    if span < 0.01:
        return 14
    if span < 0.03:
        return 13
    if span < 0.08:
        return 12
    if span < 0.15:
        return 11
    if span < 0.3:
        return 10
    return 9


def controls_panel() -> None:
    st.markdown("### 🎛️ Controls Panel")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    if col1.button("Add Ambulance", use_container_width=True):
        add_ambulance_unit()
    if col2.button("Add Batch Emergencies", use_container_width=True):
        manual_location = _manual_emergency_node()
        urgency_override = st.session_state.manual_emergency_urgency if st.session_state.manual_emergency else None
        for _ in range(4):
            add_emergency(location=manual_location, urgency=urgency_override)
    if col3.button("Increase Traffic", use_container_width=True):
        st.session_state.model.update_random_traffic(1.5)
        st.toast("Traffic congestion increased", icon="🚦")
    if col4.button("Reset Simulation", use_container_width=True):
        st.session_state.clear()
        init_state()
        st.toast("Simulation reset", icon="♻️")
    if col5.button("Show Paths" if not st.session_state.show_paths else "Hide Paths", use_container_width=True):
        st.session_state.show_paths = not st.session_state.show_paths
    with col6:
        snapshot = st.session_state.dispatcher.snapshot()
        payload = serialize_snapshot(snapshot)
        st.download_button(
            "Export Report",
            data=json.dumps(payload, indent=2),
            file_name="dispatch_report.json",
            mime="application/json",
            use_container_width=True,
        )
    st.divider()
    st.markdown("#### Custom Emergency Placement")
    manual = st.checkbox("Place emergencies manually", value=st.session_state.manual_emergency)
    st.session_state.manual_emergency = manual
    custom_cols = st.columns([1, 1, 1])
    st.session_state.manual_emergency_lat = custom_cols[0].number_input(
        "Latitude", value=float(st.session_state.manual_emergency_lat), format="%.6f"
    )
    st.session_state.manual_emergency_lon = custom_cols[1].number_input(
        "Longitude", value=float(st.session_state.manual_emergency_lon), format="%.6f"
    )
    st.session_state.manual_emergency_urgency = int(
        custom_cols[2].slider(
            "Urgency", min_value=1, max_value=5, value=int(st.session_state.manual_emergency_urgency)
        )
    )


def render_metrics(dispatch_snapshot: Dict, sim_snapshot: Dict) -> None:
    history = st.session_state.dispatcher.history
    latest = history[-1] if history else None
    metrics = sim_snapshot["metrics"]
    response_time = metrics["average_response_time"]
    distance_units = "blocks" if st.session_state.model.grid_mode else "km"
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Response Time", f"{response_time:.2f} min")
    distance_display = metrics.get("distance_display", 0.0)
    col2.metric("Distance Covered", f"{distance_display:.1f} {distance_units}")
    col3.metric("Active Emergencies", st.session_state.sim.active_emergencies())
    available = sum(1 for amb in dispatch_snapshot["ambulances"] if amb["available"])
    col4.metric("Available Ambulances", f"{available}/{len(dispatch_snapshot['ambulances'])}")
    if latest:
        st.caption(
            f"Latest optimization: {latest.method.title()} (Cost {latest.total_cost:.2f}, "
            f"{len(latest.assignments)} pairs)"
        )


def render_ambulance_status(sim_snapshot: Dict) -> None:
    ambulances = sim_snapshot.get("ambulances", [])
    if not ambulances:
        st.info("No ambulances deployed yet.")
        return
    model: GraphModel = st.session_state.model
    rows = []
    for amb in ambulances:
        status = "Idle" if amb["idle"] else "Responding"
        destination = amb["path"][-1] if amb["path"] else None
        if destination:
            if model.grid_mode:
                destination_str = f"{destination[0]:.1f}, {destination[1]:.1f}"
            else:
                destination_str = f"{destination[1]:.3f}°, {destination[0]:.3f}°"
        else:
            destination_str = "—"
        rows.append(
            {
                "Ambulance": f"#{amb['id']}",
                "Status": status,
                "Destination": destination_str,
                "Remaining Distance": f"{amb['remaining_distance']:.2f} {amb['distance_units']}",
                "ETA (min)": f"{amb['eta_minutes']:.1f}",
            }
        )
    df = pd.DataFrame(rows)
    st.subheader("Live Unit Status")
    st.dataframe(df, hide_index=True, use_container_width=True)


def render_assignment_table() -> None:
    dispatcher: RealtimeDispatcher = st.session_state.dispatcher
    active_lookup = st.session_state.emergency_lookup

    st.subheader("Live Active Assignments")
    active_rows = []
    for amb_id, emergency_id in sorted(dispatcher.active_assignments.items()):
        details = active_lookup.get(emergency_id)
        urgency = details["urgency"] if details else "—"
        coords = details["location"] if details else None
        if coords:
            if st.session_state.model.grid_mode:
                coord_str = f"{coords[0]:.1f}, {coords[1]:.1f}"
            else:
                coord_str = f"{coords[1]:.3f}°, {coords[0]:.3f}°"
        else:
            coord_str = "—"
        active_rows.append(
            {
                "Ambulance": f"#{amb_id}",
                "Emergency": emergency_id,
                "Urgency": urgency,
                "Location": coord_str,
            }
        )
    if active_rows:
        st.dataframe(pd.DataFrame(active_rows), hide_index=True, use_container_width=True)
    else:
        st.info("No ambulances currently en route.")

    history = dispatcher.history
    if not history:
        st.info("Assignments will appear once the simulation starts.")
        return
    latest = history[-1]
    if not latest.cost_matrix:
        st.warning("Waiting for sufficient emergencies to build a cost matrix.")
        return
    cost_df = pd.DataFrame(latest.cost_matrix, columns=[f"E{j+1}" for j in range(len(latest.cost_matrix[0]))])
    cost_df.index = [f"A{i+1}" for i in range(len(cost_df))]
    st.subheader(f"Cost Matrix ({latest.method.title()} strategy)")
    st.dataframe(cost_df.style.highlight_min(axis=1, color="#14532d"))
    amb_labels = latest.ambulance_labels or []
    emg_labels = latest.emergency_labels or []

    def _resolve_label(labels: List[Any], idx: int, prefix: str) -> str:
        if labels and 0 <= idx < len(labels):
            return str(labels[idx])
        return f"{prefix}{idx + 1}"

    pair_rows = []
    for amb_idx, emg_idx in latest.assignments.items():
        amb_label = _resolve_label(amb_labels, amb_idx, "A")
        emg_label = _resolve_label(emg_labels, emg_idx, "E")
        cost = latest.cost_matrix[amb_idx][emg_idx]
        pair_rows.append(
            {
                "Ambulance": f"#{amb_label}",
                "Emergency": emg_label,
                "Travel Cost": f"{cost:.1f}",
            }
        )
    if pair_rows:
        st.write("Optimal pairing (per most recent optimization window):")
        st.dataframe(pd.DataFrame(pair_rows), hide_index=True, use_container_width=True)
    else:
        st.info("No optimal pairs computed yet—waiting on additional emergencies or free ambulances.")


def analytics_tab(sim_snapshot: Dict) -> None:
    st.subheader("Response Time Trend")
    response_history = sim_snapshot.get("response_history", [])
    if response_history:
        response_df = pd.DataFrame(
            {"Response #": range(1, len(response_history) + 1), "Minutes": response_history}
        )
        st.line_chart(response_df.set_index("Response #"))
    else:
        st.info("No completed responses yet. Once ambulances resolve events, trends appear here.")

    st.subheader("Distance Covered Over Time")
    distance_history = st.session_state.get("distance_history", [])
    if distance_history:
        distance_df = pd.DataFrame(distance_history)
        st.area_chart(distance_df.set_index("Minutes"))
    else:
        st.info("Distance history populates automatically as the simulation runs.")

    st.subheader("Active Emergencies Over Time")
    active_history = st.session_state.get("active_emergencies_history", [])
    if active_history:
        active_df = pd.DataFrame(active_history)
        st.line_chart(active_df.set_index("Minutes"))
    else:
        st.info("Start the simulation to see active emergency counts.")

    dispatcher_history = st.session_state.dispatcher.history[-20:]
    if dispatcher_history:
        st.subheader("Recent Optimization Costs")
        total_runs = len(st.session_state.dispatcher.history)
        start_idx = total_runs - len(dispatcher_history) + 1
        iterations = list(range(start_idx, start_idx + len(dispatcher_history)))
        data = {
            "Iteration": iterations,
            "Total Cost": [entry.total_cost for entry in dispatcher_history],
            "Pairs": [len(entry.assignments) for entry in dispatcher_history],
        }
        df = pd.DataFrame(data)
        st.bar_chart(df.set_index("Iteration")[["Total Cost"]])
        st.caption(
            f"Average pairs per run: {sum(data['Pairs']) / len(data['Pairs']):.2f} | "
            f"Min cost: {min(data['Total Cost']):.1f}"
        )
    else:
        st.info("Optimization history will appear after the dispatcher assigns ambulances at least once.")


def sidebar_controls() -> None:
    st.sidebar.title("Simulation Controls")
    map_idx = MAP_CHOICES.index(st.session_state.map_source)
    selected_map = st.sidebar.selectbox("City Graph", MAP_CHOICES, index=map_idx)
    if selected_map != st.session_state.map_source:
        st.session_state.map_source = selected_map
        set_environment(selected_map, st.session_state.route_algo, st.session_state.assignment_method)
    st.sidebar.divider()
    route_options = ["dijkstra", "astar"]
    route_idx = route_options.index(st.session_state.route_algo)
    route_algo = st.sidebar.selectbox("Routing Algorithm", route_options, index=route_idx)
    assignment_options = ["hungarian", "random"]
    assignment_idx = assignment_options.index(st.session_state.assignment_method)
    assignment_method = st.sidebar.selectbox("Assignment Strategy", assignment_options, index=assignment_idx)
    speed_options = ["Real-time", "Fast-forward"]
    speed_idx = speed_options.index(st.session_state.speed)
    speed = st.sidebar.selectbox("Playback Speed", speed_options, index=speed_idx)
    st.sidebar.divider()
    zoom_manual = st.sidebar.checkbox("Manual Map Zoom", value=st.session_state.map_zoom is not None)
    if zoom_manual:
        default_zoom = st.session_state.map_zoom if st.session_state.map_zoom is not None else 12
        st.session_state.map_zoom = st.sidebar.slider("Zoom Level", min_value=8, max_value=18, value=int(default_zoom))
    else:
        st.session_state.map_zoom = None
    traffic = st.sidebar.selectbox("Traffic Scenario", ["Balanced", "Rush Hour", "Quiet", "Custom"])
    running = st.sidebar.toggle("Run Simulation", value=st.session_state.running)
    st.session_state.speed = speed
    st.session_state.running = running
    if route_algo != st.session_state.route_algo or assignment_method != st.session_state.assignment_method:
        st.session_state.route_algo = route_algo
        st.session_state.assignment_method = assignment_method
    dispatcher: RealtimeDispatcher = st.session_state.dispatcher
    dispatcher.algorithm = st.session_state.route_algo
    dispatcher.method = st.session_state.assignment_method
    if traffic == "Rush Hour":
        st.session_state.model.update_random_traffic(2.0)
    elif traffic == "Quiet":
        st.session_state.model.update_random_traffic(0.3)


# ------------------------------------------------------------------ #
# Layout
# ------------------------------------------------------------------ #
def main() -> None:
    init_state()
    sidebar_controls()
    st.title("🚑 Emergency Ambulance Dispatch Optimization")
    st.caption("Hybrid graph + assignment algorithms with real-time visualization.")

    home_tab, live_tab, analytics_tab_obj = st.tabs(["🏠 Home Dashboard", "🗺️ Live Map Simulation", "📊 Analytics"])

    with home_tab:
        st.header("Mission Control")
        st.write(
            "Start the dispatcher, compare routing strategies, and monitor key performance metrics. "
            "The system combines Dijkstra/A* routing with Hungarian assignment for optimal response times."
        )
        st.image("assets/ambulance_icon.png", width=72)
        col1, col2 = st.columns(2)
        col1.button("Start Simulation", on_click=lambda: st.session_state.update({"running": True}), use_container_width=True)
        col2.button(
            "Add Initial Emergencies",
            on_click=lambda: [add_emergency(urgency=random.randint(2, 5)) for _ in range(3)],
            use_container_width=True,
        )
        st.markdown(
            """
            **Highlights**
            - Live assignment updates with urgency-aware queues  
            - Smooth ambulance animation on a 10×10 city grid  
            - Analytics comparing random vs optimal strategies  
            - Exportable reports for further analysis
            """
        )

    with live_tab:
        controls_panel()
        if st.session_state.running:
            run_steps(20 if st.session_state.speed == "Fast-forward" else 8)
        sim_snapshot = st.session_state.sim.snapshot()
        _log_metric_history(sim_snapshot)
        dispatcher_snapshot = st.session_state.dispatcher.snapshot()
        render_metrics(dispatcher_snapshot, sim_snapshot)
        st.plotly_chart(render_grid(sim_snapshot, st.session_state.show_paths), use_container_width=True)
        render_ambulance_status(sim_snapshot)
        render_assignment_table()

    with analytics_tab_obj:
        analytics_tab(sim_snapshot)


if __name__ == "__main__":
    main()


"""
ui.py
-----
Enhanced Streamlit dashboard for Emergency Ambulance Dispatch Optimization
with advanced analytics, performance comparisons, and real-time monitoring.
"""
from __future__ import annotations

import json
import math
import random
import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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


def _inject_global_styles() -> None:
    """Enhanced global CSS for a modern, professional dashboard look."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        .main {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
            font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #020617 0%, #0f172a 100%) !important;
            border-right: 2px solid rgba(59, 130, 246, 0.3);
        }
        .stMetric {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.9));
            padding: 1rem 1.2rem;
            border-radius: 1rem;
            border: 1px solid rgba(59, 130, 246, 0.4);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3), 0 0 20px rgba(59, 130, 246, 0.1);
            transition: all 0.3s ease;
        }
        .stMetric:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 12px rgba(0, 0, 0, 0.4), 0 0 30px rgba(59, 130, 246, 0.2);
            border-color: rgba(59, 130, 246, 0.6);
        }
        .stMetric label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            color: #94a3b8;
            font-weight: 600;
        }
        .stMetric [data-testid="stMetricValue"] {
            font-weight: 700;
            font-size: 1.5rem;
            color: #f1f5f9;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }
        .stMetric [data-testid="stMetricDelta"] {
            font-size: 0.9rem;
        }
        div[data-testid="stDataFrame"] {
            border-radius: 1rem;
            border: 1px solid rgba(59, 130, 246, 0.4);
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        }
        .stButton>button, .stDownloadButton>button {
            border-radius: 0.75rem;
            border: 1px solid rgba(59, 130, 246, 0.5);
            background: linear-gradient(135deg, #3b82f6, #2563eb);
            color: #ffffff;
            font-weight: 600;
            padding: 0.5rem 1.5rem;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        .stButton>button:hover, .stDownloadButton>button:hover {
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            border-color: #60a5fa;
            box-shadow: 0 4px 8px rgba(59, 130, 246, 0.4);
            transform: translateY(-1px);
        }
        h1, h2, h3 {
            font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
            color: #f1f5f9;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background-color: rgba(15, 23, 42, 0.6);
            padding: 0.5rem;
            border-radius: 0.75rem;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: rgba(30, 41, 59, 0.5);
            border-radius: 0.5rem;
            padding: 0.5rem 1.5rem;
            color: #94a3b8;
            font-weight: 600;
            border: 1px solid transparent;
            transition: all 0.3s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: rgba(51, 65, 85, 0.8);
            border-color: rgba(59, 130, 246, 0.3);
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
            color: #ffffff !important;
            border-color: #60a5fa !important;
        }
        .info-card {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.9));
            padding: 1.5rem;
            border-radius: 1rem;
            border: 1px solid rgba(59, 130, 246, 0.3);
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        }
        .alert-critical {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(220, 38, 38, 0.2));
            border-left: 4px solid #ef4444;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .alert-warning {
            background: linear-gradient(135deg, rgba(251, 191, 36, 0.2), rgba(245, 158, 11, 0.2));
            border-left: 4px solid #fbbf24;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .alert-success {
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(22, 163, 74, 0.2));
            border-left: 4px solid #22c55e;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
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


def _add_event_to_timeline(event_type: str, message: str, severity: str = "info") -> None:
    """Add event to timeline with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    event = {
        "time": timestamp,
        "type": event_type,
        "message": message,
        "severity": severity
    }
    st.session_state.event_timeline.append(event)
    
    # Add to notifications if important
    if severity in ["warning", "critical"]:
        st.session_state.notifications.append(event)
        if severity == "critical":
            st.session_state.critical_events += 1


def _log_algorithm_performance(algorithm: str, execution_time: float, cost: float) -> None:
    """Log algorithm performance metrics for comparison."""
    if algorithm in st.session_state.algorithm_comparison:
        st.session_state.algorithm_comparison[algorithm]["times"].append(execution_time)
        st.session_state.algorithm_comparison[algorithm]["costs"].append(cost)
        # Keep only last 100 measurements
        if len(st.session_state.algorithm_comparison[algorithm]["times"]) > 100:
            st.session_state.algorithm_comparison[algorithm]["times"].pop(0)
            st.session_state.algorithm_comparison[algorithm]["costs"].pop(0)


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
    # Enhanced features state
    st.session_state.event_timeline = deque(maxlen=50)
    st.session_state.notifications = deque(maxlen=10)
    st.session_state.performance_log = {"dijkstra": [], "astar": []}
    st.session_state.algorithm_comparison = {"dijkstra": {"times": [], "costs": []}, "astar": {"times": [], "costs": []}}
    st.session_state.show_traffic_heatmap = False
    st.session_state.show_notifications = True
    st.session_state.critical_events = 0
    st.session_state.total_resolved = 0
    st.session_state.best_response_time = float('inf')
    st.session_state.worst_response_time = 0.0
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
    # Log event to timeline
    severity = "critical" if urgency >= 4 else "warning" if urgency >= 3 else "info"
    _add_event_to_timeline(
        "emergency",
        f"🚨 New emergency (Urgency: {urgency}) at location {location}",
        severity
    )


def cleanup_resolved_emergencies() -> None:
    sim: SimulationEngine = st.session_state.sim
    resolved_sim_ids = {
        event.id for event in sim.emergencies.values() if event.resolved_at is not None
    }
    for dispatch_id, info in list(st.session_state.emergency_lookup.items()):
        if info["sim_id"] in resolved_sim_ids:
            st.session_state.emergency_lookup.pop(dispatch_id)
            st.session_state.total_resolved += 1
            
            # Track response time
            event = sim.emergencies[info["sim_id"]]
            if hasattr(event, 'response_time') and event.response_time:
                response_time = event.response_time
                if response_time < st.session_state.best_response_time:
                    st.session_state.best_response_time = response_time
                if response_time > st.session_state.worst_response_time:
                    st.session_state.worst_response_time = response_time
            
            _add_event_to_timeline(
                "resolution",
                f"✅ Emergency {dispatch_id} resolved (Urgency: {info['urgency']})",
                "success"
            )


def run_steps(iterations: int = 15) -> None:
    sim: SimulationEngine = st.session_state.sim
    dispatcher: RealtimeDispatcher = st.session_state.dispatcher
    dt = 0.25 if st.session_state.speed == "Real-time" else 0.4
    sleep_time = 0.04 if st.session_state.speed == "Real-time" else 0.005
    
    for _ in range(iterations):
        sync_dispatcher_with_sim()
        
        # Time the rebalance operation
        start_time = time.time()
        dispatcher.rebalance_assignments()
        execution_time = time.time() - start_time
        
        # Log performance if assignments were made
        if dispatcher.history:
            latest = dispatcher.history[-1]
            _log_algorithm_performance(
                st.session_state.route_algo,
                execution_time,
                latest.total_cost
            )
        
        apply_dispatcher_paths_to_sim()
        completed = sim.tick(dt)
        
        for amb_id in completed:
            dispatcher.release_ambulance(amb_id)
            _add_event_to_timeline(
                "ambulance",
                f"🚑 Ambulance {amb_id} completed assignment",
                "info"
            )
        
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
def render_notifications() -> None:
    """Display real-time notifications sidebar."""
    if not st.session_state.show_notifications or not st.session_state.notifications:
        return
    
    st.sidebar.markdown("### 🔔 Notifications")
    for notif in reversed(list(st.session_state.notifications)):
        severity_class = f"alert-{notif['severity']}"
        icon = "🚨" if notif['severity'] == 'critical' else "⚠️" if notif['severity'] == 'warning' else "ℹ️"
        st.sidebar.markdown(
            f'<div class="{severity_class}"><small>{notif["time"]}</small><br>{icon} {notif["message"]}</div>',
            unsafe_allow_html=True
        )


def render_event_timeline() -> None:
    """Display chronological event timeline."""
    st.subheader("📋 Event Timeline")
    
    if not st.session_state.event_timeline:
        st.info("Events will appear here as the simulation runs.")
        return
    
    timeline_df = pd.DataFrame(list(st.session_state.event_timeline))
    timeline_df['severity_color'] = timeline_df['severity'].map({
        'critical': '🔴',
        'warning': '🟡',
        'info': '🟢',
        'success': '🟢'
    })
    
    display_df = timeline_df[['time', 'severity_color', 'message']].tail(20)
    display_df.columns = ['Time', 'Level', 'Event']
    st.dataframe(display_df, hide_index=True, use_container_width=True)


def render_traffic_heatmap() -> go.Figure:
    """Create traffic density heatmap overlay."""
    model: GraphModel = st.session_state.model
    
    # Get edge traffic data
    traffic_data = []
    positions = getattr(model, "positions", None) or {}
    
    for (u, v), edge_data in model.graph.edges.items():
        if u in positions and v in positions:
            # Extract numeric weight from edge data
            if isinstance(edge_data, dict):
                weight = edge_data.get('weight', 1.0)
            elif isinstance(edge_data, (int, float)):
                weight = float(edge_data)
            else:
                weight = 1.0
            
            u_pos = _coerce_point(positions[u])
            v_pos = _coerce_point(positions[v])
            mid_x = (u_pos[0] + v_pos[0]) / 2
            mid_y = (u_pos[1] + v_pos[1]) / 2
            traffic_data.append({
                'lon': mid_x,
                'lat': mid_y,
                'traffic': weight
            })
    
    if not traffic_data:
        return go.Figure()
    
    df = pd.DataFrame(traffic_data)
    
    fig = go.Figure()
    
    if not model.grid_mode:
        fig.add_trace(go.Densitymapbox(
            lon=df['lon'],
            lat=df['lat'],
            z=df['traffic'],
            radius=15,
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="Traffic Density"),
        ))
        
        center_lat = float(df['lat'].mean())
        center_lon = float(df['lon'].mean())
        
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=11,
            ),
            height=500,
            margin=dict(l=0, r=0, t=30, b=0),
            title="Traffic Density Heatmap",
            paper_bgcolor="#0f172a",
            font=dict(color="#f8fafc"),
        )
    else:
        fig = px.density_heatmap(
            df, x='lon', y='lat', z='traffic',
            color_continuous_scale='Reds',
            title="Traffic Density Heatmap"
        )
        fig.update_layout(
            height=500,
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font=dict(color="#f8fafc"),
        )
    
    return fig


def render_performance_comparison() -> None:
    """Display algorithm performance comparison dashboard."""
    st.subheader("⚡ Algorithm Performance Comparison")
    
    comparison = st.session_state.algorithm_comparison
    
    col1, col2 = st.columns(2)
    
    # Execution time comparison
    with col1:
        st.markdown("**Average Execution Time**")
        time_data = []
        for algo, data in comparison.items():
            if data["times"]:
                avg_time = np.mean(data["times"]) * 1000  # Convert to ms
                time_data.append({"Algorithm": algo.upper(), "Avg Time (ms)": avg_time})
        
        if time_data:
            df_time = pd.DataFrame(time_data)
            fig_time = px.bar(
                df_time, x='Algorithm', y='Avg Time (ms)',
                color='Algorithm',
                color_discrete_map={'DIJKSTRA': '#3b82f6', 'ASTAR': '#22c55e'}
            )
            fig_time.update_layout(
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f8fafc'),
                height=300
            )
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("Performance data will appear after running simulations.")
    
    # Travel time comparison
    with col2:
        st.markdown("**Average Travel Time**")
        cost_data = []
        for algo, data in comparison.items():
            if data["costs"]:
                avg_cost = np.mean(data["costs"])
                cost_data.append({"Algorithm": algo.upper(), "Avg Time": avg_cost})
        
        if cost_data:
            df_cost = pd.DataFrame(cost_data)
            fig_cost = px.bar(
                df_cost, x='Algorithm', y='Avg Time',
                color='Algorithm',
                color_discrete_map={'DIJKSTRA': '#3b82f6', 'ASTAR': '#22c55e'}
            )
            fig_cost.update_layout(
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f8fafc'),
                height=300
            )
            st.plotly_chart(fig_cost, use_container_width=True)
        else:
            st.info("Travel time data will appear after running simulations.")
    
    # Performance over time
    st.markdown("**Performance Trend Over Time**")
    
    fig_trend = go.Figure()
    
    for algo, data in comparison.items():
        if data["times"]:
            fig_trend.add_trace(go.Scatter(
                x=list(range(len(data["times"]))),
                y=[t * 1000 for t in data["times"]],
                mode='lines+markers',
                name=algo.upper(),
                line=dict(width=2)
            ))
    
    fig_trend.update_layout(
        xaxis_title="Iteration",
        yaxis_title="Execution Time (ms)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc'),
        height=300,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    if any(data["times"] for data in comparison.values()):
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("Performance trends will appear as you run simulations with different algorithms.")


def render_advanced_kpi_dashboard(sim_snapshot: Dict, dispatch_snapshot: Dict) -> None:
    """Display comprehensive KPI dashboard."""
    st.subheader("📊 Key Performance Indicators")
    
    metrics = sim_snapshot["metrics"]
    
    # Top row - Primary KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    
    response_time = metrics["average_response_time"]
    col1.metric(
        "Avg Response",
        f"{response_time:.1f} min",
        delta=f"{response_time - st.session_state.best_response_time:.1f}" if st.session_state.best_response_time != float('inf') else None
    )
    
    total_ambulances = len(dispatch_snapshot["ambulances"])
    available = sum(1 for amb in dispatch_snapshot["ambulances"] if amb["available"])
    utilization = ((total_ambulances - available) / total_ambulances * 100) if total_ambulances > 0 else 0
    col2.metric("Fleet Utilization", f"{utilization:.0f}%")
    
    active_emergencies = st.session_state.sim.active_emergencies()
    col3.metric("Active Calls", active_emergencies)
    
    col4.metric("Total Resolved", st.session_state.total_resolved)
    
    col5.metric("Critical Events", st.session_state.critical_events)
    
    # Second row - Secondary metrics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    distance_units = "blocks" if st.session_state.model.grid_mode else "km"
    distance_display = metrics.get("distance_display", 0.0)
    col1.metric("Distance Traveled", f"{distance_display:.1f} {distance_units}")
    
    sim_time = metrics.get("sim_time_minutes", 0.0)
    col2.metric("Simulation Time", f"{sim_time:.1f} min")
    
    if st.session_state.best_response_time != float('inf'):
        col3.metric("Best Response", f"{st.session_state.best_response_time:.1f} min")
    else:
        col3.metric("Best Response", "N/A")
    
    if st.session_state.worst_response_time > 0:
        col4.metric("Worst Response", f"{st.session_state.worst_response_time:.1f} min")
    else:
        col4.metric("Worst Response", "N/A")


def render_grid(snapshot: Dict, show_paths: bool, focus_ambulance: int | None = None) -> go.Figure:
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
            is_focus = focus_ambulance is not None and amb["id"] == focus_ambulance
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
                    marker=dict(
                        size=22 if is_focus else 18,
                        color="#facc15" if is_focus else "#2b83ba",
                    ),
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
                        line=dict(
                            color="#facc15" if is_focus else "#1f78b4",
                            width=6 if is_focus else 4,
                        ),
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
        is_focus = focus_ambulance is not None and amb["id"] == focus_ambulance
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
                marker=dict(
                    size=22 if is_focus else 18,
                    color="#facc15" if is_focus else "#2b83ba",
                ),
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
                    line=dict(
                        color="#facc15" if is_focus else "#1f78b4",
                        width=5 if is_focus else 3,
                    ),
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


def render_scenario_builder() -> None:
    """Advanced scenario builder interface."""
    st.subheader("🎬 Scenario Builder")
    
    st.markdown("Build and save custom simulation scenarios with pre-defined ambulances, emergencies, and traffic patterns.")
    
    # Scenario presets
    st.markdown("### 📋 Quick Scenarios")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏙️ Urban Rush Hour", use_container_width=True):
            st.session_state.model.update_random_traffic(2.5)
            for _ in range(8):
                add_emergency(urgency=random.randint(3, 5))
            _add_event_to_timeline("scenario", "Loaded: Urban Rush Hour scenario", "info")
            st.success("Urban Rush Hour scenario loaded!")
    
    with col2:
        if st.button("🌃 Quiet Night", use_container_width=True):
            st.session_state.model.update_random_traffic(0.3)
            for _ in range(2):
                add_emergency(urgency=random.randint(1, 3))
            _add_event_to_timeline("scenario", "Loaded: Quiet Night scenario", "info")
            st.success("Quiet Night scenario loaded!")
    
    with col3:
        if st.button("⚠️ Mass Casualty Event", use_container_width=True):
            for _ in range(15):
                add_emergency(urgency=5)
            _add_event_to_timeline("scenario", "Loaded: Mass Casualty Event scenario", "critical")
            st.warning("Mass Casualty Event scenario loaded!")
    
    # Custom scenario builder
    st.markdown("---")
    st.markdown("### ⚙️ Custom Scenario")
    
    scenario_col1, scenario_col2 = st.columns(2)
    
    with scenario_col1:
        num_emergencies = st.slider("Number of Emergencies", 1, 20, 3)
        emergency_urgency_min = st.slider("Min Urgency", 1, 5, 1)
        emergency_urgency_max = st.slider("Max Urgency", emergency_urgency_min, 5, 5)
    
    with scenario_col2:
        num_ambulances = st.slider("Additional Ambulances", 0, 10, 0)
        traffic_multiplier = st.slider("Traffic Multiplier", 0.1, 3.0, 1.0, 0.1)
        
        if st.button("🚀 Launch Custom Scenario", use_container_width=True):
            st.session_state.model.update_random_traffic(traffic_multiplier)
            
            for _ in range(num_emergencies):
                urgency = random.randint(emergency_urgency_min, emergency_urgency_max)
                add_emergency(urgency=urgency)
            
            for _ in range(num_ambulances):
                add_ambulance_unit()
            
            _add_event_to_timeline("scenario", f"Custom scenario: {num_emergencies} emergencies, {num_ambulances} ambulances", "info")
            st.success(f"Custom scenario launched with {num_emergencies} emergencies!")
    
    # Export/Import scenarios
    st.markdown("---")
    st.markdown("### 💾 Save/Load Scenarios")
    
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        scenario_data = {
            "ambulances": len(st.session_state.dispatcher.ambulances),
            "active_emergencies": st.session_state.sim.active_emergencies(),
            "traffic_state": "current",
            "timestamp": datetime.now().isoformat()
        }
        
        st.download_button(
            "📥 Export Current State",
            data=json.dumps(scenario_data, indent=2),
            file_name=f"scenario_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with export_col2:
        st.button("📤 Load Scenario", use_container_width=True, disabled=True)
        st.caption("Feature coming soon")


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
        _add_event_to_timeline("traffic", "Traffic manually increased", "warning")
        st.toast("Traffic congestion increased", icon="🚦")
    if col4.button("Reset Simulation", use_container_width=True):
        st.session_state.clear()
        init_state()
        _add_event_to_timeline("system", "Simulation reset", "info")
        st.toast("Simulation reset", icon="♻️")
    if col5.button("Show Paths" if not st.session_state.show_paths else "Hide Paths", use_container_width=True):
        st.session_state.show_paths = not st.session_state.show_paths
    with col6:
        snapshot = st.session_state.dispatcher.snapshot()
        payload = serialize_snapshot(snapshot)
        st.download_button(
            "Export Report",
            data=json.dumps(payload, indent=2),
            file_name=f"dispatch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )
    st.divider()
    st.markdown("#### 📍 Custom Emergency Placement")
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
            f"Latest optimization: {latest.method.title()} (Total Time: {latest.total_cost:.1f} min, "
            f"{len(latest.assignments)} pairs)"
        )


def render_ambulance_status(sim_snapshot: Dict) -> None:
    ambulances = sim_snapshot.get("ambulances", [])
    if not ambulances:
        st.info("No ambulances deployed yet.")
        return
    model: GraphModel = st.session_state.model
    status_filter = st.radio(
        "Filter units",
        ["All", "Idle", "Responding"],
        index=0,
        horizontal=True,
    )
    if status_filter != "All":
        want_idle = status_filter == "Idle"
        ambulances = [a for a in ambulances if a["idle"] == want_idle]
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
        st.warning("Waiting for sufficient emergencies to build a travel time matrix.")
        return
    cost_df = pd.DataFrame(latest.cost_matrix, columns=[f"E{j+1}" for j in range(len(latest.cost_matrix[0]))])
    cost_df.index = [f"A{i+1}" for i in range(len(cost_df))]
    st.subheader(f"Travel Time Matrix ({latest.method.title()} strategy)")
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
                "Travel Time (min)": f"{cost:.1f}",
            }
        )
    if pair_rows:
        st.write("Optimal pairing (per most recent optimization window):")
        st.dataframe(pd.DataFrame(pair_rows), hide_index=True, use_container_width=True)
    else:
        st.info("No optimal pairs computed yet—waiting on additional emergencies or free ambulances.")


def analytics_tab(sim_snapshot: Dict) -> None:
    st.subheader("📈 Response Time Analysis")
    response_history = sim_snapshot.get("response_history", [])
    if response_history:
        response_df = pd.DataFrame(
            {"Response #": range(1, len(response_history) + 1), "Minutes": response_history}
        )
        
        # Enhanced chart with Plotly
        fig_response = go.Figure()
        fig_response.add_trace(go.Scatter(
            x=response_df["Response #"],
            y=response_df["Minutes"],
            mode='lines+markers',
            name='Response Time',
            line=dict(color='#3b82f6', width=3),
            marker=dict(size=6)
        ))
        
        # Add average line
        avg_response = response_df["Minutes"].mean()
        fig_response.add_hline(y=avg_response, line_dash="dash", line_color="#fbbf24",
                              annotation_text=f"Avg: {avg_response:.2f} min")
        
        fig_response.update_layout(
            xaxis_title="Response Number",
            yaxis_title="Response Time (minutes)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc'),
            height=400
        )
        st.plotly_chart(fig_response, use_container_width=True)
        
        # Statistics
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        stat_col1.metric("Average", f"{avg_response:.2f} min")
        stat_col2.metric("Best", f"{min(response_history):.2f} min")
        stat_col3.metric("Worst", f"{max(response_history):.2f} min")
        stat_col4.metric("Total Responses", len(response_history))
    else:
        st.info("📊 No completed responses yet. Once ambulances resolve events, detailed analytics will appear here.")

    st.markdown("---")
    st.subheader("🚗 Distance Covered Over Time")
    distance_history = st.session_state.get("distance_history", [])
    if distance_history:
        distance_df = pd.DataFrame(distance_history)
        
        fig_distance = px.area(
            distance_df, x="Minutes", y="Distance",
            color_discrete_sequence=['#22c55e']
        )
        fig_distance.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc'),
            height=350
        )
        st.plotly_chart(fig_distance, use_container_width=True)
        
        total_distance = distance_df["Distance"].iloc[-1] if len(distance_df) > 0 else 0
        st.metric("Total Distance", f"{total_distance:.1f} {sim_snapshot['metrics'].get('distance_units', 'km')}")
    else:
        st.info("📍 Distance history populates automatically as the simulation runs.")

    st.markdown("---")
    st.subheader("🚨 Active Emergencies Over Time")
    active_history = st.session_state.get("active_emergencies_history", [])
    if active_history:
        active_df = pd.DataFrame(active_history)
        
        fig_active = go.Figure()
        fig_active.add_trace(go.Scatter(
            x=active_df["Minutes"],
            y=active_df["Active"],
            mode='lines',
            fill='tozeroy',
            line=dict(color='#ef4444', width=2),
            fillcolor='rgba(239, 68, 68, 0.3)'
        ))
        
        fig_active.update_layout(
            xaxis_title="Simulation Time (minutes)",
            yaxis_title="Active Emergencies",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc'),
            height=350
        )
        st.plotly_chart(fig_active, use_container_width=True)
        
        max_concurrent = max([e["Active"] for e in active_history])
        st.metric("Peak Concurrent Emergencies", max_concurrent)
    else:
        st.info("🔥 Start the simulation to see active emergency trends.")

    st.markdown("---")
    dispatcher_history = st.session_state.dispatcher.history[-20:]
    if dispatcher_history:
        st.subheader("🎯 Optimization Performance")
        total_runs = len(st.session_state.dispatcher.history)
        start_idx = total_runs - len(dispatcher_history) + 1
        iterations = list(range(start_idx, start_idx + len(dispatcher_history)))
        data = {
            "Iteration": iterations,
            "Total Time": [entry.total_cost for entry in dispatcher_history],
            "Pairs": [len(entry.assignments) for entry in dispatcher_history],
        }
        df = pd.DataFrame(data)
        
        # Dual-axis chart
        fig_opt = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_opt.add_trace(
            go.Bar(x=df["Iteration"], y=df["Total Time"], name="Total Response Time", marker_color='#3b82f6'),
            secondary_y=False
        )
        
        fig_opt.add_trace(
            go.Scatter(x=df["Iteration"], y=df["Pairs"], name="Pairs", 
                      line=dict(color='#22c55e', width=3), mode='lines+markers'),
            secondary_y=True
        )
        
        fig_opt.update_xaxes(title_text="Iteration")
        fig_opt.update_yaxes(title_text="Total Response Time (min)", secondary_y=False)
        fig_opt.update_yaxes(title_text="Assignment Pairs", secondary_y=True)
        
        fig_opt.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc'),
            height=400
        )
        
        st.plotly_chart(fig_opt, use_container_width=True)
        
        st.caption(
            f"📊 Average pairs per run: {sum(data['Pairs']) / len(data['Pairs']):.2f} | "
            f"Min time: {min(data['Total Time']):.1f} min | "
            f"Avg time: {sum(data['Total Time']) / len(data['Total Time']):.1f} min"
        )
    else:
        st.info("⚡ Optimization history will appear after the dispatcher assigns ambulances at least once.")


def sidebar_controls() -> None:
    st.sidebar.title("🎮 Simulation Controls")
    
    # Map selection
    map_idx = MAP_CHOICES.index(st.session_state.map_source)
    selected_map = st.sidebar.selectbox("🗺️ City Graph", MAP_CHOICES, index=map_idx)
    if selected_map != st.session_state.map_source:
        st.session_state.map_source = selected_map
        set_environment(selected_map, st.session_state.route_algo, st.session_state.assignment_method)
        _add_event_to_timeline("system", f"Switched to {selected_map}", "info")
    
    st.sidebar.divider()
    
    # Algorithm settings
    st.sidebar.markdown("### ⚙️ Algorithm Configuration")
    route_options = ["dijkstra", "astar"]
    route_idx = route_options.index(st.session_state.route_algo)
    route_algo = st.sidebar.selectbox("🛣️ Routing Algorithm", route_options, index=route_idx)
    
    assignment_options = ["hungarian", "random"]
    assignment_idx = assignment_options.index(st.session_state.assignment_method)
    assignment_method = st.sidebar.selectbox("🎯 Assignment Strategy", assignment_options, index=assignment_idx)
    
    speed_options = ["Real-time", "Fast-forward"]
    speed_idx = speed_options.index(st.session_state.speed)
    speed = st.sidebar.selectbox("⏱️ Playback Speed", speed_options, index=speed_idx)
    
    st.sidebar.divider()
    
    # Display settings
    st.sidebar.markdown("### 🎨 Display Options")
    zoom_manual = st.sidebar.checkbox("Manual Map Zoom", value=st.session_state.map_zoom is not None)
    if zoom_manual:
        default_zoom = st.session_state.map_zoom if st.session_state.map_zoom is not None else 12
        st.session_state.map_zoom = st.sidebar.slider("Zoom Level", min_value=8, max_value=18, value=int(default_zoom))
    else:
        st.session_state.map_zoom = None
    
    st.session_state.show_traffic_heatmap = st.sidebar.checkbox("Show Traffic Heatmap", value=st.session_state.show_traffic_heatmap)
    st.session_state.show_notifications = st.sidebar.checkbox("Show Notifications", value=st.session_state.show_notifications)
    
    st.sidebar.divider()
    
    # Traffic scenario
    st.sidebar.markdown("### 🚦 Traffic Control")
    traffic = st.sidebar.selectbox("Traffic Scenario", ["Balanced", "Rush Hour", "Quiet", "Custom"])
    
    # Simulation control
    st.sidebar.divider()
    st.sidebar.markdown("### ▶️ Simulation")
    running = st.sidebar.toggle("Run Simulation", value=st.session_state.running)
    
    st.session_state.speed = speed
    st.session_state.running = running
    
    if route_algo != st.session_state.route_algo or assignment_method != st.session_state.assignment_method:
        old_algo = st.session_state.route_algo
        st.session_state.route_algo = route_algo
        st.session_state.assignment_method = assignment_method
        _add_event_to_timeline("system", f"Algorithm changed: {old_algo} → {route_algo}", "info")
    
    dispatcher: RealtimeDispatcher = st.session_state.dispatcher
    dispatcher.algorithm = st.session_state.route_algo
    dispatcher.method = st.session_state.assignment_method
    
    if traffic == "Rush Hour":
        st.session_state.model.update_random_traffic(2.0)
    elif traffic == "Quiet":
        st.session_state.model.update_random_traffic(0.3)

    st.sidebar.divider()
    
    # Ambulance focus
    snap = dispatcher.snapshot()
    amb_options = ["All ambulances"] + [f"Ambulance #{amb['id']}" for amb in snap["ambulances"]]
    selected = st.sidebar.selectbox("🔍 Focus on unit", amb_options)
    if selected == "All ambulances":
        st.session_state.focus_ambulance = None
    else:
        try:
            st.session_state.focus_ambulance = int(selected.split("#")[1])
        except Exception:
            st.session_state.focus_ambulance = None
    
    # Notifications panel
    st.sidebar.divider()
    render_notifications()
    
    # Help section
    st.sidebar.divider()
    with st.sidebar.expander("ℹ️ Help & Info"):
        st.markdown(r"""
        **Quick Guide:**
        
        🚑 **Ambulances**: Blue markers on map  
        🚨 **Emergencies**: Red markers (number = urgency)  
        🟨 **Routes**: Yellow lines show paths  
        
        **Algorithms:**
        - **Dijkstra**: Guaranteed shortest path
        - **A***: Faster with heuristics
        
        **Assignment:**
        - **Hungarian**: Optimal pairing
        - **Random**: Baseline comparison
        
        **Tips:**
        - Use scenarios for quick testing
        - Monitor KPIs for performance
        - Export reports for analysis
        - Check timeline for events
        """)
        
        st.markdown("---")
        st.markdown("**System Status:**")
        model: GraphModel = st.session_state.model
        total_nodes = len(model.graph.nodes) if hasattr(model.graph, 'nodes') else 0
        total_edges = len(model.graph.edges) if hasattr(model.graph, 'edges') else 0
        st.caption(f"📍 Nodes: {total_nodes}")
        st.caption(f"🛣️ Edges: {total_edges}")
        st.caption(f"🗺️ Mode: {'Grid' if model.grid_mode else 'OSM'}")


# ------------------------------------------------------------------ #
# Layout
# ------------------------------------------------------------------ #
def main() -> None:
    init_state()
    _inject_global_styles()
    sidebar_controls()
    
    # Header with status indicators
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("🚑 Emergency Ambulance Dispatch Optimization")
        st.caption("Advanced real-time dispatch system with AI-powered routing and assignment optimization")
    with col2:
        status = "🟢 Running" if st.session_state.running else "⚪ Paused"
        st.markdown(f"### {status}")

    # Multi-tab interface
    home_tab, live_tab, analytics_tab_obj, performance_tab, timeline_tab, traffic_tab, scenario_tab = st.tabs([
        "🏠 Dashboard", 
        "🗺️ Live Map", 
        "📊 Analytics",
        "⚡ Performance",
        "📋 Timeline",
        "🚦 Traffic",
        "🎬 Scenarios"
    ])

    with home_tab:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.header("🎯 Mission Control Center")
        st.write(
            "Welcome to the advanced ambulance dispatch optimization system. "
            "This platform combines cutting-edge algorithms (Dijkstra/A*) with optimal assignment strategies (Hungarian method) "
            "to minimize emergency response times and save lives."
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("▶️ Start Simulation", use_container_width=True):
                st.session_state.running = True
                _add_event_to_timeline("system", "Simulation started", "success")
                
        with col2:
            if st.button("⏸️ Pause Simulation", use_container_width=True):
                st.session_state.running = False
                _add_event_to_timeline("system", "Simulation paused", "info")
        
        with col3:
            if st.button("➕ Add Emergencies (3x)", use_container_width=True):
                for _ in range(3):
                    add_emergency(urgency=random.randint(2, 5))
        
        with col4:
            if st.button("🚑 Add Ambulance", use_container_width=True):
                add_ambulance_unit()
        
        # System overview
        st.markdown("---")
        sim_snapshot = st.session_state.sim.snapshot()
        dispatcher_snapshot = st.session_state.dispatcher.snapshot()
        render_advanced_kpi_dashboard(sim_snapshot, dispatcher_snapshot)
        
        # Live system health
        st.markdown("---")
        st.subheader("🏥 System Health Monitor")
        
        health_col1, health_col2, health_col3, health_col4 = st.columns(4)
        
        # Calculate health metrics
        total_ambulances = len(dispatcher_snapshot["ambulances"])
        available = sum(1 for amb in dispatcher_snapshot["ambulances"] if amb["available"])
        response_time = sim_snapshot["metrics"]["average_response_time"]
        active_emergencies = st.session_state.sim.active_emergencies()
        
        # System health score (0-100)
        ambulance_score = (available / total_ambulances * 100) if total_ambulances > 0 else 0
        response_score = max(0, 100 - (response_time * 5))  # Lower time = higher score
        emergency_score = max(0, 100 - (active_emergencies * 10))  # Fewer emergencies = higher score
        overall_health = (ambulance_score + response_score + emergency_score) / 3
        
        health_status = "🟢 Excellent" if overall_health >= 70 else "🟡 Fair" if overall_health >= 40 else "🔴 Critical"
        
        health_col1.metric("Overall Health", f"{overall_health:.0f}%", help="Composite score based on availability, response time, and emergency load")
        health_col2.metric("System Status", health_status)
        health_col3.metric("Queue Pressure", f"{active_emergencies}/{total_ambulances}", help="Active emergencies vs available ambulances")
        
        # Algorithm efficiency
        if st.session_state.algorithm_comparison[st.session_state.route_algo]["times"]:
            avg_time = np.mean(st.session_state.algorithm_comparison[st.session_state.route_algo]["times"]) * 1000
            health_col4.metric("Algorithm Speed", f"{avg_time:.1f}ms", help="Average algorithm execution time")
        else:
            health_col4.metric("Algorithm Speed", "N/A")
        
        # Features showcase
        st.markdown("---")
        st.subheader("✨ System Capabilities")
        
        feat_col1, feat_col2, feat_col3 = st.columns(3)
        
        with feat_col1:
            st.markdown("""
            **🎯 Smart Routing**
            - Multiple pathfinding algorithms
            - Real-time traffic consideration
            - Dynamic rerouting capabilities
            - Optimized for speed and accuracy
            """)
        
        with feat_col2:
            st.markdown("""
            **📊 Advanced Analytics**
            - Performance comparison tools
            - Real-time KPI monitoring
            - Historical trend analysis
            - Algorithm benchmarking
            """)
        
        with feat_col3:
            st.markdown("""
            **🔔 Real-time Monitoring**
            - Live event timeline
            - Priority-based notifications
            - Traffic heatmap visualization
            - Fleet status tracking
            """)

    with live_tab:
        controls_panel()
        
        if st.session_state.running:
            run_steps(20 if st.session_state.speed == "Fast-forward" else 8)
        
        sim_snapshot = st.session_state.sim.snapshot()
        _log_metric_history(sim_snapshot)
        dispatcher_snapshot = st.session_state.dispatcher.snapshot()
        
        render_metrics(dispatcher_snapshot, sim_snapshot)
        
        # Main map
        st.plotly_chart(
            render_grid(
                sim_snapshot,
                st.session_state.show_paths,
                focus_ambulance=st.session_state.get("focus_ambulance"),
            ),
            use_container_width=True,
        )
        
        # Legend
        legend_cols = st.columns(4)
        legend_cols[0].markdown("**🟦 Ambulance** – active unit")
        legend_cols[1].markdown("**🟥 Emergency** – incident location")
        legend_cols[2].markdown("**🟨 Focus** – highlighted route")
        legend_cols[3].markdown("**🟩 Resolved** – completed calls")
        
        # Status tables
        col1, col2 = st.columns(2)
        with col1:
            render_ambulance_status(sim_snapshot)
        with col2:
            render_assignment_table()

    with analytics_tab_obj:
        analytics_tab(sim_snapshot)

    with performance_tab:
        render_performance_comparison()
        
        st.markdown("---")
        st.subheader("🎯 Optimization Efficiency")
        
        history = st.session_state.dispatcher.history
        if history and len(history) > 5:
            recent_history = history[-50:]
            
            costs = [h.total_cost for h in recent_history]
            pairs = [len(h.assignments) for h in recent_history]
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Total Response Time", "Assignments per Iteration")
            )
            
            fig.add_trace(
                go.Scatter(x=list(range(len(costs))), y=costs, mode='lines+markers', name='Response Time'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=list(range(len(pairs))), y=pairs, name='Pairs'),
                row=1, col=2
            )
            
            fig.update_layout(
                height=400,
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f8fafc')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            stat_col1.metric("Avg Time", f"{np.mean(costs):.2f} min")
            stat_col2.metric("Min Time", f"{np.min(costs):.2f} min")
            stat_col3.metric("Max Time", f"{np.max(costs):.2f} min")
            stat_col4.metric("Std Dev", f"{np.std(costs):.2f}")
        else:
            st.info("Run the simulation longer to see detailed performance analytics.")

    with timeline_tab:
        render_event_timeline()
        
        # Summary statistics
        if st.session_state.event_timeline:
            st.markdown("---")
            st.subheader("📈 Event Summary")
            
            timeline_list = list(st.session_state.event_timeline)
            severity_counts = {}
            type_counts = {}
            
            for event in timeline_list:
                severity_counts[event['severity']] = severity_counts.get(event['severity'], 0) + 1
                type_counts[event['type']] = type_counts.get(event['type'], 0) + 1
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Events by Severity**")
                severity_df = pd.DataFrame(list(severity_counts.items()), columns=['Severity', 'Count'])
                fig_severity = px.pie(severity_df, values='Count', names='Severity', 
                                     color='Severity',
                                     color_discrete_map={
                                         'critical': '#ef4444',
                                         'warning': '#fbbf24',
                                         'info': '#3b82f6',
                                         'success': '#22c55e'
                                     })
                fig_severity.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#f8fafc'),
                    height=300
                )
                st.plotly_chart(fig_severity, use_container_width=True)
            
            with col2:
                st.markdown("**Events by Type**")
                type_df = pd.DataFrame(list(type_counts.items()), columns=['Type', 'Count'])
                fig_type = px.bar(type_df, x='Type', y='Count', color='Type')
                fig_type.update_layout(
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#f8fafc'),
                    height=300
                )
                st.plotly_chart(fig_type, use_container_width=True)

    with traffic_tab:
        st.subheader("🚦 Traffic Analysis")
        
        if st.session_state.show_traffic_heatmap:
            st.plotly_chart(render_traffic_heatmap(), use_container_width=True)
        else:
            st.info("Enable 'Show Traffic Heatmap' in the sidebar to visualize traffic density.")
        
        # Traffic controls
        st.markdown("---")
        st.subheader("Traffic Simulation Controls")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚦 Increase Traffic", use_container_width=True):
                st.session_state.model.update_random_traffic(1.5)
                _add_event_to_timeline("traffic", "Traffic increased by 50%", "warning")
                st.toast("Traffic congestion increased", icon="🚦")
        
        with col2:
            if st.button("✅ Normal Traffic", use_container_width=True):
                st.session_state.model.update_random_traffic(1.0)
                _add_event_to_timeline("traffic", "Traffic normalized", "info")
                st.toast("Traffic set to normal levels", icon="✅")
        
        with col3:
            if st.button("⚡ Reduce Traffic", use_container_width=True):
                st.session_state.model.update_random_traffic(0.5)
                _add_event_to_timeline("traffic", "Traffic reduced by 50%", "success")
                st.toast("Traffic congestion reduced", icon="⚡")
        
        # Traffic statistics
        st.markdown("---")
        st.subheader("📊 Traffic Statistics")
        
        model: GraphModel = st.session_state.model
        if hasattr(model.graph, 'edges'):
            # Extract numeric weights from edges
            weights = []
            for edge_data in model.graph.edges.values():
                # Handle different data types: dict, float, int
                if isinstance(edge_data, dict):
                    weight = edge_data.get('weight', 1.0)
                elif isinstance(edge_data, (int, float)):
                    weight = float(edge_data)
                else:
                    weight = 1.0
                weights.append(weight)
            
            if weights:
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                stat_col1.metric("Avg Edge Weight", f"{np.mean(weights):.2f}")
                stat_col2.metric("Min Weight", f"{np.min(weights):.2f}")
                stat_col3.metric("Max Weight", f"{np.max(weights):.2f}")
                stat_col4.metric("Total Edges", len(weights))
                
                # Distribution chart
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(x=weights, nbinsx=30, name='Weight Distribution'))
                fig_dist.update_layout(
                    title="Edge Weight Distribution",
                    xaxis_title="Weight",
                    yaxis_title="Frequency",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#f8fafc'),
                    height=400
                )
                st.plotly_chart(fig_dist, use_container_width=True)

    with scenario_tab:
        render_scenario_builder()

    if st.session_state.running:
        st.rerun()


if __name__ == "__main__":
    main()


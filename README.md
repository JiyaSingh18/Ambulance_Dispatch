# Emergency Ambulance Dispatch Optimization 🚑

Hybrid graph and assignment algorithms that orchestrate ambulances across a 10×10 smart-city grid. Powered by NetworkX, SciPy, and Streamlit, the app animates live dispatching decisions, compares routing strategies, and surfaces rich analytics.

## Highlights
- 10×10 city lattice with stochastic traffic multipliers (1–5) per edge
- Dual routing core (Dijkstra & A*) plus Hungarian vs random assignment
- Heap-based dispatcher with urgency-aware priority queue and hybrid fallback planning
- Real-time Streamlit dashboard with animated map, metrics, and analytics
- Control panel for injecting emergencies, tweaking traffic, toggling paths, exporting reports
- Unit-tested core algorithms and sample scenario (3 ambulances / 5 emergencies)

## Project Structure
```
ambulance_dispatch/
├── analytics.py         # Data generators for the analytics page
├── assignment.py        # Hungarian + random allocation helpers
├── graph_model.py       # 10×10 grid builder and scenario utilities
├── main.py              # CLI entrypoint (demo + UI launcher)
├── realtime.py          # Priority-queue dispatcher with hybrid logic
├── routing.py           # Dijkstra/A* helpers + cost matrix builder
├── simulation.py        # Physics-lite movement engine + metrics
├── ui.py                # Streamlit app (dashboard + live map + analytics)
├── assets/
│   ├── ambulance_icon.png
│   └── emergency_icon.png
├── tests/
│   ├── test_graph_model.py
│   ├── test_realtime.py
│   └── test_routing_assignment.py
└── README.md
```

## UI Walkthrough
1. **Home Dashboard** – Mission summary, hero controls, icons, and highlights.
2. **Live Map Simulation** – Animated grid, moving ambulances, red emergencies, live metrics, cost matrix, control panel.
3. **Analytics** – Streamlit charts: random vs optimal bars, Dijkstra vs A* line chart, fairness histogram, traffic heatmap.

> _Screenshots_: replace the placeholders below with your own captures.
>
> - `assets/screens/home_placeholder.png`
> - `assets/screens/live_placeholder.png`
> - `assets/screens/analytics_placeholder.png`

## Algorithms
- **Graph Modeling**: NetworkX grid graph with per-edge traffic × distance weights. `GraphModel` exposes scenario generation, dynamic traffic updates, and heatmap exports.
- **Routing**: `routing.compute_shortest_path` for Dijkstra/A*; `route_cost_matrix` provides both cost matrix and per-pair path cache.
- **Assignment**: `assignment.assign_ambulances` orchestrates cost matrix creation and selects either Hungarian (SciPy) or random matching. `AssignmentResult` stores paths for downstream visualization.
- **Real-Time Dispatch**: `RealtimeDispatcher` maintains a heap of emergencies keyed by urgency, immediately assigns free ambulances, and when all are busy it stores hybrid global plans (future assignments + planned paths).
- **Simulation**: `SimulationEngine` advances ambulances along frame-based paths, tracks distance, resolves emergencies, and streams metrics back to the UI.

## Running the Project
### 1. Install dependencies
```bash
pip install -r requirements.txt  # or pip install streamlit networkx scipy numpy pandas plotly
```

### 2. Launch the UI
```bash
python main.py ui
# or
streamlit run ui.py
```

### 3. Console Demo (text-based)
```bash
python main.py demo --steps 40
```

### 4. Run Tests
```bash
pytest tests
```

## Controls & Interaction
- **Add Emergency**: Injects a new urgent request with random urgency or from dropdown.
- **Increase Traffic**: Raises traffic multipliers to simulate rush hour.
- **Reset Simulation**: Rebuilds the grid, respawns ambulances, clears queues.
- **Show/Hide Paths**: Toggle animated poly-lines for each ambulance.
- **Export Report**: Download the latest dispatcher snapshot as JSON.
- **Sidebar Options**:
  - Routing algorithm: Dijkstra vs A*
  - Assignment strategy: Hungarian vs random baseline
  - Traffic scenario presets (Balanced / Rush / Quiet / Custom)
  - Playback speed: real-time vs fast-forward
  - Run toggle to start/stop the animation loop

## System Requirements
- Python 3.9+
- Packages: `streamlit`, `networkx`, `numpy`, `scipy`, `pandas`, `plotly`
- GPU not required; Streamlit animates via Plotly in-browser

## Sample Scenario
`graph_model.Scenario` seeds 3 ambulances and 5 emergencies with randomized urgencies (1–5). Use the "Add Initial Emergencies" button on the Home tab to preload this scenario instantly.

## Future Enhancements
- Geographic basemap integration (Leaflet/Mapbox) for photorealistic backgrounds
- Historical data ingestion with predictive demand modeling
- Multi-hop hospital routing with availability-aware drop-off selection
- Integration hooks for live CAD/911 feeds and SMS alerts
- Reinforcement learning policy to supersede heuristic dispatch rules

Enjoy optimizing emergency response times! 🚨


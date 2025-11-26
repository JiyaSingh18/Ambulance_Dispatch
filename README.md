# Emergency Ambulance Dispatch Optimization 🚑

Hybrid graph and assignment algorithms that orchestrate ambulances across either a synthetic 10×10 grid or a real-world Mumbai road network pulled from OpenStreetMap. Powered by NetworkX, SciPy, OSMnx, and Streamlit, the app animates live dispatching decisions, compares routing strategies, and surfaces rich analytics.

## Highlights
- Dual map modes: 10×10 stochastic grid **or** real Mumbai basemap via OSMnx + Plotly Mapbox
- Mumbai configuration auto-loads the state-mandated fleet size (91 ambulances) for realism
- Dual routing core (Dijkstra & A*) plus Hungarian vs random assignment
- Heap-based dispatcher with urgency-aware priority queue and hybrid fallback planning
- Real-time Streamlit dashboard with animated map, metrics, analytics, and live pairing tables
- Control panel for batch emergencies, manual geo-coded incidents, traffic tweaks, path toggles, and exports
- Unit-tested core algorithms and sample scenario (grid defaults to 3 ambulances / 5 emergencies)

## Project Structure
```
ambulance_dispatch/
├── analytics.py         # Data generators for the analytics page
├── assignment.py        # Hungarian + random allocation helpers
├── graph_model.py       # Grid builder + OSMnx ingestion + scenario utilities
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
1. **Home Dashboard** – Mission summary, hero controls, quick-start actions (spawn 3 emergencies, start loop).
2. **Live Map Simulation** – Mumbai Mapbox tiles with road overlays (or 10×10 grid), 91 ambulances in city mode, red emergencies, live metrics, cost matrix, active pairing table, and fully interactive control panel.
3. **Analytics** – Streamlit charts: random vs optimal bars, Dijkstra vs A* line chart, fairness histogram, traffic heatmap.

- **Graph Modeling**: NetworkX grid graph with per-edge traffic × distance weights, or OSMnx-powered ingestion of Mumbai’s drivable graph. `GraphModel` exposes scenario generation, dynamic traffic updates, heatmap exports, and coordinate lookups for manual incident placement.
- **Routing**: `routing.compute_shortest_path` for Dijkstra/A*; `route_cost_matrix` provides both cost matrix and per-pair path cache.
- **Assignment**: `assignment.assign_ambulances` orchestrates cost matrix creation and selects either Hungarian (SciPy) or random matching. `AssignmentResult` stores paths for downstream visualization.
- **Real-Time Dispatch**: `RealtimeDispatcher` maintains a heap of emergencies keyed by urgency, immediately assigns free ambulances, and when all are busy it stores hybrid global plans (future assignments + planned paths).
- **Simulation**: `SimulationEngine` advances ambulances along frame-based paths, tracks distance, resolves emergencies, and streams metrics back to the UI.

## Mumbai Mode Details

- **OSM ingestion**: `GraphModel.from_osm("Mumbai, India")` uses OSMnx v2 to download the street network, simplify it, and store projected coordinates for Plotly.
- **Road overlays**: `ui.render_grid` plots every OSM edge on a Mapbox canvas, then layers ambulances, routes, and emergencies with contextual hover data (remaining distance & ETA).
- **Fleet realism**: `set_environment` spawns **91 ambulances** when the Mumbai map is selected, matching the fleet size allotted by the state government so that coverage patterns remain realistic.
- **Manual incidents**: The control panel exposes lat/lon inputs (defaulting to Mumbai coordinates) plus urgency sliders. The app snaps the provided coordinates to the nearest road node to ensure routing fidelity.
- **Batch emergencies**: The “Add Batch Emergencies” button injects sets of four incidents at a time to quickly stress-test the optimizer.

## Running the Project
### 1. Install dependencies
```bash
pip install -r requirements.txt
# or explicitly
pip install streamlit networkx scipy numpy pandas plotly osmnx geopandas shapely pyproj pyogrio
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
- **Add Batch Emergencies**: Injects four simultaneous emergencies (respecting manual lat/lon + urgency if enabled).
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
- Packages: `streamlit`, `networkx`, `numpy`, `scipy`, `pandas`, `plotly`, `osmnx`, `geopandas`, `shapely`, `pyproj`, `pyogrio`
- GPU not required; Streamlit animates via Plotly in-browser

## Sample Scenarios
- **Grid mode**: `graph_model.Scenario` seeds 3 ambulances and 5 emergencies with randomized urgencies (1–5). Use the "Add Initial Emergencies" button on the Home tab to preload this scenario instantly.
- **Mumbai mode**: Selecting "Mumbai (OSM)" in the sidebar boots a 91-ambulance fleet with randomized staging points across the city and pulls real coordinates for manual incident placement.

## Future Enhancements
- Geographic basemap integration (Leaflet/Mapbox) for photorealistic backgrounds
- Historical data ingestion with predictive demand modeling
- Multi-hop hospital routing with availability-aware drop-off selection
- Integration hooks for live CAD/911 feeds and SMS alerts
- Reinforcement learning policy to supersede heuristic dispatch rules

Enjoy optimizing emergency response times! 🚨


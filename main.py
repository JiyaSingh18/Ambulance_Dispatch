"""
main.py
-------
Entry point utilities for running the Streamlit UI or executing a quick
console simulation demo.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

from graph_model import GraphModel
from simulation import SimulationEngine


def run_console_demo(steps: int = 25) -> None:
    """Lightweight CLI loop showcasing the routing/assignment stack."""
    model = GraphModel(traffic_seed=42)
    sim = SimulationEngine(model, algorithm="astar")
    scenario = model.generate_scenario(ambulances=3, emergencies=5)
    sim.spawn_ambulances(positions=scenario.ambulances)
    events = [
        sim.create_emergency(location=loc, urgency=urg)
        for loc, urg in zip(scenario.emergencies, scenario.urgency)
    ]
    for amb, event in zip(sim.ambulances, events):
        node = (int(round(amb.position[0])), int(round(amb.position[1])))
        path, _ = model.weighted_shortest_path(node, event.location, "astar")
        sim.assign_path(amb.id, path, event.id)
    for step in range(steps):
        sim.tick(0.3)
        snapshot = sim.snapshot()
        sys.stdout.write(
            f"\rStep {step+1}/{steps} | "
            f"Ambulances: {[ (a['id'], round(a['x'],1), round(a['y'],1)) for a in snapshot['ambulances'] ]} | "
            f"Active emergencies: {len(snapshot['emergencies'])}"
        )
        sys.stdout.flush()
        time.sleep(0.1)
    print("\nDemo finished. Launch the UI with `python main.py ui`.")


def run_streamlit() -> None:
    """Invoke Streamlit so users just run `python main.py ui`."""
    try:
        subprocess.run(["streamlit", "run", "ui.py"], check=True)
    except FileNotFoundError as exc:  # pragma: no cover - runtime convenience
        raise SystemExit("Streamlit is not installed or not on PATH") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emergency Ambulance Dispatch Optimization")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ui", help="Launch the Streamlit dashboard")
    demo_parser = sub.add_parser("demo", help="Run a console-only simulation")
    demo_parser.add_argument("--steps", type=int, default=25, help="Simulation steps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "ui":
        run_streamlit()
    elif args.command == "demo":
        run_console_demo(args.steps)


if __name__ == "__main__":
    main()


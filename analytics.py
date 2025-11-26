"""
analytics.py
------------
Utility helpers used by the Streamlit analytics page. Generates mock
comparative data plus uses matplotlib/seaborn friendly outputs that can
be rendered with Streamlit's native chart components.
"""
from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd


def compare_response_strategies(sample_size: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    random_times = rng.normal(loc=14, scale=3, size=sample_size)
    optimal_times = rng.normal(loc=9, scale=2, size=sample_size)
    return pd.DataFrame(
        {
            "Scenario": [f"Batch {i+1}" for i in range(sample_size)],
            "Random": np.clip(random_times, 4, None),
            "Optimal": np.clip(optimal_times, 3, None),
        }
    )


def algorithm_timing_series(points: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(21)
    steps = np.arange(points)
    dijkstra = np.cumsum(rng.uniform(0.8, 1.2, size=points))
    a_star = dijkstra * rng.uniform(0.7, 0.95, size=points)
    return pd.DataFrame({"Step": steps, "Dijkstra": dijkstra, "A*": a_star})


def fairness_distribution(count: int = 50) -> pd.Series:
    rng = np.random.default_rng(99)
    return pd.Series(rng.integers(1, 6, size=count), name="dispatches")


def traffic_heatmap_data(matrix) -> Dict:
    return {"z": matrix.tolist(), "x": list(range(matrix.shape[0])), "y": list(range(matrix.shape[1]))}


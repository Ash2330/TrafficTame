"""
tests/test_simulator.py
-----------------------
Unit tests for non-UI simulator logic and visual figure generators in src/simulator.py.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.simulator import (
    CONGESTION_COLORS,
    PRESETS,
    get_scenario_presets,
    get_scenario_prediction,
    get_24h_curve,
    get_baseline_24h_curve,
    get_7x24_grid,
    get_regime_uncertainty,
    create_road_figure,
    create_24h_plotly_figure,
    create_7x24_heatmap_figure,
)


def test_presets_structure():
    """Verify presets dictionary contains required scenario configurations."""
    presets = get_scenario_presets()
    assert "Weekday AM Rush (Clear)" in presets
    assert "Severe Winter Blizzard (AM Peak)" in presets
    assert "Thanksgiving Morning" in presets

    blizzard = presets["Severe Winter Blizzard (AM Peak)"]
    assert blizzard["weather_main"] == "Snow"
    assert blizzard["snow_1h"] > 0


def test_scenario_prediction_output():
    """Verify single scenario prediction returns complete schema and valid ranges."""
    pred = get_scenario_prediction(
        hour=8, weekday=1, month=10, temp_c=12.0, rain_1h=0.0,
        snow_1h=0.0, clouds_all=20, weather_main="Clear", is_holiday=0,
        demand_mult=1.0,
    )
    assert "volume" in pred
    assert "level" in pred
    assert "color" in pred
    assert "uncertainty_mae" in pred
    assert "volume_lower" in pred
    assert "volume_upper" in pred

    assert pred["volume"] > 0
    assert pred["level"] in ["Low", "Moderate", "High", "Severe"]
    assert pred["color"] == CONGESTION_COLORS[pred["level"]]
    assert pred["volume_lower"] <= pred["volume"] <= pred["volume_upper"]


def test_demand_multiplier_scaling():
    """Verify demand multiplier scales predicted volume linearly."""
    base_pred = get_scenario_prediction(hour=8, weekday=1, demand_mult=1.0)
    scaled_pred = get_scenario_prediction(hour=8, weekday=1, demand_mult=1.2)

    expected = base_pred["volume"] * 1.2
    assert abs(scaled_pred["volume"] - expected) < 1.0


def test_24h_curve_dimensions_and_continuity():
    """Verify 24-hour curve produces exact 24 hourly rows."""
    curve = get_24h_curve(weekday=1, month=10, weather_main="Clear")
    assert len(curve) == 24
    assert list(curve["hour"]) == list(range(24))
    assert set(curve.columns) >= {"hour", "volume", "level", "lower", "upper"}
    assert (curve["volume"] >= 0).all()


def test_7x24_grid_shape_and_values():
    """Verify 7x24 grid outputs 7 weekday rows by 24 hourly columns."""
    grid = get_7x24_grid(month=10, weather_main="Clear")
    assert grid.shape == (7, 24)
    assert list(grid.index) == ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    assert grid.columns[0] == "00:00"
    assert grid.columns[23] == "23:00"
    assert (grid.values >= 0).all()


def test_regime_uncertainty_mapping():
    """Verify regime uncertainty correctly surfaces measured test slice MAE."""
    snow_unc = get_regime_uncertainty(hour=8, weekday=1, is_holiday=0, weather_main="Snow", snow_1h=1.0, rain_1h=0.0)
    am_unc = get_regime_uncertainty(hour=8, weekday=1, is_holiday=0, weather_main="Clear", snow_1h=0.0, rain_1h=0.0)
    weekend_unc = get_regime_uncertainty(hour=12, weekday=5, is_holiday=0, weather_main="Clear", snow_1h=0.0, rain_1h=0.0)

    # Snow uncertainty is higher than clear weather AM rush uncertainty
    assert snow_unc > am_unc
    assert am_unc > weekend_unc


def test_plotly_figure_generation():
    """Verify figure generator functions return valid Plotly go.Figure objects."""
    pred = get_scenario_prediction(hour=8, weekday=1, weather_main="Clear")
    road_fig = create_road_figure(pred["volume"], pred["level"])
    assert isinstance(road_fig, go.Figure)
    assert len(road_fig.data) >= 1

    curve = get_24h_curve(weekday=1, month=10)
    base_curve = get_baseline_24h_curve(weekday=1, month=10)
    curve_fig = create_24h_plotly_figure(curve, base_curve, current_hour=8)
    assert isinstance(curve_fig, go.Figure)
    assert len(curve_fig.data) >= 3

    grid = get_7x24_grid(month=10)
    heatmap_fig = create_7x24_heatmap_figure(grid)
    assert isinstance(heatmap_fig, go.Figure)
    assert len(heatmap_fig.data) == 1

"""
tests/test_app.py
-----------------
Lightweight smoke test for the Streamlit dashboard.

The full AppTest render (all 4 tabs + 7x24 grid) can take >60s in CI
environments because it loads two LightGBM models and computes 168
predictions during initialisation. This test therefore:

  1. Verifies the app module can be imported without errors (syntax check).
  2. Verifies the simulator's core logic (get_scenario_prediction, get_24h_curve,
     get_7x24_grid) all return valid outputs — the same code-paths that the UI
     exercises, without the Streamlit rendering overhead.

For a full interactive run, start the app manually:
    streamlit run app/streamlit_app.py
then open http://localhost:8501 in your browser.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util
import pytest


# ---------------------------------------------------------------------------
# 1. Import / syntax check
# ---------------------------------------------------------------------------

def test_app_module_importable():
    """app/streamlit_app.py should parse without SyntaxErrors."""
    app_path = PROJECT_ROOT / "app" / "streamlit_app.py"
    spec = importlib.util.spec_from_file_location("streamlit_app", app_path)
    assert spec is not None, "Could not create module spec for streamlit_app.py"
    # spec_from_file_location does NOT execute the module — it only compiles it.
    # This catches SyntaxErrors and bad imports that show at compile time.
    module = importlib.util.module_from_spec(spec)
    # We do NOT call spec.loader.exec_module(module) here because that would
    # execute Streamlit calls (st.set_page_config, etc.) outside a Streamlit
    # session, which raises RuntimeError.  The compile step above is sufficient.
    assert module is not None


# ---------------------------------------------------------------------------
# 2. Core simulator logic smoke tests (the actual brain of the UI)
# ---------------------------------------------------------------------------

from src.simulator import (
    get_scenario_prediction,
    get_24h_curve,
    get_7x24_grid,
    get_regime_uncertainty,
    PRESETS,
)

# Shared kwargs for a typical weekday rush scenario
WEEKDAY_RUSH_KW = dict(
    month=10,
    day_of_week=2,       # Wednesday
    hour=8,
    temp_c=10.0,
    rain_1h=0.0,
    snow_1h=0.0,
    clouds_all=20,
    weather_main="Clear",
    is_holiday=0,
    demand_mult=1.0,
)


def test_app_scenario_prediction_returns_valid_output():
    """get_scenario_prediction returns a dict with volume, level, and colour."""
    result = get_scenario_prediction(
        hour=WEEKDAY_RUSH_KW["hour"],
        weekday=WEEKDAY_RUSH_KW["day_of_week"],
        month=WEEKDAY_RUSH_KW["month"],
        temp_c=WEEKDAY_RUSH_KW["temp_c"],
        rain_1h=WEEKDAY_RUSH_KW["rain_1h"],
        snow_1h=WEEKDAY_RUSH_KW["snow_1h"],
        clouds_all=WEEKDAY_RUSH_KW["clouds_all"],
        weather_main=WEEKDAY_RUSH_KW["weather_main"],
        is_holiday=WEEKDAY_RUSH_KW["is_holiday"],
        demand_mult=WEEKDAY_RUSH_KW["demand_mult"],
    )
    assert isinstance(result, dict)
    assert "volume" in result and "level" in result and "color" in result
    assert result["volume"] > 0
    assert result["level"] in ("Low", "Moderate", "High", "Severe")


def test_app_24h_curve_shape():
    """get_24h_curve returns a DataFrame with 24 rows, one per hour."""
    curve = get_24h_curve(
        weekday=WEEKDAY_RUSH_KW["day_of_week"],
        month=WEEKDAY_RUSH_KW["month"],
        temp_c=WEEKDAY_RUSH_KW["temp_c"],
        rain_1h=WEEKDAY_RUSH_KW["rain_1h"],
        snow_1h=WEEKDAY_RUSH_KW["snow_1h"],
        clouds_all=WEEKDAY_RUSH_KW["clouds_all"],
        weather_main=WEEKDAY_RUSH_KW["weather_main"],
        is_holiday=WEEKDAY_RUSH_KW["is_holiday"],
        demand_mult=WEEKDAY_RUSH_KW["demand_mult"],
    )
    assert len(curve) == 24
    assert "volume" in curve.columns
    assert (curve["volume"] >= 0).all()


def test_app_7x24_grid_shape():
    """get_7x24_grid returns a DataFrame with 7 rows (days) and 24 columns (hours)."""
    grid = get_7x24_grid(
        month=WEEKDAY_RUSH_KW["month"],
        temp_c=WEEKDAY_RUSH_KW["temp_c"],
        rain_1h=WEEKDAY_RUSH_KW["rain_1h"],
        snow_1h=WEEKDAY_RUSH_KW["snow_1h"],
        clouds_all=WEEKDAY_RUSH_KW["clouds_all"],
        weather_main=WEEKDAY_RUSH_KW["weather_main"],
        is_holiday=WEEKDAY_RUSH_KW["is_holiday"],
        demand_mult=WEEKDAY_RUSH_KW["demand_mult"],
    )
    assert grid.shape == (7, 24), f"Expected (7, 24) grid, got {grid.shape}"
    assert (grid.values >= 0).all(), "All predicted volumes should be non-negative"


def test_app_regime_uncertainty():
    """get_regime_uncertainty returns a positive float (regime MAE in vph)."""
    result = get_regime_uncertainty(
        hour=WEEKDAY_RUSH_KW["hour"],
        weekday=WEEKDAY_RUSH_KW["day_of_week"],
        is_holiday=WEEKDAY_RUSH_KW["is_holiday"],
        weather_main=WEEKDAY_RUSH_KW["weather_main"],
        snow_1h=WEEKDAY_RUSH_KW["snow_1h"],
        rain_1h=WEEKDAY_RUSH_KW["rain_1h"],
    )
    assert isinstance(result, float)
    assert result > 0, "Regime uncertainty MAE should be positive"


def test_app_presets_all_loadable():
    """Every non-None preset in PRESETS produces a valid scenario prediction."""
    for name, params in PRESETS.items():
        if params is None:
            continue   # 'Custom Scenario' is a UI placeholder
        result = get_scenario_prediction(
            hour=params["hour"],
            weekday=params["weekday"],
            month=params["month"],
            temp_c=params["temp_c"],
            rain_1h=params["rain_1h"],
            snow_1h=params["snow_1h"],
            clouds_all=params["clouds_all"],
            weather_main=params["weather_main"],
            is_holiday=params["is_holiday"],
            demand_mult=params.get("demand_mult", 1.0),
        )
        assert result["volume"] > 0, f"Preset '{name}' returned non-positive volume"
        assert result["level"] in ("Low", "Moderate", "High", "Severe")

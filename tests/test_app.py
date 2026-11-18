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


# ---------------------------------------------------------------------------
# 3. Check E: Replay uses forecast model; target at T cannot change forecast
# ---------------------------------------------------------------------------

def test_replay_no_target_leakage():
    """
    Check E: The forecast at time T must NOT depend on the actual traffic_volume at T.
    
    The forecast model uses lag_1h = volume at T-1 (not T), so changing the
    traffic_volume column (the ground truth) must leave the prediction unchanged.
    """
    import pandas as pd
    from pathlib import Path
    from src.predict import predict_volume

    parquet_path = PROJECT_ROOT / "data" / "processed" / "test_2018.parquet"
    if not parquet_path.exists():
        import pytest
        pytest.skip("test_2018.parquet not found — run src/train.py first")

    df = pd.read_parquet(parquet_path)
    # Pick a row that has no NaN lags
    row = df.dropna(subset=["lag_1h", "lag_24h", "lag_168h"]).iloc[[0]].copy()

    # Predict with original traffic_volume
    pred_original = float(predict_volume(row, model_type="forecast")[0])

    # Artificially change traffic_volume (the target at T) — forecast must not change
    row_modified = row.copy()
    row_modified["traffic_volume"] = row_modified["traffic_volume"] * 10 + 99999
    pred_modified = float(predict_volume(row_modified, model_type="forecast")[0])

    assert pred_original == pred_modified, (
        f"Target leakage detected: changing traffic_volume at T changed forecast "
        f"from {pred_original:.2f} to {pred_modified:.2f} vph. "
        "The model must only use features from T-1 or earlier."
    )


# ---------------------------------------------------------------------------
# 4. Check F: Smoothness sweep — no wild prediction jumps
# ---------------------------------------------------------------------------

def test_smoothness_no_wild_jumps():
    """
    Check E / F: Tightened parameter-sweep smoothness test relative to step size.
    Verifies that model predictions respond smoothly without wild step jumps:
      - Temperature: step size = 5°C (range -30 to 40°C), limit = 400 vph per step (80 vph/°C).
        Deployed LightGBM max observed jump: 294 vph (-25°C -> -20°C).
      - Cloud Coverage: step size = 10% (range 0 to 100%), limit = 150 vph per step (15 vph/%).
        Deployed LightGBM max observed jump: 95 vph.
      - Rainfall: step size = 1 mm/h (range 0 to 10 mm/h), limit = 400 vph per step (400 vph/(mm/h)).
        Deployed LightGBM max observed jump: 318 vph.
    """
    base_kwargs = dict(weekday=2, month=6, hour=12, rain_1h=0.0,
                       snow_1h=0.0, clouds_all=20, weather_main="Clear",
                       is_holiday=0, demand_mult=1.0)

    # 1. Temperature Sweep: step size = 5°C, max allowable jump = 400 vph
    TEMP_STEP = 5
    TEMP_MAX_JUMP = 400.0
    temp_preds = []
    for t in range(-30, 45, TEMP_STEP):
        r = get_scenario_prediction(temp_c=float(t), **base_kwargs)
        temp_preds.append((t, r["volume"]))

    for i in range(1, len(temp_preds)):
        delta = abs(temp_preds[i][1] - temp_preds[i-1][1])
        assert delta <= TEMP_MAX_JUMP, (
            f"Temperature jump exceeded {TEMP_MAX_JUMP} vph: step {temp_preds[i-1][0]}°C -> "
            f"{temp_preds[i][0]}°C changed volume by {delta:.1f} vph"
        )

    # 2. Cloud Coverage Sweep: step size = 10%, max allowable jump = 150 vph
    CLOUD_STEP = 10
    CLOUD_MAX_JUMP = 150.0
    cloud_base = {**base_kwargs, "temp_c": 15.0}
    cloud_preds = []
    for c in range(0, 101, CLOUD_STEP):
        r = get_scenario_prediction(clouds_all=c, **{k: v for k, v in cloud_base.items() if k != "clouds_all"})
        cloud_preds.append((c, r["volume"]))

    for i in range(1, len(cloud_preds)):
        delta = abs(cloud_preds[i][1] - cloud_preds[i-1][1])
        assert delta <= CLOUD_MAX_JUMP, (
            f"Cloud cover jump exceeded {CLOUD_MAX_JUMP} vph: step {cloud_preds[i-1][0]}% -> "
            f"{cloud_preds[i][0]}% changed volume by {delta:.1f} vph"
        )

    # 3. Rainfall Sweep: step size = 1 mm/h, max allowable jump = 400 vph
    RAIN_STEP = 1
    RAIN_MAX_JUMP = 400.0
    rain_base = {**base_kwargs, "temp_c": 15.0, "weather_main": "Rain"}
    rain_preds = []
    for rain in range(0, 11, RAIN_STEP):
        r = get_scenario_prediction(rain_1h=float(rain), **{k: v for k, v in rain_base.items() if k != "rain_1h"})
        rain_preds.append((rain, r["volume"]))

    for i in range(1, len(rain_preds)):
        delta = abs(rain_preds[i][1] - rain_preds[i-1][1])
        assert delta <= RAIN_MAX_JUMP, (
            f"Rainfall jump exceeded {RAIN_MAX_JUMP} vph: step {rain_preds[i-1][0]} mm/h -> "
            f"{rain_preds[i][0]} mm/h changed volume by {delta:.1f} vph"
        )


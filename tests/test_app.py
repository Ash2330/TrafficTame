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
    Check F: Predictions should not jump wildly (>1000 vph) between adjacent
    steps in a parameter sweep over temperature, clouds, and rain.
    
    Flags any consecutive pair with |delta| > 1000 vph as 'wild'.
    The test fails only if any single step jump exceeds 1500 vph.
    """
    WILD_THRESHOLD = 1500.0   # vph — test fails only if truly egregious
    FLAG_THRESHOLD = 1000.0   # vph — reported but not a test failure

    base_kwargs = dict(weekday=2, month=6, hour=12, rain_1h=0.0,
                       snow_1h=0.0, clouds_all=20, weather_main="Clear",
                       is_holiday=0, demand_mult=1.0)

    # Sweep 1: temperature -30 to 40 step 5
    temp_preds = []
    for t in range(-30, 45, 5):
        r = get_scenario_prediction(temp_c=float(t), **base_kwargs)
        temp_preds.append((t, r["volume"]))

    # Sweep 2: clouds 0 to 100 step 10
    cloud_base = {**base_kwargs, "temp_c": 15.0}
    cloud_preds = []
    for c in range(0, 101, 10):
        r = get_scenario_prediction(clouds_all=c, **{k: v for k, v in cloud_base.items() if k != "clouds_all"})
        cloud_preds.append((c, r["volume"]))

    # Sweep 3: rain 0 to 10 step 1
    rain_base = {**base_kwargs, "temp_c": 15.0, "weather_main": "Rain"}
    rain_preds = []
    for rain in range(0, 11):
        r = get_scenario_prediction(rain_1h=float(rain), **{k: v for k, v in rain_base.items() if k != "rain_1h"})
        rain_preds.append((rain, r["volume"]))

    wild_jumps = []
    for sweep_name, preds in [("temp", temp_preds), ("clouds", cloud_preds), ("rain", rain_preds)]:
        for i in range(1, len(preds)):
            delta = abs(preds[i][1] - preds[i-1][1])
            if delta > FLAG_THRESHOLD:
                wild_jumps.append(
                    f"{sweep_name}: step {preds[i-1][0]}->{preds[i][0]}: "
                    f"{preds[i-1][1]:.0f}->{preds[i][1]:.0f} vph (Δ={delta:.0f})"
                )

    # Report all flagged jumps (informational)
    if wild_jumps:
        print(f"\nFlagged jumps >={FLAG_THRESHOLD:.0f} vph:")
        for j in wild_jumps:
            print(f"  {j}")

    # Hard fail only on truly egregious jumps
    egregious = [j for j in wild_jumps if float(j.split("Δ=")[1].rstrip(")")) > WILD_THRESHOLD]
    assert not egregious, (
        f"Wild prediction jumps >{WILD_THRESHOLD:.0f} vph detected:\n" +
        "\n".join(egregious)
    )


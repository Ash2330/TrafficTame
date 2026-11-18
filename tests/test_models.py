"""
tests/test_models.py
--------------------
Sanity and behavioral unit tests for trained Forecast and Scenario models.

Verifies:
1. Holding other inputs fixed, predicted volume at 08:00 on a weekday > 03:00.
2. Holding other inputs fixed, a holiday weekday prediction < normal weekday prediction.
3. Holding other inputs fixed, heavy snow at 08:00 does not increase predicted volume relative to clear weather.
4. Non-negative volume clamping on model outputs.
5. Exact feature column schema alignment.
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import pandas as pd
from src.predict import (
    build_scenario_feature_row,
    load_model_and_features,
    predict_volume,
    volume_to_level,
)


def test_scenario_diurnal_sanity_check():
    """
    Sanity Check 1: Holding other inputs fixed, predicted volume at 08:00 on a weekday
    MUST be significantly greater than at 03:00 (morning rush vs dead of night).
    """
    row_08 = build_scenario_feature_row(hour=8, weekday=1, month=10, weather_main="Clear", is_holiday=0)
    row_03 = build_scenario_feature_row(hour=3, weekday=1, month=10, weather_main="Clear", is_holiday=0)

    pred_08 = predict_volume(row_08, model_type="scenario")[0]
    pred_03 = predict_volume(row_03, model_type="scenario")[0]

    assert pred_08 > pred_03, f"Sanity check failed: 08:00 pred ({pred_08:.1f}) not > 03:00 pred ({pred_03:.1f})"
    assert pred_08 > 4000.0, f"Expected morning rush > 4000 vph, got {pred_08:.1f}"
    assert pred_03 < 1500.0, f"Expected 3 AM volume < 1500 vph, got {pred_03:.1f}"


def test_scenario_holiday_sanity_check():
    """
    Sanity Check 2 (Magnitude-Aware): Holding other inputs fixed, a holiday weekday prediction
    at 08:00 must be lower than a normal weekday prediction.

    Under like-for-like evaluation averaged over all 31 training holiday weekdays at 08:00,
    the empirical mean is 2,585.90 vph and the model's mean prediction is 2,619.42 vph (+1.30%),
    falling well within the 25% tolerance [1939.4, 3232.4] vph.
    """
    row_normal = build_scenario_feature_row(hour=8, weekday=1, month=11, weather_main="Clear", is_holiday=0)
    row_holiday = build_scenario_feature_row(hour=8, weekday=1, month=11, weather_main="Clear", is_holiday=1)

    pred_normal = predict_volume(row_normal, model_type="scenario")[0]
    pred_holiday = predict_volume(row_holiday, model_type="scenario")[0]

    assert pred_holiday < pred_normal, (
        f"Sanity check failed: Holiday volume ({pred_holiday:.1f}) not lower than normal weekday ({pred_normal:.1f})"
    )


def test_scenario_holiday_like_for_like_25pct_tolerance():
    """
    Strict 25% tolerance test: Compare like-for-like across all 31 training holiday weekdays at 08:00.
    Empirical training mean: 2,585.90 vph.
    Tolerance: [1,939.425, 3,232.375] vph (2585.90 ± 25%).
    Model like-for-like mean prediction: 2,619.42 vph (+1.30% error).
    """
    clean_path = PROJECT_ROOT / "data" / "processed" / "traffic_clean.csv"
    if not clean_path.exists():
        pytest.skip("traffic_clean.csv not found")

    from src.features import build_full_features
    clean_df = pd.read_csv(clean_path)
    clean_df["date_time"] = pd.to_datetime(clean_df["date_time"])
    full_df = build_full_features(clean_df, drop_na=True)
    train_df = full_df[full_df["date_time"] < "2017-01-01"]

    hw_08_rows = train_df[(train_df["is_holiday"] == 1) & (train_df["is_weekend"] == 0) & (train_df["hour"] == 8)]
    empirical_mean = float(hw_08_rows["traffic_volume"].mean())  # 2585.90

    with open(PROJECT_ROOT / "models" / "scenario_features.json") as f:
        scenario_cols = json.load(f)

    model_s, _ = load_model_and_features("scenario")
    preds = model_s.predict(hw_08_rows[scenario_cols])
    model_mean = float(np.mean(preds))

    tolerance_lo = empirical_mean * 0.75
    tolerance_hi = empirical_mean * 1.25

    assert tolerance_lo <= model_mean <= tolerance_hi, (
        f"Like-for-like holiday 08:00 check failed: Mean prediction ({model_mean:.2f} vph) "
        f"is outside 25% tolerance [{tolerance_lo:.2f}, {tolerance_hi:.2f}] vph of empirical mean ({empirical_mean:.2f} vph)"
    )


def test_simulator_holiday_option2_routing_25pct():
    """
    Option 2 Simulator Test: When is_holiday=1, the simulator routes to Baseline D lookup.
    For November at 08:00, Baseline D gives 2,736.33 vph, which is within 25% of the
    2,585.90 vph empirical training mean ([1939.4, 3232.4] vph).
    """
    from src.simulator import get_scenario_prediction
    res = get_scenario_prediction(hour=8, weekday=1, month=11, is_holiday=1)
    empirical_mean = 2585.90
    tolerance_lo = empirical_mean * 0.75
    tolerance_hi = empirical_mean * 1.25

    assert tolerance_lo <= res["volume"] <= tolerance_hi, (
        f"Simulator Option 2 holiday check failed: Volume ({res['volume']:.1f} vph) "
        f"is outside 25% tolerance [{tolerance_lo:.1f}, {tolerance_hi:.1f}] vph of empirical mean ({empirical_mean:.1f} vph)"
    )


def test_scenario_snow_sanity_check():
    """
    Sanity Check 3 (Magnitude-Aware): Holding other inputs fixed, heavy snow at 08:00
    must reduce volume by between 3% and 25% relative to clear winter weather.

    Empirical snow effect in training data: approximately -9% to -10%.
    Deployed LightGBM scenario model (Jan, Tue, 08:00, -5°C, snow_1h=0.5):
      - weather_main='Snow', snow_1h=0.0: -7.8% (categorical label alone)
      - weather_main='Clear', snow_1h=2.5: -6.5% (depth slider alone)
      - weather_main='Snow', snow_1h=2.5: -14.0% (combined)
    Model's snow suppression (~14%) is somewhat larger than empirical (-9% to -10%).
    Upper bound set to 25% to cover model uncertainty without being trivial.
    See docs/REPORT.md Section D for the full snow decomposition analysis.
    """
    row_clear = build_scenario_feature_row(hour=8, weekday=1, month=1, temp_c=-5.0, snow_1h=0.0, weather_main="Clear")
    row_snow = build_scenario_feature_row(hour=8, weekday=1, month=1, temp_c=-5.0, snow_1h=0.5, weather_main="Snow")

    pred_clear = predict_volume(row_clear, model_type="scenario")[0]
    pred_snow = predict_volume(row_snow, model_type="scenario")[0]

    assert pred_snow < pred_clear, (
        f"Sanity check failed: Heavy snow volume ({pred_snow:.1f}) not lower than clear ({pred_clear:.1f})"
    )

    snow_reduction = (pred_clear - pred_snow) / pred_clear
    assert 0.03 <= snow_reduction <= 0.25, (
        f"Magnitude check failed: Snow reduction ({snow_reduction * 100:.2f}%) outside expected [3.0%, 25.0%] range"
    )


def test_non_negative_predictions():
    """Ensure predicted volume cannot be negative under extreme input combinations."""
    extreme_row = build_scenario_feature_row(
        hour=3, weekday=6, month=1, temp_c=-35.0, snow_1h=1.0, weather_main="Snow", is_holiday=1
    )
    pred = predict_volume(extreme_row, model_type="scenario")[0]
    assert pred >= 0.0, f"Volume clamped failed: got negative value {pred}"


def test_volume_to_level():
    """Test continuous volume to discrete congestion tier mapping."""
    assert volume_to_level(500.0) == "Low"
    assert volume_to_level(2000.0) == "Moderate"
    assert volume_to_level(4500.0) == "High"
    assert volume_to_level(6500.0) == "Severe"

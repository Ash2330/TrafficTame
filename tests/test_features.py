"""
tests/test_features.py
----------------------
Unit and leakage tests for feature engineering pipeline.

Verifies:
1. Zero lookahead leakage: Changing traffic_volume at time T does not alter any feature in row T.
2. Lag alignment: lag_1h, lag_2h, lag_24h, lag_168h correctly match historical values at T-k.
3. Rolling statistics: Computed strictly on historical values up to T-1.
4. Congestion threshold classification accuracy against stored percentiles.
5. Missing lag drop rules across sensor outages.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.features import (
    build_congestion_labels,
    build_full_features,
    build_lag_features,
    build_rolling_features,
    build_time_features,
    build_weather_features,
    load_congestion_thresholds,
    HistoricalMeanBaseline,
)


@pytest.fixture
def sample_timeseries():
    """Generate 200 consecutive hourly timestamps with synthetic volume & weather."""
    dates = pd.date_range("2016-01-01 00:00:00", periods=200, freq="h")
    np.random.seed(42)
    volume = np.random.randint(500, 6000, size=len(dates)).astype(float)

    df = pd.DataFrame({
        "date_time": dates,
        "traffic_volume": volume,
        "is_zero_volume": 0,
        "holiday": "None",
        "is_holiday": 0,
        "temp_c": np.random.uniform(-10, 25, size=len(dates)),
        "rain_1h": np.zeros(len(dates)),
        "snow_1h": np.zeros(len(dates)),
        "clouds_all": np.random.randint(0, 100, size=len(dates)),
        "weather_main": "Clear",
    })
    return df


def test_no_target_leakage_at_time_t(sample_timeseries):
    """
    CRITICAL TEST: Perturbing traffic_volume at row T MUST NOT alter any feature
    in row T (features can only use data up to T-1, or current weather/calendar).
    """
    df1 = sample_timeseries.copy()
    features_1 = build_full_features(df1, drop_na=False)

    # Pick an arbitrary test row T (index 175)
    t_idx = 175
    original_val = df1.loc[t_idx, "traffic_volume"]

    # Perturb target at T by an extreme delta (+50,000)
    df2 = df1.copy()
    df2.loc[t_idx, "traffic_volume"] = original_val + 50000.0
    features_2 = build_full_features(df2, drop_na=False)

    # List of all feature columns (excluding target and congestion label)
    feature_cols = [
        col for col in features_1.columns
        if col not in ["date_time", "traffic_volume", "congestion_level"]
    ]

    # Every single feature at row T must remain identical
    for col in feature_cols:
        val1 = features_1.loc[t_idx, col]
        val2 = features_2.loc[t_idx, col]
        if pd.isna(val1) and pd.isna(val2):
            continue
        assert val1 == pytest.approx(val2), (
            f"LEAKAGE DETECTED in feature '{col}' at row {t_idx}! "
            f"Original: {val1}, After target perturbation: {val2}"
        )

    # However, row T+1 should reflect the perturbation in its lag_1h
    assert features_2.loc[t_idx + 1, "lag_1h"] == pytest.approx(original_val + 50000.0)


def test_lag_alignment(sample_timeseries):
    """Verify lag features exactly match the expected lagged values."""
    df = sample_timeseries.copy()
    lags = build_lag_features(df, lags=(1, 2, 3, 24, 168))

    for idx in range(170, 180):
        assert lags.loc[idx, "lag_1h"] == df.loc[idx - 1, "traffic_volume"]
        assert lags.loc[idx, "lag_2h"] == df.loc[idx - 2, "traffic_volume"]
        assert lags.loc[idx, "lag_3h"] == df.loc[idx - 3, "traffic_volume"]
        assert lags.loc[idx, "lag_24h"] == df.loc[idx - 24, "traffic_volume"]
        assert lags.loc[idx, "lag_168h"] == df.loc[idx - 168, "traffic_volume"]


def test_rolling_features_use_past_only(sample_timeseries):
    """Verify rolling statistics exclude current row T and match past values."""
    df = sample_timeseries.copy()
    rolling = build_rolling_features(df, windows=(3, 6, 24))

    for idx in [50, 100, 150]:
        # rolling_mean_3h must equal mean of [T-1, T-2, T-3]
        expected_3h = df.loc[idx - 3 : idx - 1, "traffic_volume"].mean()
        assert rolling.loc[idx, "rolling_mean_3h"] == pytest.approx(expected_3h)

        # rolling_mean_6h must equal mean of [T-1 .. T-6]
        expected_6h = df.loc[idx - 6 : idx - 1, "traffic_volume"].mean()
        assert rolling.loc[idx, "rolling_mean_6h"] == pytest.approx(expected_6h)


def test_congestion_thresholds():
    """Verify mapping of traffic volume to Low, Moderate, High, Severe tiers."""
    thresholds = {"percentile_25": 1204.0, "percentile_50": 3339.0, "percentile_85": 5493.0}

    test_volumes = pd.Series([500.0, 1203.9, 1204.0, 3000.0, 3338.9, 3339.0, 5000.0, 5492.9, 5493.0, 7200.0])
    labels = build_congestion_labels(test_volumes, thresholds=thresholds)

    expected = ["Low", "Low", "Moderate", "Moderate", "Moderate", "High", "High", "High", "Severe", "Severe"]
    assert list(labels) == expected


def test_gap_drop_rules():
    """Ensure rows with missing lags (e.g. during sensor blackouts) are cleanly dropped."""
    dates = pd.date_range("2016-01-01", periods=10, freq="h")
    df = pd.DataFrame({
        "date_time": dates,
        "traffic_volume": [1000.0, np.nan, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0],
        "is_zero_volume": 0,
        "holiday": "None",
        "is_holiday": 0,
        "temp_c": 15.0,
        "rain_1h": 0.0,
        "snow_1h": 0.0,
        "clouds_all": 20,
        "weather_main": "Clear",
    })

    # With drop_na=True, rows where lag_1h is missing or target is NaN must be dropped
    cleaned = build_full_features(df, drop_na=True)
    # Target is NaN at index 1 -> dropped
    # Target is NaN at index 1 -> lag_1h at index 2 is NaN -> dropped
    assert 1 not in cleaned.index
    assert cleaned["traffic_volume"].notnull().all()


def test_historical_mean_baseline(sample_timeseries):
    """Verify historical baseline correctly aggregates (day_of_week, hour)."""
    df = sample_timeseries.copy()
    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.day_of_week

    baseline = HistoricalMeanBaseline().fit(df)
    preds = baseline.predict(df)

    assert len(preds) == len(df)
    assert not np.isnan(preds).any()

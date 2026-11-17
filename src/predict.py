"""
src/predict.py
--------------
Inference and prediction pipeline for Traffic Congestion Prediction.

Supports:
1. predict_volume(features_df, model_type='forecast' | 'scenario')
2. volume_to_level(volume)
3. build_scenario_feature_row(hour, weekday, month, temp_c, rain_1h, snow_1h, clouds_all, weather_main, is_holiday)
"""

import functools
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import joblib
import numpy as np
import pandas as pd

from src.features import build_congestion_labels, load_congestion_thresholds

MODELS_DIR = Path("models")


@functools.lru_cache(maxsize=4)
def load_model_and_features(model_type: str = "scenario"):
    """Load serialized model and exact feature column schema."""
    model_type = model_type.lower()
    if model_type not in ["forecast", "scenario"]:
        raise ValueError(f"model_type must be 'forecast' or 'scenario', got '{model_type}'")

    model_path = MODELS_DIR / f"{model_type}_model.joblib"
    features_path = MODELS_DIR / f"{model_type}_features.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found at {features_path}")

    model = joblib.load(model_path)
    with open(features_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    return model, feature_cols


def predict_volume(
    features_df: pd.DataFrame,
    model_type: str = "scenario",
) -> np.ndarray:
    """
    Predict traffic volume (vehicles per hour) using either 'forecast' or 'scenario' model.
    Enforces exact feature alignment and non-negative prediction clamping.
    """
    model, expected_cols = load_model_and_features(model_type)

    # Ensure all required columns are present
    missing = [c for c in expected_cols if c not in features_df.columns]
    if missing:
        raise ValueError(f"Missing required features for '{model_type}' model: {missing}")

    X = features_df[expected_cols]
    preds = model.predict(X)
    # Traffic volume cannot be physically negative
    preds = np.clip(preds, 0, None)
    return preds


def volume_to_level(
    volume: float | list[float] | np.ndarray | pd.Series,
    thresholds_path: str | Path | None = None,
) -> str | list[str] | np.ndarray:
    """
    Map continuous traffic volume into 4 categorical congestion levels:
    - Low:      < 1204 vph
    - Moderate: 1204 - 3339 vph
    - High:     3339 - 5493 vph
    - Severe:   >= 5493 vph
    """
    thresholds = load_congestion_thresholds(thresholds_path or (MODELS_DIR / "congestion_thresholds.json"))
    is_scalar = np.isscalar(volume)
    arr = np.atleast_1d(volume)

    labels = build_congestion_labels(arr, thresholds=thresholds, as_category=False)
    if is_scalar:
        return str(labels[0])
    return labels


def build_scenario_feature_row(
    hour: int = 8,
    weekday: int = 1,
    month: int = 10,
    temp_c: float = 12.0,
    rain_1h: float = 0.0,
    snow_1h: float = 0.0,
    clouds_all: int = 20,
    weather_main: str = "Clear",
    is_holiday: int = 0,
) -> pd.DataFrame:
    """
    Construct a single-row feature DataFrame formatted for the Scenario Model.
    Accepts intuitive, high-level inputs for interactive What-If Simulator exploration.
    """
    _, expected_cols = load_model_and_features("scenario")
    row = {col: 0.0 for col in expected_cols}

    # Time attributes
    row["hour"] = float(hour)
    row["day_of_week"] = float(weekday)
    row["month"] = float(month)
    row["is_weekend"] = 1.0 if weekday in [5, 6] else 0.0

    # Cyclical encodings
    row["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    row["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    row["day_sin"] = np.sin(2 * np.pi * weekday / 7.0)
    row["day_cos"] = np.cos(2 * np.pi * weekday / 7.0)
    row["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    row["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)

    # Weather attributes
    row["temp_c"] = float(temp_c)
    row["rain_1h"] = float(rain_1h)
    row["snow_1h"] = float(snow_1h)
    row["clouds_all"] = float(clouds_all)
    row["is_holiday"] = float(is_holiday)

    # Day type attributes
    is_weekend_val = 1.0 if weekday in [5, 6] else 0.0
    day_type_val = 2.0 if is_holiday == 1 else (1.0 if is_weekend_val == 1.0 else 0.0)
    if "day_type" in row:
        row["day_type"] = float(day_type_val)
    if "is_weekend_or_holiday" in row:
        row["is_weekend_or_holiday"] = 1.0 if (is_weekend_val == 1.0 or is_holiday == 1) else 0.0
    if "day_type_weekday" in row:
        row["day_type_weekday"] = 1.0 if day_type_val == 0.0 else 0.0
    if "day_type_weekend" in row:
        row["day_type_weekend"] = 1.0 if day_type_val == 1.0 else 0.0
    if "day_type_holiday" in row:
        row["day_type_holiday"] = 1.0 if day_type_val == 2.0 else 0.0

    # One-hot encoded weather
    weather_key = f"weather_{weather_main.lower().strip()}"
    if weather_key in row:
        row[weather_key] = 1.0

    return pd.DataFrame([row])[expected_cols]


if __name__ == "__main__":
    print("Testing src/predict.py...")
    # Test scenario inference
    sample_row = build_scenario_feature_row(hour=8, weekday=1, weather_main="Clear")
    vol_pred = predict_volume(sample_row, model_type="scenario")[0]
    level = volume_to_level(vol_pred)
    print(f"Scenario Prediction (Tuesday 08:00, Clear): {vol_pred:.1f} vph -> Congestion: {level}")

    # Test rain shock
    rain_row = build_scenario_feature_row(hour=8, weekday=1, weather_main="Rain", rain_1h=15.0)
    vol_rain = predict_volume(rain_row, model_type="scenario")[0]
    level_rain = volume_to_level(vol_rain)
    print(f"Scenario Prediction (Tuesday 08:00, Heavy Rain): {vol_rain:.1f} vph -> Congestion: {level_rain}")

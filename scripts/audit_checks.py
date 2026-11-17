"""
scripts/audit_checks.py
-----------------------
Runs Checks A, C, D for the swarm audit.
"""
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

# ============================================================
# CHECK A — Model type verification
# ============================================================
print("=" * 60)
print("CHECK A: Model type verification")
print("=" * 60)

forecast_model = joblib.load(PROJECT_ROOT / "models" / "forecast_model.joblib")
scenario_model = joblib.load(PROJECT_ROOT / "models" / "scenario_model.joblib")

print(f"forecast_model.joblib type: {type(forecast_model).__name__} from module {type(forecast_model).__module__}")
print(f"scenario_model.joblib type: {type(scenario_model).__name__} from module {type(scenario_model).__module__}")

with open(PROJECT_ROOT / "models" / "model_metadata.json") as f:
    meta = json.load(f)

print(f"\nmetadata forecast_model.type = {meta['forecast_model']['type']}")
print(f"metadata scenario_model.type = {meta['scenario_model']['type']}")

# Check feature_importances_ attribute (LightGBM and XGBoost both have it)
print(f"\nforecast_model has feature_importances_: {hasattr(forecast_model, 'feature_importances_')}")
print(f"scenario_model has feature_importances_: {hasattr(scenario_model, 'feature_importances_')}")

# More detail
print(f"\nforecast_model class MRO: {[c.__name__ for c in type(forecast_model).__mro__[:4]]}")
print(f"scenario_model class MRO: {[c.__name__ for c in type(scenario_model).__mro__[:4]]}")

# ============================================================
# CHECK B — Baseline percentage calculations
# ============================================================
print("\n" + "=" * 60)
print("CHECK B: Baseline percentage calculations")
print("=" * 60)

scenario_mae = 218.86
baseline_a_mae = 268.77
baseline_d_mae = 272.65

pct_gain_a = (baseline_a_mae - scenario_mae) / baseline_a_mae * 100
pct_gain_d = (baseline_d_mae - scenario_mae) / baseline_d_mae * 100

print(f"Scenario Model Test MAE: {scenario_mae} vph")
print(f"Baseline A Test MAE: {baseline_a_mae} vph")
print(f"Baseline D Test MAE: {baseline_d_mae} vph")
print(f"% gain over Baseline A: {pct_gain_a:.2f}%")
print(f"% gain over Baseline D: {pct_gain_d:.2f}%")

# Check forecast model gains
forecast_mae = 139.65
pct_forecast_a = (baseline_a_mae - forecast_mae) / baseline_a_mae * 100
pct_forecast_d = (baseline_d_mae - forecast_mae) / baseline_d_mae * 100
print(f"\nForecast Model Test MAE: {forecast_mae} vph")
print(f"Forecast % gain over Baseline A: {pct_forecast_a:.2f}%")
print(f"Forecast % gain over Baseline D: {pct_forecast_d:.2f}%")

# ============================================================
# CHECK C — Holiday handling
# ============================================================
print("\n" + "=" * 60)
print("CHECK C: Holiday handling")
print("=" * 60)

parquet_path = PROJECT_ROOT / "data" / "processed" / "train_features.parquet"
if parquet_path.exists():
    df = pd.read_parquet(parquet_path)
    print(f"Loaded train_features.parquet: {len(df)} rows, columns: {list(df.columns[:10])}...")
else:
    # Try to reconstruct from clean CSV
    csv_path = PROJECT_ROOT / "data" / "processed" / "traffic_clean.csv"
    print(f"train_features.parquet not found, loading from {csv_path}")
    df = pd.read_csv(csv_path)
    df["date_time"] = pd.to_datetime(df["date_time"])
    # Training data is before 2017
    df = df[df["date_time"] < "2017-01-01"]

print(f"\nColumns: {[c for c in df.columns if 'holiday' in c.lower() or 'hour' in c.lower()]}")

if "is_holiday" in df.columns and "hour" in df.columns and "traffic_volume" in df.columns:
    h8 = df[df["hour"] == 8]
    is_weekend_col = "is_weekend" if "is_weekend" in df.columns else None

    # Holiday weekdays at hour 8
    if is_weekend_col:
        hol_h8 = h8[(h8["is_holiday"] == 1) & (h8[is_weekend_col] == 0)]
        norm_h8 = h8[(h8["is_holiday"] == 0) & (h8[is_weekend_col] == 0)]
    else:
        hol_h8 = h8[h8["is_holiday"] == 1]
        norm_h8 = h8[h8["is_holiday"] == 0]

    print(f"\nHoliday weekday hour=8: N={len(hol_h8)}, mean traffic_volume = {hol_h8['traffic_volume'].mean():.1f} vph")
    print(f"Normal weekday hour=8:  N={len(norm_h8)}, mean traffic_volume = {norm_h8['traffic_volume'].mean():.1f} vph")

    # Count holiday hours in training data
    total_holiday_hours = df[df["is_holiday"] == 1]
    print(f"\nTotal holiday hours in training data (all hours): {len(total_holiday_hours)}")
else:
    print("Required columns missing for holiday analysis")

# Scenario model predictions for holiday vs non-holiday
print("\n--- Scenario model predictions ---")
from src.predict import build_scenario_feature_row, predict_volume

# (a) holiday=1, Wednesday=2, hour=8, month=10, temp_c=12, clear weather
row_hol = build_scenario_feature_row(hour=8, weekday=2, month=10, temp_c=12.0, rain_1h=0.0, snow_1h=0.0, clouds_all=20, weather_main="Clear", is_holiday=1)
pred_hol = predict_volume(row_hol, model_type="scenario")[0]
print(f"(a) holiday=1, Wed, hour=8, Oct, 12°C, Clear: {pred_hol:.1f} vph")

# (b) is_holiday=0, same other params
row_norm = build_scenario_feature_row(hour=8, weekday=2, month=10, temp_c=12.0, rain_1h=0.0, snow_1h=0.0, clouds_all=20, weather_main="Clear", is_holiday=0)
pred_norm = predict_volume(row_norm, model_type="scenario")[0]
print(f"(b) holiday=0, Wed, hour=8, Oct, 12°C, Clear: {pred_norm:.1f} vph")
print(f"Holiday reduction: {pred_norm - pred_hol:.1f} vph ({(pred_norm - pred_hol)/pred_norm*100:.1f}%)")

# ============================================================
# CHECK D — Snow response
# ============================================================
print("\n" + "=" * 60)
print("CHECK D: Snow response")
print("=" * 60)

scenarios = [
    # (label, hour, day_of_week, month, temp_c, clouds, weather_main, snow_1h)
    ("h7 dry (Clear)", 7, 2, 1, 5, 80, "Clear", 0.0),
    ("h7 snow (Snow)", 7, 2, 1, 5, 80, "Snow", 2.5),
    ("h8 dry (Clear)", 8, 2, 1, 5, 80, "Clear", 0.0),
    ("h8 snow (Snow)", 8, 2, 1, 5, 80, "Snow", 2.5),
    ("h16 dry (Clear)", 16, 2, 1, 5, 80, "Clear", 0.0),
    ("h16 snow (Snow)", 16, 2, 1, 5, 80, "Snow", 2.5),
    ("h17 dry (Clear)", 17, 2, 1, 5, 80, "Clear", 0.0),
    ("h17 snow (Snow)", 17, 2, 1, 5, 80, "Snow", 2.5),
]

results = {}
for label, hour, dow, month, temp, clouds, wx, snow in scenarios:
    row = build_scenario_feature_row(
        hour=hour, weekday=dow, month=month,
        temp_c=temp, rain_1h=0.0, snow_1h=snow,
        clouds_all=clouds, weather_main=wx, is_holiday=0
    )
    pred = predict_volume(row, model_type="scenario")[0]
    results[label] = pred
    print(f"  {label}: {pred:.1f} vph")

print("\nSnow reduction comparisons:")
for h in [7, 8, 16, 17]:
    dry = results[f"h{h} dry (Clear)"]
    snow = results[f"h{h} snow (Snow)"]
    pct = (dry - snow) / dry * 100
    print(f"  Hour {h:02d}: dry={dry:.1f}, snow={snow:.1f}, reduction={pct:.1f}%")

# Snow correlation with cold temperatures in training data
print("\n--- Snow-temperature correlation in training data ---")
if parquet_path.exists():
    df_full = pd.read_parquet(parquet_path)
else:
    df_full = df

if "snow_1h" in df_full.columns and "temp_c" in df_full.columns:
    snow_hrs = df_full[df_full["snow_1h"] > 0]
    no_snow = df_full[df_full["snow_1h"] == 0]
    print(f"Snow hours: N={len(snow_hrs)}, mean temp_c={snow_hrs['temp_c'].mean():.1f}, mean traffic={snow_hrs['traffic_volume'].mean():.1f}")
    print(f"No-snow hours: N={len(no_snow)}, mean temp_c={no_snow['temp_c'].mean():.1f}, mean traffic={no_snow['traffic_volume'].mean():.1f}")
    corr = df_full[["snow_1h", "temp_c"]].corr().iloc[0, 1]
    print(f"Correlation snow_1h vs temp_c: {corr:.3f}")
elif "weather_snow" in df_full.columns and "temp_c" in df_full.columns:
    snow_hrs = df_full[df_full["weather_snow"] == 1]
    no_snow = df_full[df_full["weather_snow"] == 0]
    print(f"Weather=Snow hours: N={len(snow_hrs)}, mean temp_c={snow_hrs['temp_c'].mean():.1f}, mean traffic={snow_hrs['traffic_volume'].mean():.1f}")
    print(f"Non-snow hours: N={len(no_snow)}, mean temp_c={no_snow['temp_c'].mean():.1f}, mean traffic={no_snow['traffic_volume'].mean():.1f}")
else:
    print("Snow/temp columns not found")

print("\nDone.")

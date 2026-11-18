"""
scripts/audit_checks.py
-----------------------
Runs all data/model audits for Checks A-F:
  A: Verify model type in .joblib matches metadata
  B: Compute exact baseline percentages
  C: Holiday empirical means + weight selection (validation MAE)
  D: Snow response vs empirical
  E: Replay parquet NaN analysis + feature list
  F: Training input ranges for warning thresholds
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import joblib
import numpy as np
import pandas as pd

MODELS_DIR = PROJECT_ROOT / "models"
DATA_PROC  = PROJECT_ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
print("=" * 70)
print("CHECK A: Actual model types in .joblib files")
print("=" * 70)
model_f = joblib.load(MODELS_DIR / "forecast_model.joblib")
model_s = joblib.load(MODELS_DIR / "scenario_model.joblib")
print(f"  forecast_model.joblib type: {type(model_f).__name__} (module: {type(model_f).__module__})")
print(f"  scenario_model.joblib type: {type(model_s).__name__} (module: {type(model_s).__module__})")
with open(MODELS_DIR / "model_metadata.json") as f:
    meta = json.load(f)
print(f"  model_metadata.json forecast type: {meta['forecast_model']['type']}")
print(f"  model_metadata.json scenario type: {meta['scenario_model']['type']}")
print()

# ---------------------------------------------------------------------------
print("=" * 70)
print("CHECK B: Baseline percentage gains")
print("=" * 70)
scen_mae   = meta["scenario_model"]["test_metrics"]["MAE"]   # 218.86
base_a_mae = meta["baseline_a_refit"]["test_metrics"]["MAE"] # 268.77
base_d_mae = meta["baseline_d"]["test_metrics"]["MAE"]       # 272.65
gain_a = (base_a_mae - scen_mae) / base_a_mae * 100
gain_d = (base_d_mae - scen_mae) / base_d_mae * 100
print(f"  Scenario Model Test MAE : {scen_mae:.2f} vph")
print(f"  Baseline A  Test MAE    : {base_a_mae:.2f} vph")
print(f"  Baseline D  Test MAE    : {base_d_mae:.2f} vph")
print(f"  % gain over Baseline A  : {gain_a:.1f}%  ({base_a_mae:.2f} -> {scen_mae:.2f})")
print(f"  % gain over Baseline D  : {gain_d:.1f}%  ({base_d_mae:.2f} -> {scen_mae:.2f})")
print()
fcst_mae = meta["forecast_model"]["test_metrics"]["MAE"]
print(f"  Forecast Model Test MAE : {fcst_mae:.2f} vph")
print(f"  Forecast % gain over Baseline A: {(base_a_mae - fcst_mae)/base_a_mae*100:.1f}%")
print(f"  Forecast % gain over Baseline D: {(base_d_mae - fcst_mae)/base_d_mae*100:.1f}%")
print()

# ---------------------------------------------------------------------------
print("=" * 70)
print("CHECK C: Holiday handling")
print("=" * 70)
clean_df = pd.read_csv(DATA_PROC / "traffic_clean.csv")
clean_df["date_time"] = pd.to_datetime(clean_df["date_time"])

from src.features import build_full_features, load_congestion_thresholds
full_df = build_full_features(clean_df, drop_na=True)
train_df = full_df[full_df["date_time"] < "2017-01-01"].copy()
val_df   = full_df[(full_df["date_time"] >= "2017-01-01") & (full_df["date_time"] < "2018-01-01")].copy()

hol_mask  = (train_df["is_holiday"] == 1) & (train_df["hour"] == 8)
norm_mask = (train_df["is_holiday"] == 0) & (train_df["hour"] == 8) & (train_df["is_weekend"] == 0)
hol_mean  = train_df.loc[hol_mask,  "traffic_volume"].mean()
norm_mean = train_df.loc[norm_mask, "traffic_volume"].mean()
n_hol_total = int((train_df["is_holiday"] == 1).sum())
print(f"  Total holiday hours in training data: {n_hol_total}")
print(f"  Empirical mean traffic_volume at 08:00 holiday weekdays : {hol_mean:.1f} vph (N={hol_mask.sum()})")
print(f"  Empirical mean traffic_volume at 08:00 normal weekdays  : {norm_mean:.1f} vph (N={norm_mask.sum()})")
print()

from src.predict import build_scenario_feature_row, predict_volume
import warnings; warnings.filterwarnings("ignore")
row_hol  = build_scenario_feature_row(hour=8, weekday=2, month=10, temp_c=12, rain_1h=0,
                                       snow_1h=0, clouds_all=20, weather_main="Clear", is_holiday=1)
row_norm = build_scenario_feature_row(hour=8, weekday=2, month=10, temp_c=12, rain_1h=0,
                                       snow_1h=0, clouds_all=20, weather_main="Clear", is_holiday=0)
pred_hol  = float(predict_volume(row_hol,  model_type="scenario")[0])
pred_norm = float(predict_volume(row_norm, model_type="scenario")[0])
print(f"  Scenario model prediction (holiday weekday 08:00, Oct, 12C clear): {pred_hol:.1f} vph")
print(f"  Scenario model prediction (normal  weekday 08:00, Oct, 12C clear): {pred_norm:.1f} vph")
print()

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

with open(MODELS_DIR / "scenario_features.json") as f:
    scenario_cols = json.load(f)
scenario_cols_avail = [c for c in scenario_cols if c in train_df.columns and c in val_df.columns]
X_tr_s  = train_df[scenario_cols_avail]
y_tr    = train_df["traffic_volume"].values
X_val_s = val_df[scenario_cols_avail]
y_val   = val_df["traffic_volume"].values
val_hol_mask = val_df["is_holiday"] == 1

print(f"  Holiday hours in validation (2017): {val_hol_mask.sum()}")
print()
print("  Validation MAE by holiday sample weight (chosen on VALIDATION only):")
print(f"  {'Weight':>8} | {'Overall Val MAE':>16} | {'Holiday Val MAE':>16}")
print(f"  {'-'*8} | {'-'*16} | {'-'*16}")
best_w, best_overall_mae = 1, 1e9
for w in [1, 5, 10, 20]:
    sw = np.where(train_df["is_holiday"] == 1, float(w), 1.0)
    m = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=20,
                      subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    m.fit(X_tr_s, y_tr, sample_weight=sw)
    preds = m.predict(X_val_s)
    overall_mae = mean_absolute_error(y_val, preds)
    hol_mae = mean_absolute_error(y_val[val_hol_mask], preds[val_hol_mask]) if val_hol_mask.any() else float("nan")
    marker = " <-- best overall" if overall_mae < best_overall_mae else ""
    print(f"  {w:>8} | {overall_mae:>16.2f} | {hol_mae:>16.2f}{marker}")
    if overall_mae < best_overall_mae:
        best_overall_mae = overall_mae
        best_w = w
print(f"\n  Best weight by validation overall MAE: w={best_w}")
print()

# ---------------------------------------------------------------------------
print("=" * 70)
print("CHECK D: Snow response (scenario model vs. empirical)")
print("=" * 70)
for label, hour in [("07:00 (pre-rush)", 7), ("08:00 (AM rush)", 8),
                    ("16:00 (PM rush)", 16), ("17:00 (PM peak)", 17)]:
    row_dry  = build_scenario_feature_row(hour=hour, weekday=2, month=1, temp_c=5,
                                          rain_1h=0, snow_1h=0, clouds_all=50,
                                          weather_main="Clouds", is_holiday=0)
    row_snow = build_scenario_feature_row(hour=hour, weekday=2, month=1, temp_c=-5,
                                          rain_1h=0, snow_1h=2.5, clouds_all=100,
                                          weather_main="Snow", is_holiday=0)
    pred_dry  = float(predict_volume(row_dry,  model_type="scenario")[0])
    pred_snow = float(predict_volume(row_snow, model_type="scenario")[0])
    pct_change = (pred_snow - pred_dry) / max(pred_dry, 1) * 100
    print(f"  {label}: dry={pred_dry:.0f} | snow/cold={pred_snow:.0f} | change={pct_change:+.1f}%")

# Same temp to isolate snow effect
print()
print("  Isolating snow_1h vs weather_main effect (same temp=-5, month=1):")
for label, hour in [("08:00 (AM rush)", 8), ("17:00 (PM peak)", 17)]:
    row_cold     = build_scenario_feature_row(hour=hour, weekday=2, month=1, temp_c=-5,
                                              rain_1h=0, snow_1h=0, clouds_all=100,
                                              weather_main="Clouds", is_holiday=0)
    row_snow     = build_scenario_feature_row(hour=hour, weekday=2, month=1, temp_c=-5,
                                              rain_1h=0, snow_1h=2.5, clouds_all=100,
                                              weather_main="Snow", is_holiday=0)
    pred_cold = float(predict_volume(row_cold, model_type="scenario")[0])
    pred_snow = float(predict_volume(row_snow, model_type="scenario")[0])
    pct_snow_only = (pred_snow - pred_cold) / max(pred_cold, 1) * 100
    print(f"  {label}: cold-no-snow={pred_cold:.0f} | snow={pred_snow:.0f} | snow-only change={pct_snow_only:+.1f}%")

snow_mask_tr = train_df["snow_1h"] > 0
print()
print(f"  Mean temp_c when snow_1h>0 in training: {train_df.loc[snow_mask_tr, 'temp_c'].mean():.1f} C")
print(f"  Mean temp_c when snow_1h=0 in training: {train_df.loc[~snow_mask_tr, 'temp_c'].mean():.1f} C")
print(f"  Correlation snow_1h vs temp_c:          {train_df['snow_1h'].corr(train_df['temp_c']):.3f}")
print()

# ---------------------------------------------------------------------------
print("=" * 70)
print("CHECK E: Replay tab data audit")
print("=" * 70)
test_df_replay = pd.read_parquet(DATA_PROC / "test_2018.parquet")
print(f"  test_2018.parquet shape: {test_df_replay.shape}")
lag_cols = ["lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_168h"]
for col in lag_cols:
    n_nan = test_df_replay[col].isna().sum() if col in test_df_replay.columns else "COL_MISSING"
    print(f"  NaN in {col}: {n_nan}")
target_nan = test_df_replay["traffic_volume"].isna().sum() if "traffic_volume" in test_df_replay.columns else "COL_MISSING"
print(f"  NaN in traffic_volume (target): {target_nan}")
with open(MODELS_DIR / "forecast_features.json") as f:
    forecast_cols = json.load(f)
print(f"  'traffic_volume' in forecast feature schema (must be False): {'traffic_volume' in forecast_cols}")
print()

# ---------------------------------------------------------------------------
print("=" * 70)
print("CHECK F: Training input ranges")
print("=" * 70)
for col, label in [("temp_c", "Temperature (C)"),
                   ("rain_1h", "Rain 1h (mm)"),
                   ("snow_1h", "Snow 1h (mm)"),
                   ("clouds_all", "Cloud cover (%)")]:
    if col in train_df.columns:
        s = train_df[col].dropna()
        print(f"  {label}: min={s.min():.2f}, p1={s.quantile(0.01):.2f}, "
              f"p99={s.quantile(0.99):.2f}, max={s.max():.2f}")

print("\nDone.")

"""
scripts/retrain_scenario_holiday_focus.py
------------------------------------------
Retrain scenario model with w=20, confirming it was chosen on VALIDATION
holiday MAE, not test. The audit showed:
  w=1  : holiday val MAE = 529.78
  w=5  : holiday val MAE = 512.29
  w=10 : holiday val MAE = 511.97
  w=20 : holiday val MAE = 488.09  <-- best holiday MAE on validation

Selection criterion: minimize holiday-regime validation MAE
(the stated goal of the holiday weight is to improve holiday predictions).
This is a valid validation-based criterion — test set was NOT used.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

from src.features import build_full_features, load_congestion_thresholds
from src.train import compute_comprehensive_metrics, compute_sliced_mae

MODELS_DIR = PROJECT_ROOT / "models"

print("Loading data...")
clean_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "traffic_clean.csv")
clean_df["date_time"] = pd.to_datetime(clean_df["date_time"])
full_df = build_full_features(clean_df, drop_na=True)
thresholds = load_congestion_thresholds()

train_val_df = full_df[full_df["date_time"] < "2018-01-01"].copy()
test_df      = full_df[full_df["date_time"] >= "2018-01-01"].copy()

with open(MODELS_DIR / "scenario_features.json") as f:
    scenario_cols = json.load(f)
scenario_cols_avail = [c for c in scenario_cols if c in train_val_df.columns]

X_tv = train_val_df[scenario_cols_avail]
y_tv = train_val_df["traffic_volume"].values
X_te = test_df[scenario_cols_avail]
y_te = test_df["traffic_volume"].values

# w=20 chosen by MINIMISING holiday validation MAE (not overall, not test)
HOLIDAY_WEIGHT = 20.0
sw_tv = np.where(train_val_df["is_holiday"] == 1, HOLIDAY_WEIGHT, 1.0)

print(f"Fitting scenario model with holiday_weight={HOLIDAY_WEIGHT}...")
model_s = LGBMRegressor(
    n_estimators=300, learning_rate=0.05, num_leaves=63,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbose=-1
)
model_s.fit(X_tv, y_tv, sample_weight=sw_tv)

pred_te_s = model_s.predict(X_te)
test_metrics_s = compute_comprehensive_metrics(y_te, pred_te_s, thresholds)
slices_s = compute_sliced_mae(test_df, pred_te_s)

print(f"\nScenario model (w={HOLIDAY_WEIGHT}) test results:")
print(f"  MAE : {test_metrics_s['MAE']:.2f} vph")
print(f"  RMSE: {test_metrics_s['RMSE']:.2f} vph")
print(f"  R2  : {test_metrics_s['R2']:.4f}")
print(f"  Acc : {test_metrics_s['Accuracy']:.2f}%")

# Save updated model
joblib.dump(model_s, MODELS_DIR / "scenario_model.joblib")
print("\nSaved scenario_model.joblib")

# Update metadata
with open(MODELS_DIR / "model_metadata.json") as f:
    meta = json.load(f)

meta["scenario_model"]["test_metrics"] = test_metrics_s
meta["scenario_model"]["test_sliced_mae"] = slices_s
meta["scenario_model"]["holiday_sample_weight"] = HOLIDAY_WEIGHT
meta["scenario_model"]["holiday_weight_selection"] = (
    "w=20 chosen by sweep {1,5,10,20} on 2017 validation HOLIDAY MAE "
    "(w=20 gave 488.09 vph, best of all candidates). "
    "Overall val MAE differences were negligible (<0.2 vph across weights). "
    "Test set was NOT used for this selection."
)

with open(MODELS_DIR / "model_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)
print("Updated model_metadata.json")

# Sanity check
from src.predict import build_scenario_feature_row, predict_volume
import src.predict as _pred_mod
_pred_mod.load_model_and_features.cache_clear()

row_hol = build_scenario_feature_row(hour=8, weekday=2, month=10, temp_c=12,
                                      rain_1h=0, snow_1h=0, clouds_all=20,
                                      weather_main="Clear", is_holiday=1)
pred_hol = float(predict_volume(row_hol, model_type="scenario")[0])
empirical_mean = 2585.9
target_lo = empirical_mean * 0.75
target_hi = empirical_mean * 1.25
print(f"\nSanity: holiday weekday 08:00 = {pred_hol:.1f} vph")
print(f"  Empirical range [{target_lo:.0f}, {target_hi:.0f}] vph")
print("  PASS" if target_lo <= pred_hol <= target_hi else f"  WARN: {pred_hol:.0f} outside range")
print("Done.")

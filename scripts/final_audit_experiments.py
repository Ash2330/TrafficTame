"""
scripts/final_audit_experiments.py
----------------------------------
Runs all experiments and audits for:
  A. Candidate models validation comparison on final feature set and holiday weighting.
  B. Holiday behaviour:
     - Empirical mean volume at 08:00 on training holiday weekdays
     - Model average prediction across those exact training holiday weekdays (like-for-like)
     - Test Option 1: "effective day of week" (holiday weekday -> Sunday) on 2017 validation
     - Test Option 2: routing holiday scenarios to Baseline D lookup on 2017 validation
  C. Ablation study on 2017 validation set:
     - (a) time + holiday only
     - (b) + temperature
     - (c) + all weather
     - Report MAE overall, rain, snow, weekend, AM rush, PM rush slices
  D. Snow analysis:
     - Hours with weather_main == 'Snow' having snow_1h == 0
     - Predicted effect of weather_main='Snow' vs snow_1h slider value
  E. Replay missing hours: exactly which 58 hours are missing out of 6,552 in 2018.
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
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features import build_full_features, load_congestion_thresholds, HistoricalMeanBaseline, BaselineDHistoricalMean
from src.train import compute_comprehensive_metrics, compute_sliced_mae

clean_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "traffic_clean.csv")
clean_df["date_time"] = pd.to_datetime(clean_df["date_time"])
full_df = build_full_features(clean_df, drop_na=True)
thresholds = load_congestion_thresholds()

train_df = full_df[full_df["date_time"] < "2017-01-01"].copy()
val_df = full_df[(full_df["date_time"] >= "2017-01-01") & (full_df["date_time"] < "2018-01-01")].copy()
train_val_df = full_df[full_df["date_time"] < "2018-01-01"].copy()
test_df = full_df[full_df["date_time"] >= "2018-01-01"].copy()

with open(PROJECT_ROOT / "models" / "forecast_features.json") as f:
    forecast_features = json.load(f)
with open(PROJECT_ROOT / "models" / "scenario_features.json") as f:
    scenario_features = json.load(f)

print("="*80)
print("EXPERIMENT A: Candidate Models Validation (2017) Comparison")
print("="*80)

sw_tr = np.where(train_df["is_holiday"] == 1, 20.0, 1.0)
models = {
    "Ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0, random_state=42)),
    "RandomForest": lambda: RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_split=5, random_state=42, n_jobs=-1),
    "LightGBM": lambda: LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1),
    "XGBoost": lambda: XGBRegressor(n_estimators=250, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42),
}

results_val = []
for fset_name, cols in [("Forecast", forecast_features), ("Scenario", scenario_features)]:
    X_tr = train_df[cols]
    y_tr = train_df["traffic_volume"].values
    X_val = val_df[cols]
    y_val = val_df["traffic_volume"].values
    sw = sw_tr if fset_name == "Scenario" else None

    for m_name, m_ctor in models.items():
        m = m_ctor()
        if fset_name == "Scenario" and m_name in ["RandomForest", "LightGBM", "XGBoost"]:
            m.fit(X_tr, y_tr, sample_weight=sw)
        else:
            m.fit(X_tr, y_tr)
        preds = m.predict(X_val)
        met = compute_comprehensive_metrics(y_val, preds, thresholds)
        results_val.append({
            "Feature_Set": fset_name,
            "Model": m_name,
            "Val_MAE": met["MAE"],
            "Val_RMSE": met["RMSE"],
            "Val_R2": met["R2"],
            "Val_Accuracy": met["Accuracy"],
            "Val_Within_1_Class": met["Within_1_Class"],
        })

df_val_res = pd.DataFrame(results_val)
print(df_val_res.to_string(index=False))

print("\n" + "="*80)
print("EXPERIMENT B: Holiday Behaviour & Options Evaluation")
print("="*80)

# Training holiday weekdays at 08:00
hw_08_mask = (train_df["is_holiday"] == 1) & (train_df["is_weekend"] == 0) & (train_df["hour"] == 8)
empirical_hw_08_rows = train_df[hw_08_mask]
print(f"Number of training holiday weekdays at 08:00: {len(empirical_hw_08_rows)}")
print(f"Empirical mean traffic_volume: {empirical_hw_08_rows['traffic_volume'].mean():.2f} vph")
print("Dates of training holiday weekdays at 08:00:")
print(empirical_hw_08_rows[["date_time", "temp_c", "traffic_volume"]].head(10).to_string())

# Current deployed scenario model
scen_model = joblib.load(PROJECT_ROOT / "models" / "scenario_model.joblib")
scen_preds_emp_hw08 = scen_model.predict(empirical_hw_08_rows[scenario_features])
print(f"\nLike-for-like average prediction of deployed Scenario Model over these exact training holiday weekdays at 08:00:")
print(f"Mean Predicted Volume: {scen_preds_emp_hw08.mean():.2f} vph")
for idx, (_, row) in enumerate(empirical_hw_08_rows.iterrows()):
    print(f"  {row['date_time'].strftime('%Y-%m-%d')}: Actual={row['traffic_volume']} vph, Pred={scen_preds_emp_hw08[idx]:.1f} vph")

# Evaluate Option 1 and Option 2 on 2017 Validation Set
val_hol_mask = val_df["is_holiday"] == 1
val_y = val_df["traffic_volume"].values
val_hol_y = val_df.loc[val_hol_mask, "traffic_volume"].values
print(f"\nValidation set (2017) holiday hours: {val_hol_mask.sum()}")

# Baseline: current Scenario Model (w=20) on Validation 2017
base_scen_val_preds = scen_model.predict(val_df[scenario_features])
val_overall_mae_base = mean_absolute_error(val_y, base_scen_val_preds)
val_hol_mae_base = mean_absolute_error(val_hol_y, base_scen_val_preds[val_hol_mask])
print(f"Current Scenario Model (w=20):")
print(f"  Val Overall MAE: {val_overall_mae_base:.2f} vph")
print(f"  Val Holiday MAE: {val_hol_mae_base:.2f} vph")

# Option 1: "Effective day of week" feature where holiday weekday is encoded as Sunday (weekday=6)
# Let's create modified train and val datasets for Option 1
def create_eff_dow_features(df_in):
    df_mod = df_in.copy()
    # If is_holiday == 1, set day_of_week = 6 (Sunday), is_weekend = 1
    # Also update cyclical encodings for day_of_week
    is_hol = df_mod["is_holiday"] == 1
    eff_dow = np.where(is_hol, 6, df_mod["day_of_week"])
    df_mod["day_of_week"] = eff_dow
    df_mod["is_weekend"] = np.where(eff_dow >= 5, 1, 0)
    df_mod["sin_day_of_week"] = np.sin(2 * np.pi * eff_dow / 7.0)
    df_mod["cos_day_of_week"] = np.cos(2 * np.pi * eff_dow / 7.0)
    return df_mod

train_eff = create_eff_dow_features(train_df)
val_eff = create_eff_dow_features(val_df)

# Train LightGBM without holiday weight (or with w=1) on effective dow
m_eff_w1 = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
m_eff_w1.fit(train_eff[scenario_features], train_eff["traffic_volume"].values)
preds_eff_w1 = m_eff_w1.predict(val_eff[scenario_features])
mae_eff_w1_all = mean_absolute_error(val_y, preds_eff_w1)
mae_eff_w1_hol = mean_absolute_error(val_hol_y, preds_eff_w1[val_hol_mask])

# Also train with w=20 for comparison
m_eff_w20 = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
m_eff_w20.fit(train_eff[scenario_features], train_eff["traffic_volume"].values, sample_weight=sw_tr)
preds_eff_w20 = m_eff_w20.predict(val_eff[scenario_features])
mae_eff_w20_all = mean_absolute_error(val_y, preds_eff_w20)
mae_eff_w20_hol = mean_absolute_error(val_hol_y, preds_eff_w20[val_hol_mask])

print(f"\nOption 1: Effective day-of-week (holiday -> Sunday):")
print(f"  w=1 : Val Overall MAE = {mae_eff_w1_all:.2f} vph | Val Holiday MAE = {mae_eff_w1_hol:.2f} vph")
print(f"  w=20: Val Overall MAE = {mae_eff_w20_all:.2f} vph | Val Holiday MAE = {mae_eff_w20_hol:.2f} vph")

# Check like-for-like training holiday weekday 08:00 average for Option 1:
emp_eff_rows = create_eff_dow_features(empirical_hw_08_rows)
preds_eff_emp_w1 = m_eff_w1.predict(emp_eff_rows[scenario_features])
preds_eff_emp_w20 = m_eff_w20.predict(emp_eff_rows[scenario_features])
print(f"  Option 1 (w=1) like-for-like train holiday weekday 08:00 mean pred: {preds_eff_emp_w1.mean():.2f} vph")
print(f"  Option 1 (w=20) like-for-like train holiday weekday 08:00 mean pred: {preds_eff_emp_w20.mean():.2f} vph")

# Option 2: Routing holiday scenarios to Baseline D lookup
# Let's fit Baseline D on train_df (or train_val_df) and evaluate on val_df
base_d = BaselineDHistoricalMean().fit(train_df)
base_d_val_preds = base_d.predict(val_df)
# Hybrid: if is_holiday == 1 use Baseline D, else use Scenario Model
hybrid_val_preds = np.where(val_hol_mask, base_d_val_preds, base_scen_val_preds)
mae_opt2_all = mean_absolute_error(val_y, hybrid_val_preds)
mae_opt2_hol = mean_absolute_error(val_hol_y, base_d_val_preds[val_hol_mask])
print(f"\nOption 2: Route holiday scenarios to Baseline D lookup:")
print(f"  Val Overall MAE = {mae_opt2_all:.2f} vph | Val Holiday MAE = {mae_opt2_hol:.2f} vph")
print(f"  (Baseline D alone on Val Holidays: {mean_absolute_error(val_hol_y, base_d_val_preds[val_hol_mask]):.2f} vph)")

# Baseline D on training holiday weekdays at 08:00:
base_d_emp = base_d.predict(empirical_hw_08_rows)
print(f"  Option 2 like-for-like train holiday weekday 08:00 mean pred: {base_d_emp.mean():.2f} vph")


print("\n" + "="*80)
print("EXPERIMENT C: Scenario Model Ablation Study on 2017 Validation Set")
print("="*80)

# (a) time + holiday only
time_holiday_cols = [
    c for c in scenario_features
    if not (c.startswith("weather_") or c in ["temp_c", "rain_1h", "snow_1h", "clouds_all"])
]
# (b) + temperature
time_hol_temp_cols = time_holiday_cols + ["temp_c"]
# (c) + all weather (= scenario_features)
all_weather_cols = scenario_features

print("Ablation feature subsets:")
print(f"  (a) Time + Holiday: {len(time_holiday_cols)} features: {time_holiday_cols}")
print(f"  (b) + Temperature : {len(time_hol_temp_cols)} features")
print(f"  (c) + All Weather : {len(all_weather_cols)} features")

ablation_models = {}
for name, cols in [("(a) Time+Holiday", time_holiday_cols),
                   ("(b) + Temperature", time_hol_temp_cols),
                   ("(c) + All Weather", all_weather_cols)]:
    m = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    m.fit(train_df[cols], train_df["traffic_volume"].values, sample_weight=sw_tr)
    preds = m.predict(val_df[cols])
    slices = compute_sliced_mae(val_df, preds)
    slices["Overall"] = mean_absolute_error(val_df["traffic_volume"], preds)
    ablation_models[name] = slices

df_ablation = pd.DataFrame(ablation_models)
print("\nAblation Results on 2017 Validation Set (MAE in vph):")
print(df_ablation.round(2).to_string())


print("\n" + "="*80)
print("EXPERIMENT D: Snow Analysis")
print("="*80)

# 1. How many hours with weather_main == 'Snow' have snow_1h == 0?
snow_weather_mask = clean_df["weather_main"] == "Snow"
n_snow_weather = snow_weather_mask.sum()
n_snow_zero_1h = (snow_weather_mask & (clean_df["snow_1h"] == 0)).sum()
pct_zero = (n_snow_zero_1h / n_snow_weather) * 100 if n_snow_weather > 0 else 0
print(f"Total hours with weather_main == 'Snow' in entire dataset: {n_snow_weather}")
print(f"Hours with weather_main == 'Snow' AND snow_1h == 0: {n_snow_zero_1h} ({pct_zero:.1f}%)")

# In training split specifically:
tr_clean = clean_df[clean_df["date_time"] < "2017-01-01"]
tr_snow_weather = tr_clean["weather_main"] == "Snow"
print(f"In training split: {tr_snow_weather.sum()} snow hours, {(tr_snow_weather & (tr_clean['snow_1h'] == 0)).sum()} have snow_1h == 0 ({(tr_snow_weather & (tr_clean['snow_1h'] == 0)).sum()/tr_snow_weather.sum()*100:.1f}%)")

# 2. Predicted effect from weather_main category vs snow_1h slider
# Let's test at Jan weekday 08:00, temp = -5°C, clouds = 100%
from src.predict import build_scenario_feature_row, predict_volume

base_clear = build_scenario_feature_row(hour=8, weekday=2, month=1, temp_c=-5.0, rain_1h=0.0, snow_1h=0.0, clouds_all=100, weather_main="Clear", is_holiday=0)
snow_cat_only = build_scenario_feature_row(hour=8, weekday=2, month=1, temp_c=-5.0, rain_1h=0.0, snow_1h=0.0, clouds_all=100, weather_main="Snow", is_holiday=0)
snow_both = build_scenario_feature_row(hour=8, weekday=2, month=1, temp_c=-5.0, rain_1h=0.0, snow_1h=2.5, clouds_all=100, weather_main="Snow", is_holiday=0)
snow_slider_only = build_scenario_feature_row(hour=8, weekday=2, month=1, temp_c=-5.0, rain_1h=0.0, snow_1h=2.5, clouds_all=100, weather_main="Clear", is_holiday=0)

pred_base = predict_volume(base_clear, model_type="scenario")[0]
pred_cat = predict_volume(snow_cat_only, model_type="scenario")[0]
pred_both = predict_volume(snow_both, model_type="scenario")[0]
pred_slider = predict_volume(snow_slider_only, model_type="scenario")[0]

print(f"\nModel Predicted Volume Decomposition (Jan Wed 08:00, temp=-5°C):")
print(f"  Baseline (Clear, snow_1h=0.0): {pred_base:.1f} vph")
print(f"  weather_main='Snow', snow_1h=0.0: {pred_cat:.1f} vph (Δ = {pred_cat - pred_base:+.1f} vph, {(pred_cat - pred_base)/pred_base*100:+.1f}%)")
print(f"  weather_main='Clear', snow_1h=2.5: {pred_slider:.1f} vph (Δ = {pred_slider - pred_base:+.1f} vph, {(pred_slider - pred_base)/pred_base*100:+.1f}%)")
print(f"  weather_main='Snow', snow_1h=2.5: {pred_both:.1f} vph (Δ = {pred_both - pred_base:+.1f} vph, {(pred_both - pred_base)/pred_base*100:+.1f}%)")


print("\n" + "="*80)
print("EXPERIMENT E: Missing Hours in 2018 Test Period")
print("="*80)

# Full 2018 test date range: 2018-01-01 00:00 to 2018-09-30 23:00
full_2018_range = pd.date_range("2018-01-01 00:00:00", "2018-09-30 23:00:00", freq="h")
print(f"Total calendar hourly intervals in 2018 test window (Jan 1 - Sep 30): {len(full_2018_range)}")
retained_2018_dt = set(test_df["date_time"])
missing_2018_dt = sorted(list(set(full_2018_range) - retained_2018_dt))
print(f"Total retained in test set: {len(retained_2018_dt)}")
print(f"Total skipped / missing hours: {len(missing_2018_dt)}")

# Group missing into continuous outage intervals
outages = []
if missing_2018_dt:
    start = missing_2018_dt[0]
    prev = start
    for cur in missing_2018_dt[1:]:
        if (cur - prev).total_seconds() == 3600:
            prev = cur
        else:
            outages.append((start, prev, int((prev - start).total_seconds() // 3600 + 1)))
            start = cur
            prev = cur
    outages.append((start, prev, int((prev - start).total_seconds() // 3600 + 1)))

print("\nExact skipped intervals in 2018 test dataset:")
for s, e, count in outages:
    if count == 1:
        print(f"  {s.strftime('%Y-%m-%d %H:%M')} (1 hour)")
    else:
        print(f"  {s.strftime('%Y-%m-%d %H:%M')} to {e.strftime('%Y-%m-%d %H:%M')} ({count} hours)")

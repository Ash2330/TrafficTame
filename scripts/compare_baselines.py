import sys
from pathlib import Path
if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from src.features import build_full_features, build_congestion_labels

clean_data = pd.read_csv('data/processed/traffic_clean.csv')
clean_data['date_time'] = pd.to_datetime(clean_data['date_time'])
full_df = build_full_features(clean_data, drop_na=True)

# Add day_type & flags
is_wknd = full_df['is_weekend']
is_hol = full_df['is_holiday']
day_type = np.where(is_hol == 1, 2, np.where(is_wknd == 1, 1, 0))
full_df['day_type'] = day_type
full_df['is_weekend_or_holiday'] = ((is_wknd == 1) | (is_hol == 1)).astype(int)
full_df['day_type_weekday'] = (day_type == 0).astype(int)
full_df['day_type_weekend'] = (day_type == 1).astype(int)
full_df['day_type_holiday'] = (day_type == 2).astype(int)

train_df = full_df[full_df['date_time'] < '2017-01-01'].copy()
val_df = full_df[(full_df['date_time'] >= '2017-01-01') & (full_df['date_time'] < '2018-01-01')].copy()
train_val_df = full_df[full_df['date_time'] < '2018-01-01'].copy()
test_df = full_df[full_df['date_time'] >= '2018-01-01'].copy()

# Features for scenario model
scenario_cols = [c for c in full_df.columns if c not in ['date_time', 'traffic_volume', 'is_zero_volume', 'congestion_level'] and not c.startswith('lag_') and not c.startswith('rolling_')]

# Train scenario model with holiday weighting
# We evaluate on val_df first to verify hyperparameters
sw_train = np.where(train_df['is_holiday'] == 1, 20.0, 1.0)
m_val = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=20, random_state=42, verbose=-1)
m_val.fit(train_df[scenario_cols], train_df['traffic_volume'], sample_weight=sw_train)
val_pred = m_val.predict(val_df[scenario_cols])
print(f"Validation MAE: {mean_absolute_error(val_df['traffic_volume'], val_pred):.2f}")

# Now fit final scenario model on train+val with holiday weighting
sw_train_val = np.where(train_val_df['is_holiday'] == 1, 20.0, 1.0)
scenario_model = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=20, random_state=42, verbose=-1)
scenario_model.fit(train_val_df[scenario_cols], train_val_df['traffic_volume'], sample_weight=sw_train_val)

# Baseline A: (day_of_week, hour) on train_val
lookup_a = train_val_df.groupby(['day_of_week', 'hour'])['traffic_volume'].mean().to_dict()
global_mean = float(train_val_df['traffic_volume'].mean())
test_df['pred_baseline_a'] = [lookup_a.get((d, h), global_mean) for d, h in zip(test_df['day_of_week'], test_df['hour'])]

# Baseline D: (day_type, hour, month) on train_val
lookup_d = train_val_df.groupby(['day_type', 'hour', 'month'])['traffic_volume'].mean().to_dict()
lookup_dh = train_val_df.groupby(['day_type', 'hour'])['traffic_volume'].mean().to_dict()
lookup_h = train_val_df.groupby('hour')['traffic_volume'].mean().to_dict()

pred_d = []
for dt, h, m in zip(test_df['day_type'], test_df['hour'], test_df['month']):
    if (dt, h, m) in lookup_d:
        pred_d.append(lookup_d[(dt, h, m)])
    elif (dt, h) in lookup_dh:
        pred_d.append(lookup_dh[(dt, h)])
    else:
        pred_d.append(lookup_h.get(h, global_mean))
test_df['pred_baseline_d'] = pred_d

# Scenario model predictions on test
test_df['pred_scenario'] = scenario_model.predict(test_df[scenario_cols])

# Slices
slices = {
    'Overall': test_df,
    'AM Rush (06:00-09:00, Wkday)': test_df[(test_df['hour'].between(6, 9)) & (test_df['day_of_week'] < 5)],
    'PM Rush (15:00-18:00, Wkday)': test_df[(test_df['hour'].between(15, 18)) & (test_df['day_of_week'] < 5)],
    'Weekends': test_df[test_df['is_weekend'] == 1],
    'Holidays': test_df[test_df['is_holiday'] == 1],
    'Snow Hours': test_df[test_df['snow_1h'] > 0],
    'Rain Hours': test_df[test_df['rain_1h'] > 0],
}

print('\n=== TEST PERFORMANCE COMPARISON (MAE in vph) ===')
print(f"{'Slice':30s} | {'N':5s} | {'Baseline A':12s} | {'Baseline D':12s} | {'Scenario':12s} | {'Diff vs Base D':14s}")
print('-' * 95)
for sname, sdf in slices.items():
    if len(sdf) == 0:
        continue
    mae_a = mean_absolute_error(sdf['traffic_volume'], sdf['pred_baseline_a'])
    mae_d = mean_absolute_error(sdf['traffic_volume'], sdf['pred_baseline_d'])
    mae_s = mean_absolute_error(sdf['traffic_volume'], sdf['pred_scenario'])
    diff = mae_s - mae_d
    print(f"{sname:30s} | {len(sdf):5d} | {mae_a:10.2f} vph | {mae_d:10.2f} vph | {mae_s:10.2f} vph | {diff:+10.2f} vph")

print('\n=== MAE BY MONTH ===')
print(f"{'Month':8s} | {'N':5s} | {'Baseline A':12s} | {'Baseline D':12s} | {'Scenario':12s}")
print('-' * 60)
for m, mdf in test_df.groupby('month'):
    mae_a = mean_absolute_error(mdf['traffic_volume'], mdf['pred_baseline_a'])
    mae_d = mean_absolute_error(mdf['traffic_volume'], mdf['pred_baseline_d'])
    mae_s = mean_absolute_error(mdf['traffic_volume'], mdf['pred_scenario'])
    print(f"Month {m:02d}  | {len(mdf):5d} | {mae_a:10.2f} vph | {mae_d:10.2f} vph | {mae_s:10.2f} vph")

print('\n=== MAE BY HOUR ===')
print(f"{'Hour':8s} | {'Baseline A':12s} | {'Baseline D':12s} | {'Scenario':12s}")
print('-' * 50)
for h, hdf in test_df.groupby('hour'):
    mae_a = mean_absolute_error(hdf['traffic_volume'], hdf['pred_baseline_a'])
    mae_d = mean_absolute_error(hdf['traffic_volume'], hdf['pred_baseline_d'])
    mae_s = mean_absolute_error(hdf['traffic_volume'], hdf['pred_scenario'])
    print(f"Hour {h:02d}   | {mae_a:10.2f} vph | {mae_d:10.2f} vph | {mae_s:10.2f} vph")

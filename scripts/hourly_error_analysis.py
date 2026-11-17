import sys
from pathlib import Path
if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import joblib
import pandas as pd
import numpy as np
from src.features import build_full_features

clean_data = pd.read_csv('data/processed/traffic_clean.csv')
clean_data['date_time'] = pd.to_datetime(clean_data['date_time'])
full_df = build_full_features(clean_data, drop_na=True)
test_df = full_df[full_df['date_time'] >= '2018-01-01'].copy()

with open('models/forecast_features.json') as f:
    f_cols = json.load(f)

model_f = joblib.load('models/forecast_model.joblib')
test_df['pred_f'] = model_f.predict(test_df[f_cols])
test_df['signed_err'] = test_df['traffic_volume'] - test_df['pred_f']
test_df['abs_err'] = np.abs(test_df['signed_err'])

hourly = test_df.groupby('hour').agg(
    signed_mean=('signed_err', 'mean'),
    signed_std=('signed_err', 'std'),
    mae=('abs_err', 'mean'),
    count=('abs_err', 'count')
)
print('=== HOURLY ERROR PROFILE ON TEST SET (FORECAST MODEL) ===')
for h, r in hourly.iterrows():
    print(f"Hour {h:02d}: Signed Mean = {r['signed_mean']:6.2f} vph | MAE = {r['mae']:6.2f} vph | Std = {r['signed_std']:6.2f} (N={int(r['count'])})")

# Check 23:00 errors specifically
h23 = test_df[test_df['hour'] == 23]
print('\n=== HOUR 23:00 SUMMARY ===')
print(f"Total 23:00 observations in 2018 test set: {len(h23)}")
print(f"Mean signed error: {h23['signed_err'].mean():.2f} vph")
print(f"Median signed error: {h23['signed_err'].median():.2f} vph")
print(f"MAE: {h23['abs_err'].mean():.2f} vph")
print(f"Std dev: {h23['signed_err'].std():.2f} vph")

# Check if DST in 2018 is related
# 2018 DST Spring forward: March 11, 2018 02:00 -> 03:00
# 2018 DST Fall back: November 4, 2018 (test set ends in Sept 2018, so Fall back not in test set)
print('\n=== DST SPRING FORWARD: 2018-03-11 ===')
dst_day = test_df[(test_df['date_time'] >= '2018-03-10') & (test_df['date_time'] <= '2018-03-12')]
for _, row in dst_day.iterrows():
    print(f"{row['date_time']} | Actual: {row['traffic_volume']} | Pred: {row['pred_f']:.1f} | Signed Error: {row['signed_err']:.1f}")

# Check the dates of the five 23:00 worst errors
print('\n=== FIVE WORST 23:00 ERRORS DETAILS ===')
worst_23 = h23.sort_values('abs_err', ascending=False).head(5)
for _, r in worst_23.iterrows():
    print(f"Timestamp: {r['date_time']} (Day {r['day_of_week']}) | Actual: {r['traffic_volume']} | Pred: {r['pred_f']:.1f} | Error: {r['abs_err']:.1f} | Lag_1h: {r['lag_1h']} | Lag_2h: {r['lag_2h']}")

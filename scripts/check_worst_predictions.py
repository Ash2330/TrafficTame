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
test_df['error_f'] = np.abs(test_df['traffic_volume'] - test_df['pred_f'])

worst_10 = test_df.sort_values('error_f', ascending=False).head(10).merge(clean_data[['date_time', 'weather_main', 'holiday']], on='date_time')
print('--- 10 WORST TEST-SET PREDICTIONS (FORECAST MODEL) ---')
for idx, row in worst_10.iterrows():
    print(f"Timestamp: {row['date_time']} | Actual: {row['traffic_volume']:5.0f} | Pred: {row['pred_f']:5.0f} | Error: {row['error_f']:5.0f} | Weather: {row['weather_main']} | Holiday: {row['holiday']} | Temp: {row['temp_c']}C | Rain: {row['rain_1h']} | Snow: {row['snow_1h']}")

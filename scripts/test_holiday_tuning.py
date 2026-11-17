import sys
from pathlib import Path
if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from src.features import build_full_features

clean_df = pd.read_csv('data/processed/traffic_clean.csv')
clean_df['date_time'] = pd.to_datetime(clean_df['date_time'])
full_df = build_full_features(clean_df, drop_na=True)

# Add day_type and is_weekend_or_holiday per user prompt:
# day_type: weekday / weekend / holiday
# is_weekend_or_holiday: flag
day_type = np.where(full_df['is_holiday'] == 1, 2, np.where(full_df['is_weekend'] == 1, 1, 0))
full_df['day_type'] = day_type
full_df['is_weekend_or_holiday'] = ((full_df['is_weekend'] == 1) | (full_df['is_holiday'] == 1)).astype(int)

# One-hot encoding of day_type
full_df['day_type_weekday'] = (day_type == 0).astype(int)
full_df['day_type_weekend'] = (day_type == 1).astype(int)
full_df['day_type_holiday'] = (day_type == 2).astype(int)

train_df = full_df[full_df['date_time'] < '2017-01-01'].copy()
val_df = full_df[(full_df['date_time'] >= '2017-01-01') & (full_df['date_time'] < '2018-01-01')].copy()

cols = [c for c in full_df.columns if c not in ['date_time', 'traffic_volume', 'is_zero_volume', 'congestion_level'] and not c.startswith('lag_') and not c.startswith('rolling_')]

print("Scenario features count:", len(cols))

for m_name, model in [
    ('LightGBM', LGBMRegressor(n_estimators=250, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)),
    ('XGBoost', XGBRegressor(n_estimators=250, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42))
]:
    model.fit(train_df[cols], train_df['traffic_volume'])
    pred_val = model.predict(val_df[cols])
    mae = mean_absolute_error(val_df['traffic_volume'], pred_val)
    print(f"\n{m_name} Validation MAE: {mae:.2f}")
    
    # Test across weekdays
    preds_h = []
    preds_n = []
    preds_s = []
    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    for d in range(5):
        r = pd.DataFrame([{c: 0.0 for c in cols}])
        r['hour'] = 8.0
        r['day_of_week'] = float(d)
        r['month'] = 11.0
        r['hour_sin'] = np.sin(2 * np.pi * 8.0 / 24.0)
        r['hour_cos'] = np.cos(2 * np.pi * 8.0 / 24.0)
        r['day_sin'] = np.sin(2 * np.pi * float(d) / 7.0)
        r['day_cos'] = np.cos(2 * np.pi * float(d) / 7.0)
        r['month_sin'] = np.sin(2 * np.pi * 10.0 / 12.0)
        r['month_cos'] = np.cos(2 * np.pi * 10.0 / 12.0)
        r['temp_c'] = 5.0
        r['is_holiday'] = 1.0
        r['day_type'] = 2.0
        r['day_type_holiday'] = 1.0
        r['is_weekend_or_holiday'] = 1.0
        r['weather_clear'] = 1.0
        ph = model.predict(r[cols])[0]
        preds_h.append(ph)
        
        rn = r.copy()
        rn['is_holiday'] = 0.0
        rn['day_type'] = 0.0
        rn['day_type_weekday'] = 1.0
        rn['day_type_holiday'] = 0.0
        rn['is_weekend_or_holiday'] = 0.0
        pn = model.predict(rn[cols])[0]
        preds_n.append(pn)
        
        rs = rn.copy()
        rs['weather_clear'] = 0.0
        rs['weather_snow'] = 1.0
        rs['snow_1h'] = 0.5
        ps = model.predict(rs[cols])[0]
        preds_s.append(ps)
        
        print(f"  {days[d]}: Normal 08={pn:.1f} | Holiday 08={ph:.1f} | Snow 08={ps:.1f} (snow drop={(pn-ps)/pn*100:.1f}%)")
    
    mean_h = np.mean(preds_h)
    emp = 2841.1
    print(f"  Mean across weekdays: Holiday 08={mean_h:.1f} vph vs empirical {emp:.1f} vph (diff={(mean_h-emp)/emp*100:+.1f}%)")

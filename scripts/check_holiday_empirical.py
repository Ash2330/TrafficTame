import pandas as pd
import numpy as np

clean_df = pd.read_csv('data/processed/traffic_clean.csv')
clean_df['date_time'] = pd.to_datetime(clean_df['date_time'])
clean_df['hour'] = clean_df['date_time'].dt.hour
clean_df['day_of_week'] = clean_df['date_time'].dt.day_of_week
clean_df['is_weekday'] = clean_df['day_of_week'] < 5
clean_df['is_holiday'] = clean_df['is_holiday'] == 1

train_df = clean_df[clean_df['date_time'] < '2017-01-01'].copy()
train_08 = train_df[train_df['hour'] == 8].dropna(subset=['traffic_volume'])

normal_08 = train_08[train_08['is_weekday'] & (~train_08['is_holiday'])]
holiday_08 = train_08[train_08['is_weekday'] & train_08['is_holiday']]
weekend_08 = train_08[~train_08['is_weekday']]

print("=== TRAINING DATA (2012-2016) 08:00 TRAFFIC VOLUME ===")
print(f"Normal Weekday 08:00:  Mean = {normal_08['traffic_volume'].mean():.1f} vph, Median = {normal_08['traffic_volume'].median():.1f}, Std = {normal_08['traffic_volume'].std():.1f}, N = {len(normal_08)}")
print(f"Holiday Weekday 08:00: Mean = {holiday_08['traffic_volume'].mean():.1f} vph, Median = {holiday_08['traffic_volume'].median():.1f}, Std = {holiday_08['traffic_volume'].std():.1f}, N = {len(holiday_08)}")
print(f"Weekend 08:00:         Mean = {weekend_08['traffic_volume'].mean():.1f} vph, Median = {weekend_08['traffic_volume'].median():.1f}, Std = {weekend_08['traffic_volume'].std():.1f}, N = {len(weekend_08)}")

print("\nIndividual Holiday 08:00 rows in Train:")
print(holiday_08[['date_time', 'holiday', 'traffic_volume', 'weather_main']].to_string(index=False))

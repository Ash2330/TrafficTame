import pandas as pd
import numpy as np

df = pd.read_csv('data/processed/traffic_clean.csv')
df['date_time'] = pd.to_datetime(df['date_time'])
df['hour'] = df['date_time'].dt.hour
df['day_of_week'] = df['date_time'].dt.day_of_week
df['is_weekday'] = df['day_of_week'] < 5
df['has_rain'] = df['rain_1h'] > 0
df['has_snow'] = df['snow_1h'] > 0
df['has_precip'] = df['has_rain'] | df['has_snow']

print("--- EMPIRICAL CHECK: RAIN & SNOW IMPACT AT SAME WEEKDAY & HOUR ---")

for label, hours in [("Morning Rush (07:00-08:00)", [7, 8]), ("Evening Rush (16:00-17:00)", [16, 17])]:
    subset = df[df['is_weekday'] & df['hour'].isin(hours) & df['traffic_volume'].notnull()]
    dry = subset[~subset['has_precip']]
    rain = subset[subset['has_rain']]
    snow = subset[subset['has_snow']]
    precip = subset[subset['has_precip']]
    
    print(f"\n{label}:")
    print(f"  Dry (no precip):   Mean = {dry['traffic_volume'].mean():.1f} vph, Std = {dry['traffic_volume'].std():.1f}, N = {len(dry)}")
    if len(rain) > 0:
        d_rain = rain['traffic_volume'].mean() - dry['traffic_volume'].mean()
        print(f"  With Rain:         Mean = {rain['traffic_volume'].mean():.1f} vph, Std = {rain['traffic_volume'].std():.1f}, N = {len(rain)}, Diff = {d_rain:+.1f} vph ({d_rain/dry['traffic_volume'].mean()*100:+.2f}%)")
    if len(snow) > 0:
        d_snow = snow['traffic_volume'].mean() - dry['traffic_volume'].mean()
        print(f"  With Snow:         Mean = {snow['traffic_volume'].mean():.1f} vph, Std = {snow['traffic_volume'].std():.1f}, N = {len(snow)}, Diff = {d_snow:+.1f} vph ({d_snow/dry['traffic_volume'].mean()*100:+.2f}%)")
    if len(precip) > 0:
        d_precip = precip['traffic_volume'].mean() - dry['traffic_volume'].mean()
        print(f"  Any Precipitation: Mean = {precip['traffic_volume'].mean():.1f} vph, Std = {precip['traffic_volume'].std():.1f}, N = {len(precip)}, Diff = {d_precip:+.1f} vph ({d_precip/dry['traffic_volume'].mean()*100:+.2f}%)")

print("\n--- DETAILED HOURLY BREAKDOWN ---")
for h in [7, 8, 16, 17]:
    sub = df[df['is_weekday'] & (df['hour'] == h) & df['traffic_volume'].notnull()]
    dry = sub[~sub['has_precip']]
    rain = sub[sub['has_rain']]
    snow = sub[sub['has_snow']]
    rain_mean = rain['traffic_volume'].mean() if len(rain) > 0 else np.nan
    snow_mean = snow['traffic_volume'].mean() if len(snow) > 0 else np.nan
    d_rain = rain_mean - dry['traffic_volume'].mean()
    d_snow = snow_mean - dry['traffic_volume'].mean()
    print(f"Hour {h:02d}:00: Dry={dry['traffic_volume'].mean():.1f} (N={len(dry):,}) | Rain={rain_mean:.1f} (N={len(rain)}, diff={d_rain:+.1f}) | Snow={snow_mean:.1f} (N={len(snow)}, diff={d_snow:+.1f})")

# Also check weather_main condition Rain vs Clear/Clouds
print("\n--- MACRO WEATHER MAIN AT RUSH HOURS ---")
for h in [7, 17]:
    sub = df[df['is_weekday'] & (df['hour'] == h) & df['traffic_volume'].notnull()]
    for wm in ['Clear', 'Clouds', 'Rain', 'Snow']:
        wm_sub = sub[sub['weather_main'] == wm]
        if len(wm_sub) > 0:
            print(f"Hour {h:02d}:00 [{wm}]: Mean = {wm_sub['traffic_volume'].mean():.1f} vph, Std = {wm_sub['traffic_volume'].std():.1f}, N = {len(wm_sub)}")

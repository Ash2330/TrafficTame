"""
src/preprocess.py
-----------------
Data preprocessing pipeline for Metro Interstate Traffic Volume dataset.

Handles:
1. Loading raw CSV data.
2. Cleaning impossible sensor values (temp = 0K, extreme rain_1h outliers).
3. Deduplicating timestamps with multiple weather reports per hour using
   weather severity priority and sensible aggregations.
4. Propagating holiday annotations across all 24 hours of each holiday date.
5. Reindexing to a complete continuous hourly time series.
6. Interpolating short gaps (<= 2-3 hours) while leaving multi-day sensor
   blackouts untouched to prevent data distortion.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# Weather severity ranking for deduplication
WEATHER_SEVERITY = {
    "Squall": 11,
    "Thunderstorm": 10,
    "Snow": 9,
    "Rain": 8,
    "Drizzle": 7,
    "Fog": 6,
    "Mist": 5,
    "Haze": 4,
    "Smoke": 3,
    "Clouds": 2,
    "Clear": 1,
}


def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    """Load raw dataset preserving 'None' as string in holiday."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at {path}")

    df = pd.read_csv(path, keep_default_na=False)
    df["date_time"] = pd.to_datetime(df["date_time"])
    # Convert numeric columns
    numeric_cols = ["temp", "rain_1h", "snow_1h", "clouds_all", "traffic_volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure empty holiday strings are represented as 'None'
    df["holiday"] = df["holiday"].replace("", "None")
    return df


def clean_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle sensor glitches and physical anomalies:
    - Temperature: Replace impossible values (temp <= 0 Kelvin) with NaN.
    - Rain: Replace extreme sensor outlier (9831.3 mm) with NaN.
    """
    df = df.copy()

    # Temperature anomalies (0 K is absolute zero, physical impossibility)
    temp_anomalies = (df["temp"] <= 0) | (df["temp"] < 200) | (df["temp"] > 330)
    df.loc[temp_anomalies, "temp"] = np.nan

    # Precipitation anomalies (> 100 mm/h is physically implausible for the region)
    rain_anomalies = df["rain_1h"] > 100.0
    df.loc[rain_anomalies, "rain_1h"] = np.nan

    return df


def _pick_dominant_weather(group: pd.DataFrame) -> pd.Series:
    """Helper to pick representative weather row based on severity."""
    if len(group) == 1:
        row = group.iloc[0]
        return pd.Series({
            "temp": row["temp"],
            "rain_1h": row["rain_1h"],
            "snow_1h": row["snow_1h"],
            "clouds_all": row["clouds_all"],
            "weather_main": row["weather_main"],
            "weather_description": row["weather_description"],
            "holiday": row["holiday"],
            "traffic_volume": row["traffic_volume"],
        })

    # Pick weather record with highest severity
    best_idx = group["weather_main"].map(WEATHER_SEVERITY).fillna(0).idxmax()
    dominant_weather = group.loc[best_idx]

    # Non-empty holiday if present
    hols = group["holiday"][group["holiday"] != "None"]
    holiday_val = hols.iloc[0] if len(hols) > 0 else "None"

    return pd.Series({
        "temp": group["temp"].mean(),
        "rain_1h": group["rain_1h"].max(),
        "snow_1h": group["snow_1h"].max(),
        "clouds_all": group["clouds_all"].mean(),
        "weather_main": dominant_weather["weather_main"],
        "weather_description": dominant_weather["weather_description"],
        "holiday": holiday_val,
        "traffic_volume": group["traffic_volume"].mean(),
    })


def deduplicate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate timestamps having multiple weather entries per hour.
    Aggregates numeric fields and selects the dominant weather condition.
    """
    df = df.sort_values("date_time").reset_index(drop=True)
    # Fast path for non-duplicated rows
    is_dupe = df["date_time"].duplicated(keep=False)
    unique_rows = df[~is_dupe].copy()
    dupe_rows = df[is_dupe]

    if len(dupe_rows) > 0:
        aggregated_dupes = (
            dupe_rows.groupby("date_time", group_keys=False)
            .apply(_pick_dominant_weather)
            .reset_index()
        )
        combined = pd.concat([unique_rows, aggregated_dupes], ignore_index=True)
    else:
        combined = unique_rows

    combined = combined.sort_values("date_time").reset_index(drop=True)
    return combined


def propagate_holidays(df: pd.DataFrame) -> pd.DataFrame:
    """
    The raw dataset tags holidays only at hour 00:00:00.
    Propagate the holiday name to all 24 hours of that calendar day.
    """
    df = df.copy()
    holiday_dates = df[df["holiday"] != "None"][["date_time", "holiday"]].copy()
    holiday_dates["date"] = holiday_dates["date_time"].dt.date
    holiday_map = holiday_dates.drop_duplicates(subset=["date"]).set_index("date")["holiday"].to_dict()

    df["date"] = df["date_time"].dt.date
    df["holiday"] = df["date"].map(holiday_map).fillna("None")
    df["is_holiday"] = (df["holiday"] != "None").astype(int)
    df = df.drop(columns=["date"])
    return df


def reindex_and_handle_gaps(
    df: pd.DataFrame,
    max_traffic_interp_hours: int = 2,
    max_weather_interp_hours: int = 6,
) -> pd.DataFrame:
    """
    Reindex to continuous hourly series from min to max date.
    - Interpolates short gaps (<= max_traffic_interp_hours) for traffic_volume.
    - Leaves large gaps (> max_traffic_interp_hours) as NaN to prevent fabricating data.
    - Interpolates temperature and clouds for short weather gaps.
    - Fills unrecorded rain/snow with 0.0.
    """
    df = df.set_index("date_time").sort_index()
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h", name="date_time")
    reindexed = df.reindex(full_idx)

    # Track originally recorded vs imputed traffic
    reindexed["traffic_imputed"] = reindexed["traffic_volume"].isnull()

    # Short-gap interpolation for traffic volume
    reindexed["traffic_volume"] = reindexed["traffic_volume"].interpolate(
        method="time", limit=max_traffic_interp_hours
    )
    # Only keep traffic_imputed True where it was filled
    reindexed["traffic_imputed"] = (
        reindexed["traffic_imputed"] & reindexed["traffic_volume"].notnull()
    )

    # Weather feature interpolation
    reindexed["temp"] = reindexed["temp"].interpolate(
        method="time", limit=max_weather_interp_hours
    )
    # Forward-fill remaining small temp gaps if any, or leave
    reindexed["temp"] = reindexed["temp"].ffill(limit=2)

    reindexed["clouds_all"] = (
        reindexed["clouds_all"]
        .interpolate(method="time", limit=max_weather_interp_hours)
        .ffill(limit=2)
        .round()
    )

    # Precipitation: unrecorded hours during dry periods are 0.0
    reindexed["rain_1h"] = reindexed["rain_1h"].fillna(0.0)
    reindexed["snow_1h"] = reindexed["snow_1h"].fillna(0.0)

    # Categorical weather forward fill for short periods
    reindexed["weather_main"] = reindexed["weather_main"].ffill(limit=max_weather_interp_hours)
    reindexed["weather_description"] = reindexed["weather_description"].ffill(
        limit=max_weather_interp_hours
    )

    # For any remaining gaps in categorical weather (e.g. during 10-month outage), default to 'Clear'
    reindexed["weather_main"] = reindexed["weather_main"].fillna("Clear")
    reindexed["weather_description"] = reindexed["weather_description"].fillna("clear sky")

    reindexed = reindexed.reset_index()

    # Re-propagate holidays across any newly added hours
    reindexed = propagate_holidays(reindexed)

    # Add useful human-readable temperature in Celsius
    reindexed["temp_c"] = (reindexed["temp"] - 273.15).round(2)

    # Flag zero-volume hours (do not silently drop them; keep for inspection)
    reindexed["is_zero_volume"] = (reindexed["traffic_volume"] == 0).astype(int)

    return reindexed


def run_pipeline(
    raw_path: str | Path = "data/raw/Metro_Interstate_Traffic_Volume.csv",
    output_path: str | Path = "data/processed/traffic_clean.csv",
) -> pd.DataFrame:
    """Run end-to-end preprocessing pipeline."""
    print("Step 1: Loading raw data...")
    raw_df = load_raw_data(raw_path)
    print(f"  Raw records: {len(raw_df):,}")

    print("Step 2: Cleaning sensor glitches (temp <= 0K, rain outlier)...")
    cleaned_df = clean_anomalies(raw_df)

    print("Step 3: Deduplicating hourly timestamps...")
    dedup_df = deduplicate_timestamps(cleaned_df)
    print(f"  Unique timestamps: {len(dedup_df):,}")

    print("Step 4: Propagating holidays across full calendar days...")
    holiday_df = propagate_holidays(dedup_df)
    print(f"  Total holiday hours: {(holiday_df['is_holiday'] == 1).sum():,}")

    print("Step 5: Reindexing to continuous hourly series & short-gap imputation...")
    final_df = reindex_and_handle_gaps(holiday_df, max_traffic_interp_hours=2)
    print(f"  Continuous hourly series length: {len(final_df):,}")
    print(f"  Valid traffic volume records: {final_df['traffic_volume'].notnull().sum():,}")
    print(f"  Imputed traffic hours: {final_df['traffic_imputed'].sum():,}")
    print(f"  Zero-volume hours flagged: {(final_df['is_zero_volume'] == 1).sum():,}")
    print(f"  Missing traffic hours remaining: {final_df['traffic_volume'].isnull().sum():,}")

    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_p, index=False)
    print(f"Saved processed dataset to {output_p.resolve()}")

    return final_df


if __name__ == "__main__":
    run_pipeline()

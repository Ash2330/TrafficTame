"""
src/features.py
---------------
Feature engineering and baseline evaluation module for Traffic Congestion Prediction.

Rules enforced:
1. Target at time T is traffic_volume at T.
2. Most recent traffic volume allowed as feature is T-1. No lookahead leakage.
3. Weather and holiday at time T are allowed as scenario inputs.
4. Lags across sensor outages are NOT imputed; rows with NaN lags are dropped.
5. Historical baseline computed strictly on the training partition.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score

WEATHER_CATEGORIES = [
    "Clear",
    "Clouds",
    "Rain",
    "Drizzle",
    "Mist",
    "Haze",
    "Fog",
    "Thunderstorm",
    "Snow",
    "Squall",
    "Smoke",
]

DEFAULT_THRESHOLDS_PATH = Path("models/congestion_thresholds.json")


def load_congestion_thresholds(
    thresholds_path: str | Path = DEFAULT_THRESHOLDS_PATH,
) -> dict:
    """Load precomputed training set congestion thresholds."""
    path = Path(thresholds_path)
    if not path.exists():
        return {
            "percentile_25": 1204.0,
            "percentile_50": 3339.0,
            "percentile_85": 5493.0,
            "labels": ["Low", "Moderate", "High", "Severe"],
        }
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_congestion_labels(
    volume: pd.Series | np.ndarray,
    thresholds: dict | None = None,
    as_category: bool = True,
) -> pd.Series | np.ndarray:
    """
    Convert continuous traffic volume into 4 categorical congestion levels:
    - Low:      volume < p25 (1204.0)
    - Moderate: p25 <= volume < p50 (3339.0)
    - High:     p50 <= volume < p85 (5493.0)
    - Severe:   volume >= p85 (5493.0)
    """
    if thresholds is None:
        thresholds = load_congestion_thresholds()

    p25 = thresholds.get("percentile_25", 1204.0)
    p50 = thresholds.get("percentile_50", 3339.0)
    p85 = thresholds.get("percentile_85", 5493.0)
    labels = thresholds.get("labels", ["Low", "Moderate", "High", "Severe"])

    arr = np.asarray(volume)
    conds = [
        arr < p25,
        (arr >= p25) & (arr < p50),
        (arr >= p50) & (arr < p85),
        arr >= p85,
    ]
    result = np.select(conds, labels, default="Moderate")

    if isinstance(volume, pd.Series) and as_category:
        return pd.Series(
            pd.Categorical(result, categories=labels, ordered=True),
            index=volume.index,
            name="congestion_level",
        )
    return result


def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate calendar and cyclical time features:
    - hour (0-23), day_of_week (0-6), month (1-12), is_weekend (0/1)
    - sin/cos cyclical encodings for hour, day_of_week, month
    """
    out = pd.DataFrame(index=df.index)
    dt = df["date_time"].dt

    out["hour"] = dt.hour
    out["day_of_week"] = dt.day_of_week
    out["month"] = dt.month
    out["is_weekend"] = dt.day_of_week.isin([5, 6]).astype(int)

    # Cyclical encodings
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)

    out["day_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7.0)
    out["day_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7.0)

    out["month_sin"] = np.sin(2 * np.pi * (out["month"] - 1) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * (out["month"] - 1) / 12.0)

    return out


def build_weather_features(
    df: pd.DataFrame,
    categories: list[str] = WEATHER_CATEGORIES,
) -> pd.DataFrame:
    """
    Extract weather attributes, holiday/day_type indicators, and encode macro weather categories.
    These values at time T are allowed as scenario inputs for simulator forecasting.
    """
    out = pd.DataFrame(index=df.index)
    out["temp_c"] = df["temp_c"]
    out["rain_1h"] = df["rain_1h"]
    out["snow_1h"] = df["snow_1h"]
    out["clouds_all"] = df["clouds_all"]
    out["is_holiday"] = df["is_holiday"]

    # Day type: 0 = weekday, 1 = weekend, 2 = holiday
    is_weekend = df["date_time"].dt.day_of_week.isin([5, 6]).astype(int)
    day_type = np.where(df["is_holiday"] == 1, 2, np.where(is_weekend == 1, 1, 0))
    out["day_type"] = day_type
    out["is_weekend_or_holiday"] = ((is_weekend == 1) | (df["is_holiday"] == 1)).astype(int)
    out["day_type_weekday"] = (day_type == 0).astype(int)
    out["day_type_weekend"] = (day_type == 1).astype(int)
    out["day_type_holiday"] = (day_type == 2).astype(int)

    # One-hot encode weather_main with consistent column alignment
    weather_series = df["weather_main"].fillna("Clear")
    for cat in categories:
        out[f"weather_{cat.lower()}"] = (weather_series == cat).astype(int)

    return out


def build_lag_features(
    df: pd.DataFrame,
    lags: tuple[int, ...] = (1, 2, 3, 24, 168),
    target_col: str = "traffic_volume",
) -> pd.DataFrame:
    """
    Construct strictly past lag features:
    lag_1h (T-1), lag_2h (T-2), lag_3h (T-3), lag_24h (T-24), lag_168h (T-168).
    Assumes continuous hourly series sorted by date_time.
    """
    out = pd.DataFrame(index=df.index)
    for lag in lags:
        out[f"lag_{lag}h"] = df[target_col].shift(lag)
    return out


def build_rolling_features(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (3, 6, 24),
    target_col: str = "traffic_volume",
) -> pd.DataFrame:
    """
    Rolling summary statistics computed ONLY from data up to T-1.
    Uses shift(1) prior to rolling window aggregation to guarantee zero leakage.
    """
    out = pd.DataFrame(index=df.index)
    # Strictly past values
    shifted_target = df[target_col].shift(1)

    for w in windows:
        out[f"rolling_mean_{w}h"] = shifted_target.rolling(window=w, min_periods=w).mean()

    # Past 24h volatility
    out["rolling_std_24h"] = shifted_target.rolling(window=24, min_periods=24).std()
    return out


def build_full_features(
    df: pd.DataFrame,
    drop_na: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assemble complete feature dataset with zero lookahead leakage.
    Returns:
    - features_df: Matrix X containing time, weather, lag, and rolling features.
    - full_df: Combined dataframe with target traffic_volume and congestion labels.
    """
    df_sorted = df.sort_values("date_time").reset_index(drop=True)

    time_feats = build_time_features(df_sorted)
    weather_feats = build_weather_features(df_sorted)
    lag_feats = build_lag_features(df_sorted)
    rolling_feats = build_rolling_features(df_sorted)

    combined = pd.concat(
        [
            df_sorted[["date_time", "traffic_volume", "is_zero_volume"]],
            time_feats,
            weather_feats,
            lag_feats,
            rolling_feats,
        ],
        axis=1,
    )

    combined["congestion_level"] = build_congestion_labels(combined["traffic_volume"])

    if drop_na:
        # Require all lag features, rolling features, and the target to be non-null
        feature_cols = [c for c in combined.columns if c not in ["date_time", "congestion_level"]]
        combined = combined.dropna(subset=feature_cols).reset_index(drop=True)

    return combined


class HistoricalMeanBaseline:
    """
    Baseline A: Historical average traffic volume by (day_of_week, hour),
    fit strictly on the training partition (or train+val for final comparison).
    """

    def __init__(self):
        self.lookup: dict[tuple[int, int], float] = {}
        self.global_mean: float = 0.0

    def fit(self, train_df: pd.DataFrame, target_col: str = "traffic_volume"):
        grouped = train_df.groupby(["day_of_week", "hour"])[target_col].mean()
        self.lookup = grouped.to_dict()
        self.global_mean = float(train_df[target_col].mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = []
        for d, h in zip(df["day_of_week"], df["hour"]):
            preds.append(self.lookup.get((d, h), self.global_mean))
        return np.array(preds)


class BaselineDHistoricalMean:
    """
    Baseline D: Historical average traffic volume by (day_type, hour, month),
    where day_type is weekday (0), weekend (1), or holiday (2).
    Includes hierarchical fallbacks to ensure robust prediction with zero missing values.
    """

    def __init__(self):
        self.lookup_dhm: dict[tuple[int, int, int], float] = {}
        self.lookup_dh: dict[tuple[int, int], float] = {}
        self.lookup_h: dict[int, float] = {}
        self.global_mean: float = 0.0

    def fit(self, train_df: pd.DataFrame, target_col: str = "traffic_volume"):
        df = train_df.copy()
        if "day_type" not in df.columns:
            is_wknd = df["is_weekend"] if "is_weekend" in df.columns else df["date_time"].dt.dayofweek.isin([5, 6]).astype(int)
            is_hol = df["is_holiday"]
            df["day_type"] = np.where(is_hol == 1, 2, np.where(is_wknd == 1, 1, 0))

        self.lookup_dhm = df.groupby(["day_type", "hour", "month"])[target_col].mean().to_dict()
        self.lookup_dh = df.groupby(["day_type", "hour"])[target_col].mean().to_dict()
        self.lookup_h = df.groupby("hour")[target_col].mean().to_dict()
        self.global_mean = float(df[target_col].mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = []
        dt_col = df["day_type"] if "day_type" in df.columns else np.zeros(len(df))
        for dt, h, m in zip(dt_col, df["hour"], df["month"]):
            if (dt, h, m) in self.lookup_dhm:
                preds.append(self.lookup_dhm[(dt, h, m)])
            elif (dt, h) in self.lookup_dh:
                preds.append(self.lookup_dh[(dt, h)])
            else:
                preds.append(self.lookup_h.get(h, self.global_mean))
        return np.array(preds)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: dict | None = None,
) -> dict[str, float]:
    """Calculate regression and 4-class classification metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    cls_true = build_congestion_labels(y_true, thresholds=thresholds, as_category=False)
    cls_pred = build_congestion_labels(y_pred, thresholds=thresholds, as_category=False)
    acc = float(accuracy_score(cls_true, cls_pred))

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "Accuracy": round(acc * 100, 2),
    }


def evaluate_baselines(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    thresholds: dict | None = None,
) -> pd.DataFrame:
    """
    Evaluate Baselines A, B, and C across Validation (2017) and Test (2018) sets.
    """
    hist_model = HistoricalMeanBaseline().fit(train_df)
    results = []

    for split_name, split_data in [("Validation (2017)", val_df), ("Test (2018)", test_df)]:
        y_true = split_data["traffic_volume"].values

        # Baseline A: Historical Mean by (day_of_week, hour)
        pred_a = hist_model.predict(split_data)
        metrics_a = compute_metrics(y_true, pred_a, thresholds)
        results.append({"Split": split_name, "Baseline": "Baseline A (Historical Mean)", **metrics_a})

        # Baseline B: Persistence (T-1 volume)
        pred_b = split_data["lag_1h"].values
        metrics_b = compute_metrics(y_true, pred_b, thresholds)
        results.append({"Split": split_name, "Baseline": "Baseline B (Persistence)", **metrics_b})

        # Baseline C: Seasonal Naive (T-168 volume)
        pred_c = split_data["lag_168h"].values
        metrics_c = compute_metrics(y_true, pred_c, thresholds)
        results.append({"Split": split_name, "Baseline": "Baseline C (Seasonal Naive 168h)", **metrics_c})

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("Building full engineered feature dataset...")
    raw_clean_path = Path("data/processed/traffic_clean.csv")
    clean_data = pd.read_csv(raw_clean_path)
    clean_data["date_time"] = pd.to_datetime(clean_data["date_time"])

    full_features = build_full_features(clean_data, drop_na=True)
    print(f"Total usable engineered records: {len(full_features):,}")
    print(f"Engineered columns: {len(full_features.columns)} columns")

    train = full_features[full_features["date_time"] < "2017-01-01"]
    val = full_features[(full_features["date_time"] >= "2017-01-01") & (full_features["date_time"] < "2018-01-01")]
    test = full_features[full_features["date_time"] >= "2018-01-01"]

    print(f"Train (2012-2016): {len(train):,} rows")
    print(f"Val (2017):        {len(val):,} rows")
    print(f"Test (2018):       {len(test):,} rows")

    baseline_results = evaluate_baselines(train, val, test)
    print("\nBaseline Results:")
    print(baseline_results.to_string(index=False))

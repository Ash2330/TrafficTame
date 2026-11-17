"""
src/train.py
------------
Model training, evaluation, and serialization pipeline for Traffic Congestion Prediction.

Trains two models:
1. Forecast Model: Uses all features (time, weather, lags T-1..T-168, rolling stats).
2. Scenario Model: Uses NO lag or rolling features (time, weather, holiday).
   Designed specifically to power the interactive What-If Simulator dashboard.

Protocol:
- Trains and tunes on Train partition (2012–2016).
- Evaluates Ridge, RandomForest, LightGBM, and XGBoost on Validation (2017).
- Selects best model per feature set on Validation.
- Refits best models on combined Train + Validation (2012–2017).
- Evaluates on Test (2018) ONCE.
- Serializes models, feature column schemas, and metadata.
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import joblib
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.features import (
    build_congestion_labels,
    build_full_features,
    load_congestion_thresholds,
    HistoricalMeanBaseline,
    BaselineDHistoricalMean,
)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: dict | None = None,
) -> dict:
    """Compute regression metrics, 4-class accuracy, within-one-class accuracy, and confusion matrix."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    cls_true = build_congestion_labels(y_true, thresholds=thresholds, as_category=False)
    cls_pred = build_congestion_labels(y_pred, thresholds=thresholds, as_category=False)

    cls_order = ["Low", "Moderate", "High", "Severe"]
    cls_map = {lbl: i for i, lbl in enumerate(cls_order)}

    acc = float(accuracy_score(cls_true, cls_pred))

    true_indices = np.array([cls_map[c] for c in cls_true])
    pred_indices = np.array([cls_map[c] for c in cls_pred])
    within_one = float(np.mean(np.abs(true_indices - pred_indices) <= 1))

    cm = confusion_matrix(cls_true, cls_pred, labels=cls_order).tolist()

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "Accuracy": round(acc * 100, 2),
        "Within_1_Class": round(within_one * 100, 2),
        "Confusion_Matrix": cm,
    }


def compute_sliced_mae(
    df: pd.DataFrame,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute sliced MAE across operational traffic regimes."""
    df_eval = df.copy()
    df_eval["pred"] = y_pred
    df_eval["error"] = np.abs(df_eval["traffic_volume"] - df_eval["pred"])

    slices = {}

    # Weekday morning rush (07:00-08:00)
    am_mask = (df_eval["is_weekend"] == 0) & (df_eval["hour"].isin([7, 8]))
    slices["Weekday_AM_Rush (07-08)"] = round(float(df_eval.loc[am_mask, "error"].mean()), 2) if am_mask.any() else np.nan

    # Weekday evening rush (16:00-17:00)
    pm_mask = (df_eval["is_weekend"] == 0) & (df_eval["hour"].isin([16, 17]))
    slices["Weekday_PM_Rush (16-17)"] = round(float(df_eval.loc[pm_mask, "error"].mean()), 2) if pm_mask.any() else np.nan

    # Weekends
    weekend_mask = df_eval["is_weekend"] == 1
    slices["Weekends"] = round(float(df_eval.loc[weekend_mask, "error"].mean()), 2) if weekend_mask.any() else np.nan

    # Holidays
    holiday_mask = df_eval["is_holiday"] == 1
    slices["Holidays"] = round(float(df_eval.loc[holiday_mask, "error"].mean()), 2) if holiday_mask.any() else np.nan

    # Snow hours
    snow_mask = (df_eval["snow_1h"] > 0) | (df_eval["weather_snow"] == 1)
    slices["Snow_Hours"] = round(float(df_eval.loc[snow_mask, "error"].mean()), 2) if snow_mask.any() else np.nan

    # Rain hours
    rain_mask = (df_eval["rain_1h"] > 0) | (df_eval["weather_rain"] == 1)
    slices["Rain_Hours"] = round(float(df_eval.loc[rain_mask, "error"].mean()), 2) if rain_mask.any() else np.nan

    # Overall
    slices["Overall"] = round(float(df_eval["error"].mean()), 2)

    return slices


def train_and_evaluate_all():
    """Execute complete training, comparison, refitting, and serialization."""
    print("Loading preprocessed dataset and generating features...")
    raw_clean_path = Path("data/processed/traffic_clean.csv")
    clean_data = pd.read_csv(raw_clean_path)
    clean_data["date_time"] = pd.to_datetime(clean_data["date_time"])

    full_features = build_full_features(clean_data, drop_na=True)
    thresholds = load_congestion_thresholds()

    # Define feature schemas
    forecast_features = [
        c for c in full_features.columns
        if c not in ["date_time", "traffic_volume", "is_zero_volume", "congestion_level"]
    ]
    scenario_features = [
        c for c in forecast_features
        if not c.startswith("lag_") and not c.startswith("rolling_")
    ]

    # Save feature schemas
    with open(MODELS_DIR / "forecast_features.json", "w", encoding="utf-8") as f:
        json.dump(forecast_features, f, indent=2)
    with open(MODELS_DIR / "scenario_features.json", "w", encoding="utf-8") as f:
        json.dump(scenario_features, f, indent=2)

    # Chronological partition
    train_df = full_features[full_features["date_time"] < "2017-01-01"].copy()
    val_df = full_features[(full_features["date_time"] >= "2017-01-01") & (full_features["date_time"] < "2018-01-01")].copy()
    train_val_df = full_features[full_features["date_time"] < "2018-01-01"].copy()
    test_df = full_features[full_features["date_time"] >= "2018-01-01"].copy()

    print(f"Train split: {len(train_df):,} | Val: {len(val_df):,} | Train+Val: {len(train_val_df):,} | Test: {len(test_df):,}")

    # Model definitions
    candidate_models = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0, random_state=42)),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_split=5, random_state=42, n_jobs=-1),
        "LightGBM": LGBMRegressor(n_estimators=250, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1),
        "XGBoost": XGBRegressor(n_estimators=250, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42),
    }

    val_records = []
    trained_candidates = {}

    # Sample weights for holiday conditioning on scenario models
    sw_tr = np.where(train_df["is_holiday"] == 1, 20.0, 1.0)
    sw_tv = np.where(train_val_df["is_holiday"] == 1, 20.0, 1.0)

    print("\nPhase 1: Evaluating Candidate Models on Validation Set (2017)...")
    for fset_name, cols in [("Forecast", forecast_features), ("Scenario", scenario_features)]:
        X_tr, y_tr = train_df[cols], train_df["traffic_volume"].values
        X_val, y_val = val_df[cols], val_df["traffic_volume"].values
        sw = sw_tr if fset_name == "Scenario" else None

        for m_name, model in candidate_models.items():
            print(f"  Training {fset_name} :: {m_name}...")
            # Clone model
            if m_name == "Ridge":
                m = make_pipeline(StandardScaler(), Ridge(alpha=10.0, random_state=42))
                m.fit(X_tr, y_tr)
            elif m_name == "RandomForest":
                m = RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_split=5, random_state=42, n_jobs=-1)
                m.fit(X_tr, y_tr, sample_weight=sw)
            elif m_name == "LightGBM":
                m = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
                m.fit(X_tr, y_tr, sample_weight=sw)
            elif m_name == "XGBoost":
                m = XGBRegressor(n_estimators=250, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
                m.fit(X_tr, y_tr, sample_weight=sw)

            preds_val = m.predict(X_val)
            metrics = compute_comprehensive_metrics(y_val, preds_val, thresholds)
            trained_candidates[(fset_name, m_name)] = m

            val_records.append({
                "Feature_Set": fset_name,
                "Model": m_name,
                **{k: v for k, v in metrics.items() if k != "Confusion_Matrix"},
            })

    val_comparison_df = pd.DataFrame(val_records)
    print("\nValidation Comparison:")
    print(val_comparison_df.to_string(index=False))

    # Identify winners on Validation set (lowest MAE)
    best_forecast_name = val_comparison_df[val_comparison_df["Feature_Set"] == "Forecast"].sort_values("MAE").iloc[0]["Model"]
    best_scenario_name = val_comparison_df[val_comparison_df["Feature_Set"] == "Scenario"].sort_values("MAE").iloc[0]["Model"]
    print(f"\nSelected Best Forecast Model: {best_forecast_name}")
    print(f"Selected Best Scenario Model: {best_scenario_name}")

    print("\nPhase 2: Refitting Best Models on Train + Validation (2012–2017)...")
    # Instantiate best models with chosen hyperparameters
    if best_forecast_name == "XGBoost":
        best_forecast_model = XGBRegressor(n_estimators=250, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
    else:
        best_forecast_model = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)

    if best_scenario_name == "XGBoost":
        best_scenario_model = XGBRegressor(n_estimators=250, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
    else:
        best_scenario_model = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)

    # Refit on combined Train + Validation
    X_tv_f = train_val_df[forecast_features]
    X_tv_s = train_val_df[scenario_features]
    y_tv = train_val_df["traffic_volume"].values

    best_forecast_model.fit(X_tv_f, y_tv)
    best_scenario_model.fit(X_tv_s, y_tv, sample_weight=sw_tv)

    print("Phase 3: Final Single Evaluation on 2018 Test Set...")
    X_te_f = test_df[forecast_features]
    X_te_s = test_df[scenario_features]
    y_te = test_df["traffic_volume"].values

    pred_test_f = best_forecast_model.predict(X_te_f)
    pred_test_s = best_scenario_model.predict(X_te_s)

    test_metrics_f = compute_comprehensive_metrics(y_te, pred_test_f, thresholds)
    test_metrics_s = compute_comprehensive_metrics(y_te, pred_test_s, thresholds)

    # Fit and evaluate baselines on Train+Val for fair comparison
    baseline_a = HistoricalMeanBaseline().fit(train_val_df)
    pred_test_base_a = baseline_a.predict(test_df)
    test_metrics_base_a = compute_comprehensive_metrics(y_te, pred_test_base_a, thresholds)
    slices_base_a = compute_sliced_mae(test_df, pred_test_base_a)

    baseline_d = BaselineDHistoricalMean().fit(train_val_df)
    pred_test_base_d = baseline_d.predict(test_df)
    test_metrics_base_d = compute_comprehensive_metrics(y_te, pred_test_base_d, thresholds)
    slices_base_d = compute_sliced_mae(test_df, pred_test_base_d)

    # Sliced MAE
    slices_f = compute_sliced_mae(test_df, pred_test_f)
    slices_s = compute_sliced_mae(test_df, pred_test_s)

    # Save models
    joblib.dump(best_forecast_model, MODELS_DIR / "forecast_model.joblib")
    joblib.dump(best_scenario_model, MODELS_DIR / "scenario_model.joblib")
    print("Serialized models to models/forecast_model.joblib and models/scenario_model.joblib")

    # Save metadata
    metadata = {
        "dataset": "UCI Metro Interstate Traffic Volume",
        "training_window": "2012-10-02 to 2017-12-31 (Train + Validation refit)",
        "test_window": "2018-01-01 to 2018-09-30 (Strictly single evaluation)",
        "thresholds_file": "models/congestion_thresholds.json",
        "baseline_a_refit": {
            "name": "Historical Mean by (Day_of_Week, Hour) on Train+Val",
            "test_metrics": test_metrics_base_a,
            "test_sliced_mae": slices_base_a,
        },
        "baseline_d": {
            "name": "Historical Mean by (Day_Type, Hour, Month) on Train+Val",
            "test_metrics": test_metrics_base_d,
            "test_sliced_mae": slices_base_d,
        },
        "forecast_model": {
            "type": best_forecast_name,
            "feature_count": len(forecast_features),
            "features_file": "models/forecast_features.json",
            "validation_metrics": val_comparison_df[(val_comparison_df["Feature_Set"] == "Forecast") & (val_comparison_df["Model"] == best_forecast_name)].iloc[0].to_dict(),
            "test_metrics": test_metrics_f,
            "test_sliced_mae": slices_f,
        },
        "scenario_model": {
            "type": best_scenario_name,
            "feature_count": len(scenario_features),
            "features_file": "models/scenario_features.json",
            "validation_metrics": val_comparison_df[(val_comparison_df["Feature_Set"] == "Scenario") & (val_comparison_df["Model"] == best_scenario_name)].iloc[0].to_dict(),
            "test_metrics": test_metrics_s,
            "test_sliced_mae": slices_s,
        },
    }

    with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print("Serialized metadata to models/model_metadata.json")

    return {
        "val_comparison": val_comparison_df,
        "test_metrics_forecast": test_metrics_f,
        "test_metrics_scenario": test_metrics_s,
        "slices_forecast": slices_f,
        "slices_scenario": slices_s,
        "best_forecast_name": best_forecast_name,
        "best_scenario_name": best_scenario_name,
    }


if __name__ == "__main__":
    train_and_evaluate_all()

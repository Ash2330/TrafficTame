"""
src/simulator.py
----------------
Core logic and numerical computation engine for the What-If Traffic Simulator.
All business logic, curve generation, grid synthesis, and Plotly visualizations
are isolated here for clean modularity and comprehensive unit testing.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib

from src.predict import (
    build_scenario_feature_row,
    load_model_and_features,
    predict_volume,
    volume_to_level,
)
from src.features import load_congestion_thresholds, BaselineDHistoricalMean

MODELS_DIR = Path("models")
_BASELINE_D_MODEL = None


def _get_baseline_d():
    """Lazy loader for Baseline D model (used for holiday scenario routing)."""
    global _BASELINE_D_MODEL
    if _BASELINE_D_MODEL is None:
        p = MODELS_DIR / "baseline_d.joblib"
        if p.exists():
            try:
                _BASELINE_D_MODEL = joblib.load(p)
            except Exception:
                _BASELINE_D_MODEL = None
    return _BASELINE_D_MODEL

# Color palette mapped to congestion tiers
CONGESTION_COLORS = {
    "Low": "#2ecc71",       # Emerald Green
    "Moderate": "#f1c40f",  # Sunflower Yellow
    "High": "#e67e22",      # Pumpkin Orange
    "Severe": "#e74c3c",    # Crimson Red
}

PRESETS = {
    "Custom Scenario": None,
    "Weekday AM Rush (Clear)": {
        "hour": 8, "weekday": 1, "month": 10, "temp_c": 12.0,
        "rain_1h": 0.0, "snow_1h": 0.0, "clouds_all": 20, "weather_main": "Clear",
        "is_holiday": 0, "demand_mult": 1.0,
    },
    "Weekday PM Rush (Clear)": {
        "hour": 17, "weekday": 3, "month": 6, "temp_c": 24.0,
        "rain_1h": 0.0, "snow_1h": 0.0, "clouds_all": 10, "weather_main": "Clear",
        "is_holiday": 0, "demand_mult": 1.0,
    },
    "Severe Winter Blizzard (AM Peak)": {
        "hour": 8, "weekday": 2, "month": 1, "temp_c": -12.0,
        "rain_1h": 0.0, "snow_1h": 2.5, "clouds_all": 100, "weather_main": "Snow",
        "is_holiday": 0, "demand_mult": 1.0,
    },
    "Summer Friday Getaway Surge": {
        "hour": 16, "weekday": 4, "month": 7, "temp_c": 29.0,
        "rain_1h": 0.0, "snow_1h": 0.0, "clouds_all": 30, "weather_main": "Clear",
        "is_holiday": 0, "demand_mult": 1.15,
    },
    "Thanksgiving Morning": {
        "hour": 9, "weekday": 3, "month": 11, "temp_c": 1.0,
        "rain_1h": 0.0, "snow_1h": 0.0, "clouds_all": 60, "weather_main": "Clouds",
        "is_holiday": 1, "demand_mult": 1.0,
    },
    "Dead of Night Free Flow": {
        "hour": 3, "weekday": 6, "month": 5, "temp_c": 14.0,
        "rain_1h": 0.0, "snow_1h": 0.0, "clouds_all": 0, "weather_main": "Clear",
        "is_holiday": 0, "demand_mult": 1.0,
    },
}


def get_scenario_presets() -> dict:
    """Return dictionary of predefined What-If scenarios."""
    return PRESETS


def get_regime_uncertainty(
    hour: int,
    weekday: int,
    is_holiday: int,
    weather_main: str,
    snow_1h: float,
    rain_1h: float,
    metadata_path: str | Path | None = None,
) -> float:
    """
    Retrieve empirical regime-based MAE uncertainty (in vph) from test set evaluation.
    Matches the active scenario condition against measured test set slices.
    """
    meta_file = Path(metadata_path or (MODELS_DIR / "model_metadata.json"))
    default_uncertainty = 218.86  # Default overall test MAE
    if not meta_file.exists():
        return default_uncertainty

    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        slices = meta.get("scenario_model", {}).get("test_sliced_mae", {})

        if snow_1h > 0 or weather_main.lower() == "snow":
            return float(slices.get("Snow_Hours", 535.65))
        if is_holiday == 1:
            return float(slices.get("Holidays", 357.66))
        if weekday < 5 and hour in [7, 8]:
            return float(slices.get("Weekday_AM_Rush (07-08)", 327.17))
        if weekday < 5 and hour in [16, 17]:
            return float(slices.get("Weekday_PM_Rush (16-17)", 322.03))
        if weekday in [5, 6]:
            return float(slices.get("Weekends", 275.01))
        if rain_1h > 0 or weather_main.lower() == "rain":
            return float(slices.get("Rain_Hours", 172.44))

        return float(slices.get("Overall", 218.86))
    except Exception:
        return default_uncertainty


def get_scenario_prediction(
    hour: int = 8,
    weekday: int = 1,
    month: int = 10,
    temp_c: float = 12.0,
    rain_1h: float = 0.0,
    snow_1h: float = 0.0,
    clouds_all: int = 20,
    weather_main: str = "Clear",
    is_holiday: int = 0,
    demand_mult: float = 1.0,
) -> dict:
    """
    Compute single-point scenario prediction with demand multiplier and uncertainty.
    """
    # Option 2 Routing: If is_holiday == 1, route to Baseline D historical lookup table
    # Selected on 2017 Validation Set (Val Holiday MAE 404.8 vph vs 485.8 vph for ML model)
    base_d = _get_baseline_d() if is_holiday == 1 else None
    if is_holiday == 1 and base_d is not None:
        df_h = pd.DataFrame([{"day_type": 2, "hour": hour, "month": month}])
        raw_vol = float(base_d.predict(df_h)[0])
        is_holiday_routed = True
    else:
        feat_row = build_scenario_feature_row(
            hour=hour,
            weekday=weekday,
            month=month,
            temp_c=temp_c,
            rain_1h=rain_1h,
            snow_1h=snow_1h,
            clouds_all=clouds_all,
            weather_main=weather_main,
            is_holiday=is_holiday,
        )
        raw_vol = float(predict_volume(feat_row, model_type="scenario")[0])
        is_holiday_routed = False

    scaled_vol = float(np.clip(raw_vol * demand_mult, 0, None))
    level = str(volume_to_level(scaled_vol))
    color = CONGESTION_COLORS.get(level, "#95a5a6")
    uncertainty = get_regime_uncertainty(hour, weekday, is_holiday, weather_main, snow_1h, rain_1h)

    return {
        "raw_volume": round(raw_vol, 1),
        "volume": round(scaled_vol, 1),
        "level": level,
        "color": color,
        "uncertainty_mae": round(uncertainty, 1),
        "volume_lower": max(0.0, round(scaled_vol - uncertainty, 1)),
        "volume_upper": round(scaled_vol + uncertainty, 1),
        "demand_multiplier": demand_mult,
        "is_holiday_routed": is_holiday_routed,
    }


def get_24h_curve(
    weekday: int = 1,
    month: int = 10,
    temp_c: float = 12.0,
    rain_1h: float = 0.0,
    snow_1h: float = 0.0,
    clouds_all: int = 20,
    weather_main: str = "Clear",
    is_holiday: int = 0,
    demand_mult: float = 1.0,
) -> pd.DataFrame:
    """
    Generate complete 24-hour diurnal volume curve (hours 0..23) for the scenario.
    """
    records = []
    for h in range(24):
        pred = get_scenario_prediction(
            hour=h,
            weekday=weekday,
            month=month,
            temp_c=temp_c,
            rain_1h=rain_1h,
            snow_1h=snow_1h,
            clouds_all=clouds_all,
            weather_main=weather_main,
            is_holiday=is_holiday,
            demand_mult=demand_mult,
        )
        records.append({
            "hour": h,
            "volume": pred["volume"],
            "level": pred["level"],
            "lower": pred["volume_lower"],
            "upper": pred["volume_upper"],
        })
    return pd.DataFrame(records)


def get_baseline_24h_curve(weekday: int = 1, month: int = 10, is_holiday: int = 0) -> pd.DataFrame:
    """
    Generate typical baseline 24-hour curve for comparison using clear weather baseline.
    """
    records = []
    for h in range(24):
        pred = get_scenario_prediction(
            hour=h,
            weekday=weekday,
            month=month,
            temp_c=15.0,
            rain_1h=0.0,
            snow_1h=0.0,
            clouds_all=20,
            weather_main="Clear",
            is_holiday=is_holiday,
            demand_mult=1.0,
        )
        records.append({
            "hour": h,
            "baseline_volume": pred["volume"],
        })
    return pd.DataFrame(records)


def get_7x24_grid(
    month: int = 10,
    temp_c: float = 12.0,
    rain_1h: float = 0.0,
    snow_1h: float = 0.0,
    clouds_all: int = 20,
    weather_main: str = "Clear",
    is_holiday: int = 0,
    demand_mult: float = 1.0,
) -> pd.DataFrame:
    """
    Generate 7-day x 24-hour grid of traffic predictions for the selected weather conditions.
    Rows: Days of Week (Monday to Sunday)
    Columns: Hours (00:00 to 23:00)
    """
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    grid_rows = []

    for d in range(7):
        row_vals = []
        for h in range(24):
            pred = get_scenario_prediction(
                hour=h,
                weekday=d,
                month=month,
                temp_c=temp_c,
                rain_1h=rain_1h,
                snow_1h=snow_1h,
                clouds_all=clouds_all,
                weather_main=weather_main,
                is_holiday=is_holiday,
                demand_mult=demand_mult,
            )
            row_vals.append(pred["volume"])
        grid_rows.append(row_vals)

    col_names = [f"{h:02d}:00" for h in range(24)]
    return pd.DataFrame(grid_rows, index=day_names, columns=col_names)


def create_road_figure(volume: float, level: str) -> go.Figure:
    """
    Build interactive Plotly visual representing an I-94 westbound multi-lane highway segment.
    Lane color matches the current congestion level.
    """
    color = CONGESTION_COLORS.get(level, "#95a5a6")

    # Map volume to vehicle icon density
    # Maximum physical capacity ~6,000 vph
    num_cars = max(1, min(14, int(round((volume / 6000.0) * 14))))

    fig = go.Figure()

    # Highway roadbed
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=10, y1=4,
        fillcolor="#2c3e50",
        line=dict(color="#1a252f", width=3),
    )

    # Lane dividers (dashed yellow/white lines)
    for y in [1, 2, 3]:
        fig.add_shape(
            type="line",
            x0=0, y0=y, x1=10, y1=y,
            line=dict(color="#ecf0f1", width=2, dash="dash"),
        )

    # Road shoulder bands
    fig.add_shape(type="line", x0=0, y0=0.05, x1=10, y1=0.05, line=dict(color=color, width=6))
    fig.add_shape(type="line", x0=0, y0=3.95, x1=10, y1=3.95, line=dict(color=color, width=6))

    # Scatter cars across lanes
    np.random.seed(int(volume) % 1000)
    car_x = np.linspace(0.8, 9.2, num_cars)
    car_y = [0.5, 1.5, 2.5, 3.5] * (num_cars // 4 + 1)
    car_y = car_y[:num_cars]

    fig.add_trace(go.Scatter(
        x=car_x,
        y=car_y,
        mode="markers+text",
        marker=dict(
            symbol="triangle-left",
            size=22,
            color=color,
            line=dict(color="#ffffff", width=1.5),
        ),
        text=["🚗" for _ in range(num_cars)],
        textposition="middle center",
        hoverinfo="text",
        hovertext=[f"I-94 Westbound | {level} Flow ({volume:,.0f} vph)" for _ in range(num_cars)],
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>I-94 Westbound Segment Simulation</b>: <span style='color:{color}'>{level.upper()} CONGESTION</span> ({volume:,.0f} vph)",
            font=dict(size=15),
            x=0.02,
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.2, 10.2]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.2, 4.2]),
        margin=dict(l=10, r=10, t=45, b=10),
        height=190,
        plot_bgcolor="#1e272e",
        paper_bgcolor="#1e272e",
    )
    return fig


def create_24h_plotly_figure(
    df_curve: pd.DataFrame,
    df_baseline: pd.DataFrame | None = None,
    thresholds: dict | None = None,
    current_hour: int = 8,
) -> go.Figure:
    """
    Build Plotly 24-hour diurnal profile with shaded congestion threshold zones,
    the active scenario curve, error envelope, and optional baseline overlay.
    """
    if thresholds is None:
        thresholds = load_congestion_thresholds()

    p25 = thresholds.get("percentile_25", 1204.0)
    p50 = thresholds.get("percentile_50", 3339.0)
    p85 = thresholds.get("percentile_85", 5493.0)
    max_y = max(7000.0, float(df_curve["upper"].max() * 1.1))

    fig = go.Figure()

    # Shaded threshold zones
    # Low zone: 0 to p25
    fig.add_hrect(y0=0, y1=p25, fillcolor="rgba(46, 204, 113, 0.12)", line_width=0, annotation_text="Low (< 1,204)", annotation_position="top left")
    # Moderate zone: p25 to p50
    fig.add_hrect(y0=p25, y1=p50, fillcolor="rgba(241, 196, 15, 0.12)", line_width=0, annotation_text="Moderate (1,204–3,339)", annotation_position="top left")
    # High zone: p50 to p85
    fig.add_hrect(y0=p50, y1=p85, fillcolor="rgba(230, 126, 34, 0.12)", line_width=0, annotation_text="High (3,339–5,493)", annotation_position="top left")
    # Severe zone: p85 to max_y
    fig.add_hrect(y0=p85, y1=max_y, fillcolor="rgba(231, 76, 60, 0.14)", line_width=0, annotation_text="Severe (≥ 5,493)", annotation_position="top left")

    # Uncertainty envelope
    fig.add_trace(go.Scatter(
        x=list(df_curve["hour"]) + list(df_curve["hour"][::-1]),
        y=list(df_curve["upper"]) + list(df_curve["lower"][::-1]),
        fill="toself",
        fillcolor="rgba(52, 152, 219, 0.20)",
        line=dict(color="rgba(255, 255, 255, 0)"),
        hoverinfo="skip",
        showlegend=True,
        name="Regime Uncertainty (±MAE)",
    ))

    # Baseline overlay if available
    if df_baseline is not None and "baseline_volume" in df_baseline.columns:
        fig.add_trace(go.Scatter(
            x=df_baseline["hour"],
            y=df_baseline["baseline_volume"],
            mode="lines",
            line=dict(color="#95a5a6", width=2, dash="dash"),
            name="Typical Clear Baseline",
            hovertemplate="Baseline Hour %{x:02d}:00: <b>%{y:,.0f} vph</b><extra></extra>",
        ))

    # Scenario curve
    fig.add_trace(go.Scatter(
        x=df_curve["hour"],
        y=df_curve["volume"],
        mode="lines+markers",
        line=dict(color="#2980b9", width=3),
        marker=dict(size=6, color="#2980b9"),
        name="What-If Scenario Prediction",
        hovertemplate="Scenario Hour %{x:02d}:00: <b>%{y:,.0f} vph</b><extra></extra>",
    ))

    # Current selected hour marker
    if current_hour in df_curve["hour"].values:
        cur_row = df_curve[df_curve["hour"] == current_hour].iloc[0]
        cur_col = CONGESTION_COLORS.get(cur_row["level"], "#e74c3c")
        fig.add_trace(go.Scatter(
            x=[current_hour],
            y=[cur_row["volume"]],
            mode="markers",
            marker=dict(size=14, color=cur_col, line=dict(color="#ffffff", width=2)),
            name=f"Selected Hour ({current_hour:02d}:00)",
            hovertemplate=f"Selected Hour {current_hour:02d}:00: <b>{cur_row['volume']:,.0f} vph</b> ({cur_row['level']})<extra></extra>",
        ))

    fig.update_layout(
        title="<b>24-Hour Diurnal Volume Curve with Congestion Bands</b>",
        xaxis=dict(
            title="Hour of Day",
            tickmode="linear",
            tick0=0,
            dtick=1,
            range=[-0.5, 23.5],
            gridcolor="rgba(200, 200, 200, 0.15)",
        ),
        yaxis=dict(
            title="Predicted Volume (vph)",
            range=[0, max_y],
            gridcolor="rgba(200, 200, 200, 0.15)",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=70, b=40),
        height=400,
        template="plotly_dark",
    )
    return fig


def create_7x24_heatmap_figure(grid_df: pd.DataFrame, thresholds: dict | None = None) -> go.Figure:
    """
    Build Plotly 7x24 hourly heatmap across days of the week.
    """
    if thresholds is None:
        thresholds = load_congestion_thresholds()

    p25 = thresholds.get("percentile_25", 1204.0)
    p50 = thresholds.get("percentile_50", 3339.0)
    p85 = thresholds.get("percentile_85", 5493.0)

    # Custom colorscale mapped to congestion levels
    # Low: green, Moderate: yellow, High: orange, Severe: red
    colorscale = [
        [0.0, "#2ecc71"],
        [p25 / 6500.0, "#2ecc71"],
        [p25 / 6500.0 + 0.001, "#f1c40f"],
        [p50 / 6500.0, "#f1c40f"],
        [p50 / 6500.0 + 0.001, "#e67e22"],
        [p85 / 6500.0, "#e67e22"],
        [p85 / 6500.0 + 0.001, "#e74c3c"],
        [1.0, "#922b21"],
    ]

    fig = go.Figure(data=go.Heatmap(
        z=grid_df.values,
        x=grid_df.columns,
        y=grid_df.index,
        colorscale=colorscale,
        zmin=0,
        zmax=6500,
        colorbar=dict(
            title="Volume (vph)",
            tickvals=[600, 2270, 4416, 5800],
            ticktext=["Low", "Moderate", "High", "Severe"],
        ),
        hovertemplate="Day: <b>%{y}</b><br>Hour: <b>%{x}</b><br>Predicted Volume: <b>%{z:,.0f} vph</b><extra></extra>",
    ))

    fig.update_layout(
        title="<b>Weekly Traffic Congestion Heatmap (7 Days × 24 Hours)</b>",
        xaxis=dict(title="Hour of Day", dtick=2),
        yaxis=dict(title="Day of Week", autorange="reversed"),
        margin=dict(l=60, r=20, t=50, b=40),
        height=320,
        template="plotly_dark",
    )
    return fig

"""
app/streamlit_app.py
--------------------
Interactive Streamlit Dashboard for Traffic Congestion Prediction & What-If Simulation.
Built for Interstate 94 (I-94 Westbound in Minneapolis–St. Paul, Minnesota).

Features 4 Comprehensive Tabs:
1. What-If Scenario Simulator: Interactive scenario explorer with road graphic, 24h diurnal curve, weekly heatmap, and Scenario A vs B comparison.
2. Historical Replay Mode: Step through 2018 test-set days hour-by-hour with 1h-ahead Forecast Model.
3. Model Insights & Benchmarks: Test metrics, sliced regime errors, feature importances, and confusion matrix.
4. About & Operational Limitations: Scope, methodology, volume-vs-speed caveat, and academic citation.
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.features import load_congestion_thresholds
from src.predict import (
    build_scenario_feature_row,
    predict_volume,
    volume_to_level,
)
from src.simulator import (
    CONGESTION_COLORS,
    PRESETS,
    create_24h_plotly_figure,
    create_7x24_heatmap_figure,
    create_road_figure,
    get_24h_curve,
    get_7x24_grid,
    get_baseline_24h_curve,
    get_regime_uncertainty,
    get_scenario_prediction,
    get_scenario_presets,
)

# Page Configuration
st.set_page_config(
    page_title="I-94 Traffic Congestion Simulator",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1e272e;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        border-left: 5px solid #3498db;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 700;
        font-size: 14px;
        color: white;
    }
    .flow-note {
        font-size: 12px;
        color: #bdc3c7;
        font-style: italic;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models_and_metadata():
    """Load pre-trained models, feature column schemas, thresholds, and metadata."""
    models_dir = Path("models")
    with open(models_dir / "forecast_features.json", "r", encoding="utf-8") as f:
        f_cols = json.load(f)
    with open(models_dir / "scenario_features.json", "r", encoding="utf-8") as f:
        s_cols = json.load(f)
    with open(models_dir / "model_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    thresholds = load_congestion_thresholds(models_dir / "congestion_thresholds.json")
    model_f = joblib.load(models_dir / "forecast_model.joblib")
    model_s = joblib.load(models_dir / "scenario_model.joblib")

    return {
        "model_f": model_f,
        "model_s": model_s,
        "f_cols": f_cols,
        "s_cols": s_cols,
        "metadata": metadata,
        "thresholds": thresholds,
    }


@st.cache_data
def load_replay_data():
    """Load cached 2018 test dataset for instantaneous Historical Replay exploration."""
    parquet_path = Path("data/processed/test_2018.parquet")
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    else:
        from src.features import build_full_features
        clean_df = pd.read_csv("data/processed/traffic_clean.csv")
        clean_df["date_time"] = pd.to_datetime(clean_df["date_time"])
        full_df = build_full_features(clean_df, drop_na=True)
        df = full_df[full_df["date_time"] >= "2018-01-01"].copy()

    df["date_time"] = pd.to_datetime(df["date_time"])
    df["date_str"] = df["date_time"].dt.strftime("%Y-%m-%d")
    return df


res = load_models_and_metadata()
thresholds = res["thresholds"]
metadata = res["metadata"]

# App Title & Navigation Header
st.title("🚗 I-94 Traffic Congestion Prediction & What-If Simulator")
st.caption(
    "Westbound Interstate 94 (Minneapolis–St. Paul, MN) | Single Road Segment Machine Learning System"
)

# Main Application Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Tab 1: Scenario Simulator",
    "⏪ Tab 2: Historical Replay (2018)",
    "📊 Tab 3: Model Insights & Benchmarks",
    "ℹ️ Tab 4: About & Limitations",
])


# ==============================================================================
# TAB 1: SCENARIO SIMULATOR (WHAT-IF DASHBOARD)
# ==============================================================================
with tab1:
    st.subheader("Interactive What-If Scenario Exploration")
    st.markdown(
        "Evaluate highway congestion under hypothetical weather conditions, temporal demand profiles, "
        "and traffic shock scenarios using the **Scenario Model** (conditioning on time & weather with zero lag leakage)."
    )

    # Preset selection
    presets = get_scenario_presets()
    preset_names = list(presets.keys())
    selected_preset = st.selectbox("⚡ Quick Scenario Presets:", preset_names, index=1)

    default_vals = presets[selected_preset] if selected_preset != "Custom Scenario" else presets["Weekday AM Rush (Clear)"]

    # Sidebar Scenario Controls
    with st.sidebar:
        st.header("🎛️ Scenario Controls")
        st.caption("Adjust environmental and temporal parameters:")

        c_time1, c_time2 = st.columns(2)
        with c_time1:
            sel_hour = st.slider("Hour of Day (0–23):", 0, 23, default_vals["hour"])
            days_map = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            sel_day_name = st.selectbox("Day of Week:", days_map, index=default_vals["weekday"])
            sel_day = days_map.index(sel_day_name)
        with c_time2:
            months_map = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            sel_month_name = st.selectbox("Month:", months_map, index=default_vals["month"] - 1)
            sel_month = months_map.index(sel_month_name) + 1
            sel_holiday = st.checkbox("Federal / State Holiday?", value=bool(default_vals["is_holiday"]))

        st.markdown("---")
        st.subheader("🌦️ Weather Conditions")
        c_w1, c_w2 = st.columns(2)
        with c_w1:
            sel_temp = st.slider("Temperature (°C):", -35.0, 40.0, float(default_vals["temp_c"]), step=0.5)
            sel_rain = st.slider("Rainfall (mm/h):", 0.0, 50.0, float(default_vals["rain_1h"]), step=0.5)
        with c_w2:
            sel_snow = st.slider("Snowfall (mm/h):", 0.0, 10.0, float(default_vals["snow_1h"]), step=0.2)
            sel_clouds = st.slider("Cloud Coverage (%):", 0, 100, int(default_vals["clouds_all"]), step=5)

        weather_cats = ["Clear", "Clouds", "Rain", "Snow", "Drizzle", "Mist", "Haze", "Fog", "Thunderstorm", "Squall", "Smoke"]
        sel_weather_idx = weather_cats.index(default_vals["weather_main"]) if default_vals["weather_main"] in weather_cats else 0
        sel_weather = st.selectbox("Weather Condition (Macro):", weather_cats, index=sel_weather_idx)

        st.markdown("---")
        st.subheader("📈 Demand Scaling")
        sel_demand = st.slider(
            "Latent Demand Multiplier:",
            0.80, 1.30, float(default_vals.get("demand_mult", 1.0)), step=0.05,
            help="Simulates regional traffic growth or holiday egress surge."
        )
        st.markdown(
            "<div class='flow-note'><b>Flow Saturation Note</b>: The physical capacity of this 4-lane westbound I-94 segment is ~6,000 vph "
            "(99th percentile observed volume is 6,056 vph). Multipliers exceeding 1.0x represent theoretical latent demand; "
            "in practice, physical flow breaks down into queues rather than sustained super-capacity volume.</div>",
            unsafe_allow_html=True,
        )

    # Compute Scenario Prediction
    pred = get_scenario_prediction(
        hour=sel_hour,
        weekday=sel_day,
        month=sel_month,
        temp_c=sel_temp,
        rain_1h=sel_rain,
        snow_1h=sel_snow,
        clouds_all=sel_clouds,
        weather_main=sel_weather,
        is_holiday=int(sel_holiday),
        demand_mult=sel_demand,
    )

    # Top KPI Metrics Display
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Predicted Traffic Volume", f"{pred['volume']:,.0f} vph", delta=f"{pred['volume'] - 3339:,.0f} vs median")
    with kpi2:
        badge_html = f"<div style='background-color:{pred['color']};' class='badge-pill'>{pred['level'].upper()} CONGESTION</div>"
        st.markdown("<b>Congestion Tier:</b>", unsafe_allow_html=True)
        st.markdown(badge_html, unsafe_allow_html=True)
    with kpi3:
        st.metric("Regime Uncertainty (±MAE)", f"±{pred['uncertainty_mae']:.0f} vph", help="Empirical error measured on matching test set regime slice.")
    with kpi4:
        flow_status = "Free Flowing" if pred["level"] == "Low" else ("Normal Commute" if pred["level"] == "Moderate" else ("Heavy Bottleneck" if pred["level"] == "High" else "Breakdown / Queue"))
        st.metric("Operational Status", flow_status)

    # Highway Segment Visual
    st.plotly_chart(create_road_figure(pred["volume"], pred["level"]), use_container_width=True)

    # 24-Hour Diurnal Curve
    curve_df = get_24h_curve(
        weekday=sel_day,
        month=sel_month,
        temp_c=sel_temp,
        rain_1h=sel_rain,
        snow_1h=sel_snow,
        clouds_all=sel_clouds,
        weather_main=sel_weather,
        is_holiday=int(sel_holiday),
        demand_mult=sel_demand,
    )
    base_curve_df = get_baseline_24h_curve(weekday=sel_day, month=sel_month, is_holiday=int(sel_holiday))
    st.plotly_chart(create_24h_plotly_figure(curve_df, base_curve_df, thresholds, current_hour=sel_hour), use_container_width=True)

    # 7x24 Weekly Heatmap
    st.plotly_chart(
        create_7x24_heatmap_figure(
            get_7x24_grid(
                month=sel_month,
                temp_c=sel_temp,
                rain_1h=sel_rain,
                snow_1h=sel_snow,
                clouds_all=sel_clouds,
                weather_main=sel_weather,
                is_holiday=int(sel_holiday),
                demand_mult=sel_demand,
            ),
            thresholds,
        ),
        use_container_width=True,
    )

    # Scenario A vs Scenario B Comparison Expander
    with st.expander("⚖️ Compare Scenario A vs. Scenario B (Diff Analysis)"):
        st.markdown("Compare the active scenario (**Scenario A**) against a counterfactual (**Scenario B**):")
        sb_col1, sb_col2, sb_col3, sb_col4 = st.columns(4)
        with sb_col1:
            sb_weather = st.selectbox("Scenario B Weather:", weather_cats, index=3)  # Snow
        with sb_col2:
            sb_snow = st.slider("Scenario B Snow (mm/h):", 0.0, 10.0, 1.5, step=0.2)
        with sb_col3:
            sb_temp = st.slider("Scenario B Temp (°C):", -35.0, 40.0, -8.0, step=0.5)
        with sb_col4:
            sb_holiday = st.checkbox("Scenario B Holiday?", value=False)

        pred_b = get_scenario_prediction(
            hour=sel_hour,
            weekday=sel_day,
            month=sel_month,
            temp_c=sb_temp,
            rain_1h=0.0,
            snow_1h=sb_snow,
            clouds_all=90,
            weather_main=sb_weather,
            is_holiday=int(sb_holiday),
            demand_mult=1.0,
        )

        diff_vol = pred_b["volume"] - pred["volume"]
        diff_pct = (diff_vol / pred["volume"]) * 100 if pred["volume"] > 0 else 0.0

        cmp1, cmp2, cmp3 = st.columns(3)
        cmp1.metric("Scenario A Volume", f"{pred['volume']:,.0f} vph", f"Level: {pred['level']}")
        cmp2.metric("Scenario B Volume", f"{pred_b['volume']:,.0f} vph", f"Level: {pred_b['level']}")
        cmp3.metric("Volume Impact (B vs. A)", f"{diff_vol:+,.0f} vph", f"{diff_pct:+.1f}%")


# ==============================================================================
# TAB 2: HISTORICAL REPLAY MODE (2018 TEST SET)
# ==============================================================================
with tab2:
    st.subheader("Historical Replay: 1-Hour-Ahead Forecast Model")
    st.markdown(
        "Inspect the performance of the **Production Forecast Model** (which utilizes strictly past lags $T-1..T-168$ and rolling statistics) "
        "on the held-out **2018 Test Set** (Jan 01, 2018 to Sep 30, 2018)."
    )

    replay_df = load_replay_data()
    available_dates = sorted(replay_df["date_str"].unique())

    r_col1, r_col2 = st.columns([1, 3])
    with r_col1:
        default_idx = available_dates.index("2018-05-28") if "2018-05-28" in available_dates else 0
        chosen_date_str = st.selectbox(
            "Select Test Date (2018):",
            available_dates,
            index=default_idx,
            help="Choose any date from the held-out 2018 Test partition."
        )
        day_records = replay_df[replay_df["date_str"] == chosen_date_str].sort_values("date_time").reset_index(drop=True)

        if len(day_records) == 0:
            st.warning("No continuous sensor records available for this selected date.")
            step_hour = 0
        else:
            avail_hours = sorted(day_records["hour"].unique())
            step_hour = st.slider("Hour Stepper (0–23):", 0, 23, int(avail_hours[len(avail_hours) // 2]) if avail_hours else 8)

    with r_col2:
        if len(day_records) > 0:
            # Generate 1-hour-ahead predictions for the day
            day_records["forecast_vol"] = predict_volume(day_records, model_type="forecast")
            day_records["abs_error"] = np.abs(day_records["traffic_volume"] - day_records["forecast_vol"])

            # Plot Forecast vs Actual
            fig_rep = go.Figure()
            fig_rep.add_trace(go.Scatter(
                x=day_records["hour"],
                y=day_records["traffic_volume"],
                mode="lines+markers",
                name="Ground Truth Actual",
                line=dict(color="#ffffff", width=2.5),
                marker=dict(size=6),
            ))
            fig_rep.add_trace(go.Scatter(
                x=day_records["hour"],
                y=day_records["forecast_vol"],
                mode="lines+markers",
                name="1h-Ahead Model Forecast",
                line=dict(color="#3498db", width=2, dash="dash"),
                marker=dict(size=6),
            ))

            # Current Hour Highlight
            if step_hour in day_records["hour"].values:
                step_row = day_records[day_records["hour"] == step_hour].iloc[0]
                fig_rep.add_vline(x=step_hour, line_width=1.5, line_dash="dot", line_color="#f1c40f")
                fig_rep.add_trace(go.Scatter(
                    x=[step_hour],
                    y=[step_row["traffic_volume"]],
                    mode="markers",
                    marker=dict(size=13, color="#e74c3c", line=dict(color="#ffffff", width=2)),
                    name=f"Current Step ({step_hour:02d}:00)",
                ))

            fig_rep.update_layout(
                title=f"<b>Hourly Volume: Forecast vs Actual on {chosen_date_str}</b>",
                xaxis=dict(title="Hour of Day", dtick=1, range=[-0.5, 23.5]),
                yaxis=dict(title="Traffic Volume (vph)", range=[0, 7000]),
                height=340,
                margin=dict(l=30, r=20, t=50, b=30),
                template="plotly_dark",
            )
            st.plotly_chart(fig_rep, use_container_width=True)

    # Step KPIs and Input Features
    if len(day_records) > 0 and step_hour in day_records["hour"].values:
        cur_row = day_records[day_records["hour"] == step_hour].iloc[0]
        actual_val = float(cur_row["traffic_volume"])
        fcst_val = float(cur_row["forecast_vol"])
        err_val = fcst_val - actual_val
        running_mae = float(day_records[day_records["hour"] <= step_hour]["abs_error"].mean())

        sk1, sk2, sk3, sk4 = st.columns(4)
        sk1.metric(f"Hour {step_hour:02d}:00 Actual", f"{actual_val:,.0f} vph", f"Level: {volume_to_level(actual_val)}")
        sk2.metric(f"Hour {step_hour:02d}:00 Forecast", f"{fcst_val:,.0f} vph", f"Level: {volume_to_level(fcst_val)}")
        sk3.metric("Signed Error (Pred - Act)", f"{err_val:+,.0f} vph", f"{abs(err_val) / max(1, actual_val) * 100:.1f}% error")
        sk4.metric("Day Running MAE", f"{running_mae:.1f} vph")

        with st.expander("🔍 Inspect Strict Past Features Driving This Forecast"):
            f_cols_display = {
                "Lag T-1 (Previous Hour)": f"{cur_row.get('lag_1h', np.nan):,.0f} vph",
                "Lag T-2 (2 Hours Prior)": f"{cur_row.get('lag_2h', np.nan):,.0f} vph",
                "Lag T-24 (Same Hour Yesterday)": f"{cur_row.get('lag_24h', np.nan):,.0f} vph",
                "Lag T-168 (Same Hour Last Week)": f"{cur_row.get('lag_168h', np.nan):,.0f} vph",
                "Past 3h Rolling Mean": f"{cur_row.get('rolling_mean_3h', np.nan):,.0f} vph",
                "Temperature": f"{cur_row.get('temp_c', np.nan):.1f} °C",
                "Rainfall (1h)": f"{cur_row.get('rain_1h', np.nan):.1f} mm",
                "Snowfall (1h)": f"{cur_row.get('snow_1h', np.nan):.1f} mm",
                "Holiday Status": "Yes" if cur_row.get("is_holiday", 0) == 1 else "No",
            }
            st.table(pd.DataFrame(list(f_cols_display.items()), columns=["Input Feature", "Value at Time T"]))


# ==============================================================================
# TAB 3: MODEL INSIGHTS & BENCHMARKS
# ==============================================================================
with tab3:
    st.subheader("Model Performance, Baselines & Operational Diagnostics")
    st.markdown(
        "Detailed comparison of the **Forecast Model**, **Scenario Model**, **Baseline A (Refit)**, "
        "and **Baseline D** on the held-out 2018 Test Set."
    )

    # Overall Test Set Comparison Table
    st.markdown("#### 1. Overall 2018 Test Set Performance")
    comp_data = {
        "Model / Benchmark": [
            "Forecast Model (LightGBM)",
            "Scenario Model (LightGBM)",
            "Baseline A (Refit on Train+Val)",
            "Baseline D (Day_Type, Hour, Month)",
        ],
        "Input Feature Scope": [
            "All Features (Lags T-1..T-168 + Rolling + Weather)",
            "No Lags (Time, Weather, Holiday, Day-Type)",
            "Historical Mean by (Day_of_Week, Hour)",
            "Historical Mean by (Day_Type, Hour, Month)",
        ],
        "Test MAE (vph)": [
            metadata["forecast_model"]["test_metrics"]["MAE"],
            metadata["scenario_model"]["test_metrics"]["MAE"],
            metadata["baseline_a_refit"]["test_metrics"]["MAE"],
            metadata["baseline_d"]["test_metrics"]["MAE"],
        ],
        "Test RMSE (vph)": [
            metadata["forecast_model"]["test_metrics"]["RMSE"],
            metadata["scenario_model"]["test_metrics"]["RMSE"],
            metadata["baseline_a_refit"]["test_metrics"]["RMSE"],
            metadata["baseline_d"]["test_metrics"]["RMSE"],
        ],
        "R² Score": [
            metadata["forecast_model"]["test_metrics"]["R2"],
            metadata["scenario_model"]["test_metrics"]["R2"],
            metadata["baseline_a_refit"]["test_metrics"]["R2"],
            metadata["baseline_d"]["test_metrics"]["R2"],
        ],
        "4-Class Accuracy (%)": [
            f"{metadata['forecast_model']['test_metrics']['Accuracy']:.2f}%",
            f"{metadata['scenario_model']['test_metrics']['Accuracy']:.2f}%",
            f"{metadata['baseline_a_refit']['test_metrics']['Accuracy']:.2f}%",
            f"{metadata['baseline_d']['test_metrics']['Accuracy']:.2f}%",
        ],
        "Within 1 Class (%)": [
            f"{metadata['forecast_model']['test_metrics']['Within_1_Class']:.2f}%",
            f"{metadata['scenario_model']['test_metrics']['Within_1_Class']:.2f}%",
            f"{metadata['baseline_a_refit']['test_metrics']['Within_1_Class']:.2f}%",
            f"{metadata['baseline_d']['test_metrics']['Within_1_Class']:.2f}%",
        ],
    }
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

    # Sliced MAE Table
    st.markdown("#### 2. Sliced MAE Across Operational Traffic Regimes (vph)")
    fcst_slices = metadata["forecast_model"]["test_sliced_mae"]
    scen_slices = metadata["scenario_model"]["test_sliced_mae"]
    base_a_slices = metadata["baseline_a_refit"]["test_sliced_mae"]
    base_d_slices = metadata["baseline_d"]["test_sliced_mae"]

    slice_names = [
        ("Weekday Morning Rush (07:00–08:00)", "Weekday_AM_Rush (07-08)"),
        ("Weekday Evening Rush (16:00–17:00)", "Weekday_PM_Rush (16-17)"),
        ("Weekends (Saturday–Sunday)", "Weekends"),
        ("Holidays", "Holidays"),
        ("Snow Hours", "Snow_Hours"),
        ("Rain Hours", "Rain_Hours"),
        ("Overall Test Set", "Overall"),
    ]

    sliced_rows = []
    for disp_name, key in slice_names:
        sliced_rows.append({
            "Traffic Regime Slice": disp_name,
            "Forecast Model": f"{fcst_slices[key]:.1f} vph",
            "Scenario Model": f"{scen_slices[key]:.1f} vph",
            "Baseline D": f"{base_d_slices[key]:.1f} vph",
            "Baseline A (Refit)": f"{base_a_slices[key]:.1f} vph",
            "Forecast Gain vs. Base D": f"{(base_d_slices[key] - fcst_slices[key]) / base_d_slices[key] * 100:+.1f}%",
        })
    st.dataframe(pd.DataFrame(sliced_rows), use_container_width=True, hide_index=True)

    # Feature Importance & Confusion Matrix
    c_fi, c_cm = st.columns([1, 1])
    with c_fi:
        st.markdown("#### 3. Top Feature Importances (Scenario Model)")
        model_s = res["model_s"]
        s_cols = res["s_cols"]
        fi_s = pd.Series(model_s.feature_importances_, index=s_cols).sort_values(ascending=False).head(10)
        fig_fi = px.bar(
            x=fi_s.values,
            y=fi_s.index,
            orientation="h",
            labels={"x": "Importance (Split Gain)", "y": "Feature"},
            color=fi_s.values,
            color_continuous_scale="Blues",
        )
        fig_fi.update_layout(yaxis=dict(autorange="reversed"), height=330, margin=dict(l=20, r=20, t=20, b=20), template="plotly_dark")
        st.plotly_chart(fig_fi, use_container_width=True)

    with c_cm:
        st.markdown("#### 4. Congestion Confusion Matrix (Forecast Model)")
        cm = metadata["forecast_model"]["test_metrics"]["Confusion_Matrix"]
        labels = ["Low", "Moderate", "High", "Severe"]
        fig_cm = px.imshow(
            cm,
            x=[f"Pred {lbl}" for lbl in labels],
            y=[f"True {lbl}" for lbl in labels],
            color_continuous_scale="Viridis",
            text_auto=True,
        )
        fig_cm.update_layout(height=330, margin=dict(l=20, r=20, t=20, b=20), template="plotly_dark")
        st.plotly_chart(fig_cm, use_container_width=True)


# ==============================================================================
# TAB 4: ABOUT & LIMITATIONS
# ==============================================================================
with tab4:
    st.subheader("About This System & Operational Limitations")
    st.markdown(
        r"""
        ### 1. Geographical & Corridor Scope
        - **Facility**: Interstate 94 (I-94) Westbound in Minneapolis–St. Paul, Minnesota.
        - **Source**: UCI Machine Learning Repository (Dataset ID #492).
        - **Instrumentation**: Automatic loop detectors maintained by the Minnesota Department of Transportation (MnDOT), recording hourly traffic counts alongside meteorological records from Minneapolis–St. Paul International Airport (KMSP).
        - **Single Road Segment Scope**: All models and simulations apply strictly to this specific corridor segment and cannot be transferred without retraining to other highway segments.

        ---

        ### 2. Methodological Distinctions: Forecast Model vs. Scenario Model
        - **Forecast Model (Operational Forecaster)**:
          - Trained with strictly past lags ($T-1$, $T-2$, $T-3$, $T-24$, $T-168$) and rolling statistics computed prior to $T$.
          - Delivers superior 1-hour-ahead predictive accuracy (**Test MAE: 139.65 vph**, 93.69% accuracy).
          - Evaluated on observed weather at hour $T$; in live deployment, weather inputs would be supplied by short-term numerical weather prediction (NWP) feeds.
        - **Scenario Model (What-If Simulator)**:
          - Trained **without any lag or rolling features**, using only calendar time, weather conditions, holiday status, and day-type indicators.
          - Enables users to explore arbitrary counterfactual conditions (e.g. blizzard shocks, demand scaling) without requiring fabricated past volume series.

        ---

        ### 3. Critical Operational Limitations & Caveats

        > [!WARNING]
        > **Volume Percentiles ≠ Vehicular Speed or Travel Delay**:
        > This dataset records exclusively **traffic volume (vehicles per hour)**. It contains **no speed measurements, travel time index (TTI), or queue delay data**. While top-tier volume ($\ge 5,493$ vph) indicates capacity saturation, an acute drop in volume can indicate either free-flow dead-of-night travel ($< 1,200$ vph) OR a severe jam where vehicles are stopped over the loop detectors.

        > [!CAUTION]
        > **Predictive ML Correlations ≠ Causal Infrastructure Evaluations**:
        > This simulator captures historical statistical associations. It **cannot simulate infrastructure modifications** such as adding lanes, constructing flyovers, or altering interchange geometry. In real-world transportation engineering, adding lanes induces additional travel demand and alters network equilibrium, which requires causal macroscopic or microscopic dynamic simulation models (e.g. SUMO, VISSIM).

        ---

        ### 4. Academic Citation & Data Provenance
        - **Dataset**: Hogue, K. (2019). *Metro Interstate Traffic Volume*. UCI Machine Learning Repository. [https://doi.org/10.24432/C56S34](https://doi.org/10.24432/C56S34).
        - **Licensing**: Creative Commons Attribution 4.0 International (CC BY 4.0).
        """
    )

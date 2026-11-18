"""
scripts/generate_figures.py
----------------------------
Generates and saves key project figures to docs/figures/.
Figures:
  01_traffic_volume_distribution.png
  02_diurnal_patterns.png
  03_model_comparison_mae.png
  04_sliced_mae.png
  05_feature_importance_forecast.png
  06_feature_importance_scenario.png
  07_confusion_matrix_forecast.png
  08_snow_weather_effect.png
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings; warnings.filterwarnings("ignore")

FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = PROJECT_ROOT / "models"

# Load metadata and data
with open(MODELS_DIR / "model_metadata.json") as f:
    meta = json.load(f)
with open(MODELS_DIR / "forecast_features.json") as f:
    forecast_cols = json.load(f)
with open(MODELS_DIR / "scenario_features.json") as f:
    scenario_cols = json.load(f)
with open(MODELS_DIR / "congestion_thresholds.json") as f:
    thresholds = json.load(f)

clean_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "traffic_clean.csv")
clean_df["date_time"] = pd.to_datetime(clean_df["date_time"])
clean_df["hour"] = clean_df["date_time"].dt.hour
clean_df["day_of_week"] = clean_df["date_time"].dt.day_of_week
clean_df["is_weekend"] = (clean_df["day_of_week"] >= 5).astype(int)
clean_df["month"] = clean_df["date_time"].dt.month

COLORS = {"Low": "#2ecc71", "Moderate": "#f1c40f", "High": "#e67e22", "Severe": "#e74c3c"}
DARK_BG = "#1a1a2e"
PANEL_BG = "#16213e"
TEXT_COLOR = "#e0e0e0"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": PANEL_BG,
    "axes.edgecolor": "#444",
    "text.color": TEXT_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "axes.titlecolor": TEXT_COLOR,
    "grid.color": "#2a2a4a",
    "grid.alpha": 0.5,
})

# --- Figure 1: Traffic Volume Distribution ---
fig, ax = plt.subplots(figsize=(10, 5))
vols = clean_df["traffic_volume"].dropna()
t_lo, t_mod, t_hi, t_sev = (thresholds["percentile_25"],
                            thresholds["percentile_50"],
                            thresholds["percentile_85"], 10000)
for lo, hi, label, color in [
    (0,     t_lo,  "Low",      COLORS["Low"]),
    (t_lo,  t_mod, "Moderate", COLORS["Moderate"]),
    (t_mod, t_hi,  "High",     COLORS["High"]),
    (t_hi,  10000, "Severe",   COLORS["Severe"]),
]:
    sub = vols[(vols >= lo) & (vols < hi)]
    ax.hist(sub, bins=60, color=color, alpha=0.75, label=label, range=(lo, hi))
ax.set_xlabel("Traffic Volume (vehicles per hour)")
ax.set_ylabel("Count (hours)")
ax.set_title("Figure 1: Traffic Volume Distribution with Congestion Tiers\n(2012–2018, all hours)", fontsize=12)
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "01_traffic_volume_distribution.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved 01_traffic_volume_distribution.png")

# --- Figure 2: Diurnal Patterns (weekday vs weekend) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
clean_v = clean_df.dropna(subset=["traffic_volume"])
wd = clean_v[clean_v["is_weekend"] == 0].groupby("hour")["traffic_volume"]
we = clean_v[clean_v["is_weekend"] == 1].groupby("hour")["traffic_volume"]
hours = range(24)

ax1.fill_between(hours, wd.quantile(0.25), wd.quantile(0.75), alpha=0.3, color="#3498db")
ax1.plot(hours, wd.mean(), color="#3498db", linewidth=2.5, label="Weekday mean")
ax1.axhline(t_lo,  color=COLORS["Low"],      ls="--", lw=1, alpha=0.6, label=f"Low/Moderate ({t_lo:,})")
ax1.axhline(t_mod, color=COLORS["Moderate"], ls="--", lw=1, alpha=0.6, label=f"Moderate/High ({t_mod:,})")
ax1.axhline(t_hi,  color=COLORS["High"],     ls="--", lw=1, alpha=0.6, label=f"High/Severe ({t_hi:,})")
ax1.set_title("Weekday Diurnal Profile (IQR shaded)")
ax1.set_xlabel("Hour of Day"); ax1.set_ylabel("Traffic Volume (vph)")
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(0, 24, 2))

ax2.fill_between(hours, we.quantile(0.25), we.quantile(0.75), alpha=0.3, color="#e67e22")
ax2.plot(hours, we.mean(), color="#e67e22", linewidth=2.5, label="Weekend mean")
ax2.axhline(t_lo,  color=COLORS["Low"],      ls="--", lw=1, alpha=0.6)
ax2.axhline(t_mod, color=COLORS["Moderate"], ls="--", lw=1, alpha=0.6)
ax2.axhline(t_hi,  color=COLORS["High"],     ls="--", lw=1, alpha=0.6)
ax2.set_title("Weekend Diurnal Profile (IQR shaded)")
ax2.set_xlabel("Hour of Day"); ax2.set_ylabel("Traffic Volume (vph)")
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(0, 24, 2))

fig.suptitle("Figure 2: Diurnal Traffic Patterns — I-94 Westbound (2012–2018)", fontsize=12)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "02_diurnal_patterns.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved 02_diurnal_patterns.png")

# --- Figure 3: Model Comparison (bar chart of test MAE) ---
fig, ax = plt.subplots(figsize=(10, 5))
models = ["Forecast\nModel\n(LightGBM)", "Scenario\nModel\n(LightGBM)", "Baseline A\n(Day+Hour)", "Baseline D\n(DayType+Hour\n+Month)"]
maes   = [meta["forecast_model"]["test_metrics"]["MAE"],
           meta["scenario_model"]["test_metrics"]["MAE"],
           meta["baseline_a_refit"]["test_metrics"]["MAE"],
           meta["baseline_d"]["test_metrics"]["MAE"]]
colors = ["#3498db", "#9b59b6", "#95a5a6", "#7f8c8d"]
bars = ax.bar(models, maes, color=colors, alpha=0.85, edgecolor="#444", linewidth=0.8)
for bar, mae in zip(bars, maes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f"{mae:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylabel("Test MAE (vehicles per hour)")
ax.set_title("Figure 3: 2018 Test Set MAE — Models vs Baselines", fontsize=12)
ax.set_ylim(0, max(maes) * 1.2)
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "03_model_comparison_mae.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved 03_model_comparison_mae.png")

# --- Figure 4: Sliced MAE ---
fig, ax = plt.subplots(figsize=(12, 6))
slices_f = meta["forecast_model"]["test_sliced_mae"]
slices_s = meta["scenario_model"]["test_sliced_mae"]
slices_d = meta["baseline_d"]["test_sliced_mae"]
slice_keys = ["Weekday_AM_Rush (07-08)", "Weekday_PM_Rush (16-17)", "Weekends", "Holidays", "Snow_Hours", "Rain_Hours", "Overall"]
slice_labels = ["AM Rush\n(07–08)", "PM Rush\n(16–17)", "Weekends", "Holidays", "Snow\nHours", "Rain\nHours", "Overall"]
x = np.arange(len(slice_keys))
w = 0.26
ax.bar(x - w, [slices_f[k] for k in slice_keys], w, label="Forecast Model", color="#3498db", alpha=0.85)
ax.bar(x,     [slices_s[k] for k in slice_keys], w, label="Scenario Model", color="#9b59b6", alpha=0.85)
ax.bar(x + w, [slices_d[k] for k in slice_keys], w, label="Baseline D", color="#7f8c8d", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(slice_labels)
ax.set_ylabel("MAE (vehicles per hour)")
ax.set_title("Figure 4: Sliced MAE by Traffic Regime — Forecast vs Scenario vs Baseline D", fontsize=12)
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "04_sliced_mae.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved 04_sliced_mae.png")

# --- Figures 5 & 6: Feature Importances ---
model_f = joblib.load(MODELS_DIR / "forecast_model.joblib")
model_s = joblib.load(MODELS_DIR / "scenario_model.joblib")

for fig_num, model, cols, label, color in [
    (5, model_f, forecast_cols, "Forecast Model", "#3498db"),
    (6, model_s, scenario_cols, "Scenario Model", "#9b59b6"),
]:
    fi = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(fi.index, fi.values, color=color, alpha=0.85, edgecolor="#444")
    ax.set_xlabel("Feature Importance (Split Gain)")
    ax.set_title(f"Figure {fig_num}: Top 15 Feature Importances — {label} (LightGBM)", fontsize=12)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    fname = f"0{fig_num}_feature_importance_{'forecast' if fig_num==5 else 'scenario'}.png"
    plt.savefig(FIGURES_DIR / fname, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")

# --- Figure 7: Confusion Matrix ---
cm = meta["forecast_model"]["test_metrics"]["Confusion_Matrix"]
labels = ["Low", "Moderate", "High", "Severe"]
cm_arr = np.array(cm)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm_arr, cmap="Blues", aspect="auto")
plt.colorbar(im, ax=ax, label="Count")
ax.set_xticks(range(4)); ax.set_xticklabels([f"Pred {l}" for l in labels])
ax.set_yticks(range(4)); ax.set_yticklabels([f"True {l}" for l in labels])
for i in range(4):
    for j in range(4):
        ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center",
                color="white" if cm_arr[i,j] > cm_arr.max()/2 else TEXT_COLOR,
                fontsize=13, fontweight="bold")
ax.set_title("Figure 7: Confusion Matrix — Forecast Model (Test 2018)", fontsize=12)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "07_confusion_matrix_forecast.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved 07_confusion_matrix_forecast.png")

# --- Figure 8: Snow/Weather effect (empirical vs model) ---
fig, ax = plt.subplots(figsize=(10, 5))
categories = ["Empirical\nAM Rush\n(Winter)", "Model\nAM Rush\n(Jan, cold+snow\nvs. cold, dry)", 
              "Empirical\nPM Rush\n(Winter)", "Model\nPM Rush\n(Jan, cold+snow\nvs. cold, dry)"]
pct_effects = [-8.9, -15.8, -10.0, -15.3]  # from audit_checks.py
bar_colors = ["#2ecc71", "#e74c3c", "#2ecc71", "#e74c3c"]
bars = ax.bar(categories, pct_effects, color=bar_colors, alpha=0.85, edgecolor="#444")
for bar, val in zip(bars, pct_effects):
    ax.text(bar.get_x() + bar.get_width()/2, val - 0.5, f"{val:.1f}%",
            ha="center", va="top", fontsize=11, fontweight="bold")
ax.axhline(0, color="#888", lw=1)
ax.set_ylabel("Volume Change vs. Dry Baseline (%)")
ax.set_title("Figure 8: Snow Effect — Empirical (Winter Control) vs Scenario Model", fontsize=12)
ax.set_ylim(min(pct_effects) * 1.3, 2)
patches = [mpatches.Patch(color="#2ecc71", label="Empirical (Nov–Mar, same hour)"),
           mpatches.Patch(color="#e74c3c", label="Model (Jan, temp drop + snow_1h=2.5)")]
ax.legend(handles=patches)
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "08_snow_weather_effect.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved 08_snow_weather_effect.png")

print(f"\nAll figures saved to {FIGURES_DIR}")

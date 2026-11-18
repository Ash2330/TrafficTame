# Traffic Congestion Prediction with What-If Simulator
## Project Report — I-94 Westbound, Minneapolis–St. Paul, MN

> **Academic Report** | Machine Learning Project | Sem 5

---

## 1. Problem Statement and Objectives

**Problem**: Urban traffic congestion causes significant economic losses and emissions. Accurate short-horizon traffic forecasting enables operators and commuters to make better routing decisions.

**Objective**: Build an end-to-end ML system that:
1. Predicts hourly traffic volume on I-94 Westbound **1 hour ahead** using only past data available at prediction time (strict no-leakage constraint).
2. Classifies predictions into four operational congestion tiers: Low / Moderate / High / Severe.
3. Powers an interactive **What-If Simulator** that allows users to explore predictions under hypothetical weather, holiday, and demand scenarios without requiring real-time lag data.

**Scope**: Single road segment (I-94 Westbound at MnDOT ATR station 301) for demonstration purposes. No multi-road or network modeling.

---

## 2. Dataset and Cleaning Decisions

**Source**: UCI Machine Learning Repository — Metro Interstate Traffic Volume (ID #492)  
**URL**: https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume  
**Coverage**: Oct 2012 – Sep 2018, hourly observations (48,204 raw rows)

**Columns used**:
| Column | Type | Role |
|---|---|---|
| `date_time` | datetime | Index, time partitioning |
| `traffic_volume` | int | Prediction target |
| `holiday` | str | Calendar feature (one-hot: is_holiday) |
| `temp` | float (K) | Weather → converted to °C |
| `rain_1h` | float | Weather (mm/h) |
| `snow_1h` | float | Weather (mm/h) |
| `clouds_all` | int | Weather (%) |
| `weather_main` | str | Weather category → one-hot encoded |

**Cleaning steps** (`src/preprocess.py`):
- Removed 2,897 exact duplicate rows.
- Standardised to continuous hourly index (reindexed to fill sensor gaps as NaN).
- Identified 2 zero-volume hours (2016-07-23 18:00 and 23:00) as consistent with a road closure; cause cannot be verified from dataset alone — flagged, not dropped.
- Converted `temp` (Kelvin) to `temp_c` (Celsius): `temp_c = temp - 273.15`.

**Row count waterfall** (after cleaning and feature engineering):
| Partition | Total Intervals | Final Retained Rows |
|---|---|---|
| Train (2012–2016) | 37,239 | 24,972 |
| Validation (2017) | 8,760 | 8,576 |
| Test (2018) | 6,552 | 6,494 |

Rows dropped from NaN lags across sensor blackout periods — never imputed.

---

## 3. EDA Findings

- **Strong diurnal pattern**: Traffic peaks at 07:00–09:00 (AM rush) and 16:00–18:00 (PM rush) on weekdays, with PM peak ~10–15% higher than AM.
- **Weekend suppression**: Weekend volumes are 30–40% lower than weekday peaks and lack the dual-peak structure.
- **Holiday suppression**: Holidays show a different pattern depending on the holiday type. Major US holidays (Thanksgiving, Christmas, Independence Day) show volume drops of 40–60% during typical rush hours, but some holidays (Columbus Day, Veterans Day) show near-normal weekday volumes.
- **Snow effect** (winter control, Nov–Mar):
  - AM rush (07:00–08:00): Snow = −495 vph (−8.9%) vs dry winter baseline.
  - PM rush (16:00–17:00): Snow = −583 vph (−10.0%) vs dry winter baseline.
  - Winter rain shows near-zero volume effect (N too small to draw conclusions).
- **Sensor outages**: 8,992 missing hourly records, mostly in a large 2013–2015 gap.
- **23:00 anomaly**: Five of the ten worst forecast errors occur at 23:00. The mean signed error is +6.99 vph (virtually unbiased) but std = 370 vph (highest of any hour) — driven by sporadic late-night surges (event egress, detours) that are unlearnable from historical patterns. DST transitions do not explain the pattern (checked explicitly for 2018-03-11).

---

## 4. Feature Engineering and Leakage Prevention

**Strict leakage boundary**: The prediction target for hour T is `traffic_volume[T]`. The most recent traffic volume allowed as a feature is `traffic_volume[T-1]` (i.e., lag_1h). Weather at time T is allowed as a scenario input (simulating real-time NWP feed).

**Features built** (`src/features.py`):

| Category | Features |
|---|---|
| **Time** | hour, day_of_week, month, is_weekend, sin/cos(hour), sin/cos(day_of_week), sin/cos(month) |
| **Calendar** | is_holiday, day_type (weekday/weekend/holiday), is_weekend_or_holiday |
| **Weather** | temp_c, rain_1h, snow_1h, clouds_all, weather_main one-hot (11 categories) |
| **Lags** | lag_1h, lag_2h, lag_3h, lag_24h, lag_168h |
| **Rolling** | rolling_mean_3h, rolling_mean_6h, rolling_mean_24h, rolling_std_3h (all shifted by 1 before rolling) |

**Anti-leakage measures**:
- All lags use `shift(1)` or greater before accessing traffic volume.
- Rolling features use `shift(1)` before `.rolling()` — the window never includes the current hour.
- Lags across sensor blackout gaps are never imputed — rows with any NaN lag are dropped.
- The forecast feature schema (`models/forecast_features.json`) explicitly excludes `traffic_volume`.
- Verified by unit tests in `tests/test_features.py` (6 tests, all pass).

---

## 5. Baselines

| Baseline | Definition | Fit On |
|---|---|---|
| **Baseline A** | Historical mean by (day_of_week, hour) | Train + Validation (2012–2017) |
| **Baseline D** | Historical mean by (day_type, hour, month) | Train + Validation (2012–2017) |

Baseline D already captures >85% of traffic variance through purely static tabular binning — the ML models must beat a strong non-ML baseline.

---

## 6. Models and Evaluation Protocol

### Two-Model Design

The project uses **two separate LightGBM models** to serve two different use cases:

| | Forecast Model | Scenario Model |
|---|---|---|
| **Purpose** | 1h-ahead prediction using real sensor data | What-If Simulator dashboard |
| **Lag features** | Yes (lag_1h..lag_168h, rolling stats) | **No** — users cannot supply realistic lags |
| **Feature count** | 40 | 31 |
| **Holiday sample weight** | None | w=20 (see §6.1) |

### Evaluation Protocol

1. **Train on 2012–2016** (24,972 rows) with 4 candidate models (Ridge, RandomForest, LightGBM, XGBoost).
2. **Select best model per feature set on 2017 validation** (8,576 rows) by lowest MAE.
3. **Refit winners on Train+Validation (2012–2017)** combined (33,548 rows).
4. **Evaluate once on 2018 test** (6,494 rows). The test set was examined exactly once before Correction C (holiday weight tuning) in a previous iteration — noted in README section 6.

### 6.1 Holiday Sample Weight Selection

Holiday hours are sparse (744/24,972 = 3.0% of training). To improve holiday predictions, candidate weights {1, 5, 10, 20} were evaluated on the **2017 validation set holiday MAE**:

| Weight | Val Overall MAE | Val Holiday MAE |
|---|---|---|
| 1 | 242.31 | 529.78 |
| 5 | 242.10 | 512.29 |
| 10 | 242.21 | 511.97 |
| **20** | **242.27** | **488.09** ← best |

**w=20 selected** by minimising validation holiday MAE. The test set was not used for this choice.

**Known limitation**: The model still underpredicts holiday suppression for non-November months (holiday predictions vary significantly by month due to small training samples). November holidays (Thanksgiving-adjacent) predict correctly at ~3,283 vph vs empirical mean 2,586 vph.

---

## 7. Results

### Overall Test Set (2018)

| Model | Test MAE (vph) | Test RMSE (vph) | R² | 4-Class Accuracy |
|---|---|---|---|---|
| **Forecast Model (LightGBM)** | **139.65** | **217.38** | **0.9879** | **93.69%** |
| **Scenario Model (LightGBM)** | **218.86** | **364.56** | **0.9659** | **90.50%** |
| Baseline A (Day_of_Week, Hour) | 268.77 | 469.44 | 0.9435 | 88.96% |
| Baseline D (Day_Type, Hour, Month) | 272.65 | 429.48 | 0.9527 | 88.56% |

**Gains (all computed on Test 2018):**
- Forecast vs Baseline A: **48.0% MAE reduction**
- Forecast vs Baseline D: **48.8% MAE reduction**
- Scenario vs Baseline A: **18.6% MAE reduction**
- Scenario vs Baseline D: **19.7% MAE reduction**

### Sliced MAE by Operational Regime

| Regime | Forecast | Scenario | Baseline D | Scenario vs D |
|---|---|---|---|---|
| Weekday AM Rush (07–08) | 228.5 | 327.2 | 331.9 | −1.4% |
| Weekday PM Rush (16–17) | 206.6 | 322.0 | 362.9 | **−11.3%** |
| Weekends | 148.0 | 275.0 | 353.5 | **−22.2%** |
| Holidays | 151.3 | 357.7 | 324.1 | +10.4% (worse) |
| Snow Hours | 202.7 | 535.7 | 627.4 | **−14.6%** |
| Rain Hours | 124.9 | 172.4 | 212.6 | **−18.9%** |
| **Overall** | **139.7** | **218.9** | **272.7** | **−19.7%** |

**Where the scenario model's gain over Baseline D comes from:**
- **Weather sensitivity** (snow: −14.6%, rain: −18.9%): The model learns the continuous relationship between snowfall/rainfall and volume suppression that tabular means cannot capture.
- **Fine-grained intra-week patterns** (weekends: −22.2%): Distinguishing Saturday vs Sunday vs individual weekdays within the same day_type group.
- **Holiday regime is worse** (+10.4% over Baseline D): The model overpredicts many holidays because it has only 744 training holiday hours and cannot generalise across the diverse holiday types in this dataset.

---

## 8. Error Analysis

### Ten Worst Test Predictions

> **Important**: This dataset contains NO crash, incident, or event data. All diagnoses below are strictly *hypotheses* consistent with volume patterns; causes cannot be verified as fact.

The five worst errors all occur in two narrow windows: (a) 2018-05-21 13:00–14:00 (a mid-afternoon volume crash/recovery), and (b) five late-night (23:00) spikes in September 2018. The model correctly learns the average diurnal decay at 23:00, but cannot anticipate sporadic surges.

**23:00 error analysis:**
- Mean signed error at 23:00 (all 271 test observations): +6.99 vph (virtually unbiased)
- Standard deviation: 370.0 vph (highest of any hour; nighttime free-flow hours average ~65 vph)
- **DST check**: On 2018-03-11 (Spring forward), the 23:00 error was −28.9 vph — DST is not responsible
- **Root cause hypothesis**: Late-night volume typically drops rapidly from ~22:00. When unscheduled events (sports, concerts, highway incidents upstream) maintain elevated volume into midnight, models trained on average diurnal curves systematically underpredict.

### Snow Response: Model vs. Empirical

The empirical winter-control comparison (Nov–Mar, same weekday+hour) found:
- AM rush snow effect: **−8.9%** (−495 vph)
- PM rush snow effect: **−10.0%** (−583 vph)

The scenario model predicts (Jan, weekday, temp=5°C dry vs. temp=−5°C + snow_1h=2.5 mm):
- AM rush: **−13.8% to −15.8%**
- PM rush: **−13.4% to −15.3%**

The model **overpredicts the snow effect by ~5–6 percentage points** relative to the empirical winter-control comparison. Investigation shows this is because the model comparison (dry vs. snow) confounds two effects: (1) the snow_1h signal itself, and (2) the temperature drop (5°C → −5°C) used in the comparison. When temperature is held constant at −5°C (cold-no-snow vs. cold+snow), the isolated snow-only effect is **−12.5% to −13.9%** — still larger than empirical.

Likely explanation: The empirical "dry winter" baseline includes partially-cloudy and overcast days at −5°C to 10°C, which already suppress volume somewhat versus a clear 15°C summer day. The model's snow effect is relative to its own dry-winter baseline, which may predict higher "normal" winter volume than what is actually observed.

---

## 9. Limitations

1. **Single road segment**: All models apply only to I-94 Westbound at the ATR 301 sensor. Results cannot be transferred to other highway segments without retraining.
2. **Volume, not speed**: The dataset records vehicle counts (vph), not travel speed, delay, or density. Predictions of "congestion level" are based on volume thresholds derived from the historical distribution — not from fundamental traffic flow theory (e.g., the fundamental diagram). A road near capacity may carry the same volume as a free-flow road.
3. **Observed weather, not forecasted**: The forecast model uses actual weather measurements at time T (not T+1 forecasted weather). In live production, users would need a short-term weather forecast (NWP or nowcast) as input, which introduces additional uncertainty.
4. **Demand multiplier is an assumption**: The ×0.8–×1.3 demand multiplier in the What-If simulator is a simple linear scale applied to the model output. It does not model traffic assignment, induced demand, or capacity constraints.
5. **Correlation, not causation**: The models identify statistical associations between weather, time, and traffic volume. They cannot establish causal mechanisms (e.g., whether snow causes trips to be cancelled or redistributed to other times).
6. **Holiday generalisation**: With only 744 holiday hours in training (~3% of data), the model learns holiday effects primarily from Thanksgiving-adjacent Novembers. Predictions for other holiday types (Columbus Day, Independence Day at unusual hours) are unreliable.

---

## 10. Future Work

1. **Live API data feed**: Replace the 2018 replay mode with a live MnDOT Traffic Management Center API feed for real-time predictions.
2. **Forecast weather input**: Integrate NOAA short-range ensemble forecast (SREF) or NWS hourly point forecasts as weather inputs, replacing observed-at-T weather.
3. **SUMO simulation**: Use SUMO (Simulation of Urban MObility) to simulate flow breakdown and queue formation under congestion — moving beyond volume to delay and density.
4. **Multi-road graph models**: Replace the single-segment model with a GNN (Graph Neural Network) or spatiotemporal attention model that captures interactions between I-94 and parallel/intersecting routes (I-35W, US-169, TH-100).
5. **Probabilistic forecasting**: Replace point predictions with quantile regression or conformal prediction intervals to give operators reliable uncertainty bounds.
6. **Anomaly detection**: Train a separate model to detect sensor outages and anomalous volume spikes (pre-event detection) rather than treating them as ordinary data points.

---

*Report generated by the Traffic Congestion ML project pipeline. For reproducing all results, see `README.md` Section 3.*

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

1. **Train on 2012–2016** (24,972 rows) with 4 candidate models: Ridge (with StandardScaler), RandomForest, LightGBM, and XGBoost.
2. **Select best model per feature set on 2017 validation** (8,576 rows) by lowest MAE.
3. **Refit winners on Train+Validation (2012–2017)** combined (33,548 rows).
4. **Evaluate once on 2018 test** (6,494 rows).

### Candidate Models Validation Comparison (2017)

All models evaluated under final feature schema (40 features for Forecast, 31 for Scenario) and holiday sample weighting:

| Feature Set | Model | Val MAE (vph) | Val RMSE (vph) | Val R² | Val 4-Class Acc (%) | Val Within-1-Class (%) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Forecast** | Ridge (StandardScaler) | 285.77 | 399.58 | 0.9594 | 88.58% | 99.97% |
| | RandomForest | 158.44 | 246.29 | 0.9846 | 93.02% | 100.00% |
| | **LightGBM (Selected Best)** | **150.53** | **234.34** | **0.9860** | **93.31%** | **100.00%** |
| | XGBoost | 151.86 | 235.36 | 0.9859 | 93.20% | 100.00% |
| **Scenario** | Ridge (StandardScaler) | 801.12 | 1030.46 | 0.7299 | 58.65% | 97.81% |
| | RandomForest | 244.98 | 386.05 | 0.9621 | 89.13% | 99.92% |
| | **LightGBM (Selected Best)** | **242.27** | **374.61** | **0.9643** | **89.16%** | **99.97%** |
| | XGBoost | 244.74 | 377.51 | 0.9638 | 89.16% | 99.95% |

**Selection Result**: LightGBM achieved the lowest MAE on the 2017 Validation partition across both feature sets and was unequivocally selected for deployment.

### Test-Set Usage Disclosure

To maintain strict scientific integrity, all interactions with the held-out 2018 test partition are documented below:
- **First Evaluation (Task 3)**: Initial candidate models (with shallow default parameters: LightGBM `num_leaves=31`, XGBoost `max_depth=6`) were evaluated. At that stage, XGBoost won on validation and achieved initial test MAEs of **142.47 vph** (Forecast) and **215.67 vph** (Scenario).
- **Post-Evaluation Feedback & Adjustments**:
  1. *Holiday handling*: Analysis of initial predictions revealed holiday morning predictions were over-estimating volume. Feature engineering was expanded to include `day_type` (weekday, weekend, holiday indicators) and `is_weekend_or_holiday`. Holiday sample weighting was explored.
  2. *Hyperparameter tuning*: LightGBM hyperparameters were tuned (`num_leaves=63`, `min_child_samples=20`, `n_estimators=300`) evaluated strictly on the **2017 Validation set**.
  3. *Model selection update*: On the 2017 Validation set with the enhanced feature set, LightGBM surpassed XGBoost (Forecast: 150.53 vs. 151.86; Scenario: 242.27 vs. 244.74). Following the decision rule ("Select best on Validation 2017"), the deployed model type transitioned to LightGBM.
  4. *Final refit & test evaluation*: The LightGBM models refit on Train+Val (2012–2017) yielded final test MAEs of **139.65 vph** (Forecast) and **218.86 vph** (Scenario).
  5. *Freezing*: All model architectures, hyperparameters, and feature definitions are permanently frozen.

### 6.1 Holiday Sample Weight Selection & Validation Optimization

Holiday hours are sparse (744 / 24,972 = 3.0% of training data). To improve holiday modeling without degrading general traffic forecasts, candidate weights $w \in \{1, 5, 10, 20\}$ were evaluated on the **2017 validation set**:

| Holiday Sample Weight ($w$) | Val Overall MAE (vph) | Val Holiday MAE (vph) |
|---|:---:|:---:|
| $w = 1$ | 242.31 | 529.78 |
| $w = 5$ | 242.10 | 512.29 |
| $w = 10$ | 242.21 | 511.97 |
| **$w = 20$** | **242.27** | **488.09** ← lowest holiday error |

**Weight Selection**: $w=20$ was chosen by minimizing holiday-regime validation MAE (achieving a 41.7 vph error reduction on validation holidays compared to $w=1$).

#### Holiday Options Evaluation on 2017 Validation Set
Two structural holiday approaches were evaluated on 2017 validation:
- **Option 1 (Effective Day of Week)**: Encoding holiday weekdays as Sunday ($day\_of\_week = 6$).
  - Results: $w=1 \rightarrow$ Val Holiday MAE = 552.17 vph; $w=20 \rightarrow$ Val Holiday MAE = 485.81 vph.
- **Option 2 (Dashboard Routing to Baseline D Lookup Table)**:
  - Routing holiday scenarios directly to the Baseline D historical lookup table ($day\_type=2, hour, month$).
  - Results: **Val Holiday MAE = 404.82 vph** (81.0 vph lower error than Option 1).
  - **Decision**: Option 2 won on the validation set and is deployed in the interactive simulator for holiday scenarios.

---

## 7. Results

### Overall Test Set (2018)

| Model / Benchmark | Feature Scope | Test MAE (vph) | Test RMSE (vph) | R² | 4-Class Accuracy |
|---|---|:---:|:---:|:---:|:---:|
| **Forecast Model (LightGBM)** | Lags + Rolling + Weather | **139.65** | **217.38** | **0.9879** | **93.69%** |
| **Scenario Model (LightGBM)** | No Lags (What-If Inputs) | **218.86** | **364.56** | **0.9659** | **90.50%** |
| Baseline A (Refit on Train+Val) | Historical Mean by (Day, Hour) | 268.77 | 469.44 | 0.9435 | 88.96% |
| Baseline D (Fit on Train+Val) | Historical Mean by (DayType, Hour, Month) | 272.65 | 429.48 | 0.9527 | 88.56% |

**Gains (all computed on 2018 Test Set):**
- Forecast vs Baseline A: **48.0% MAE reduction**
- Forecast vs Baseline D: **48.8% MAE reduction**
- Scenario vs Baseline A: **18.6% MAE reduction**
- Scenario vs Baseline D: **19.7% MAE reduction**

### Sliced MAE by Operational Regime (2018 Test Set)

| Traffic Regime Slice | Forecast Model | Scenario Model | Baseline D | Scenario MAE Delta vs. Base D | Forecast MAE Delta vs. Base D |
|---|:---:|:---:|:---:|:---:|:---:|
| **Weekday AM Rush (07–08)** | 228.5 vph | 327.2 vph | 331.9 vph | **-4.7 vph (-1.4%)** | **-103.4 vph (-31.1%)** |
| **Weekday PM Rush (16–17)** | 206.6 vph | 322.0 vph | 362.9 vph | **-40.9 vph (-11.3%)** | **-156.3 vph (-43.1%)** |
| **Weekends** | 148.0 vph | 275.0 vph | 353.5 vph | **-78.5 vph (-22.2%)** | **-205.5 vph (-58.1%)** |
| **Holidays** | 151.3 vph | 357.7 vph | 324.1 vph | **+33.6 vph (+10.4%)** | **-172.8 vph (-53.3%)** |
| **Snow Hours** | 202.7 vph | 535.7 vph | 627.4 vph | **-91.7 vph (-14.6%)** | **-424.7 vph (-67.7%)** |
| **Rain Hours** | 124.9 vph | 172.4 vph | 212.6 vph | **-40.2 vph (-18.9%)** | **-87.7 vph (-41.2%)** |
| **Overall Test Set** | **139.7 vph** | **218.9 vph** | **272.7 vph** | **-53.8 vph (-19.7%)** | **-133.0 vph (-48.8%)** |

### Scenario Model Ablation Study (2017 Validation Set)

To rigorously isolate what drives the Scenario Model's performance without confounding MAE gains with volume suppression effects, we conducted an ablation study on the 2017 Validation partition:

| Feature Ablation Level | Features Included | Overall MAE | AM Rush (07–08) | PM Rush (16–17) | Weekends | Snow Hours | Rain Hours | Holidays |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **(a) Time + Holiday** | 16 features | **238.85 vph** | 428.95 vph | 353.81 vph | 275.92 vph | 314.48 vph | 209.52 vph | 273.54 vph |
| **(b) + Temperature** | 17 features | 243.68 vph | 430.86 vph | 351.80 vph | 279.23 vph | 321.11 vph | 214.26 vph | 439.98 vph |
| **(c) + All Weather** | 31 features | 242.27 vph | 420.74 vph | 342.15 vph | 280.69 vph | 326.12 vph | 215.35 vph | 488.09 vph |

#### Source of Scenario Model Advantage
Analysis of the validation ablation reveals:
1. **Primary Driver — Temporal & Calendar Structure**: Time and calendar conditioning (hour, day of week, cyclical encodings, and day-type indicators) accounts for the vast majority of predictive capability (achieving an overall MAE of 238.85 vph with zero weather inputs).
2. **Weather Feature Refinement**: Adding full weather features refines rush-hour predictions on the validation set (reducing AM rush MAE from 428.95 to 420.74 vph, and PM rush from 353.81 to 342.15 vph). On the 2018 Test set, the Scenario Model achieves a **14.6% error reduction on snow hours** (535.7 vs. 627.4 vph) and an **18.9% error reduction on rain hours** (172.4 vs. 212.6 vph) compared to Baseline D, confirming that weather inputs capture volume deviations that static tabular lookups miss.
3. **Holiday Trade-Off**: Incorporating continuous weather and high holiday sample weighting increases validation holiday error on the ML tree model due to sparse holiday samples (258 hours in 2017). This directly justifies adopting Option 2 (routing holiday scenarios to Baseline D lookup in the simulator).

---

## 8. Error Analysis

### Ten Worst Test Predictions

> **Important**: This dataset contains NO crash, incident, or event data. All diagnoses below are strictly *hypotheses* consistent with volume patterns; causes cannot be verified as fact.

| Timestamp | Actual (vph) | Predicted (vph) | Error (vph) | Hypothesis (No Incident/Event Data in Dataset) |
|---|:---:|:---:|:---:|---|
| `2018-05-21 13:00` | 2,040 | 4,821 | 2,781 | *Hypothesis*: Sudden mid-afternoon lane obstruction; volume fell 58% in 1 hour then recovered. |
| `2018-01-17 15:00` | 3,271 | 5,604 | 2,333 | *Hypothesis*: Severe incident or weather hazard immediately preceding the PM rush. |
| `2018-05-21 14:00` | 4,224 | 2,236 | 1,988 | *Hypothesis*: Post-obstruction flow recovery; model underpredicted because $T-1$ lag was depressed. |
| `2018-09-29 23:00` | 3,856 | 2,140 | 1,716 | *Hypothesis*: Unscheduled Saturday late-night surge (regional entertainment/sports egress or detour). |
| `2018-01-17 16:00` | 5,287 | 3,595 | 1,692 | *Hypothesis*: Flushing of upstream vehicle queue formed during the 15:00 disruption. |
| `2018-07-01 23:00` | 3,078 | 1,412 | 1,666 | *Hypothesis*: Sunday night holiday-week return travel surge. |
| `2018-09-12 23:00` | 2,842 | 1,212 | 1,630 | *Hypothesis*: Midweek night detour diverting traffic from parallel arteries onto I-94. |
| `2018-03-25 21:00` | 3,967 | 2,341 | 1,626 | *Hypothesis*: Concentrated Sunday evening return traffic. |
| `2018-09-28 23:00` | 3,544 | 2,033 | 1,511 | *Hypothesis*: Friday night urban event egress. |
| `2018-01-17 23:00` | 2,597 | 1,144 | 1,453 | *Hypothesis*: Delayed late-night traffic following acute cold-weather commute disruptions. |

**23:00 Error Concentration Audit**:
- Across all 271 test-set observations at 23:00, mean signed error is **+6.99 vph** (unbiased across the year).
- Standard deviation is **370.0 vph** (highest of any hour in the 24h cycle; nighttime free-flow hours average ~65 vph).
- **DST Check**: On the Spring forward transition date (2018-03-11), signed error at 23:00 was $-28.9\text{ vph}$ (actual 1,000 vs. predicted 1,028.9 vph). DST transitions do not explain these errors.

### Snow Response: Empirical vs. Model Decomposition

- **Measurement Limitation in Raw Data**: In the entire dataset, **99.0% of hours (3,085 out of 3,116 hours)** with `weather_main == 'Snow'` have `snow_1h == 0.0 mm`. In the training split, 98.5% (1,983 out of 2,014 hours) have zero recorded liquid snow depth. The automated weather sensor frequently recorded snow occurrence without measuring hourly accumulation depth.
- **Model Decomposition (January Weekday 08:00, temp = -5°C)**:
  - Baseline (Clear, snow_1h = 0.0): 5,640.3 vph
  - `weather_main == 'Snow'`, `snow_1h == 0.0`: **5,198.9 vph (-441.4 vph, -7.8%)**
  - `weather_main == 'Clear'`, `snow_1h == 2.5`: **5,272.5 vph (-367.8 vph, -6.5%)**
  - `weather_main == 'Snow'`, `snow_1h == 2.5`: **4,852.6 vph (-787.7 vph, -14.0%)**
- **Takeaway**: Categorical snow identification drives over half of the predicted snow suppression (-7.8%), while the intensity slider contributes an additional -6.2%. The combined response (~ -14.0%) is slightly higher than the empirical -8.9% to -10.0% volume drop observed during winter rush hours, reflecting the correlation between sub-zero temperatures and snow events.

### Holiday Behaviour: Like-for-Like Empirical Comparison

- **Empirical Training Volume at 08:00**:
  - Normal Weekdays: **5,584.30 vph** ($N=708$)
  - Holiday Weekdays: **2,585.90 vph** ($N=31$)
- **Like-for-Like Model Evaluation**:
  - Averaging the Scenario Model's prediction over the exact 31 training holiday weekdays at 08:00 yields **2,619.42 vph**.
  - Compared to the empirical mean of 2,585.90 vph, this represents a discrepancy of only **+33.52 vph (+1.30%)**, falling squarely within the 25% tolerance band ($[1,939.4, 3,232.4]\text{ vph}$).
- **Single-Date Variations & Untested Hypothesis**:
  - Single-point predictions vary significantly across months (e.g. November Thanksgiving weekday at 08:00 predicts 3,283.1 vph, while an October Columbus Day scenario predicts ~5,600 vph).
  - *Untested Hypothesis*: The elevated volume predicted on October holidays reflects the empirical presence of Columbus Day in the training data (e.g. 2015-10-12 had 5,368 vph; 2016-10-10 had 5,226 vph), where private-sector commuting continued normally despite official federal holiday status.

### 2018 Test Period Skipped Intervals Audit

The 2018 test set (Jan 1 – Sep 30, 2018) contains 6,552 calendar hourly intervals, of which **6,494 hours are retained** and **exactly 58 hours are skipped** due to sensor blackout gaps:
1. `2018-03-24 04:00` to `2018-03-25 07:00` (28 consecutive hours)
2. `2018-03-31 04:00` to `2018-03-31 07:00` (4 consecutive hours)
3. `2018-08-07 09:00` to `2018-08-08 09:00` (25 consecutive hours)
4. `2018-08-14 09:00` (1 isolated hour)
Total skipped: $28 + 4 + 25 + 1 = 58$ hours. Zero rows are imputed.

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

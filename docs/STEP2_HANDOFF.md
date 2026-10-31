# Step 2 handoff: time series models

**Status:** Final for Step 2. Models and settings are frozen. Step 3 (ML) and Step 4 (dashboard) can start from here.

**One-paragraph answer to the research question.** Economic indicators and regional data **do** improve NYUR forecasts when the labor market is turning: in the 2022 post-COVID decline, an ARX with NY claims, NY payrolls and US unemployment, and a VAR of NY/NJ/CT/PA unemployment, cut 3-month RMSE by about 25–35% versus naive. In calm periods they **don't help**: they turn noise into trends, and at 6 months nothing reliably beats "no change". A combination shrunk toward naive is the most robust model and is the only one that beats naive with statistical significance (1-month horizon, post-COVID, Diebold-Mariano p = 0.02). Most other differences are not significant with ~50 test months.

---

## 1. What to use

| You are | Use | Notes |
|---|---|---|
| Dashboard (Step 4) | [`results/forecasts.csv`](../results/forecasts.csv) | One row per forecast. `period = live` rows are the current forecasts. |
| Dashboard / slides | [`results/metrics.csv`](../results/metrics.csv), [`results/scorecard.png`](../results/scorecard.png) | Accuracy tables by period and horizon |
| ML (Step 3) | [`modeling/`](../modeling/) framework, `src/experiments/improved_models.py` as the template | Use the same origins, timing and scoring so results are comparable |

### `results/forecasts.csv` columns

| Column | Meaning |
|---|---|
| `date` | Target month being forecast (reference month, `YYYY-MM-01`) |
| `model` | `naive`, `arima`, `arx`, `arx_s`, `var_s`, `combo_s` (see section 3) |
| `horizon` | 1, 3 or 6: months after the **forecast origin month** (see section 2) |
| `yhat` | Forecast NYUR, % |
| `lower`, `upper` | Empirical 95% interval (section 5) |
| `origin` | Forecast date: last day of a month. Only data published by that day is used |
| `actual` | Published NYUR, blank if not out yet (or Oct 2025, never published) |
| `period` | `validation` (targets 2016–19), `covid` (2020–21), `post_covid` (2022–Aug 2026), `live` |

### Current (live) forecasts, recommended model `combo_s`

| Origin | Latest NYUR known | Target | Forecast | 95% interval |
|---|---|---|---|---|
| 2026-08-31 | Jul 2026 = 4.4% | **Sep 2026** (released Oct 20) | **4.31%** | 3.96–4.47 |
| 2026-09-30 | Aug 2026 = 4.3% | Oct 2026 | 4.28% | 3.93–4.43 |
| 2026-09-30 | Aug 2026 = 4.3% | Dec 2026 | 4.27% | 3.44–4.55 |
| 2026-09-30 | Aug 2026 = 4.3% | Mar 2027 | 4.30% | 2.38–4.69 |

All models agree September comes in at roughly 4.1–4.4%. The September forecast was made with information as of August 31, before the October 20 release.

---

## 2. Forecast design (pseudo-real-time)

The framework in `modeling/` was built by a teammate (see [`modeling/START_HERE.md`](../modeling/START_HERE.md)); we kept it unchanged and added experiments in `modeling/src/experiments/`. Its rules:

- **Origins** are month ends. Each origin uses only values published by that day, according to a release calendar (verified BLS/DOL dates, with documented estimated rules where unverified).
- **Horizon h** targets origin month + h. Because NYUR comes out about 3 weeks after the month ends, a 1-month forecast is usually 2 steps beyond the last known NYUR.
- **NY claims seasonal adjustment** is re-estimated at every origin with only the claims available then. The panel's `ny_claims_avg_sa` (full-sample STL) is never used, because it looks ahead.
- **Expanding window**: models are refit at each origin with every label published by then.
- **Direct forecasting** for regressions: a separate equation per horizon, using only predictors known at the origin. No future regressor values are ever plugged in.
- **Scoring on common samples**: a target counts only if every model issued a forecast and the actual exists. October 2025 (never published) is not scored.

**Splits** are by target month: train 1990–2015, validation 2016–2019 (used to pick settings), COVID test 2020–2021, post-COVID test 2022–Aug 2026.

---

## 3. Final models

| Model | What it is | Settings (frozen) |
|---|---|---|
| `naive` | Latest published NYUR | none |
| `arima` | ARIMA(0,1,2) on published NYUR, missing months handled by the state-space model | Order chosen by the teammate on validation |
| `arx` | Direct regression of the NYUR change on latest NYUR level/change, NY claims (real-time STL, log change), NY payroll growth, US unemployment | Features chosen by the teammate on validation. **Our changes:** COVID rows dropped from training; current inputs clipped to the training range |
| `arx_s` | `arx` shrunk toward naive: `naive + λ·(arx − naive)` | λ = 0.6 / 0.2 / 0.0 for h = 1 / 3 / 6 |
| `var_s` | VAR(3) on monthly changes of NY, NJ, CT, PA unemployment rates, COVID rows dropped, iterated to the target, then shrunk | Lag 3 by HQIC on 1990–2015; λ = 0.7 / 0.5 / 0.3 |
| `combo_s` | Mean of naive, arima, arx, then shrunk. **Recommended** | λ = 1.0 / 0.8 / 0.1 |

All λ were chosen on **validation only** (grid 0, 0.1, …, 1) and then frozen before scoring the test periods.

---

## 4. Results

RMSE relative to naive with the same information (below 1 = better than naive). Full numbers: [`results/metrics.csv`](../results/metrics.csv).

| Model | Validation 2016–19<br>h = 1 / 3 / 6 | COVID 2020–21 | Post-COVID 2022–26 | Calm 2023–26 |
|---|---|---|---|---|
| naive RMSE (pp) | 0.117 / 0.202 / 0.293 | 3.72 / 4.91 / 5.68 | 0.206 / 0.385 / 0.763 | 0.110 / 0.156 / 0.200 |
| arima | 0.80 / 0.91 / 0.96 | 1.33 / 1.26 / 1.23 | 1.12 / 1.08 / 1.05 | 1.03 / 1.01 / 1.01 |
| arx | 0.95 / 1.16 / 1.35 | 0.98 / 0.99 / 0.96 | 0.80 / 0.66 / 0.73 | 0.92 / 1.25 / 1.66 |
| arx_s | 0.88 / 0.98 / 1.00 | 0.99 / 1.00 / 1.00 | 0.83 / 0.89 / 1.00 | 0.89 / 0.94 / 1.00 |
| var_s | 0.73 / 0.86 / 0.93 | 1.40 / 1.45 / 1.32 | 0.74 / 0.66 / 0.79 | 0.90 / 1.02 / 1.08 |
| **combo_s** | 0.85 / 0.96 / 1.00 | 1.07 / 1.05 / 1.00 | **0.92** / 0.88 / 0.99 | 0.92 / 0.93 / 0.99 |

Validation numbers for `_s` models are optimistic because λ was chosen there.

**Statistical significance** (Diebold-Mariano vs naive, HLN-corrected): only `combo_s` at h = 1 post-COVID is significant (p = 0.02); `arx_s` h = 1 is borderline (p = 0.06). Everything else has p > 0.1.

**Model scorecard** (our judgment: calm accuracy 25%, turning-point accuracy 25%, shock robustness 15%, statistical evidence 20%, simplicity 15%):

| Model | Score /10 | Strength | Weakness |
|---|---|---|---|
| combo_s | 7.5 | Never loses badly; only significant win | Modest gains |
| var_s | 7.5 | Best single model outside COVID | Breaks down in shocks (COVID 1.3–1.5× naive) |
| arx_s | 7.4 | Robust, interpretable | At h = 6 it is just naive (λ = 0) |
| naive | 6.8 | Hard to beat in calm times and in COVID | Slow at turning points |
| arx | 6.7 | Strongest at turning points | Over-reacts in calm periods |
| arima | 6.3 | Good in validation | Loses to naive after 2022 |
| teammate's original macro ARX | 3.1 | Correct design | COVID in training distorts it (up to 3.7× naive; one 71% forecast) |

---

## 5. Processing history: what changed and why

1. **Data pipeline (Step 1):** `data/processed/monthly_panel.csv`. Unchanged.
2. **v0 exploration (ours, superseded):** `src/features/`, `src/models/`, outputs in `results/v0_exploratory/`, data overview plots in `figures/`. It built stationary features and a train/validation/test split, and ran SARIMA, ARX, VAR. **Do not use its numbers:** it used the full-sample STL claims column (look-ahead) and ignored publication lags. It is kept because it found the COVID-training problem and the value of VAR and Empire.
3. **Teammate framework:** `modeling/`, a strict pseudo-real-time setup with release calendar, real-time STL and tests. Frozen baselines (`results/modeling/`) showed naive winning everywhere.
4. **Why the two disagreed.** Two issues in the framework explained most of it:
   - **Release timing too conservative.** Only 9 NYUR release dates were verified, and the fallback rule was month end + 45 days, so most origins treated last month's NYUR as unpublished. NYUR and NY payrolls come out in the same BLS release, and 67 of 67 verified non-January, non-shutdown payroll releases came out within the following month. **Fix** ([`release_timing_v2.py`](../modeling/src/experiments/release_timing_v2.py)): NY/NJ/CT/PA rates inherit the 72 verified BLS payroll dates; the fallback becomes "by the end of the following month" (January: by the end of March). Verified holdbacks such as shutdowns still override it. Result: naive RMSE fell about 25–30% because the information is genuinely available.
   - **COVID in regression training.** Claims rose roughly tenfold in 2020; once those rows entered the expanding window, ARX coefficients were dominated by them. **Fix** ([`improved_models.py`](../modeling/src/experiments/improved_models.py)): drop training rows whose target, or whose 3-month feature window, falls in Mar 2020–Dec 2021, and clip inputs to the training range. Post-COVID ARX went from 2.7–3.7× naive to 0.65–0.80×.
5. **Added from our v0:** VAR of neighboring states, Empire survey (`M4x` in the experiment: small, unclear gain, so not in the final set), combination.
6. **Shrinkage** ([`shrink_to_naive.py`](../modeling/src/experiments/shrink_to_naive.py)): validation showed longer horizons should trust models less. This fixed calm-period over-forecasting at the cost of some turning-point gains.
7. **Final outputs** ([`final_forecasts.py`](../modeling/src/experiments/final_forecasts.py)): backtest + live forecasts, empirical 95% intervals, metrics.

---

## 6. Reproduce

```sh
cd modeling
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # on Python 3.9 use matplotlib<3.10
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m src.handoff inspect                        # teammate baselines unchanged
python -m src.experiments.release_timing_v2          # data/experiments/*_v2.csv
python -m src.experiments.improved_models            # ~10 min, all models, all origins
python -m src.experiments.evaluate_improved          # metrics + DM tests vs frozen baselines
python -m src.experiments.shrink_to_naive            # choose λ on validation, apply
python -m src.experiments.final_forecasts            # final forecasts.csv / metrics.csv (~10 min)
python -m src.experiments.plot_improved              # scorecard.png
```

Then copy `modeling/results/experiments/final_v1/{forecasts,metrics}.csv` to `results/`.

---

## 7. Limitations (say these in the report)

- **Not a fresh out-of-sample test.** 2020–2026 results have been looked at by both of us. The clean test is prospective: score the frozen models on each new month starting with September 2026.
- **Pseudo-real-time, not real-time.** Values are today's revised data, not what was published at the time (no ALFRED vintages).
- **Release timing for unverified months is an estimated rule**, documented in `modeling/data/experiments/release_rules_v2.csv`.
- **Intervals** are empirical 2.5/97.5% quantiles of validation + post-COVID backtest errors for each model and horizon. They exclude COVID, so they understate risk in a shock. They are asymmetric (wider on the downside) because 2022 had large declines.
- **Small samples:** 48 validation and about 50 post-COVID targets per horizon; multi-step errors overlap. Rankings are suggestive, not proof.

---

## 8. Notes for the next steps

**Step 3 (ML).**
- Reuse `improved_models.py`: same origins, `build_origin_bank` features, `expanding_training_sample`, COVID-free rows and common-sample scoring. Add a model alongside the others rather than building a separate evaluation, or the numbers won't be comparable.
- Regularization (Ridge, LightGBM) is the natural next test; fit scaling and tuning inside each training window.
- Pick any hyperparameters on validation 2016–2019 only.
- The LLM forecast comparison can use the same `live` origins.

**Step 4 (dashboard).**
- Show `combo_s` as the headline with its interval; let users switch models.
- Plot backtests from `period != live`, and shade COVID.
- Compare the September forecast (4.31%) with the actual after October 20.

**Open issues for the framework owner.**
- `tests/test_handoff_integrity.py::test_core_input_closure_is_explicit` expects exactly 5 CSVs under `data/`, but `START_HERE.md` says experiments should put new inputs in `data/experiments/`. With our two timing files it fails (7 ≠ 5). Please decide which rule wins.
- Two integrity tests import `src/models/validation_report.py`, which uses Python 3.12 f-string syntax, so they error on Python 3.9/3.11. Model tests pass (71/73 before our files were added).
- `inspect` still verifies all saved baseline results. Our work only adds files and changes no original file.

# nys-unemployment-forecast

Can economic indicators and alternative data improve short-term forecasts of the New York State unemployment rate (FRED `NYUR`)?

## Data pipeline (Step 1)

### Running it

With the virtual environment active and `FRED_API_KEY` set in `.env` (see `.env.example`), run from the repo root:

```
python src/data/fetch_fred.py
python src/data/fetch_yfinance.py
python src/data/fetch_indeed.py
python src/data/build_panel.py
python src/data/check_panel.py
python src/data/fetch_taxable_sales.py
python src/data/build_taxable_sales.py
```

The fetch scripts save untouched source data to `data/raw/`. `build_panel.py` converts everything to monthly and writes `data/processed/monthly_panel.csv`. `check_panel.py` runs sanity checks and saves a plot of every column to `data/processed/panel_plots.png`.

You don't need to rerun anything to use the data: the committed `monthly_panel.csv` is a frozen snapshot. Rerunning the fetch scripts pulls the latest data, and past values can change because FRED revises them.

**Data as of:** 2026-10-19.

### Panel format

- One row per month, `date` as `YYYY-MM-01`, from 1976-01 (the start of `NYUR`) to the last complete month.
- **Rows are dated by reference month, not release date.** The August unemployment rate is in the `2026-08-01` row even though it was published in September. In backtests, only use values that would have been published by the forecast date, or you will leak future information.
- **Levels only.** Compute growth rates, differences and lags in your own code.
- **Missing values are NaN and are never filled.** Series start at different dates, and the latest month is often missing for slower series.

### Columns

Full details for every column in both output files (units, series IDs, transformations, start dates and typical release lags) are in [`data/processed/data_dictionary.csv`](../data/processed/data_dictionary.csv). `check_panel.py` fails if a panel column is missing from it.

| Column | Source | How it becomes monthly |
|---|---|---|
| `ny_ur` | FRED `NYUR`, NY unemployment rate, SA. **Target.** | Monthly already |
| `ny_claims` | FRED `NYICLAIMS`, NY initial claims, NSA | Week containing the 12th (see below) |
| `ny_claims_avg` | FRED `NYICLAIMS` | Average of all weeks ending in the month; NaN if fewer than 4 weeks |
| `ny_claims_avg_sa` | `ny_claims_avg`, seasonally adjusted with STL | **Use this one for NY claims in models** |
| `ny_payrolls` | FRED `NYNA`, NY nonfarm employees, SA, thousands | Monthly already |
| `ny_coincident` | FRED `NYPHCI`, Philly Fed NY coincident index | Monthly already |
| `us_ur` | FRED `UNRATE`, US unemployment rate, SA | Monthly already |
| `us_claims` | FRED `ICSA`, US initial claims, SA | Week containing the 12th |
| `spread_10y3m` | FRED `T10Y3M`, 10-year minus 3-month Treasury yield | Monthly average of daily values |
| `fedfunds` | FRED `FEDFUNDS`, effective federal funds rate | Monthly already |
| `nj_ur`, `ct_ur`, `pa_ur` | FRED `NJUR`, `CTUR`, `PAUR`, neighboring states' unemployment rates, SA | Monthly already |
| `sp500` | Yahoo Finance `^GSPC`, S&P 500 close | Monthly average of daily values |
| `empire_general` | FRED `GACDISA066MSFRBNY`, NY Fed Empire State Manufacturing Survey, current general business conditions, SA. From 2001-07. | Monthly already |
| `indeed_postings` | Indeed Hiring Lab job postings index, New York, SA, Feb 1 2020 = 100. From 2020-02. | Monthly average of daily values |

`indeed_postings` data is from [Indeed Hiring Lab](https://github.com/hiring-lab/job_postings_tracker), licensed CC BY 4.0. It is already seasonally adjusted and smoothed with a 7-day trailing average, and Indeed revises its history. It is genuinely missing before February 2020, so models that use it are limited to 2020 onward.

`empire_general` is a diffusion index: the percent of NY manufacturers reporting better conditions minus the percent reporting worse. Above 0 means conditions are improving; below 0 means they're getting worse.

### Weekly claims: why the 12th, and why NY claims are different

**Why the week containing the 12th.** The unemployment rate comes from the Current Population Survey, which asks people about their work status during the week (Sunday to Saturday) that contains the 12th of the month. Taking claims for that same week measures layoffs over the period the unemployment rate describes. This is standard practice in labor-market forecasting.

FRED dates each claims week by the Saturday it ends on. The week containing the 12th is always the one whose Saturday falls on the 12th to 18th (Saturday the 12th if the 12th is a Saturday, Saturday the 18th if it is a Sunday), and every month has exactly one such Saturday.

**Why NY claims also get a monthly average.** `NYICLAIMS` is not seasonally adjusted, and NY has short spikes of about two weeks in early January and early July (likely end-of-holiday jobs and school-year layoffs). Whether the reference week lands inside a spike depends on the calendar, not the economy. For example:

| | July 2019 | July 2026 |
|---|---|---|
| Reference week | Jul 13: 25,416 (inside spike) | Jul 18: 14,735 (spike just ended) |
| Average of all July weeks | 18,550 | 19,680 |

The reference week suggests July claims fell 42%. The weekly data shows both Julys had a similar spike. Seasonal adjustment then makes this worse: STL learns that July is usually about 1.8x normal and divides every July by that, so July 2026 comes out at roughly half its true level. The monthly average always includes the spike weeks, so it adjusts cleanly. That's why the panel has no seasonally adjusted reference-week NY column: `ny_claims` is kept for reference, but model with `ny_claims_avg_sa`.

`us_claims` doesn't have this problem because `ICSA` is already seasonally adjusted week by week by the Department of Labor.

### Quarterly file: NYS taxable sales

`data/processed/quarterly_taxable_sales.csv` is kept separate from the monthly panel because it is quarterly, and spreading it across months would invent values.

- **Source:** NY open data, [Taxable Sales And Purchases Quarterly Data](https://data.ny.gov/d/ny73-2j3u), statewide rows only. From March 2013.
- **Quarters are NY sales-tax quarters, not calendar quarters.** The sales-tax year runs March to February: Mar–May, Jun–Aug, Sep–Nov, Dec–Feb. `date` is the first month of the quarter (`2026-03-01` = Mar–May 2026).
- **`taxable_sales`** is the sum of all industry groups, in dollars, NSA. A few small industries are suppressed for confidentiality (`suppressed_industries`, 0–3 per quarter) and count as zero, so the total is slightly understated.
- **`taxable_sales_sa`** is STL-adjusted in logs with `seasonal=13`. The default setting let the 2020 crash leak into the seasonal pattern, so the stiffer setting keeps the COVID drop in the adjusted series where it belongs.
- **`status`** is `P` (preliminary, can still be revised) or `F` (final).

### Sources considered but not included

- **BLS JOLTS state estimates (NY openings, hires, quits, layoffs).** Available from the BLS API, but as of October 2026 state data stops at December 2025, for every state checked (NY and CA), while national JOLTS is current through August 2026. A feature with no 2026 values can't help forecast current months, so it's paused until BLS resumes state publication.
- **NYS DOL WARN notices.** Notices before April 2025 are individual web pages, with no bulk export. Since April 2025 they're in a Tableau dashboard that does allow downloads, but that's only about 18 months of history, too short to model with.

### Known issues for modelers

- **October 2025 is missing for all unemployment rates** (`ny_ur`, `us_ur`, `nj_ur`, `ct_ur`, `pa_ur`). The federal government shutdown stopped BLS from collecting the household survey that month, so the value was never published. Models that need an unbroken series (ARIMA, ETS) have to handle this gap themselves, for example by interpolating inside the model code.
- **STL seasonal adjustment looks ahead.** STL estimates each month's seasonal pattern partly from later years, so `ny_claims_avg_sa` contains a small amount of future information. For a strict backtest, re-run the adjustment on each training window.

## Time series models (Step 2)

**Start with [docs/STEP2_HANDOFF.md](STEP2_HANDOFF.md).** It summarizes the findings, the final frozen models, and the forecast file for the dashboard ([`results/forecasts.csv`](../results/forecasts.csv)). The final models run in the pseudo-real-time framework in [`modeling/`](../modeling/).

### v0 exploratory features (superseded)

The section below documents our first feature pass (`src/features/`). It is kept for reference; its claims seasonal adjustment and timing assumptions are not used by the final models.


```
python src/features/build_features.py
python src/features/check_features.py
```

`build_features.py` reads the processed panel and writes `data/features/model_features.csv` (one row per month, 1976-01 on) plus `data/features/feature_dictionary.csv`. `check_features.py` runs ADF + KPSS tests on every level and transformed series (`data/features/stationarity.csv`) and plots them (`data/features/feature_plots.png`).

- **October 2025 gap:** `ny_ur_raw` keeps it as NaN. Use it for SARIMA/ARIMAX: statsmodels `SARIMAX` skips missing values on its own. The interpolated unemployment-rate columns (`ny_ur`, `nj_ur`, ...; flagged by `ur_imputed`) are for VAR, which cannot take NaN.
- **Naming:** `d_` = first difference (percentage points), `g_` = 100 × log difference (monthly % growth), `_lagK` = lagged K months. Levels keep their panel names. `empire_general` is already stationary, so it is used in levels.
- **Stationarity:** `ny_ur` is non-stationary in levels and stationary after one difference, so use d = 1 (or model `d_ny_ur`). Every `d_` / `g_` column passes ADF and KPSS.
- **`yoy_ny_claims`:** 12-month log change of the NSA NY claims average, with no STL look-ahead. Prefer `g_ny_claims_avg_sa` in the main models and use this as a robustness check (it has a base effect: +300% in 2020, about −200% in 2021).
- **Multi-step forecasts (direct method):** lags 1–6 and 12 of `d_ny_ur` and the main regressors. Fit a separate ARIMAX for each horizon h and use only `_lagK` columns with K ≥ h, so every input exists when the forecast is made. Do not feed actual future regressor values into a backtest.
- **Splits (`split` column):** all three avoid COVID and the October 2025 gap, and have no missing values in any core feature.

  | split | months | use |
  |---|---|---|
  | `train` | 1990-02 to 2016-12 (323) | fit models |
  | `validation` | 2017-01 to 2020-02 (38) | pick orders, lags and regressors |
  | `excluded` | 2020-03 to 2021-12 (22) | COVID; dropped from fitting and scoring (`covid = 1`) |
  | `test` | 2022-01 to 2025-09 (45) | score once at the end, after choices are frozen |
  | `live` | 2025-10 on | refit on everything and produce the real forecast |
  | `before` | before 1990-02 | not every core regressor exists yet |

  Rows in 2022 have `_lag` values that reach back into COVID months. Drop the first 12 test months, or treat that as a known limitation.
- **Samples:** `sample_long` (1990-02 on, all core regressors), `sample_empire` (2001-07 on), `sample_indeed` (2020-03 on). Indeed does not reach the train split, so it can only be used in short post-COVID comparisons.
- **Taxable sales:** quarterly growth placed on the monthly grid from the month it is published (assumed 3 months after the quarter ends) and held until the next release. Nothing is interpolated between quarters.
- **Publication lag:** `pub_lag_months` in the feature dictionary is how many months after the reference month a value is published, net of any lag. For a forecast made at the start of month t+1, only use features with `pub_lag_months` ≤ 0 at row t.

## Machine learning and validation (Step 3)

See [the Step 3 handoff](STEP3_HANDOFF.md) for methods, limitations,
reproduction commands and results, or [the walkthrough](STEP3_GUIDE.md).
Source code is in `modeling/src/experiments/step3_ml/`; generated forecasts,
common-sample metrics and figures are in `modeling/results/experiments/step3_ml/`.
This experiment preserves the Step 1 data and frozen Step 2 code/results.

## Dashboard (Step 4) and AI assistant (Step 5)

A Streamlit + Plotly app in [`app/`](../app/) that reads only committed outputs (the Step 1 panel, Step 2 `results/`, Step 3 `modeling/results/experiments/step3_ml/`), so it runs offline.

```
pip install -r requirements.txt
streamlit run app/app.py
```

Tabs: **Dashboard** (headline tiles, the forecast explorer with the assistant panel beside it, live forecasts, findings), **Model scorecard** (error vs. naive by period and horizon, plus the with/without-indicators comparison), **Indicators** (a tile per panel series tagged by how the models use it, and a chart comparing any series with NY unemployment) and **Methodology**. The original overlay Indicators page is kept in `app/app.py` behind `SHOW_OLD_INDICATORS = False`.

The Google search trends tile is a **simulated placeholder** (`placeholder_trends()` in `app/app.py`), not real data.

**Ask the forecast** is a Claude (`claude-opus-5-5`) assistant whose system prompt contains the Step 2 and Step 3 handoffs, the accuracy table, the live forecasts and the latest panel values, so it answers from our results. It needs `ANTHROPIC_API_KEY` in `.env` locally, or in the app's secrets on Streamlit Community Cloud (`.streamlit/secrets.toml` is gitignored). Without a key the panel shows a preview conversation and everything else still works.

Files: `app/data.py` (loaders, model names and colors), `app/charts.py` (Plotly figures), `app/assistant.py` (Claude prompt and streaming), `app/app.py` (layout), `app/style.css` (styling).

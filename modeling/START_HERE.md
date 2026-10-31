# NYS Unemployment: model improvement handoff

**Research question:** Can traditional economic indicators and alternative data improve 1-, 3-, and 6-month forecasts of New York State unemployment?

This is a small offline package of the existing baselines, fixed data snapshot and observed results. Read [MODEL_RESULTS.md](MODEL_RESULTS.md) before proposing extensions. No API account, credentials, data download or original project checkout is needed. The package has no data-fetching or historical cleaning pipeline.

## Data and scope

Dates in the panel identify reference months, not publication dates. Coverage below is the nonmissing coverage in the supplied snapshot, **not** a promise of complete monthly history or availability at each origin. Modeling begins in January 1990. NYUR's latest observed target is **August 2026**.

| Core variable | Original source | Nonmissing monthly span | Units / candidate transformations |
|---|---|---|---|
| NYUR (`ny_ur`) | [BLS LAUS via FRED NYUR](https://fred.stlouisfed.org/series/NYUR) | Jan 1976–Aug 2026 | Percent, seasonally adjusted (SA); level and monthly difference in percentage points |
| NY Initial Claims (`ny_claims_avg`) | [US DOL ETA via FRED NYICLAIMS](https://fred.stlouisfed.org/series/NYICLAIMS) | Mar 1986–Sep 2026 | Claims/week, originally not SA; mean of weeks ending in each month; training-window log-STL adjusted log level and first log difference |
| NY Payroll Employment (`ny_payrolls`) | [BLS CES via FRED NYNA](https://fred.stlouisfed.org/series/NYNA) | Jan 1990–Aug 2026 | Thousands of employees/persons, SA total nonfarm; 100 × monthly log difference |
| USUR (`us_ur`) | [BLS CPS via FRED UNRATE](https://fred.stlouisfed.org/series/UNRATE) | Jan 1976–Sep 2026 | Percent, SA; level and monthly difference in percentage points |

The unchanged [monthly panel](data/processed/monthly_panel.csv) retains extra original columns to preserve the snapshot; the modeling loader explicitly selects only these four series. [The original dictionary](data/processed/data_dictionary.csv) describes every column. Its advice to use `ny_claims_avg_sa` is obsolete for backtesting: that column uses full-sample STL and is always excluded. Only [weekly NY claims](data/raw/fred_NYICLAIMS.csv) is required as a raw input. The monthly claims aggregate requires every expected Saturday, at least four weekly observations, and matching raw values. No partial-month proxy is substituted.

## Forecasting design

- Origins are the last calendar day of each month, at New York day-end. Horizon h targets origin reference month +h, irrespective of reporting delays.
- Initial training history: 1990–2015. Validation targets: 2016–2019. COVID test targets: 2020–2021. Post-COVID targets: January 2022–August 2026 in this snapshot.
- Each origin uses only released observations. Verified calendar entries override estimated fallback rules; monthly claims wait for all weekly constituents. See [RELEASE_POLICY.md](RELEASE_POLICY.md), [rules](data/processed/release_rules.csv), and [calendar](data/processed/release_calendar.csv).
- Expanding-window estimation re-estimates robust log-STL inside each historical feature origin's own available claims history (minimum 36 months, period 12). That historical predictor row is never replaced by a later STL estimate.
- Each historical training row retains its original predictors. Its future NYUR label becomes eligible only when published by the current fitting origin. Direct regressions standardize on eligible training rows only and do no imputation.
- Preserve target NYUR levels in percent for evaluation. October 2025 NYUR is genuinely missing; its forecasts are retained but unscored. Monthly differences crossing the gap stay NaN, and state-space ARIMA retains the internal monthly calendar gap.

This is **pseudo-real-time**: dates are screened, but snapshot values contain later revisions and seasonal adjustments. It does not reconstruct historical numerical vintages.

## Frozen baseline models

| Model | Fixed specification |
|---|---|
| M0 Naive | Latest actually published NYUR, repeated for each horizon |
| M1 ARIMA | ARIMA(0,1,2), no drift, state-space handling of missing months; forecast steps include publication age |
| M2 Core Direct ARX | Latest published NYUR level, calendar first difference and age; claims STL log difference at origin Lag 1; payroll log growth at origin Lag 2 |
| M3 Macro Direct ARX | M2 structure plus USUR; h1 uses Payroll Lag 3 and USUR difference, h3 Payroll Lag 3 and USUR level, h6 Payroll Lag 2 and USUR difference |

Direct ARX estimates a separate equation at each horizon for the change from the latest published NYUR, then adds that baseline back. It uses explanatory information known at the origin. This avoids assuming future realized claims, payrolls or USUR are already known. Direct OLS uses at least 60 complete training rows; unavailable current predictors mean no forecast, with no replacement model. The saved [selections](results/modeling/selected_specifications.csv) and [frozen lock](results/modeling/test_frozen_specifications.json) are fixed references, not search instructions.

## How to get started

Use Python 3.12. From the extracted folder (the commands use no personal paths):

```sh
cd NYS_Model_Improvement_Handoff
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python -m unittest discover -s tests -v
python -m src.handoff inspect
python -m src.handoff features --origin 2023-12-31
```

On Windows activate `.venv\Scripts\activate` and set the same environment variables with the shell's syntax. The Unix commands are the tested reference. `inspect` checks included-file hashes, all saved metric rows, the twelve fixed choices and the unscored October 2025 records; it does not fit models.

To refit the saved baselines from the included data, with **no grid search or new model training**:

```sh
python tools/run_isolated.py reproduce --output results/reproduced
```

The isolation harness calls `src.handoff reproduce` and blocks external project/home file reads and network connections. This builds the historical feature bank from raw constituents, refits the fixed twelve specifications, writes 576 validation and 960 test forecast records, and checks every prediction/status against the saved baselines (absolute tolerance 0.000001 percentage point plus 1e-8 relative tolerance). Expect several minutes. Output must be a **new** directory inside the package; a repeat can use `results/reproduced_again`. Canonical results stay unchanged. Use this runner, rather than the original archival `run_validation`/`out_of_sample` main programs, whose cache/audit-file dependencies are deliberately omitted. Model and eligibility functions are reused unchanged. See [PACKAGING_NOTES.md](PACKAGING_NOTES.md) for the exact adaptations.

The full reproduction and retained tests were checked in an isolated directory. [VERIFICATION.md](VERIFICATION.md) records the tested environment and limitations of dependency installation. Installed Python libraries are external runtime dependencies; no data or project code is read from the original full project.

## How to Improve the Models

**Start an explicitly named experiment; preserve the baselines.** Add experiment modules under `src/experiments/`, new inputs under `data/experiments/`, and outputs under `results/experiments/<name>/`. Do not edit the frozen selections, core functions or canonical results just to obtain a better comparison.

1. **New economic indicator:** store its values and source/units/reference definition separately. Add observation-specific publication evidence and documented estimated fallbacks in experiment-specific metadata. Use `available_panel` in [data_availability.py](src/models/data_availability.py) before transforming it. For weekly/daily aggregates, supply every required constituent and make availability the latest constituent publication plus completion of the reference period. Unknown release timing must not silently become same-month availability.
2. **Transformation or lag:** inspect `build_feature_window` in [feature_engineering.py](src/models/feature_engineering.py), then `origin_row` and `build_origin_bank` in [rolling_validation.py](src/models/rolling_validation.py). Extend in a separate experiment feature builder; keep calendar gaps and original-origin snapshots. Origin Lag 2 means origin calendar month minus two, never target month minus two. Never use `ny_claims_avg_sa`, future test scaling or full-sample imputation.
3. **Alternative regression/ML model:** use `training_sample` for validation-boundary eligibility or `expanding_training_sample` in [out_of_sample.py](src/models/out_of_sample.py) for later released labels. Fit preprocessing/regularization inside each eligible training fold. Direct forecasting should use only origin-known predictors. A recursive SARIMAX must forecast its future exogenous path using origin-known information, or clearly label an assumed scenario.
4. **Execute and compare:** the runnable reference loop is [src/handoff.py](src/handoff.py). Preserve `forecast_origin`, `target_month`, `horizon`, `model`, `spec_id`, `yhat`, `actual`, `status` and publication/eligibility metadata in new forecasts. Use identical origin/target/horizon keys and intersect finite, observed forecasts across baselines **and** the new model. Report exclusions and own coverage alongside common-sample MAE/RMSE/bias. Do not use a different easy subset for each model; keep horizon and evaluation-view filters explicit.

Research directions include alternative claims transformations, payroll level versus growth, Ridge/Elastic Net, alternative ARIMA/SARIMAX structures with valid future exogenous paths, additional leading indicators, and robust handling of extreme shocks. **None has been validated by this handoff.**

**Evaluation warning:** validation 2016–2019 has already selected models, and test 2020–2026 outcomes have now been inspected. New models can use these periods for development and retrospective benchmarking, but improved results on them are not fresh untouched out-of-sample confirmation. Predeclare the next protocol and tuning limits; where possible use new prospective observations or an explicitly reserved additional holdout. Keep the earlier frozen baselines fixed.

The original validation information cutoff was January 31, 2020. Some requested early COVID target/horizon pairs have origins before the final selections could be known. Saved tests flag those origins and supply a separate `post_selection_origins` view; neither view is chosen to favor a model. See MODEL_RESULTS for details.

## Package map

`data/`: five input CSVs. `src/`: preserved model implementations plus the portable runner. `results/modeling/`: canonical metrics, forecasts, fixed specifications, coverage and three traceable comparison figures. `tests/`: retained core regressions and handoff integrity checks. [FILE_INVENTORY.csv](FILE_INVENTORY.csv) gives every included file's purpose, bytes and SHA256 (excluding itself); [provenance/source_manifest.json](provenance/source_manifest.json) identifies copied source files and packaging adjustments.

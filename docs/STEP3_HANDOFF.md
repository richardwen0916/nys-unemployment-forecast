# Step 3: machine learning and validation

Implemented on `step3-ml-validation`, based on main commit
`391860f231aa8f626a90fb4ee252b0149c382668`. All forecasting code, data and
canonical results from Steps 1 and 2 are preserved. Step 3 code and outputs
are stored separately under `step3_ml` experiment directories.

## Findings

The model selected on 2016–2019 validation is regional Ridge at horizons 1 and
3, and Gradient Boosting at horizon 6. These choices remain fixed even where
another model subsequently performs better on test data.

Post-COVID (2022–August 2026), on identical finite targets shared by all six
Step 2 models and all six Step 3 variants:

| Horizon | Validation-selected ML model | n | RMSE (pp) | MAE (pp) | RMSE / naive |
|---|---|---:|---:|---:|---:|
| 1 | ridge_regional | 50 | 0.1576 | 0.1071 | 0.764 |
| 3 | ridge_regional | 50 | 0.2616 | 0.2005 | 0.679 |
| 6 | gradient_boosting | 51 | 0.6298 | 0.3845 | 0.825 |

This does not establish that ML universally beats Step 2. Its frozen `var_s`
beats the selected ML at h=1 on this sample, and its `arx` beats the selected
ML at h=3 and h=6. Core Ridge has stronger retrospective h=3/6 results than
the validation-selected models, but is NOT substituted after viewing test errors.

For calm targets from 2023 onward, selected-model RMSE / naive is 0.879, 1.174
and 1.142 for h=1/3/6. Thus short-horizon gains coexist with weaker calm-period
medium/long-horizon performance. COVID shock errors are approximately as large
as naive; none of these models anticipates an unprecedented shutdown.

## Models and features

All regressions predict the change from the latest **published** NYUR to the
target NYUR level, then add that published baseline back. Separate models are
estimated for h=1,3,6. Each horizon is measured from the origin calendar month,
not from the last observed unemployment month.

| Name | Estimator | Information |
|---|---|---|
| ridge | StandardScaler + Ridge | 10 core origin features |
| lasso | StandardScaler + Lasso | same core features |
| random_forest | 80 trees, shallow/large leaves | same core features |
| gradient_boosting | 80 stages, Huber loss, shallow trees | same core features |
| ridge_ar_only | StandardScaler + Ridge | latest NYUR level, monthly change, publication age |
| ridge_regional | StandardScaler + Ridge | core plus NJ/CT/PA latest rates/changes and Empire survey |

Core features: latest NYUR level/change/publication age; origin-lag 1/2/3
log changes of own-origin STL-adjusted claims; payroll log growth at origin-lag
2/3; latest published US unemployment level/change. `features.py` declares the
exact names. Regional variables are publication-masked before selecting their
latest levels and computing calendar differences. Empire uses the existing
release policy, including the first-publication gate.

Use `feature_ablation.csv` for paired predictive comparisons with target-only
Ridge. Evaluation targets match within each pair, but training spans and
selected regularization strengths differ. Do not interpret these as a causal
or otherwise fully controlled estimate of adding a particular feature.

Indeed is excluded because its conservative publication gate leaves no
2016–2019 validation history. Quarterly taxable sales are excluded because no
compatible audited quarterly-release input exists in the inherited framework.
Neither is artificially filled into the long historical training set.

## Validation and safeguards

1. Reuse Step 2 v2 release rules/calendar and unchanged `build_origin_bank`,
   `training_sample`, and `expanding_training_sample` functions.
2. Build 441 original-origin snapshots, January 1990–September 2026. Claims
   robust log-STL is recomputed using only each original origin's available
   history. Never use the panel's full-sample `ny_claims_avg_sa` column.
3. Only train on prior origin rows whose target label was published by the
   fitting origin. Historical X is never recomputed with a later information set.
4. Follow Step 2's COVID training exclusions: target or origin feature reach
   intersects March 2020–December 2021. COVID targets remain in the evaluation.
5. No imputation. Missing current inputs produce an explicit missing forecast.
   October 2025 targets remain unobserved and unscored. Monthly differences
   crossing that calendar gap remain missing.
6. Fit scaling only on each eligible training window. Clip current inputs to
   that training range, following Step 2's experimental robustness policy.
7. Declare small grids in `models.py`; tune each model/horizon on validation
   targets 2016–2019 using common candidate rows. Save `protocol.json` before
   scoring, then freeze `selected_models.json` before retrospective tests.
8. Use `metrics.csv` rows with `sample=common_all_models` for cross-model
   comparisons. `own_available` rows answer a different coverage question.
9. Early COVID forecasts whose origins precede the January 31, 2020 selection
   cutoff are flagged. `post_selection_covid` is reported separately.

## Files to inspect

Source: `modeling/src/experiments/step3_ml/`

- `features.py`: source hashes, per-origin features and release audit.
- `models.py`: feature groups, estimator grids, eligible samples, train-only scaling.
- `run.py`: rolling validation, locked selection, tests, live forecasts, provenance.
- `report.py`: common-sample metrics, coverage, paired comparisons and charts.
- `modeling/tests/test_step3_ml.py`: focused regression tests.

Outputs: `modeling/results/experiments/step3_ml/`

- `RESULTS.md`: generated concise result summary.
- `forecasts.csv`: 2,340 planned ML records, including 40 missing-current-input
  records; status and per-fit publication/eligibility metadata are retained.
- `live_forecasts.csv`: 36 live rows; `recommended` is determined by validation.
- `metrics.csv`: RMSE, MAE, bias, ratio to naive; own and common samples separated.
- `coverage.csv`: planned, forecast, actual, own-scored and common-scored counts.
- `feature_ablation.csv`: paired Ridge comparisons with the caveat above.
- `feature_importance.csv`: standardized linear coefficients or training impurity
  importance at the September 2026 origin, by horizon. Not causal importance.
- `model_comparison.png`, `forecast_paths.png`, `feature_importance.png`.
- `protocol.json`, `selected_models.json`, `recommended_models.json`,
  `tuning_scores.csv`, `run_metadata.json`, `feature_manifest.json`.
- `origin_features.csv`, `origin_metadata.csv`, `labels.csv`,
  `extra_release_audit.csv`: auditable intermediate data, not raw-source replacements.

Dashboard can read `live_forecasts.csv` columns
`date,model,horizon,yhat,lower,upper,origin,actual,period` just as in Step 2;
additional columns include `recommended`, `status` and fitting metadata.
Keep the Step 2 headline model available rather than silently replacing it.

## Reproduce

From the repository root, create a Python 3.12 environment and install:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r modeling/requirements-step3.txt
cd modeling
python -m src.experiments.step3_ml.run
```

On Windows use `.venv\Scripts\activate`. After installation, execution is offline
and reads the committed snapshots. `--phase tune` performs validation only;
`--phase test` requires the unchanged existing selection lock. Hash-verified
feature caches are reused. If changing a forecasting specification, use a new
experiment directory rather than overwriting the lock to chase better test errors.
To regenerate only charts/tables: `python -m src.experiments.step3_ml.report`.

Verified environment: Python 3.12.14, numpy 2.3.5, pandas 2.2.3, scipy 1.15.3,
scikit-learn 1.7.2, statsmodels 0.14.6, matplotlib 3.10.9.

Verification: 5 Step 3 tests plus 61 inherited tests across availability (21),
feature engineering (13), rolling validation (15), and out-of-sample eligibility
(12) passed. These are targeted tests, not a claim that the archival handoff's
entire integrity suite passes. Plot layouts were visually inspected.

## Limitations and next step

This is retrospective **pseudo-real-time**, not an untouched out-of-sample
test: the team has already examined the 2020–2026 history. Revised numerical
vintages and some estimated release dates remain. The Step 2 comparison uses
its existing four-decimal forecast export; it is not refitted under a new library
version. Small, overlapping test samples do not establish statistical superiority.

Live intervals are validation residual 2.5%/97.5% quantiles added to the point
forecast; no test residuals calibrate them. They reflect 48 calm validation
targets and may severely understate shock or regime-change uncertainty.

Keep the frozen specifications and evaluate subsequent actual releases
prospectively. Any change to the selected models requires an explicitly new
experiment, not relabeling a test-selected winner as validation-selected.

## References used

- Course Assignment 3 Part A: lagged predictors, Ridge Pipeline, random forest,
  forecast comparison, recursive target discipline. Step 3 uses direct horizons
  instead of assuming future exogenous values are known.
- Class 4 State Space lecture, pages 10–11: information sets and comparable
  rolling versus fixed-origin validation.
- Chapter 12b / RNN examples: chronological splits and training-only scaling.
- Assignment 5 electricity benchmark: shared metric/export layout. Its RMSE
  helper omitted the square root; this implementation computes RMSE correctly.
- Chapter 7 factor-model lecture: shrinkage and correlated predictors.

# Step 3 walkthrough and reading guide

## What this experiment does

This experiment uses the frozen Step 1 data and Step 2 publication rules to
forecast New York State unemployment at horizons of 1, 3 and 6 months. Each
month-end forecast uses only information published by that origin. Developed on
`step3-ml-validation`, this experiment preserves the Step 1 data and Step 2 results.

## Suggested reading order

1. [Models and parameter grids](../modeling/src/experiments/step3_ml/models.py).
2. [Features and publication controls](../modeling/src/experiments/step3_ml/features.py).
3. [Validation, selection and rolling forecasts](../modeling/src/experiments/step3_ml/run.py).
4. [Technical handoff and reproduction instructions](STEP3_HANDOFF.md).
5. [Model comparison chart](../modeling/results/experiments/step3_ml/model_comparison.png).
6. [Actual and forecast paths](../modeling/results/experiments/step3_ml/forecast_paths.png).
7. [Full error tables](../modeling/results/experiments/step3_ml/metrics.csv).
8. [Current forecasts](../modeling/results/experiments/step3_ml/live_forecasts.csv).

## How the models work

- Ridge penalizes linear regression coefficients to reduce overfitting with correlated predictors.
- Lasso also penalizes coefficients and can set some of them to zero.
- Random Forest averages constrained decision trees to learn nonlinear relationships.
- Gradient Boosting sequentially fits small trees to correct errors, using Huber loss to reduce sensitivity to extreme residuals.
- `ridge_ar_only` uses only the latest published unemployment level, its change and its publication age as a reference model.
- `ridge_regional` adds NJ, CT and PA unemployment and the Empire manufacturing survey to the core predictors.

The response is the target unemployment rate minus the latest unemployment rate
published at the forecast origin. After predicting that change, the model adds
the published baseline back. For example, a latest published rate of 4.3% and a
predicted change of -0.1 percentage points produce a forecast of 4.2%.

## Validation safeguards

- Select parameters on 2016-2019 validation targets, then freeze them before retrospective test evaluation.
- Refit at each origin using only historical target labels already published then.
- Recompute NY claims seasonal adjustment separately at each historical origin.
- Fit scaling only on the eligible training sample.
- Compare models on identical target months, and report coverage separately.
- Leave October 2025 actuals missing. Do not fill differences that cross the missing month.

## How to explain the results

For post-COVID targets with a published actual and a forecast from all 12 Step 2
and Step 3 model variants:

| Horizon | Validation-selected ML model | RMSE in percentage points | Reduction relative to naive |
|---|---|---:|---:|
| 1 month | Regional Ridge | 0.1576 | Approximately 23.6% |
| 3 months | Regional Ridge | 0.2616 | Approximately 32.1% |
| 6 months | Gradient Boosting | 0.6298 | Approximately 17.5% |

The naive forecast holds the latest published unemployment rate constant.
Lower RMSE is better; 0.1576 is measured in percentage points, not 15.76%.

These results do not show that ML universally beats traditional models. Some
Step 2 models remain better on the same sample. For calmer targets from 2023
onward, the selected 3- and 6-month ML models underperform naive. The models can
capture some economic changes but may overpredict movement in stable periods.

`feature_ablation.csv` compares target-only Ridge with expanded Ridge models on
matching evaluation months. Training spans and selected penalties can differ,
so this is a comparison of predictive specifications, not a causal estimate of
the contribution of an individual indicator.

## How to read the current forecasts

Filter `live_forecasts.csv` to `recommended=True` and `origin=2026-09-30`:

- October 2026: approximately 4.16%.
- December 2026: approximately 4.07%.
- March 2027: approximately 4.16%.

These are forecasts using information available at the end of September 2026,
not observed outcomes. Bounds use validation errors, not test errors. Because
the validation period was relatively calm, the intervals do not describe the
risk of a pandemic-scale shock.

## Limitations

Historical values reflect later revisions rather than original publication
vintages. Some release dates use estimated rules, and the team has previously
examined the test history. This is a retrospective pseudo-real-time experiment,
not evidence from an untouched test. Keep the model specifications fixed and
evaluate subsequent unemployment releases prospectively.

## Running the experiment

Follow the environment setup in [the handoff](STEP3_HANDOFF.md), then run from
the `modeling` directory with the repository's environment activated:

```sh
python -m src.experiments.step3_ml.run
```

The first run constructs historical features. Subsequent runs reuse caches only
when their hashes match. Changes to the model code or relevant inputs invalidate
the frozen selection lock; use a separately named experiment for a new model
specification rather than overwriting the existing selection.

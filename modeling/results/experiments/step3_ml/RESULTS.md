# Step 3 results

Retrospective pseudo-real-time benchmark. All errors are percentage points.
Selections use 2016–2019 only; test history was already inspected by the team.

## Models selected on validation

| Horizon | Model | Post-COVID common n | RMSE | MAE | RMSE / naive |
|---|---|---|---|---|---|
| 1 | ridge_regional | 50 | 0.1576 | 0.1071 | 0.764 |
| 3 | ridge_regional | 50 | 0.2616 | 0.2005 | 0.679 |
| 6 | gradient_boosting | 51 | 0.6298 | 0.3845 | 0.825 |

These are validation-selected models, not the best models chosen after seeing test errors.

## Reproduction and interpretation

- Run from `modeling/`: `../.venv/bin/python -m src.experiments.step3_ml.run`.
- `metrics.csv` distinguishes common-sample comparisons from own-available coverage.
- `coverage.csv` reports missing forecasts and unpublished actuals separately.
- `feature_ablation.csv` compares macro/regional Ridge against target-history-only Ridge on identical evaluation keys. Training spans and validation-selected penalties differ; this is a predictive comparison, not an isolated feature-effect estimate.
- `live_forecasts.csv` follows the Step 2 date/model/horizon/yhat/lower/upper/origin/actual/period schema.
- Live intervals use only validation residual quantiles; no post-COVID outcomes calibrate them.
- October 2025 is never imputed or scored. Features crossing the gap remain missing.
- COVID training exclusions and input clipping follow the frozen Step 2 experiment policy.
- Values are revised snapshots; some release dates are estimated. This is not a historical-vintage backtest.
- Pre-selection COVID origins are flagged; `post_selection_covid` is an additional evaluation view.
- Feature coefficients and impurity importance are descriptive, not causal evidence.
- Indeed and quarterly taxable sales are excluded: no compatible long validation history / release audit.
- Reserve new releases for prospective confirmation; do not retune on the reported test ranking.

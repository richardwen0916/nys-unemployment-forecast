# Observed model results

All errors below are NYUR percentage points. Tables use the same scored target months across the four models within each horizon. Full precision, MAE, bias and coverage are retained in the linked canonical CSVs. No parameters or features were changed in response to test errors.

## Validation: 2016–2019

| Model | 1M | 3M | 6M |
|---|---:|---:|---:|
| Naive | 0.1620 | 0.2358 | 0.3192 |
| ARIMA | **0.1387** | **0.2200** | **0.3070** |
| Core ARX | 0.1647 | 0.2839 | 0.4460 |
| Macro ARX | 0.1604 | 0.2767 | 0.4283 |

Source: [validation_metrics.csv](results/modeling/validation_metrics.csv), 48 common targets per horizon; [selected specifications](results/modeling/selected_specifications.csv). ARIMA(0,1,2) had the lowest validation RMSE at all three horizons. Those outcomes were used for model selection.

## Post-COVID test: January 2022–August 2026

| Model | 1M | 3M | 6M |
|---|---:|---:|---:|
| Naive | **0.286** | **0.502** | **0.881** |
| ARIMA | 0.315 | 0.537 | 0.916 |
| Core ARX | 1.021 | 0.983 | 1.240 |
| Macro ARX | 0.778 | 0.713 | 1.118 |

Source: [test_metrics.csv](results/modeling/test_metrics.csv), `period=post_covid`, `evaluation_view=requested_full_period`, `sample=common_four_models`. Common targets: 51, 51 and 52 for horizons 1, 3 and 6. There are 56 planned target months and 55 observed NYUR actuals, with October 2025 missing. Prediction coverage differs from common scored coverage; see [coverage_summary.csv](results/modeling/coverage_summary.csv).

## COVID and combined tests

COVID 2020–2021 RMSE (1M / 3M / 6M), with 24 common targets at each horizon:

- Naive: **4.399 / 5.235 / 5.834**.
- ARIMA: 5.635 / 6.543 / 7.148.
- Core ARX: 7.252 / 10.108 / 14.124.
- Macro ARX: 6.359 / 8.782 / 13.191.

Combined January 2020–August 2026 RMSE: Naive **2.499 / 2.990 / 3.358**; ARIMA 3.198 / 3.728 / 4.087; Core ARX 4.188 / 5.775 / 8.003; Macro ARX 3.654 / 5.003 / 7.470. Common counts are 75 / 75 / 76 out of 79 observed target months. The same [official project test metric file](results/modeling/test_metrics.csv) verifies every value; filter `period=covid` or `combined` and the requested-full-period view. [test_forecasts.csv](results/modeling/test_forecasts.csv) is the underlying forecast-level evidence, copied unchanged from the full project.

Naive had the lowest test MAE and RMSE at all horizons in COVID, post-COVID and combined periods. Neither ARX improved RMSE over either benchmark. Macro ARX generally reduced error relative to Core ARX, but its payroll lag differs at horizons 1 and 3, so these differences do not isolate USUR's contribution or establish causality. One narrower exception: Macro ARX's COVID 1M MAE beats ARIMA (3.272 versus 3.442), while still losing to Naive and having worse RMSE.

**Severe extrapolation example:** at origin **June 30, 2020**, Core ARX's 6M forecast for **December 2020** was **71.428%**, versus actual **8.700%**. This is the frozen linear equation extrapolating far beyond its pre-crisis predictor range. It is not an observed or fabricated NYUR value. It remains in the errors, without clipping or retrospective repair. See [extreme forecasts](results/modeling/test_extreme_forecasts.csv) and [independently reconciled contributions](results/modeling/test_shock_forecast_contributions.csv).

## Limitations and interpretation

- **Pseudo-real-time:** release filtering cannot recover numerical vintages. Later revisions, benchmarks and SA revisions remain in the snapshot. Some publication dates are conservative estimates; verified public-by dates need not be the earliest possible release or FRED ingestion time.
- **STL:** each training-window estimate uses only then-available raw claims, but short-window and endpoint seasonal estimates are uncertain and can shift across origins.
- **Coverage:** missing predictors produce explicit no-forecast records. Across all four models there are 960 planned model/horizon/target records, of which 17 Direct ARX predictions are unavailable (943 finite forecasts overall). Common comparisons omit a target for all models when any model cannot issue a forecast; full counts are in the CSVs. Do not mistake issued-forecast coverage for scored coverage.
- **October 2025:** the NYUR actual remains NaN. All twelve forecasts are retained, none is scored. ARIMA keeps the missing monthly measurement; Direct ARX does not treat September-to-November as a monthly difference. Payroll release holdbacks and missing NYUR/USUR differences affect later forecasts.
- **Selection chronology:** the 2020–2026 values were reserved during original validation selection, but early COVID origins precede the January 31, 2020 validation information cutoff. The requested full COVID period is a retrospective frozen-specification boundary evaluation. A predeclared `post_selection_origins` view has COVID common counts 23 / 21 / 18 and also favors Naive. Do not mix the views or select a favorable one after seeing results.
- **Limited evidence:** validation has only 48 target months per horizon and a relatively calm regime. Multi-step errors overlap and are serially dependent; common-sample sizes differ across horizons and tests cover distinct regimes. No significance test or causal improvement claim follows from these rankings.
- **Future research:** these periods are now inspected. Reusing them for exploratory improvements is retrospective benchmarking, not a new untouched test. Preserve the frozen references and predeclare a new confirmation protocol.

## Figure provenance

[COVID comparison](results/modeling/test_model_comparison_covid.png), [post-COVID comparison](results/modeling/test_model_comparison_post_covid.png), and [combined comparison](results/modeling/test_model_comparison_combined.png) are the original project PNGs. Each plots RMSE from `test_metrics.csv`, `evaluation_view=requested_full_period`, the matching `period`, all models and horizons. Bars start at zero; sample counts appear in the panels. No duplicate SVGs, chart-data copies or dashboard are included.

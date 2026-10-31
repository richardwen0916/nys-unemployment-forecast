"""Static dashboard-ready chart exports and the final test report, never selection."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .validation_report import PALETTE,STYLES,NAMES

PERIOD_NAMES={'covid':'COVID: 2020–2021','post_covid':'Post-COVID: 2022–August 2026','combined':'Combined: 2020–August 2026'}


def figures(out,forecasts,metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    primary=metrics.loc[metrics.evaluation_view.eq('requested_full_period')]
    manifest=[]
    for period in ('covid','post_covid'):
        part=forecasts.loc[forecasts.test_period.eq(period)]
        fig,axes=plt.subplots(3,1,figsize=(14,11),sharex=True,constrained_layout=True)
        for h,ax in zip((1,3,6),axes):
            sub=part.loc[part.horizon.eq(h)].sort_values('target_month')
            actual=sub.drop_duplicates('target_month').set_index('target_month').actual.asfreq('MS')
            ax.plot(actual.index,actual,color='#161616',lw=2.5,label='Actual NYUR',zorder=6)
            for model in ('M0','M1','M2','M3'):
                group=sub.loc[sub.model.eq(model)].set_index('target_month').asfreq('MS')
                ax.plot(group.index,group.yhat,color=PALETTE[model],ls=STYLES[model],lw=1.6,label=f'{model}: {NAMES[model]}')
            ax.set_title(f'{h}-month horizon | all issued forecasts; unavailable predictions remain gaps',fontsize=11)
            ax.set_ylabel('NYUR (%)');ax.grid(axis='y',alpha=.18)
            if period=='post_covid':
                ax.axvspan(pd.Timestamp('2025-10-01'),pd.Timestamp('2025-11-01'),color='#888888',alpha=.14)
            ax.xaxis.set_major_locator(mdates.YearLocator() if period=='post_covid' else mdates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y' if period=='post_covid' else '%b %Y'))
        axes[0].legend(ncol=3,fontsize=9,loc='best')
        axes[-1].set_xlabel('Target reference month; grey shading = missing October 2025 actual' if period=='post_covid' else 'Target reference month')
        caveat='Includes early-2020 pre-selection boundary origins' if period=='covid' else 'October 2025 forecasts retained; missing actual is never scored'
        fig.suptitle(f'{PERIOD_NAMES[period]} | frozen-model forecasts\nPseudo-real-time • {caveat}',fontsize=14)
        name=f'test_actual_vs_predicted_{period}'
        for extension in ('png','svg'):
            fig.savefig(out/f'{name}.{extension}',dpi=150,bbox_inches='tight')
        plt.close(fig)
        manifest.append(dict(chart=name,kind='line_small_multiples',data='test_forecasts.csv',period=period,
                             files=[name+'.png',name+'.svg'],metric='NYUR level, percent',missing_policy='Preserve line gaps, including October 2025 actual'))
    hatches={'M0':'','M1':'//','M2':'..','M3':'xx'}
    for period in ('covid','post_covid','combined'):
        fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True,sharey=True)
        sub=primary.loc[primary.period.eq(period)]
        limit=float(sub.rmse.max())*1.24
        for h,ax in zip((1,3,6),axes):
            group=sub.loc[sub.horizon.eq(h)].set_index('model').reindex(['M0','M1','M2','M3'])
            bars=ax.bar(range(4),group.rmse,color=[PALETTE[m] for m in group.index],edgecolor='#333333',lw=.5)
            for bar,model,value in zip(bars,group.index,group.rmse):
                bar.set_hatch(hatches[model]);ax.text(bar.get_x()+bar.get_width()/2,value+limit*.018,f'{value:.3f}',ha='center',fontsize=9)
            ax.set_xticks(range(4),['Naive','ARIMA','Core ARX','Macro ARX'],rotation=15)
            ax.set_ylim(0,limit);ax.set_title(f'Horizon {h} | common n={int(group.n_scored.iloc[0])}/{int(group.n_actual.iloc[0])} observed')
            ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        axes[0].set_ylabel('RMSE (percentage points; lower is better)')
        boundary='\nIncludes early-2020 pre-selection boundary origins' if period in ('covid','combined') else ''
        fig.suptitle(f'{PERIOD_NAMES[period]} | frozen-model comparison\nSame scored targets within each horizon • pseudo-real-time'+boundary,fontsize=13)
        name=f'test_model_comparison_{period}'
        for extension in ('png','svg'):
            fig.savefig(out/f'{name}.{extension}',dpi=150,bbox_inches='tight')
        plt.close(fig)
        manifest.append(dict(chart=name,kind='bar_small_multiples',data='test_metrics.csv',period=period,
                             evaluation_view='requested_full_period',sample='common_four_models',
                             files=[name+'.png',name+'.svg'],metric='RMSE, percentage points; zero baseline'))
    # Tidy, machine-readable records are the canonical inputs for later dashboard work.
    primary.to_csv(out/'test_chart_comparison_data.csv',index=False)
    forecasts[['forecast_origin','target_month','horizon','test_period','model','spec_id','actual','yhat','status','common_scored','origin_after_validation_cutoff']].to_csv(out/'test_chart_forecast_data.csv',index=False)
    (out/'test_chart_manifest.json').write_text(json.dumps(dict(charts=manifest,
        palette=PALETTE,model_names=NAMES,scope='Frozen-specification final test evaluation; no dashboard constructed',
        scoring='Use explicit evaluation_view and sample; do not mix own and common samples'),indent=2)+'\n')


def write_report(out,forecasts,metrics):
    out=Path(out)
    figures(out,forecasts,metrics)
    primary=metrics.loc[metrics.evaluation_view.eq('requested_full_period')]
    strict=metrics.loc[metrics.evaluation_view.eq('post_selection_origins')]
    lock=json.loads((out/'test_frozen_specifications.json').read_text())
    lines=['# Frozen-model out-of-sample evaluation','',
        '**Final evaluation of validation-selected specifications; no test-driven model selection.**',
        '', 'Forecasts are pseudo-real-time views of a revised data snapshot. COVID targets are January 2020–December 2021; post-COVID targets are January 2022–August 2026, the latest NYUR reference month available in the supplied panel as of October 19, 2026.',
        '', '## Test findings on common targets', '']
    minimum=primary.loc[primary.groupby(['period','horizon']).rmse.idxmin()]
    if minimum.model.eq('M0').all():
        lines += ['**Naive has the lowest common-sample RMSE at every horizon in COVID, post-COVID and the combined test period. Neither economic-indicator specification improves RMSE over Naive or ARIMA.** This reverses the validation ranking, where ARIMA was strongest.', '']
    for period in ('covid','post_covid','combined'):
        lines += [f'### {PERIOD_NAMES[period]}','',
            '| Horizon | Model | Common n | Forecasts / planned | MAE (pp) | RMSE (pp) | Bias (pp) | RMSE gain vs Naive | RMSE gain vs ARIMA |',
            '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
        for row in primary.loc[primary.period.eq(period)].sort_values(['horizon','model']).itertuples():
            lines.append(f'| {row.horizon} | {NAMES[row.model]} | {row.n_scored} | {row.n_forecasts}/{row.n_planned} | {row.mae:.3f} | {row.rmse:.3f} | {row.bias:+.3f} | {row.rmse_improvement_vs_naive_pct:+.1f}% | {row.rmse_improvement_vs_arima_pct:+.1f}% |')
        lines.append('')
        for h in (1,3,6):
            sub=primary.loc[primary.period.eq(period)&primary.horizon.eq(h)].set_index('model')
            core=sub.loc['M2'];macro=sub.loc['M3']
            best=sub.rmse.idxmin()
            def result(row):
                a='improves' if row.rmse_improvement_vs_naive_pct>0 else 'does not improve'
                b='improves' if row.rmse_improvement_vs_arima_pct>0 else 'does not improve'
                return f'{a} on Naive and {b} on ARIMA'
            lines.append(f'- Horizon {h}: lowest observed test RMSE is {NAMES[best]}. Core ARX {result(core)}; Macro ARX {result(macro)}.')
        lines.append('')
    largest=forecasts.assign(absolute_error_pp=(forecasts.yhat-forecasts.actual).abs()).nlargest(1,'absolute_error_pp').iloc[0]
    lines += ['## Shock extrapolation remains in the results', '',
        f"The largest error is {largest.model}, horizon {largest.horizon}, origin {largest.forecast_origin:%Y-%m-%d}: forecast {largest.yhat:.3f}% versus actual {largest.actual:.3f}% for {largest.target_month:%Y-%m}. Large COVID changes put the frozen linear equations far outside their pre-crisis predictor range. Forecasts were retained without clipping or retrospective repair. Independent unscaled OLS reconciliation and `test_shock_forecast_contributions.csv` reproduce the arithmetic; `test_extreme_forecasts.csv` preserves the largest errors.", '']
    lines += ['Positive RMSE gains indicate lower error on exactly the same scored rows. Bias is forecast minus actual: positive means overprediction. Coverage counts issued finite forecasts, including forecasts for the missing October 2025 actual; common accuracy excludes any row unavailable for one of the four models. `test_own_sample_metrics.csv` is supplemental and must not be used for unequal-sample rankings.',
        '', '## Validation findings versus test findings', '',
        'Validation selected ARIMA(0,1,2) at all horizons, with RMSE 0.139, 0.220 and 0.307 pp for 2016–2019. Core ARX did not beat the validation benchmarks; USUR offered modest improvements within the ARX variants. The tables above are the final test findings, without revising those earlier choices. Describing the smallest test error is not selecting a replacement specification.',
        '', 'M2 evaluates the Claims/Payroll bundle against Naive and ARIMA; these comparisons do not isolate either indicator individually or establish causality. M3 adds USUR, but its frozen payroll lag differs from M2 at horizons 1 and 3, so their difference cannot be attributed solely to USUR. At horizon 6 both use Payroll Lag 2 and the M3 extension is the USUR first difference.',
        '', '## Specification freeze and refitting', '',
        'The saved selection and full declaration files were hashed before test fitting in `test_frozen_specifications.json`; the 12 frozen model/horizon rows are in `test_frozen_specifications.csv`. Original validation artifacts remain unchanged. No candidate grid, selection function, parameter/lag/transform search, forecast clipping or fallback model was used.',
        '', '| Horizon | M0 | M1 | M2 | M3 |','|---:|---|---|---|---|']
    for h in (1,3,6):
        chosen={r['model']:r['spec_id'] for r in lock['specifications'] if r['horizon']==h}
        lines.append(f"| {h} | {chosen['M0']} | ARIMA(0,1,2) | {chosen['M2']} | {chosen['M3']} |")
    lines += ['', 'Orders, features, lags, transformations, trend, minimum sample and estimation policy stay frozen. Coefficients may be re-estimated on expanding published history. Earlier test outcomes can enter later fits only after publication; they never trigger a new specification choice. Each direct row retains its original historical predictors and baseline. OLS scaling uses only the eligible training rows. Historical Claims STL is estimated at the row’s own origin, never with the later fit’s information window.',
        '', 'The Phase 1 helper intentionally capped training labels at 2019. A separate out-of-sample adapter removes that validation-only ceiling while retaining the identical response, features, complete-case policy and forecasting functions. All original code remains unchanged.',
        '', '## Chronology caveat at the 2019/2020 boundary', '',
        'The validation scoring cutoff was January 31, 2020. January 2020 targets require December 2019, October 2019 and July 2019 origins at horizons 1, 3 and 6, before the chosen specifications could have been established from all validation outcomes. Those required boundary forecasts are retained and flagged `origin_after_validation_cutoff=False`. The full requested COVID period is therefore a retrospective frozen-specification test, not a claim that the final specification was prospectively known at every boundary origin.',
        '', 'A predeclared supplemental view restricts origins to January 31, 2020 or later. This changes no model or forecast; it only makes specification chronology explicit. COVID common-row results for that view:',
        '', '| Horizon | Common n | Naive RMSE | ARIMA RMSE | Core ARX RMSE | Macro ARX RMSE |','|---:|---:|---:|---:|---:|---:|']
    for h in (1,3,6):
        group=strict.loc[strict.period.eq('covid')&strict.horizon.eq(h)].set_index('model')
        lines.append(f"| {h} | {int(group.n_scored.iloc[0])} | {group.at['M0','rmse']:.3f} | {group.at['M1','rmse']:.3f} | {group.at['M2','rmse']:.3f} | {group.at['M3','rmse']:.3f} |")
    lines += ['', 'Both views are saved in `test_metrics.csv` under an explicit `evaluation_view`; do not combine or choose a view based on which favors a model. Post-COVID findings have no such boundary restriction.',
        '', '## October 2025 missing observation and shutdown audit', '',
        'October 2025 NYUR stays NaN in the original panel and the separate label bank. Its 12 model/horizon forecast records are retained in `test_october_2025_forecasts.csv` but are excluded from every accuracy calculation. No observed value is interpolated, fabricated or inferred from a forecast.',
        '', 'State-space ARIMA receives a regular monthly index. Once November 2025 is published, October remains an internal NaN between September and November; the Kalman filter propagates its latent state across the missing measurement. No row compression or insertion of a synthetic observed NYUR occurs. `test_arima_calendar_audit.csv` and `test_arima_missing_reference_months.csv` record the calendar and missing references at every test origin.',
        '', 'The original feature module computes differences on a complete monthly index before selecting the latest available observation. November minus October is NaN when October is missing; September-to-November is never substituted as a one-month difference. The same protection applies to USUR, whose October value is also missing. If a frozen predictor is missing at an origin, the corresponding Direct ARX forecast remains NaN with its missing feature names recorded; no carry-forward or alternative specification is used.',
        '', 'Origin-level coverage for September 2025 through January 2026:',
        '', '| Origin | Horizon | Model | Target | Forecast status | Affected frozen features |','|---|---:|---|---|---|---|']
    shutdown=forecasts.loc[forecasts.forecast_origin.between('2025-09-30','2026-01-31')]
    for row in shutdown.loc[shutdown.status.ne('ok')].sort_values(['forecast_origin','horizon','model']).itertuples():
        missing=', '.join(json.loads(row.missing_current_features))
        lines.append(f'| {row.forecast_origin:%Y-%m-%d} | {row.horizon} | {row.model} | {row.target_month:%Y-%m} | {row.status} | {missing} |')
    lines += ['', 'All other origin/model/horizon rows in this five-origin audit issued finite forecasts. Payroll release holdbacks and USUR/NYUR missing first differences cause different exclusions by frozen specification; these are availability effects, not retuning decisions. `test_shutdown_feature_audit.csv`, `test_shutdown_reference_audit.csv` and `test_shutdown_forecasts.csv` show the available levels, calendar differences, fixed lags and forecast statuses. The complete exclusion file additionally records affected later origins and common-sample exclusions for otherwise valid benchmark forecasts.',
        '', '## Plots and dashboard-ready comparisons', '',
        'Actual-versus-predicted figures retain forecasts for missing actuals and show unavailable predictions as line gaps. Grey shading marks the October 2025 missing actual. Comparison bars start at zero and use common-scored RMSE within each horizon. COVID charts use the full requested period, with the boundary caveat above.',
        '', '![COVID forecasts](test_actual_vs_predicted_covid.png)',
        '', '![Post-COVID forecasts](test_actual_vs_predicted_post_covid.png)',
        '', '![COVID model comparison](test_model_comparison_covid.png)',
        '', '![Post-COVID model comparison](test_model_comparison_post_covid.png)',
        '', '![Combined model comparison](test_model_comparison_combined.png)',
        '', 'Each figure is supplied as PNG and SVG, with tidy chart inputs and `test_chart_manifest.json` for later Streamlit embedding. No dashboard has been built.',
        '', '## Limitations and reproducibility', '',
        'The panel contains later numerical revisions, benchmarks and seasonal adjustments. Calendar-first verified dates plus estimated conservative fallbacks define this pseudo-real-time policy; unknown historical release dates may still differ. Claims STL endpoint estimates can shift inside each available window. Frozen linear equations are allowed to extrapolate during COVID; forecasts are not clipped or repaired after seeing errors. Common-case comparisons omit targets where a frozen ARX cannot issue a forecast, so coverage is part of the result and own-sample metrics are separately identified.',
        '', 'No test outcome was used to change orders, features, lags, transformations, scaling policy or selection rules. Do not treat favorable test results as permission to select another specification on this same test set. All original datasets, release metadata, code and validation outputs are preserved; nothing is pushed to GitHub.',
        '', 'Reproduce from the project root:', '', '```sh',
        'PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR="$PWD/work/out_of_sample/matplotlib" python -m src.models.out_of_sample',
        'PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v','```',
        '', 'Frozen provenance, training eligibility, coefficients, fit warnings, excluded rows, historical features/labels and chart inputs are saved with the `test_` prefix. Test logs, preservation checks and the exact file inventory are saved separately with the `out_of_sample_` prefix to preserve Phase 1 outputs.']
    checks=out/'out_of_sample_validation_checks.json'
    tests=out/'out_of_sample_test_results.txt'
    if checks.exists():
        verified=json.loads(checks.read_text())
        lines += ['', '## Verification and files', '',
            f"Independent checks reconciled {verified['independent_direct_prediction_reconciliations']} finite direct-model forecasts, all {verified['all_metric_rows_recomputed']} common-metric rows, scaler parameters, publication cutoffs, horizon alignment, the immutable validation-feature prefix and all 87 pre-existing protected files. Missing predictor exclusions are intentional; no model-fit failure was replaced by a fallback."]
    if tests.exists():
        import re
        match=re.search(r'Ran (\d+) tests',tests.read_text())
        if match and tests.read_text().rstrip().endswith('OK'):
            lines += ['', 'Baseline: 65 tests passed. Final: '+match.group(1)+' existing and new tests passed; see `out_of_sample_test_results.txt`.']
    lines += ['', 'New source files: `src/models/out_of_sample.py`, `src/models/out_of_sample_report.py`, and `tests/test_out_of_sample.py`. The exact source/output inventory and preservation evidence are `out_of_sample_file_inventory.csv` and `out_of_sample_preservation_check.json`. No pre-existing file was changed.']
    (out/'OUT_OF_SAMPLE_REPORT.md').write_text('\n'.join(lines)+'\n')

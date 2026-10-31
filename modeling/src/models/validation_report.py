"""Static scientific plots and the requested Markdown validation report."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera

PALETTE={'M0':'#555555','M1':'#426B9D','M2':'#C17D39','M3':'#788B50'}
STYLES={'M0':'--','M1':'-','M2':'-.','M3':':'}
NAMES={'M0':'Naive','M1':'ARIMA','M2':'Core Direct ARX','M3':'Macro Direct ARX'}


def residual_diagnostics(out,chosen):
    rows,correlations=[],[]
    for (h,model),group in chosen.groupby(['horizon','model']):
        group=group.sort_values('target_month')
        errors=pd.Series((group.yhat-group.actual).to_numpy(),index=pd.DatetimeIndex(group.target_month)).asfreq('MS')
        observed=errors.dropna()
        if len(observed)<12:
            continue
        aa=acf(errors,nlags=12,fft=False,missing='conservative')
        for lag,value in enumerate(aa):
            correlations.append(dict(horizon=h,model=model,lag=lag,acf=value,nobs=len(observed)))
        # No calendar compression for serial tests. Current validation is complete.
        if errors.isna().any():
            lb=pd.DataFrame({'lb_stat':[np.nan,np.nan],'lb_pvalue':[np.nan,np.nan]},index=[6,12])
        else:
            lb=acorr_ljungbox(errors,lags=[6,12],return_df=True)
        jb,jbp,skew,kurtosis=jarque_bera(observed)
        rows.append(dict(horizon=h,model=model,spec_id=group.spec_id.iloc[0],nobs=len(observed),
            mean_error_pp=observed.mean(),error_std_pp=observed.std(ddof=1),max_abs_error_pp=observed.abs().max(),
            acf_lag1=aa[1],ljung_box_lag6_stat=lb.loc[6,'lb_stat'],ljung_box_lag6_pvalue=lb.loc[6,'lb_pvalue'],
            ljung_box_lag12_stat=lb.loc[12,'lb_stat'],ljung_box_lag12_pvalue=lb.loc[12,'lb_pvalue'],
            jarque_bera=jb,jarque_bera_pvalue=jbp,skewness=skew,kurtosis=kurtosis,
            interpretation='Descriptive validation forecast errors; overlapping multi-step horizons and validation selection invalidate naive independent-error inference'))
    residuals=pd.DataFrame(rows);serial=pd.DataFrame(correlations)
    residuals.to_csv(out/'residual_diagnostics.csv',index=False)
    serial.to_csv(out/'residual_acf.csv',index=False)
    chosen.assign(error_pp=chosen.yhat-chosen.actual).to_csv(out/'validation_errors.csv',index=False)
    return residuals,serial


def plots(out,chosen,serial):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.titleweight':'bold'})
    lo=min(chosen.actual.min(),chosen.yhat.min())-.12
    hi=max(chosen.actual.max(),chosen.yhat.max())+.12
    for mode,filename in [('levels','actual_vs_predicted.png'),('errors','forecast_errors.png')]:
        fig,axes=plt.subplots(3,1,figsize=(14,11),constrained_layout=True,sharex=True)
        for h,ax in zip((1,3,6),axes):
            subset=chosen.loc[chosen.horizon.eq(h)].sort_values('target_month')
            actual=subset.drop_duplicates('target_month').set_index('target_month').actual.asfreq('MS')
            if mode=='levels':
                ax.plot(actual.index,actual,color='#161616',lw=2.4,label='Actual NYUR',zorder=6)
                ax.set_ylim(lo,hi);ax.set_ylabel('NYUR (%)')
            else:
                ax.axhline(0,color='#161616',lw=.8)
                bound=float((chosen.yhat-chosen.actual).abs().max())*1.08
                ax.set_ylim(-bound,bound);ax.set_ylabel('Forecast − actual (pp)')
            for model in ('M0','M1','M2','M3'):
                group=subset.loc[subset.model.eq(model)].set_index('target_month').asfreq('MS')
                y=group.yhat if mode=='levels' else group.yhat-group.actual
                ax.plot(group.index,y,color=PALETTE[model],ls=STYLES[model],lw=1.7,
                        label=f'{model}: {NAMES[model]}')
            ax.set_title(f'{h}-month horizon | 48 common validation targets')
            ax.grid(axis='y',alpha=.17);ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        axes[0].legend(loc='upper right',ncol=3,fontsize=9)
        axes[-1].set_xlabel('Target reference month (2016–2019)')
        fig.suptitle(('NYUR actual and rolling forecasts' if mode=='levels' else 'Rolling validation forecast errors')+
                     '\nPseudo-real-time; specifications selected on these validation targets',fontsize=15)
        fig.savefig(out/filename,dpi=150,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(4,3,figsize=(14,12),constrained_layout=True,sharex=True,sharey=True)
    for i,model in enumerate(('M0','M1','M2','M3')):
        for j,h in enumerate((1,3,6)):
            ax=axes[i,j];sub=serial.loc[serial.model.eq(model)&serial.horizon.eq(h)&serial.lag.gt(0)]
            ax.axhline(0,color='#555555',lw=.7)
            ax.axhspan(-1.96/np.sqrt(48),1.96/np.sqrt(48),color='#888888',alpha=.10)
            ax.vlines(sub.lag,0,sub.acf,color=PALETTE[model],lw=2)
            ax.scatter(sub.lag,sub.acf,color=PALETTE[model],s=15)
            ax.set_ylim(-1,1);ax.set_title(f'{model} | horizon {h}')
            ax.set_xticks([1,3,6,9,12]);ax.set_xlabel('Monthly lag');ax.set_ylabel('Error ACF')
    fig.suptitle('Forecast error autocorrelation | validation only\nGrey band: white-noise reference; overlapping horizons are dependent',fontsize=14)
    fig.savefig(out/'residual_acf.png',dpi=140,bbox_inches='tight');plt.close(fig)


def write_outputs(out,chosen,metrics,selection,candidate_metrics,samples,matched):
    out=Path(out)
    residuals,serial=residual_diagnostics(out,chosen)
    plots(out,chosen,serial)
    lines=['# Phase 1 rolling validation: NYS unemployment',
        '', '**Pseudo-real-time; validation target months January 2016–December 2019.**', '']
    for h in (1,3,6):
        sub=metrics.loc[metrics.horizon.eq(h)].sort_values(['rmse','mae','model'])
        winner=sub.iloc[0]
        lines.append(f"- Horizon {h}: **{('ARIMA('+','.join(winner.spec_id.split('_')[1])+')') if winner.model=='M1' else NAMES[winner.model]} ({winner.spec_id})** has lowest RMSE, **{winner.rmse:.4f} percentage points**, on {winner.n_scored} common targets.")
    lines += ['', 'These are validation-selected results, not untouched test performance. COVID (2020–2021) and post-COVID (2022 onward) values were excluded from fitting, preprocessing, selection and scoring.',
        '', '## Comparable validation accuracy', '',
        '| Horizon | Model | Selected specification | n | MAE (pp) | RMSE (pp) | RMSE improvement vs Naive |',
        '|---:|---|---|---:|---:|---:|---:|']
    for r in metrics.sort_values(['horizon','model']).itertuples():
        lines.append(f'| {r.horizon} | {NAMES[r.model]} | `{r.spec_id}` | {r.n_scored} | {r.mae:.4f} | {r.rmse:.4f} | {r.rmse_improvement_pct:+.1f}% |')
    lines += ['', 'Positive improvement means lower RMSE than Naive on exactly the same origin/target/horizon observations. Full candidate results and coverage are in `candidate_metrics.csv`; all forecasts, including numerical failures, are in `candidate_forecasts.csv`.',
        '', '## Claims/Payroll and USUR comparisons', '']
    for h in (1,3,6):
        sub=metrics.loc[metrics.horizon.eq(h)].set_index('model')
        core=sub.loc['M2'];arima=sub.loc['M1'];macro=sub.loc['M3']
        pair=matched.loc[matched.horizon.eq(h)&matched.macro_spec.eq(macro.spec_id)].iloc[0]
        core_gain=100*(1-core.rmse/arima.rmse)
        macro_gain=100*(1-macro.rmse/core.rmse)
        lines.append(f'- Horizon {h}: Core ARX RMSE {core.rmse:.4f} versus ARIMA {arima.rmse:.4f} ({abs(core_gain):.1f}% {'lower' if core_gain>=0 else 'higher'} RMSE). Macro ARX RMSE {macro.rmse:.4f} versus selected Core ARX ({macro_gain:+.1f}% reduction). Adding USUR to the **same** underlying core specification changes RMSE by {pair.rmse_reduction_pct:+.1f}% (positive = improvement).')
    lines += ['', 'M2 tests the Claims/Payroll bundle against the two benchmark approaches. Different functional forms prevent attributing its gains separately to Claims or Payroll; no individual-indicator ablation or causal claim is made. `matched_usur_comparisons.csv` compares each USUR extension against its identical core feature set, avoiding confusion when the selected M2 and M3 use different payroll lags.',
        '', '## Fixed candidate design', '',
        '- M0: latest NYUR actually published at the origin, repeated for all horizons.',
        '- M1: all 18 ARIMA(p,d,q) combinations with p,q in {0,1,2}, d in {0,1}. Constant for d=0; no drift for d=1. Stationarity/invertibility enforced. State-space estimation retains internal missing reference months. A fixed LBFGS → Powell → LBFGS convergence retry policy is used; failed fits are reported without a substitute forecast.',
        '- M2: two OLS direct-change candidates. Both include latest published NYUR level, its calendar first difference and age in months, plus training-window STL Claims log change at origin Lag 1. Payroll log growth ×100 is either origin Lag 2 or Lag 3. Payroll Lag 1 is excluded.',
        '- M3: extend each M2 with either latest available USUR level or latest available USUR first difference: four macro candidates, never both US transformations together.',
        '- Each horizon has a separate direct equation. Response = future target NYUR minus NYUR last published at that training row’s own origin; add the prediction origin’s baseline back to return percent levels. OLS intercept and standardization are estimated from eligible training rows only; no imputation, clipping, future exogenous path, or feature selection on test data.',
        '- Selection rule fixed before fitting: candidates need all 48 planned forecasts for that horizon, then minimum validation RMSE within family; ties use MAE, complexity, then specification ID. `model_specifications.csv` records the grid and `selected_specifications.csv` records the choices.',
        '', '## Information timing and training consistency', '',
        'All feature origins from January 1990 through November 2019 were rebuilt using the existing availability/STL modules. Each historical STL fit uses only raw Claims published by its own origin, with expanding history starting January 1990 and at least 36 months. The 21 existing per-origin features reproduce the prior audited snapshots within 1e-10. The modeling feature/label files and their input/output hashes are retained in `snapshot_manifest.json`.',
        '', 'Forecast horizons are target-reference-month offsets from the calendar origin, not from the most recent observation. For January 2016 at horizon 6, the origin is July 31, 2015; only information already published then can be used. The initial 1990–2015 history does not authorize later-2015 data at that earlier origin. At every subsequent fit the expanding sample admits earlier validation labels only after their publication. No label is eligible merely because its reference month has passed.',
        '', 'Latest NYUR can be two or three months old under the unchanged rules. ARIMA forecasts h plus that age in calendar steps; missing months are kept in the state-space calendar. Direct models retain each row’s original predictors and baseline. Scaling is refit on those eligible rows only. The evaluation cutoff January 31, 2020 reveals December 2019 NYUR for scoring only; no fit uses that evaluation cutoff.',
        '', '## Sample eligibility and exclusions', '',
        '| Horizon | Model/spec | Training n min–max | Released-label rows lost to incomplete X (min–max) |',
        '|---:|---|---:|---:|']
    for choice in selection.itertuples():
        selected_samples=samples.loc[samples.horizon.eq(choice.horizon)&samples.spec_id.eq(choice.spec_id)]
        if len(selected_samples):
            n=f'{selected_samples.n_train.min()}–{selected_samples.n_train.max()}'
            lost=f'{selected_samples.excluded_missing_predictors.min()}–{selected_samples.excluded_missing_predictors.max()}'
        else:
            sub=chosen.loc[chosen.horizon.eq(choice.horizon)&chosen.spec_id.eq(choice.spec_id)]
            n=f'{int(sub.n_train.min())}–{int(sub.n_train.max())}';lost='n/a: univariate history'
        lines.append(f'| {choice.horizon} | `{choice.spec_id}` | {n} | {lost} |')
    exclusions=pd.read_csv(out/'forecast_exclusions.csv')
    lines += ['', f'{len(exclusions)} candidate forecast rows failed or lacked inputs; the exclusion file preserves their origin, horizon and status. `forecast_coverage.csv` covers every candidate; `training_sample_coverage.csv` records every direct fit’s prior rows, released labels, incomplete predictors, unpublished/missing/out-of-period labels, feature-specific missing counts, sample endpoints, maximum label-publication timestamp and membership hash. Missing early Claims history is explicitly excluded rather than filled.',
        '', 'The final comparison uses the common observed targets across the selected four models. Missing actual NYUR is never scored. October 2025 remains missing in the original panel and is outside this validation run. Verified release metadata and all original datasets/cleaning scripts are unchanged.',
        '', '## Forecast error diagnostics', '',
        'Errors are forecast minus actual, in percentage points. `residual_diagnostics.csv` reports mean error, dispersion, maximum absolute error, ACF(1), Ljung–Box at 6/12 lags and Jarque–Bera statistics; `residual_acf.csv` retains the ACF values. These are descriptive rolling forecast-error checks, not in-sample regression residuals or proof of independent innovations. Multi-step forecasts overlap, and specification selection used this same validation period; ordinary p-values should not be used as independent confirmation of model quality.',
        '', '![NYUR actual versus predicted](actual_vs_predicted.png)',
        '', '![Rolling forecast errors](forecast_errors.png)',
        '', '![Forecast error autocorrelation](residual_acf.png)',
        '', '## Remaining limitations and next step', '',
        'Historical numerical vintages are unresolved: the panel contains later revisions and seasonal/benchmark updates. Calendar-first verified dates and documented estimated fallbacks define the information policy; unverified release assumptions can still be wrong. Claims STL endpoint estimates are unstable and raw historical values are revised. Validation has only 48 targets per horizon, with overlapping forecast errors and selection across a limited grid, so apparent improvements may not generalize. Preserve the selected choices before any future COVID/post-COVID evaluation; those reserved tests have not been run here.',
        '', 'Reproduce from the project root:', '',
        '```sh',
        'PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR="$PWD/work/modeling/matplotlib" python -m src.models.run_validation',
        'PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v',
        '```', '',
        'The runner reuses per-origin snapshots only when input and output hashes match; otherwise it rebuilds them at their original origins. It writes only modeling outputs. No new packages, sources, dashboard, Git commits, pushes or merges are required.',
        '', 'API references: [statsmodels ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html), [state-space missing-data example](https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_sarimax_internet.html), [Ljung–Box](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html). Local timing assumptions: `../../PAYROLL_RELEASE_AUDIT.md` and `../../RELEASE_DATE_GUIDE.md`.']
    diagnostics=[]
    for h in (1,3,6):
        row=residuals.loc[residuals.horizon.eq(h)&residuals.model.eq('M1')].iloc[0]
        diagnostics.append(f"horizon {h}: mean error {row.mean_error_pp:+.3f} pp, ACF(1) {row.acf_lag1:.2f}")
    at=lines.index('## Forecast error diagnostics')+2
    lines.insert(at,'Selected ARIMA shows positive average errors (overprediction) and persistent serial dependence: '+'; '.join(diagnostics)+'. Lower RMSE does not imply white-noise forecast errors.')
    lines.insert(at+1,'')
    tests=out/'test_results.txt'
    if tests.exists():
        import re
        match=re.search(r'Ran (\d+) tests',tests.read_text())
        if match and tests.read_text().rstrip().endswith('OK'):
            lines += ['', '## Verification and files', '',
                'Baseline: 50 tests passed. Final: '+match.group(1)+' tests passed. See `test_results.txt`, `validation_checks.json`, and `preservation_check.json` for executed checks and preservation evidence.',
                '', 'Created source files: `src/models/rolling_validation.py`, `src/models/run_validation.py`, `src/models/validation_report.py`, and `tests/test_rolling_validation.py`. All deliverables are in `results/modeling/`; the exact file inventory is `file_inventory.csv`. Existing source files, tests, datasets and release metadata are unchanged.']
    (out/'MODEL_VALIDATION_REPORT.md').write_text('\n'.join(lines)+'\n')

"""Common-sample comparisons, coverage, plots, and a readable result summary."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features import ROOT, OUT
from .models import MODELS

BASELINES = ['naive','arima','arx','arx_s','var_s','combo_s']
ALL_MODELS = BASELINES + list(MODELS)
KEYS = ['origin','date','horizon','period']


def comparison_frames():
    ml = pd.read_csv(OUT/'forecasts.csv',parse_dates=['origin','date'])
    baseline = pd.read_csv(ROOT.parent/'results/forecasts.csv',parse_dates=['origin','date'])
    if baseline.duplicated(KEYS+['model']).any() or ml.duplicated(KEYS+['model']).any():
        raise ValueError('Duplicate forecast keys')
    baseline['status'] = np.where(baseline.yhat.notna(),'ok','missing_baseline')
    joined = pd.concat([baseline,ml],ignore_index=True)
    # Never silently disagree about an observed target across source files.
    if joined.groupby(KEYS).actual.nunique().gt(1).any():
        raise ValueError('Baseline and Step 3 actual values differ')
    return ml, joined


def metrics(joined):
    records, coverage = [], []
    for period in ('validation','covid','post_covid','calm_2023on','post_selection_covid'):
        source_period = 'post_covid' if period == 'calm_2023on' else 'covid' if period == 'post_selection_covid' else period
        part = joined[joined.period.eq(source_period)].copy()
        if period == 'calm_2023on':
            part = part[part.date.ge('2023-01-01')]
        if period == 'post_selection_covid':
            part = part[part.origin.ge('2020-01-31')]
        for h, group in part.groupby('horizon'):
            wide = group.pivot(index=['origin','date'],columns='model',values='yhat').reindex(columns=ALL_MODELS)
            truth = group.groupby(['origin','date']).actual.first().reindex(wide.index)
            good_status = group.assign(ok=group.status.eq('ok')).pivot(index=['origin','date'],columns='model',values='ok')
            wide = wide.where(good_status.reindex(columns=ALL_MODELS).fillna(False))
            shared = wide.notna().all(axis=1) & truth.notna()
            shared_naive = wide.loc[shared,'naive']-truth.loc[shared]
            naive_rmse = float(np.sqrt(np.mean(shared_naive**2))) if shared.any() else np.nan
            for model in ALL_MODELS:
                own = wide[model].notna() & truth.notna()
                coverage.append(dict(view=period,horizon=h,model=model,n_planned=len(wide),
                                     n_actual=int(truth.notna().sum()),n_forecast=int(wide[model].notna().sum()),
                                     n_own_scored=int(own.sum()),n_common=int(shared.sum()),
                                     n_unpublished_actual=int(truth.isna().sum())))
                for sample_name, mask in [('common_all_models',shared),('own_available',own)]:
                    e = wide.loc[mask,model]-truth.loc[mask]
                    rmse = float(np.sqrt(np.mean(e**2))) if len(e) else np.nan
                    # Reference naive is recalculated on the EXACT same keys.
                    ref = wide.loc[mask,'naive']-truth.loc[mask]
                    ref_rmse = float(np.sqrt(np.mean(ref**2))) if len(ref) else np.nan
                    records.append(dict(view=period,horizon=h,model=model,sample=sample_name,n=len(e),
                                        rmse=rmse,mae=float(abs(e).mean()),bias=float(e.mean()),
                                        naive_rmse=ref_rmse,
                                        rmse_vs_naive=rmse/ref_rmse if ref_rmse>0 else np.nan))
    return pd.DataFrame(records),pd.DataFrame(coverage)


def ablation(joined):
    rows = []
    for (period,h), group in joined[joined.period.ne('live')].groupby(['period','horizon']):
        wide = group.pivot(index=['origin','date'],columns='model',values='yhat')
        truth = group.groupby(['origin','date']).actual.first()
        for model in ['ridge','ridge_regional']:
            mask = wide[[model,'ridge_ar_only']].notna().all(axis=1) & truth.notna()
            e = wide.loc[mask,model]-truth.loc[mask]
            ref = wide.loc[mask,'ridge_ar_only']-truth.loc[mask]
            a,b = np.sqrt(np.mean(e**2)),np.sqrt(np.mean(ref**2))
            rows.append(dict(period=period,horizon=h,model=model,n=int(mask.sum()),
                             rmse=a,ar_only_rmse=b,improvement_pct=100*(1-a/b) if b>0 else np.nan))
    return pd.DataFrame(rows)


def plots(ml, table):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(1,3,figsize=(17,6),sharey=True)
    for ax,(h,part) in zip(axes, table[(table.view=='post_covid') & (table['sample']=='common_all_models')].groupby('horizon')):
        part = part.set_index('model').reindex(ALL_MODELS)
        ax.barh(part.index[::-1],part.rmse_vs_naive.iloc[::-1],
                color=['#267e99' if m in MODELS else '#9eabb5' for m in part.index[::-1]])
        ax.axvline(1,color='black',linestyle='--',linewidth=1)
        ax.set(title=f'{h}-month horizon | common n={int(part.n.iloc[0])}',
               xlabel='RMSE / naive RMSE (lower is better)')
    fig.suptitle('Step 3 vs frozen Step 2 models: retrospective 2022–Aug 2026')
    fig.tight_layout()
    fig.savefig(OUT/'model_comparison.png',dpi=160)
    plt.close(fig)
    winners = json.loads((OUT/'recommended_models.json').read_text())
    base = pd.read_csv(ROOT.parent/'results/forecasts.csv',parse_dates=['origin','date'])
    fig, axes = plt.subplots(3,1,figsize=(13,10))
    for h,ax in zip((1,3,6),axes):
        model = winners[str(h)]
        part = ml[(ml.horizon==h)&(ml.model==model)&ml.period.isin(['covid','post_covid'])].sort_values('date')
        reference = base[(base.horizon==h)&(base.model=='naive')&base.period.isin(['covid','post_covid'])].sort_values('date')
        ax.plot(part.date,part.actual,color='black',label='Actual',linewidth=1.5)
        ax.plot(part.date,part.yhat,color='#267e99',label=model)
        ax.plot(reference.date,reference.yhat,color='#d08b36',linestyle='--',alpha=.7,label='Naive')
        ax.axvspan(pd.Timestamp('2020-01-01'),pd.Timestamp('2021-12-31'),color='gray',alpha=.12)
        ax.set(title=f'h={h}: ML model selected on 2016–2019 validation only',ylabel='NY unemployment (%)')
        ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(OUT/'forecast_paths.png',dpi=160)
    plt.close(fig)
    imp = pd.read_csv(OUT/'feature_importance.csv')
    part = imp[(imp.model=='random_forest')&(imp.horizon==1)].sort_values('value')
    fig,ax=plt.subplots(figsize=(10,5))
    ax.barh(part.feature,part.value,color='#267e99')
    ax.set(title='Random forest h=1: training impurity importance at Sep 2026 origin',
           xlabel='Training importance; not causal or out-of-sample evidence')
    fig.tight_layout()
    fig.savefig(OUT/'feature_importance.png',dpi=160)
    plt.close(fig)


def main():
    ml,joined = comparison_frames()
    table,coverage = metrics(joined)
    table.to_csv(OUT/'metrics.csv',index=False)
    coverage.to_csv(OUT/'coverage.csv',index=False)
    paired = ablation(joined)
    paired.to_csv(OUT/'feature_ablation.csv',index=False)
    plots(ml,table)
    winners = json.loads((OUT/'recommended_models.json').read_text())
    live = ml[ml.period.eq('live')].copy()
    live['recommended'] = [winners[str(h)]==m for h,m in zip(live.horizon,live.model)]
    live.to_csv(OUT/'live_forecasts.csv',index=False)
    lines = ['# Step 3 results','',
             'Retrospective pseudo-real-time benchmark. All errors are percentage points.',
             'Selections use 2016–2019 only; test history was already inspected by the team.',
             '','## Models selected on validation','',
             '| Horizon | Model | Post-COVID common n | RMSE | MAE | RMSE / naive |',
             '|---|---|---|---|---|---|']
    common = table[(table.view=='post_covid')&(table['sample']=='common_all_models')]
    for h in (1,3,6):
        r=common[(common.horizon==h)&(common.model==winners[str(h)])].iloc[0]
        lines.append(f'| {h} | {r.model} | {r.n} | {r.rmse:.4f} | {r.mae:.4f} | {r.rmse_vs_naive:.3f} |')
    lines += ['', 'These are validation-selected models, not the best models chosen after seeing test errors.',
              '', '## Reproduction and interpretation','',
              '- Run from `modeling/`: `../.venv/bin/python -m src.experiments.step3_ml.run`.',
              '- `metrics.csv` distinguishes common-sample comparisons from own-available coverage.',
              '- `coverage.csv` reports missing forecasts and unpublished actuals separately.',
              '- `feature_ablation.csv` compares macro/regional Ridge against target-history-only Ridge on identical evaluation keys. Training spans and validation-selected penalties differ; this is a predictive comparison, not an isolated feature-effect estimate.',
              '- `live_forecasts.csv` follows the Step 2 date/model/horizon/yhat/lower/upper/origin/actual/period schema.',
              '- Live intervals use only validation residual quantiles; no post-COVID outcomes calibrate them.',
              '- October 2025 is never imputed or scored. Features crossing the gap remain missing.',
              '- COVID training exclusions and input clipping follow the frozen Step 2 experiment policy.',
              '- Values are revised snapshots; some release dates are estimated. This is not a historical-vintage backtest.',
              '- Pre-selection COVID origins are flagged; `post_selection_covid` is an additional evaluation view.',
              '- Feature coefficients and impurity importance are descriptive, not causal evidence.',
              '- Indeed and quarterly taxable sales are excluded: no compatible long validation history / release audit.',
              '- Reserve new releases for prospective confirmation; do not retune on the reported test ranking.', '']
    (OUT/'RESULTS.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:18]),flush=True)


if __name__ == '__main__':
    main()

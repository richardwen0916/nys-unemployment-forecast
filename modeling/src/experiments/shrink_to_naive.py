"""Experiment `shrink_v1`: pull model forecasts part of the way back to naive.

    yhat_shrunk = naive + lambda_h * (yhat_model - naive),   0 <= lambda_h <= 1

lambda_h is chosen per model and horizon on the VALIDATION period (2016-2019)
only, from the grid 0, 0.1, ..., 1, minimising RMSE on the validation common
sample. It is then frozen and applied unchanged to the COVID and post-COVID
test periods. No test error enters the choice.

Input: results/experiments/improved_v1/forecasts.csv (no refitting needed).
Output: results/experiments/shrink_v1/{lambdas.csv, forecasts.csv, metrics.csv}
Run from the package root:  python -m src.experiments.shrink_to_naive
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.evaluate_improved import dm_test

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'results/experiments/improved_v1/forecasts.csv'
OUT = ROOT / 'results/experiments/shrink_v1'
KEYS = ['forecast_origin', 'target_month', 'horizon']
SHRINK = ['M3x', 'M4x', 'VAR', 'COMBO']
GRID = np.round(np.arange(0, 1.01, 0.1), 2)
ALL_NEW = ['M0', 'M1', 'M2', 'M3', 'M2x', 'M3x', 'M4x', 'VAR', 'COMBO']

f = pd.read_csv(SRC, parse_dates=['forecast_origin', 'target_month'])
f.loc[f.status.ne('ok'), 'yhat'] = np.nan
wide = f.pivot_table(index=KEYS + ['period'], columns='model', values='yhat', aggfunc='first')
wide = wide.join(f.groupby(KEYS + ['period']).actual.first())

# 1. choose lambda on validation
val = wide[wide.index.get_level_values('period') == 'validation'].dropna(subset=ALL_NEW + ['actual'])
lambdas = []
for h, s in val.groupby(level='horizon'):
    for m in SHRINK:
        rmse = [np.sqrt(((s.M0 + lam * (s[m] - s.M0) - s.actual) ** 2).mean()) for lam in GRID]
        lambdas.append(dict(model=m, horizon=h, lam=GRID[int(np.argmin(rmse))], val_rmse=min(rmse),
                            val_rmse_unshrunk=rmse[-1], val_rmse_naive=rmse[0]))
lambdas = pd.DataFrame(lambdas)

# 2. apply frozen lambdas everywhere
for row in lambdas.itertuples():
    on_h = wide.index.get_level_values('horizon') == row.horizon
    wide.loc[on_h, f'{row.model}_s'] = wide.loc[on_h, 'M0'] + row.lam * (wide.loc[on_h, row.model] - wide.loc[on_h, 'M0'])

# 3. score on common samples, including a calm 2023+ view
models = ALL_NEW + [f'{m}_s' for m in SHRINK]
views = {'validation': ('validation', None), 'covid': ('covid', None),
         'post_covid': ('post_covid', None), 'calm_2023on': ('post_covid', '2023-01-01')}
rows = []
for view, (period, start) in views.items():
    sub = wide[wide.index.get_level_values('period') == period]
    if start:
        sub = sub[sub.index.get_level_values('target_month') >= start]
    sub = sub.dropna(subset=models + ['actual'])
    for h, s in sub.groupby(level='horizon'):
        e0 = (s.M0 - s.actual).to_numpy()
        for m in models:
            e = (s[m] - s.actual).to_numpy()
            stat, p = dm_test(e, e0, h) if m != 'M0' else (np.nan, np.nan)
            rows.append(dict(view=view, horizon=h, model=m, n=len(s), rmse=np.sqrt((e ** 2).mean()),
                             rmse_vs_naive=np.sqrt((e ** 2).mean()) / np.sqrt((e0 ** 2).mean()),
                             dm_stat=stat, dm_p=p))
metrics = pd.DataFrame(rows)

OUT.mkdir(parents=True, exist_ok=True)
lambdas.to_csv(OUT / 'lambdas.csv', index=False)
wide.reset_index().to_csv(OUT / 'forecasts.csv', index=False)
metrics.to_csv(OUT / 'metrics.csv', index=False)

pd.set_option('display.width', 160)
print('lambda chosen on validation:')
print(lambdas.pivot(index='model', columns='horizon', values='lam').to_string())
show = ['M1', 'M3x', 'M3x_s', 'M4x_s', 'VAR', 'VAR_s', 'COMBO', 'COMBO_s']
for view in views:
    t = metrics[metrics.view == view]
    print(f'\n=== {view}: RMSE / naive  [DM p]  n={t.groupby("horizon").n.first().tolist()} ===')
    r = t.pivot(index='model', columns='horizon', values='rmse_vs_naive').loc[show].round(2).astype(str)
    p = t.pivot(index='model', columns='horizon', values='dm_p').loc[show].round(2).astype(str)
    print((r + ' [' + p + ']').to_string())

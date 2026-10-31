"""Score experiment improved_v1 against the frozen baselines on common samples.

For each period (validation 2016-2019, covid 2020-2021, post_covid 2022+) and
horizon, only targets where EVERY model (the 8 experiment models and the 4
frozen baselines under their original timing) issued a finite forecast and the
actual is observed are scored. Exclusions are reported as n_scored.

Diebold-Mariano test vs naive (M0 under v2 timing) on squared errors, with a
Newey-West variance (lag = horizon) and the Harvey-Leybourne-Newbold small-sample
correction. Positive dm_stat = model has lower squared error than naive.

Outputs in results/experiments/improved_v1/: metrics.csv, dm_tests.csv
Run from the package root:  python -m src.experiments.evaluate_improved
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / 'results/experiments/improved_v1'
KEYS = ['forecast_origin', 'target_month', 'horizon']


def load():
    new = pd.read_csv(EXP / 'forecasts.csv', parse_dates=['forecast_origin', 'target_month'])
    old = pd.concat([
        pd.read_csv(ROOT / 'results/modeling/validation_forecasts.csv', parse_dates=['forecast_origin', 'target_month'])
          .assign(period='validation'),
        pd.read_csv(ROOT / 'results/modeling/test_forecasts.csv', parse_dates=['forecast_origin', 'target_month'])
          .rename(columns={'test_period': 'period'}),
    ])
    old = old.assign(model='old_' + old.model)[KEYS + ['model', 'yhat', 'actual', 'period', 'status']]
    frame = pd.concat([new[KEYS + ['model', 'yhat', 'actual', 'period', 'status']], old], ignore_index=True)
    frame.loc[frame.status.ne('ok'), 'yhat'] = np.nan
    return frame


def dm_test(e_model, e_naive, h):
    d = e_naive ** 2 - e_model ** 2
    n = len(d)
    gamma = [np.mean((d[k:] - d.mean()) * (d[:n - k] - d.mean())) for k in range(h + 1)]
    var = (gamma[0] + 2 * sum((1 - k / (h + 1)) * gamma[k] for k in range(1, h + 1))) / n
    if var <= 0:
        return np.nan, np.nan
    stat = d.mean() / np.sqrt(var)
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)          # HLN correction
    return stat, 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))


def main():
    frame = load()
    wide = frame.pivot_table(index=KEYS + ['period'], columns='model', values='yhat', aggfunc='first')
    actual = frame.groupby(KEYS + ['period']).actual.first()
    wide = wide.join(actual)
    models = [c for c in wide.columns if c != 'actual']
    metrics, dm = [], []
    for (period, h), sub in wide.groupby(level=['period', 'horizon']):
        planned = len(sub)
        common = sub.dropna(subset=models + ['actual'])
        for m in models:
            e = (common[m] - common.actual).to_numpy()
            metrics.append(dict(period=period, horizon=h, model=m, n_planned=planned, n_scored=len(common),
                                rmse=np.sqrt(np.mean(e ** 2)), mae=np.mean(np.abs(e)), bias=np.mean(e)))
            if m != 'M0':
                stat, p = dm_test(e, (common['M0'] - common.actual).to_numpy(), h)
                dm.append(dict(period=period, horizon=h, model=m, dm_stat=stat, p_value=p))
    metrics = pd.DataFrame(metrics)
    naive = metrics[metrics.model == 'M0'].set_index(['period', 'horizon']).rmse
    metrics['rmse_vs_naive'] = metrics.rmse / [naive[(p, h)] for p, h in zip(metrics.period, metrics.horizon)]
    metrics.to_csv(EXP / 'metrics.csv', index=False)
    pd.DataFrame(dm).to_csv(EXP / 'dm_tests.csv', index=False)

    order = ['old_M0', 'old_M1', 'old_M2', 'old_M3', 'M0', 'M1', 'M2', 'M3', 'M2x', 'M3x', 'M4x', 'VAR', 'COMBO']
    pd.set_option('display.width', 160)
    for period in ['validation', 'covid', 'post_covid']:
        t = metrics[metrics.period == period].pivot(index='model', columns='horizon', values='rmse').loc[order]
        n = metrics[metrics.period == period].groupby('horizon').n_scored.first().to_dict()
        print(f'\n=== {period} RMSE (n scored per horizon {n}) ===')
        print(t.round(3).to_string())
    d = pd.DataFrame(dm)
    print('\n=== DM vs naive (post_covid): stat (p) ===')
    print(d[d.period == 'post_covid'].pivot(index='model', columns='horizon', values='p_value').round(3).to_string())


if __name__ == '__main__':
    main()

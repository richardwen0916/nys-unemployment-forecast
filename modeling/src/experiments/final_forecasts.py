"""Final Step 2 deliverable: frozen model set, backtest + live forecasts with 95% intervals.

Final models (all frozen; nothing here is tuned):
  naive     M0, latest published NYUR
  arima     M1, frozen ARIMA(0,1,2)
  arx       M3x, frozen macro ARX features, COVID-free training, clipped inputs
  arx_s     M3x shrunk to naive with the validation-chosen lambda (shrink_v1)
  var_s     VAR(3) of NY/NJ/CT/PA rates, shrunk to naive (shrink_v1 lambda)
  combo_s   mean(naive, arima, arx) shrunk to naive (shrink_v1 lambda)  <- recommended

Backtest rows come from improved_v1 (+ shrink_v1 lambdas). Live rows are new
origins after the last scored target, built with the same functions:
  2026-08-31 -> Sep 2026 (h=1), Nov 2026 (h=3), Feb 2027 (h=6)
  2026-09-30 -> Oct 2026 (h=1), Dec 2026 (h=3), Mar 2027 (h=6)

95% intervals are empirical: yhat + the 2.5% / 97.5% quantiles of that model's
backtest errors at that horizon, using validation and post-COVID targets only
(COVID errors would make intervals useless in normal times; say so on the dashboard).

Output (results/experiments/final_v1/):
  forecasts.csv  date, model, horizon, yhat, lower, upper, origin, actual, period
  metrics.csv    RMSE / MAE / RMSE vs naive / DM p-value by period x horizon x model
Run from the package root:  python -m src.experiments.final_forecasts
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.data_availability import available_panel
from src.models.feature_engineering import load_core_inputs
from src.models.rolling_validation import build_origin_bank, released_nyur, arima_forecasts
from src.models.out_of_sample import freeze_specs, make_label_bank, expanding_training_sample
from src.experiments.improved_models import covid_free_rows, ols_forecast, var_forecast
from src.experiments.evaluate_improved import dm_test

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/experiments/final_v1'
AS_OF = '2026-10-19'
LIVE_ORIGINS = ['2026-08-31', '2026-09-30']
HORIZONS = (1, 3, 6)
NAMES = {'M0': 'naive', 'M1': 'arima', 'M3x': 'arx', 'M3x_s': 'arx_s', 'VAR_s': 'var_s', 'COMBO_s': 'combo_s'}


def live_rows(lambdas):
    rules = pd.read_csv(ROOT / 'data/experiments/release_rules_v2.csv')
    calendar = pd.read_csv(ROOT / 'data/experiments/release_calendar_v2.csv')
    panel, _, _, raw = load_core_inputs(ROOT)
    panel_full = pd.read_csv(ROOT / 'data/processed/monthly_panel.csv', parse_dates=['date'])
    specs, _ = freeze_specs()
    actual_view = available_panel(panel[['date', 'ny_ur']], AS_OF, rules, releases=calendar)
    origins = pd.date_range('1990-01-31', LIVE_ORIGINS[-1], freq='ME')
    features, _, _ = build_origin_bank(panel, origins, rules, calendar, raw, progress=print)
    labels = make_label_bank(features, panel, actual_view)
    lam = lambdas.set_index(['model', 'horizon']).lam
    rows = []
    for origin in pd.to_datetime(LIVE_ORIGINS):
        current = features.loc[origin]
        targets = {h: (origin.to_period('M') + h).to_timestamp() for h in HORIZONS}
        series, _ = released_nyur(panel, origin, rules, calendar)
        arima, _, _ = arima_forecasts(series, origin, (0, 1, 2), targets)
        var, _ = var_forecast(panel_full, origin, targets, rules, calendar)
        for h, target in targets.items():
            X, y, _, _ = expanding_training_sample(features, labels, origin, h, list(specs[(h, 'M3')].features))
            keep = covid_free_rows(X, labels.xs(h, level='horizon').reindex(X.index).target_reference)
            arx, _ = ols_forecast(X[keep], y[keep], current, clip=True)
            naive = float(current.latest_nyur_level)
            out = {'M0': naive, 'M1': arima.get(h, np.nan), 'M3x': arx}
            out['M3x_s'] = naive + lam[('M3x', h)] * (arx - naive)
            out['VAR_s'] = naive + lam[('VAR', h)] * (var.get(h, np.nan) - naive)
            combo = np.mean([naive, out['M1'], arx])
            out['COMBO_s'] = naive + lam[('COMBO', h)] * (combo - naive)
            for m, v in out.items():
                rows.append(dict(origin=origin, date=target, horizon=h, model=m, yhat=v,
                                 actual=actual_view.data.ny_ur.get(target, np.nan), period='live'))
    return pd.DataFrame(rows)


def main():
    shrink = ROOT / 'results/experiments/shrink_v1'
    lambdas = pd.read_csv(shrink / 'lambdas.csv')
    wide = pd.read_csv(shrink / 'forecasts.csv', parse_dates=['forecast_origin', 'target_month'])
    back = wide.melt(id_vars=['forecast_origin', 'target_month', 'horizon', 'period', 'actual'],
                     value_vars=list(NAMES), var_name='model', value_name='yhat')
    back = back.rename(columns={'forecast_origin': 'origin', 'target_month': 'date'})
    live = live_rows(lambdas)

    # Empirical 95% intervals from validation + post-COVID backtest errors
    calm = back[back.period.isin(['validation', 'post_covid'])].dropna(subset=['yhat', 'actual'])
    q = (calm.assign(err=calm.actual - calm.yhat).groupby(['model', 'horizon']).err
         .quantile([0.025, 0.975]).unstack())
    frame = pd.concat([back, live], ignore_index=True)
    frame = frame.join(q, on=['model', 'horizon'])
    frame['lower'] = frame.yhat + frame[0.025]
    frame['upper'] = frame.yhat + frame[0.975]
    frame['model'] = frame.model.map(NAMES)
    cols = ['date', 'model', 'horizon', 'yhat', 'lower', 'upper', 'origin', 'actual', 'period']
    OUT.mkdir(parents=True, exist_ok=True)
    frame[cols].sort_values(['period', 'model', 'horizon', 'date']).to_csv(OUT / 'forecasts.csv', index=False,
                                                                         float_format='%.4f')

    # Metrics on common samples (backtest only)
    metrics = []
    views = {'validation': ('validation', None), 'covid': ('covid', None),
             'post_covid': ('post_covid', None), 'calm_2023on': ('post_covid', '2023-01-01')}
    for view, (period, start) in views.items():
        sub = wide[wide.period.eq(period)]
        if start:
            sub = sub[sub.target_month >= start]
        sub = sub.dropna(subset=list(NAMES) + ['actual'])
        for h, s in sub.groupby('horizon'):
            e0 = (s.M0 - s.actual).to_numpy()
            for m in NAMES:
                e = (s[m] - s.actual).to_numpy()
                inside = ((s.actual >= s[m] + q.loc[(m, h), 0.025]) & (s.actual <= s[m] + q.loc[(m, h), 0.975])).mean()
                stat, p = dm_test(e, e0, h) if m != 'M0' else (np.nan, np.nan)
                metrics.append(dict(view=view, horizon=h, model=NAMES[m], n=len(s),
                                    rmse=np.sqrt((e ** 2).mean()), mae=np.abs(e).mean(), bias=e.mean(),
                                    rmse_vs_naive=np.sqrt((e ** 2).mean()) / np.sqrt((e0 ** 2).mean()),
                                    dm_p_vs_naive=p, coverage95=inside))
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(OUT / 'metrics.csv', index=False, float_format='%.4f')

    pd.set_option('display.width', 160)
    print(live.pivot_table(index=['origin', 'model'], columns='horizon', values='yhat').round(2).to_string())
    print(frame[frame.period == 'live'][['origin', 'date', 'model', 'yhat', 'lower', 'upper']].round(2).to_string(index=False))
    print(metrics.pivot_table(index=['view', 'model'], columns='horizon', values='rmse_vs_naive').round(2).to_string())


if __name__ == '__main__':
    main()

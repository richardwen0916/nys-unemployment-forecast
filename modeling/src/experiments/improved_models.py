"""Experiment `improved_v1`: fix release timing and COVID training, add VAR and a combination.

Everything reuses the frozen framework (pseudo-real-time availability, own-origin
claims STL, published-label gating, direct OLS). Frozen specifications, core
functions and canonical results are not modified. Specifications below were
declared before running this experiment; nothing is tuned on test errors.

Models (all at every origin/horizon in the validation and test plans)
  M0, M1, M2, M3   frozen baselines, refit under the v2 release timing
  M2x, M3x         same features as M2/M3, but training rows whose target or
                   origin-window touches COVID (2020-03..2021-12) are dropped,
                   and the current predictors are clipped to the training range
                   before forecasting (no extrapolation beyond seen values)
  M4x              M3x features + latest published Empire State survey level
  VAR              VAR(3) in first differences of NY, NJ, CT, PA unemployment
                   rates, released data only, COVID rows dropped, iterated to
                   the target month
  COMBO            equal-weight mean of M0, M1 and M3x

Timing v2 (src/experiments/release_timing_v2.py) makes last month's NYUR
available at most month-end origins, as the verified BLS dates show.

Outputs: results/experiments/improved_v1/{forecasts.csv, metrics.csv, dm_tests.csv}

Run from the package root:  python -m src.experiments.improved_models
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import stats

from src.models.data_availability import available_panel
from src.models.feature_engineering import load_core_inputs
from src.models.rolling_validation import (
    build_origin_bank, training_sample, released_nyur, arima_forecasts, validation_plan,
)
from src.models.out_of_sample import freeze_specs, test_plan, make_label_bank, expanding_training_sample

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/experiments/improved_v1'
AS_OF = '2026-10-19'
COVID = (pd.Timestamp('2020-03-01'), pd.Timestamp('2021-12-01'))
FEATURE_REACH = 3            # origin features look back up to 3 months (lag3)
VAR_STATES = ['ny_ur', 'nj_ur', 'ct_ur', 'pa_ur']
VAR_LAGS = 3                 # HQIC on 1990-2015 differences
MIN_TRAIN = 60


def in_covid(month):
    return COVID[0] <= month <= COVID[1]


def covid_free_rows(X, target_reference):
    """Drop rows whose target is a COVID month or whose origin features reach into COVID."""
    keep = []
    for origin, target in zip(X.index, target_reference):
        origin_month = origin.to_period('M').to_timestamp()
        window = pd.date_range(origin_month - pd.DateOffset(months=FEATURE_REACH), origin_month, freq='MS')
        keep.append(not in_covid(target) and not any(in_covid(m) for m in window))
    return np.array(keep, dtype=bool)


def ols_forecast(X, y, current, *, clip):
    """Same OLS as the frozen direct_forecast (fit-only standardization), optional clipping."""
    if len(X) < max(MIN_TRAIN, X.shape[1] + 2):
        return np.nan, 'insufficient_training'
    x_now = current.reindex(X.columns).astype(float)
    if x_now.isna().any():
        return np.nan, 'missing_current_predictor'
    if clip:
        x_now = x_now.clip(X.min(), X.max())
    means, scales = X.mean(), X.std(ddof=0).replace(0, 1)
    design = np.column_stack([np.ones(len(X)), ((X - means) / scales).to_numpy()])
    beta, _, rank, _ = np.linalg.lstsq(design, y.to_numpy(), rcond=None)
    if rank < design.shape[1]:
        return np.nan, 'rank_deficient'
    vector = np.r_[1.0, ((x_now - means) / scales).to_numpy()]
    return float(current['latest_nyur_level'] + vector @ beta), 'ok'


def empire_bank(panel_full, origins, rules, calendar):
    """Latest published Empire State survey value at each origin."""
    values = {}
    frame = panel_full[['date', 'empire_general']]
    for origin in origins:
        data = available_panel(frame, origin, rules, releases=calendar).data.empire_general.dropna()
        values[origin.to_period('M').to_timestamp() + pd.offsets.MonthEnd(0)] = data.iloc[-1] if len(data) else np.nan
    return pd.Series(values, name='empire_latest')


def var_forecast(panel_full, origin, targets, rules, calendar):
    """VAR(p) on released differences, COVID rows dropped; returns {h: level forecast}."""
    view = available_panel(panel_full[['date'] + VAR_STATES], origin, rules, releases=calendar).data
    levels = view.loc['1990-01-01':, VAR_STATES]
    complete = levels.dropna()
    if complete.empty:
        return {}, 'no_data'
    last = complete.index.max()
    levels = levels.loc[:last].asfreq('MS')
    diffs = levels.diff()
    lagged = pd.concat({k: diffs.shift(k) for k in range(1, VAR_LAGS + 1)}, axis=1)
    reach = pd.Series([any(in_covid(m - pd.DateOffset(months=k)) for k in range(VAR_LAGS + 1))
                       for m in diffs.index], index=diffs.index)
    ok = diffs.notna().all(axis=1) & lagged.notna().all(axis=1) & ~reach
    if ok.sum() < MIN_TRAIN:
        return {}, 'insufficient_training'
    X = np.column_stack([np.ones(ok.sum()), lagged[ok].to_numpy()])
    B, *_ = np.linalg.lstsq(X, diffs[ok].to_numpy(), rcond=None)
    hist = diffs.iloc[-VAR_LAGS:].to_numpy()[::-1]
    if np.isnan(hist).any():
        return {}, 'gap_in_recent_history'
    n = len(VAR_STATES)
    A = [B[1 + i * n:1 + (i + 1) * n].T for i in range(VAR_LAGS)]
    steps = {h: t.to_period('M').ordinal - last.to_period('M').ordinal for h, t in targets.items()}
    path, history = [], list(hist)
    for _ in range(max(steps.values())):
        step = B[0] + sum(A[i] @ history[i] for i in range(VAR_LAGS))
        path.append(step)
        history = [step] + history[:-1]
    cumulative = levels.loc[last, 'ny_ur'] + np.cumsum([p[0] for p in path])
    return {h: float(cumulative[s - 1]) for h, s in steps.items()}, 'ok'


def run():
    rules = pd.read_csv(ROOT / 'data/experiments/release_rules_v2.csv')
    calendar = pd.read_csv(ROOT / 'data/experiments/release_calendar_v2.csv')
    panel, _, _, raw = load_core_inputs(ROOT)
    panel_full = pd.read_csv(ROOT / 'data/processed/monthly_panel.csv', parse_dates=['date'])
    specs, cutoff = freeze_specs()

    actual_view = available_panel(panel[['date', 'ny_ur']], AS_OF, rules, releases=calendar)
    last_target = actual_view.data.ny_ur.dropna().index.max()
    plans = [('validation', validation_plan()), ('test', test_plan(last_target, cutoff))]
    last_origin = max(p.forecast_origin.max() for _, p in plans)
    origins = pd.date_range('1990-01-31', last_origin, freq='ME')
    features, meta, _ = build_origin_bank(panel, origins, rules, calendar, raw, progress=print)
    features['empire_latest'] = empire_bank(panel_full, features.index, rules, calendar).reindex(features.index).values
    labels = make_label_bank(features, panel, actual_view)

    rows = []
    for period, plan in plans:
        sampler = training_sample if period == 'validation' else expanding_training_sample
        for origin, tasks in plan.groupby('forecast_origin', sort=True):
            current = features.loc[origin]
            targets = {int(r.horizon): r.target_month for r in tasks.itertuples()}
            series, _ = released_nyur(panel, origin, rules, calendar)
            arima, arima_info, _ = arima_forecasts(series, origin, (0, 1, 2), targets)
            var, var_status = var_forecast(panel_full, origin, targets, rules, calendar)
            for task in tasks.itertuples():
                h, target = int(task.horizon), task.target_month
                out = {'M0': (float(current.latest_nyur_level), 'ok'),
                       'M1': (arima.get(h, np.nan), arima_info['status']),
                       'VAR': (var.get(h, np.nan), var_status)}
                for name, base, extra, covid_free in [('M2', 'M2', (), False), ('M3', 'M3', (), False),
                                                      ('M2x', 'M2', (), True), ('M3x', 'M3', (), True),
                                                      ('M4x', 'M3', ('empire_latest',), True)]:
                    columns = list(specs[(h, base)].features) + list(extra)
                    X, y, _, _ = sampler(features, labels, origin, h, columns)
                    if covid_free and len(X):
                        target_ref = labels.xs(h, level='horizon').reindex(X.index).target_reference
                        keep = covid_free_rows(X, target_ref)
                        X, y = X[keep], y[keep]
                    out[name] = ols_forecast(X, y, current, clip=covid_free)
                parts = [out[m][0] for m in ('M0', 'M1', 'M3x')]
                out['COMBO'] = (float(np.mean(parts)), 'ok') if np.isfinite(parts).all() else (np.nan, 'missing_component')
                for model, (yhat, status) in out.items():
                    rows.append(dict(forecast_origin=origin, target_month=target, horizon=h, model=model,
                                     yhat=yhat, status=status, actual=actual_view.data.ny_ur.get(target, np.nan),
                                     period=period if period == 'validation' else task.test_period,
                                     nyur_age_months=current.nyur_age_months))
        print(f'{period}: done', flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    forecasts = pd.DataFrame(rows)
    forecasts.to_csv(OUT / 'forecasts.csv', index=False)
    print(json.dumps({'forecasts': len(forecasts), 'finite': int(forecasts.yhat.notna().sum())}))


if __name__ == '__main__':
    run()

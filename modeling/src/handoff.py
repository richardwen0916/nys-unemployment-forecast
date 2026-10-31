"""Portable, offline entry point for the fixed historical baselines.

No selection/search is performed. Original modeling functions remain unchanged.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

from .models.data_availability import available_panel
from .models.feature_engineering import load_core_inputs, build_feature_window
from .models.rolling_validation import (
    build_origin_bank, training_sample, direct_forecast, released_nyur,
    arima_forecasts, validation_plan, scored_metrics,
)
from .models.out_of_sample import (
    freeze_specs, test_plan, make_label_bank, expanding_training_sample,
    evaluate_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / 'results/modeling'


def verify_saved():
    """Reconcile saved metrics from forecasts, without any model fitting."""
    manifest = pd.read_csv(ROOT / 'FILE_INVENTORY.csv', keep_default_na=False)
    for row in manifest.itertuples():
        path = ROOT / row.path
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row.sha256:
            raise ValueError(f'Inventory mismatch: {row.path}')
    v = pd.read_csv(SAVED / 'validation_forecasts.csv', parse_dates=['forecast_origin', 'target_month'])
    t = pd.read_csv(SAVED / 'test_forecasts.csv', parse_dates=['forecast_origin', 'target_month'])
    vm = pd.read_csv(SAVED / 'validation_metrics.csv')
    tm = pd.read_csv(SAVED / 'test_metrics.csv')
    recomputed = scored_metrics(v, common=True)
    a = vm.sort_values(['horizon', 'model']).reset_index(drop=True)
    b = recomputed.sort_values(['horizon', 'model']).reset_index(drop=True)
    np.testing.assert_allclose(a[['mae', 'rmse']], b[['mae', 'rmse']], atol=1e-12)
    for restricted, view in [(False, 'requested_full_period'), (True, 'post_selection_origins')]:
        a = tm.loc[tm.evaluation_view.eq(view)].sort_values(['period', 'horizon', 'model']).reset_index(drop=True)
        b = evaluate_metrics(t, post_selection_only=restricted).sort_values(['period', 'horizon', 'model']).reset_index(drop=True)
        np.testing.assert_allclose(a[['mae', 'rmse', 'bias', 'forecast_coverage', 'n_scored']],
                                   b[['mae', 'rmse', 'bias', 'forecast_coverage', 'n_scored']], atol=1e-12)
    gap = t.loc[t.target_month.eq('2025-10-01')]
    assert len(gap) == 12 and gap.actual.isna().all() and gap.yhat.notna().all() and not gap.common_scored.any()
    specs, _ = freeze_specs()
    assert len(specs) == 12
    print(json.dumps(dict(inventory_files=len(manifest), validation_forecasts=len(v),
        test_forecasts=len(t), test_metric_rows=len(tm), frozen_specifications=12,
        october_2025_forecasts_retained_unscored=12, saved_results_verified=True), indent=2))


def compare_forecasts(actual, expected):
    keys = ['forecast_origin', 'target_month', 'horizon', 'model']
    merged = actual.merge(expected[keys + ['yhat', 'status']], on=keys, suffixes=('', '_saved'), validate='one_to_one')
    assert len(merged) == len(actual) == len(expected)
    assert merged.status.eq(merged.status_saved).all()
    np.testing.assert_allclose(merged.yhat, merged.yhat_saved, rtol=1e-8, atol=1e-6, equal_nan=True)
    return float((merged.yhat - merged.yhat_saved).abs().max())


def reproduce(output):
    """Refit only the twelve saved choices; write into a new in-package folder."""
    verify_saved()
    out = (ROOT / output).resolve()
    if not out.is_relative_to(ROOT) or out == ROOT:
        raise ValueError('Output must be a new directory inside this package')
    out.mkdir(parents=True, exist_ok=False)
    specs, cutoff = freeze_specs()
    panel, rules, calendar, raw = load_core_inputs(ROOT)
    actual_view = available_panel(panel[['date', 'ny_ur']], '2026-10-19', rules, releases=calendar)
    last_target = actual_view.data.ny_ur.dropna().index.max()
    plans = [('validation', validation_plan()), ('test', test_plan(last_target, cutoff))]
    last_origin = max(p.forecast_origin.max() for _, p in plans)
    origins = pd.date_range('1990-01-31', last_origin, freq='ME')
    features, meta, windows = build_origin_bank(panel, origins, rules, calendar, raw, progress=print)
    labels = make_label_bank(features, panel, actual_view)
    # The same origin has the same predictors regardless of evaluation period.
    del windows
    discrepancy = {}
    for period, plan in plans:
        rows = []
        grouped = list(plan.groupby('forecast_origin', sort=True))
        for i, (origin, tasks) in enumerate(grouped):
            current = features.loc[origin]
            series, max_release = released_nyur(panel, origin, rules, calendar)
            targets = {int(r.horizon): r.target_month for r in tasks.itertuples()}
            prediction, arima_info, _ = arima_forecasts(series, origin, (0, 1, 2), targets)
            for task in tasks.itertuples():
                h = int(task.horizon)
                for model in ('M0', 'M1', 'M2', 'M3'):
                    spec = specs[(h, model)]
                    if model == 'M0':
                        yhat, info = float(current.latest_nyur_level), dict(status='ok', n_train=int(series.notna().sum()), max_train_label_available_at=max_release)
                    elif model == 'M1':
                        yhat, info = prediction[h], dict(arima_info, max_train_label_available_at=max_release)
                    else:
                        sampler = training_sample if period == 'validation' else expanding_training_sample
                        X, y, sample, _ = sampler(features, labels, origin, h, spec.features)
                        yhat, fit_info, _ = direct_forecast(X, y, current)
                        info = dict(sample, **fit_info)
                    rows.append(dict(forecast_origin=origin, target_month=task.target_month,
                        horizon=h, model=model, spec_id=spec.spec_id, yhat=yhat,
                        actual=actual_view.data.ny_ur.get(task.target_month, np.nan),
                        test_period=getattr(task, 'test_period', 'validation'),
                        origin_after_validation_cutoff=getattr(task, 'origin_after_validation_cutoff', False), **info))
            if (i + 1) % 12 == 0 or i + 1 == len(grouped):
                print(f'{period}: {i + 1}/{len(grouped)} frozen refit origins', flush=True)
        frame = pd.DataFrame(rows)
        frame.to_csv(out / f'{period}_forecasts.csv', index=False)
        metrics = (scored_metrics(frame, common=True) if period == 'validation' else
                   pd.concat([evaluate_metrics(frame), evaluate_metrics(frame, post_selection_only=True)], ignore_index=True))
        metrics.to_csv(out / f'{period}_metrics.csv', index=False)
        saved = pd.read_csv(SAVED / f'{period}_forecasts.csv', parse_dates=['forecast_origin', 'target_month'])
        discrepancy[period] = compare_forecasts(frame, saved)
    receipt = dict(forecasts_refitted=1536, historical_origins=len(features), specifications_changed=False,
        max_absolute_prediction_difference_vs_saved=discrepancy, tolerance_pp=1e-6,
        policy='Same model functions, own-origin STL and published-label gating; no grid or selection; no external project files')
    (out / 'reproduction_checks.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['inspect', 'reproduce', 'features'])
    parser.add_argument('--output', default='results/reproduced')
    parser.add_argument('--origin', default='2023-12-31')
    args = parser.parse_args()
    if args.command == 'inspect':
        verify_saved()
    elif args.command == 'reproduce':
        reproduce(args.output)
    else:
        panel, rules, calendar, raw = load_core_inputs(ROOT)
        window = build_feature_window(panel, args.origin, rules, releases=calendar, raw=raw)
        print(window.features.to_string())
        print(f'STL window: {window.stl_first_reference} through {window.stl_last_reference}; n={window.stl_nobs}')


if __name__ == '__main__':
    main()

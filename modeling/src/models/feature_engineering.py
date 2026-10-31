"""Core-only pseudo-real-time features and explicitly separate forecast labels.

No fitting of forecasting models, scaling, filling, or feature selection occurs.
Every origin rebuilds its information set and re-estimates claims STL.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data_availability import available_panel, forecast_origin, target_months
from .rolling_stl import rolling_claims_stl

CORE = ('ny_ur', 'ny_claims_avg', 'ny_payrolls', 'us_ur')
TRANSFORMS = {
    'ny_ur_level': ('ny_ur', 'level', 'percent'),
    'ny_ur_diff_pp': ('ny_ur', 'first difference', 'percentage points'),
    'ny_claims_sa_log': ('ny_claims_avg', 'log of training-window STL adjusted level', 'natural log claims/week'),
    'ny_claims_sa_log_diff': ('ny_claims_avg', 'first log difference after training-window STL', 'log points'),
    'ny_payrolls_log_growth_pct': ('ny_payrolls', '100 times first log difference', 'approx percent/month'),
    'us_ur_level': ('us_ur', 'level', 'percent'),
    'us_ur_diff_pp': ('us_ur', 'first difference', 'percentage points'),
}
DEFAULT_LAGS = (1, 2, 3)


def _panel(panel):
    frame = panel.copy()
    if 'date' in frame.columns:
        frame = frame.set_index('date')
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index), name='date')
    if (frame.index.tz is not None or frame.index.has_duplicates or frame.index.hasnans
            or not frame.index.is_month_start.all()
            or not (frame.index == frame.index.normalize()).all()):
        raise ValueError('Panel needs unique naive month-start dates')
    missing = set(CORE) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing core variables: {sorted(missing)}')
    return frame.loc[:, list(CORE)].sort_index()


def split_for_target(reference):
    """Predefined split is determined by target reference month, never origin."""
    year = pd.Timestamp(reference).year
    if 1990 <= year <= 2015:
        return 'initial_train'
    if 2016 <= year <= 2019:
        return 'validation'
    if 2020 <= year <= 2021:
        return 'covid_test'
    if 2022 <= year <= 2026:
        return 'post_covid_test'
    return 'outside_design'


@dataclass
class FeatureWindow:
    origin: pd.Timestamp
    history: pd.DataFrame
    features: pd.Series
    audit: pd.DataFrame
    target_levels: pd.Series
    stl_first_reference: pd.Timestamp
    stl_last_reference: pd.Timestamp
    stl_nobs: int
    stl_status: str


@dataclass
class TrainingExamples:
    X: pd.DataFrame
    labels: pd.DataFrame
    audit: pd.DataFrame
    as_of: pd.Timestamp

    def eligible(self, split='initial_train', columns=None, require_complete=True):
        """Return X/y only where labels were published by the explicit cutoff.

        Choose columns explicitly for later validation. Never fills ragged edges.
        Split purging uses target dates, including across year-end boundaries.
        """
        columns = list(self.X.columns) if columns is None else list(columns)
        X = self.X.loc[:, columns]
        keep = self.labels.label_known & self.labels.split.eq(split)
        if require_complete:
            keep &= X.notna().all(axis=1)
        return X.loc[keep].copy(), self.labels.loc[keep, 'target_ny_ur_level'].copy()


def build_feature_window(panel, origin, rules, *, releases=None, raw=None,
                         training_start='1990-01-01', lags=DEFAULT_LAGS,
                         allow_estimated=True, stl_min_months=36):
    """Build seven candidate transforms and fixed reference lags at month-end.

    Lag L always means origin calendar month minus L, not last nonmissing value
    and not target month minus L. Missing lagged values remain missing. The STL
    sample starts at training_start and ends at the last available claims month.
    """
    cutoff = forecast_origin(origin)
    if not cutoff.is_month_end:
        raise ValueError('Forecast origins must be the last calendar day of a month')
    if not lags or len(set(lags)) != len(lags) or any(type(lag) is not int or lag < 0 for lag in lags):
        raise ValueError('Lags must be distinct nonnegative integer month offsets')
    start = pd.Timestamp(training_start)
    if start.tzinfo is not None or not start.is_month_start or start != start.normalize():
        raise ValueError('training_start must be a naive month-start date')
    source = _panel(panel).loc[start:]
    # Claims are filtered by the existing STL entry point; avoid filtering twice.
    view = available_panel(source[['ny_ur', 'ny_payrolls', 'us_ur']], origin, rules, releases=releases, raw=raw,
                           allow_estimated=allow_estimated)
    last_month = cutoff.tz_localize(None).to_period('M').to_timestamp()
    dates = pd.date_range(start, last_month, freq='MS', name='date')
    observed = view.data.reindex(dates)
    claims = rolling_claims_stl(source, origin, rules, releases=releases, raw=raw,
        training_start=start, min_months=stl_min_months, allow_estimated=allow_estimated)
    payrolls = observed.ny_payrolls
    if ((payrolls.dropna() <= 0) | ~np.isfinite(payrolls.dropna())).any():
        raise ValueError('Payroll log growth needs strictly positive finite values')
    log_claims = np.log(claims.series.reindex(dates))
    history = pd.DataFrame({
        'ny_ur_level': observed.ny_ur,
        'ny_ur_diff_pp': observed.ny_ur.diff(),
        'ny_claims_sa_log': log_claims,
        'ny_claims_sa_log_diff': log_claims.diff(),
        'ny_payrolls_log_growth_pct': 100 * np.log(payrolls).diff(),
        'us_ur_level': observed.us_ur,
        'us_ur_diff_pp': observed.us_ur.diff(),
    }, index=dates)
    values, records = {}, []
    for lag in lags:
        reference = (last_month.to_period('M') - lag).to_timestamp()
        for name, (variable, transformation, unit) in TRANSFORMS.items():
            feature = f'{name}_lag{lag}'
            value = history.at[reference, name] if reference in history.index else np.nan
            values[feature] = value
            stamp, status, reason = pd.NaT, 'missing', 'unavailable_or_source_missing'
            previous = reference - pd.DateOffset(months=1)
            if pd.notna(value):
                if variable == 'ny_claims_avg':
                    # Every adjusted value depends on the entire available STL fit.
                    stamp, status = cutoff, 'derived_from_origin_training_window'
                else:
                    dependencies = [reference] if transformation == 'level' else [previous, reference]
                    details = [view.audit.loc[(date, variable)] for date in dependencies]
                    stamp = max(item.available_at for item in details)
                    status = ('verified_public_by' if all(item.status == 'verified_public_by' for item in details)
                              else 'estimated')
                if stamp > cutoff:
                    raise AssertionError('A feature dependency exceeds the forecast origin')
                reason = 'available'
            records.append(dict(feature=feature, variable=variable, transformation=transformation,
                unit=unit, reference_month=reference, reference_lag_months=lag,
                previous_reference=previous if transformation != 'level' else pd.NaT,
                available_at=stamp, timing_status=status, reason=reason,
                stl_last_reference=claims.last_reference if variable == 'ny_claims_avg' else pd.NaT))
    return FeatureWindow(cutoff, history, pd.Series(values, name=cutoff.tz_localize(None).normalize()),
        pd.DataFrame(records).set_index('feature'), observed.ny_ur.copy(),
        claims.first_reference, claims.last_reference, claims.nobs, claims.status)


def align_training_examples(panel, windows, as_of, rules, *, releases=None,
                            horizons=(1, 3, 6), allow_estimated=True):
    """Attach separate horizon labels to already origin-built feature windows.

    as_of is the training/evaluation label-information cutoff, not a permission
    to refit historical predictors. Unpublished or absent labels stay NaN.
    """
    cutoff = forecast_origin(as_of)
    windows = list(windows)
    if not horizons or len(set(horizons)) != len(horizons):
        raise ValueError('Horizons must be distinct positive integer offsets')
    target_months(as_of, horizons)  # validate
    source = _panel(panel)
    target_view = available_panel(source[['ny_ur']], as_of, rules, releases=releases,
                                   allow_estimated=allow_estimated)
    feature_rows, label_rows, audit_rows, indices = [], [], [], []
    seen = set()
    for window in windows:
        if window.origin > cutoff:
            raise ValueError('Feature origin is later than the label-information cutoff')
        origin = window.origin.tz_localize(None).normalize()
        if origin in seen:
            raise ValueError('Duplicate forecast origin')
        seen.add(origin)
        for horizon, reference in target_months(window.origin, horizons).items():
            key = (origin, horizon)
            known = reference in target_view.data.index and pd.notna(target_view.data.at[reference, 'ny_ur'])
            outcome = target_view.data.at[reference, 'ny_ur'] if known else np.nan
            stamp = target_view.audit.loc[(reference, 'ny_ur'), 'available_at'] if reference in target_view.data.index else pd.NaT
            indices.append(key)
            feature_rows.append(window.features.to_dict())
            label_rows.append(dict(target_reference=reference, target_ny_ur_level=outcome,
                label_available_at=stamp, label_known=bool(known), split=split_for_target(reference)))
            audit_rows.append(window.audit.reset_index().assign(origin=origin, horizon=horizon))
    index = pd.MultiIndex.from_tuples(indices, names=['origin', 'horizon'])
    X = pd.DataFrame(feature_rows, index=index)
    labels = pd.DataFrame(label_rows, index=index,
        columns=['target_reference', 'target_ny_ur_level', 'label_available_at', 'label_known', 'split'])
    audit = pd.concat(audit_rows, ignore_index=True) if audit_rows else pd.DataFrame()
    return TrainingExamples(X, labels, audit, cutoff)


def build_training_examples(panel, origins, as_of, rules, *, releases=None, raw=None,
                            horizons=(1, 3, 6), **feature_options):
    """Convenience interface; every historical origin gets its own STL fit."""
    cutoff = forecast_origin(as_of)
    origins = list(origins)
    if any(forecast_origin(origin) > cutoff for origin in origins):
        raise ValueError('Feature origin is later than the label-information cutoff')
    windows = [build_feature_window(panel, origin, rules, releases=releases, raw=raw,
                                    **feature_options) for origin in origins]
    return align_training_examples(panel, windows, as_of, rules, releases=releases,
        horizons=horizons, allow_estimated=feature_options.get('allow_estimated', True))


def load_core_inputs(root):
    """Read the panel/rules/calendar and only the required raw claims series."""
    root = Path(root)
    panel = pd.read_csv(root/'data/processed/monthly_panel.csv', parse_dates=['date'], usecols=['date', *CORE])
    rules = pd.read_csv(root/'data/processed/release_rules.csv')
    calendar = pd.read_csv(root/'data/processed/release_calendar.csv')
    claims = pd.read_csv(root/'data/raw/fred_NYICLAIMS.csv', parse_dates=['date']).set_index('date').iloc[:, 0]
    return panel, rules, calendar, {'NYICLAIMS': claims}

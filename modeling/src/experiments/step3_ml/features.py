"""Build immutable origin snapshots; keep labels and publication times separate.

Run from modeling/: python -m src.experiments.step3_ml.features
No network calls, full-sample STL, interpolation or data-source changes.
"""
from pathlib import Path
import hashlib
import json

import pandas as pd

from src.models.data_availability import available_panel
from src.models.feature_engineering import load_core_inputs
from src.models.rolling_validation import build_origin_bank
from src.models.out_of_sample import make_label_bank

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/experiments/step3_ml'
AS_OF = '2026-10-19'
LAST_ORIGIN = '2026-09-30'
AR_FEATURES = ['latest_nyur_level', 'latest_nyur_diff_pp', 'nyur_age_months']
CORE_FEATURES = AR_FEATURES + [
    'ny_claims_sa_log_diff_lag1', 'ny_claims_sa_log_diff_lag2',
    'ny_claims_sa_log_diff_lag3', 'ny_payrolls_log_growth_pct_lag2',
    'ny_payrolls_log_growth_pct_lag3', 'latest_usur_level', 'latest_usur_diff_pp',
]
EXTRA_VARIABLES = ['nj_ur', 'ct_ur', 'pa_ur', 'empire_general']
EXTRA_FEATURES = [f'{state}_{kind}' for state in EXTRA_VARIABLES[:3]
                  for kind in ('latest', 'diff_pp')] + ['empire_latest']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_paths():
    return [ROOT / name for name in (
        'data/processed/monthly_panel.csv', 'data/processed/data_dictionary.csv',
        'data/raw/fred_NYICLAIMS.csv', 'data/experiments/release_rules_v2.csv',
        'data/experiments/release_calendar_v2.csv',
        'src/models/data_availability.py', 'src/models/feature_engineering.py',
        'src/models/rolling_stl.py', 'src/models/rolling_validation.py',
        'src/models/out_of_sample.py', 'src/experiments/step3_ml/features.py',
    )]


def regional_row(panel, origin, rules, calendar):
    """Mask before differencing; never bridge a missing calendar month."""
    view = available_panel(panel[['date'] + EXTRA_VARIABLES], origin, rules, releases=calendar)
    values, audit = {}, []
    for variable in EXTRA_VARIABLES:
        series = view.data[variable].asfreq('MS')
        known = series.dropna()
        if known.empty:
            values['empire_latest' if variable == 'empire_general' else variable + '_latest'] = float('nan')
            if variable != 'empire_general':
                values[variable + '_diff_pp'] = float('nan')
            continue
        ref = known.index[-1]
        values['empire_latest' if variable == 'empire_general' else variable + '_latest'] = known.iloc[-1]
        if variable != 'empire_general':
            values[variable + '_diff_pp'] = series.diff().at[ref]
        detail = view.audit.loc[(ref, variable)]
        audit.append(dict(forecast_origin=origin, variable=variable, reference=ref,
                          available_at=detail.available_at, timing_status=detail.status))
    return values, audit


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    fingerprints = {str(p.relative_to(ROOT)): sha(p) for p in input_paths()}
    manifest_path = OUT / 'feature_manifest.json'
    names = ['origin_features.csv', 'origin_metadata.csv', 'labels.csv', 'extra_release_audit.csv']
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['inputs'] == fingerprints and all(
                (OUT/n).exists() and sha(OUT/n) == manifest['outputs'][n] for n in names):
            print('Reusing hash-verified Step 3 origin snapshots.', flush=True)
            return
    panel, _, _, raw = load_core_inputs(ROOT)
    full = pd.read_csv(ROOT/'data/processed/monthly_panel.csv', parse_dates=['date'])
    rules = pd.read_csv(ROOT/'data/experiments/release_rules_v2.csv')
    calendar = pd.read_csv(ROOT/'data/experiments/release_calendar_v2.csv')
    origins = pd.date_range('1990-01-31', LAST_ORIGIN, freq='ME')
    features, metadata, windows = build_origin_bank(panel, origins, rules, calendar, raw, progress=print)
    del windows
    extra_rows, audit = [], []
    for origin in origins:
        values, details = regional_row(full, origin, rules, calendar)
        extra_rows.append(values)
        audit.extend(details)
    features = features.join(pd.DataFrame(extra_rows, index=features.index))
    actual = available_panel(panel[['date', 'ny_ur']], AS_OF, rules, releases=calendar)
    labels = make_label_bank(features, panel, actual)
    features.to_csv(OUT/names[0])
    metadata.to_csv(OUT/names[1])
    labels.to_csv(OUT/names[2])
    pd.DataFrame(audit).to_csv(OUT/names[3], index=False)
    manifest = dict(as_of=AS_OF, last_origin=LAST_ORIGIN, inputs=fingerprints,
                    outputs={n: sha(OUT/n) for n in names},
                    policy='Step 2 v2 timing; immutable own-origin robust STL; published labels only; no filling')
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Saved {len(features)} origin snapshots.', flush=True)


def load():
    prepare()
    features = pd.read_csv(OUT/'origin_features.csv', parse_dates=['forecast_origin']).set_index('forecast_origin')
    labels = pd.read_csv(OUT/'labels.csv', parse_dates=['forecast_origin', 'target_reference'])
    labels['label_available_at'] = pd.to_datetime(labels.label_available_at, utc=True, format='mixed')
    return features, labels.set_index(['forecast_origin', 'horizon'])


if __name__ == '__main__':
    prepare()

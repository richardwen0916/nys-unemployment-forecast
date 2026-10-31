"""Experiment metadata: better-supported release timing for the BLS state release.

Problem in the frozen baseline: only 9 NYUR release dates are verified, so most
months fall back to "reference month-end + 45 days" (January + 75). The verified
evidence says NYUR normally comes out about 20 days after month end, so most
origins wrongly treat last month's NYUR as unpublished.

Evidence used (all already in data/processed/release_calendar.csv):
  * NY payrolls and the state unemployment rates are published in the same BLS
    "State Employment and Unemployment" release. All 9 verified NYUR dates equal
    the verified payroll date for the same month (one payroll date is a day
    earlier because it comes from the NYS DOL report, not BLS).
  * 79 payroll dates are verified from BLS release pages (bls.gov/news.release).
    Excluding shutdown months, non-January lags are 15-30 days after month end
    and January lags (annual benchmark) are 31-61 days.

Changes (written to data/experiments/, the original files are untouched):
  1. NYUR, NJUR, CTUR and PAUR inherit every BLS-sourced verified payroll date
     for the same reference month, unless they already have their own entry.
  2. Fallback rule for those four rates and NY payrolls: month end + 28 days
     (January + 59 days). Origins are month ends, so this means "published by
     the end of the following month" (January: by the end of March). All 67
     verified non-January, non-shutdown releases met this; 12 of 13 January
     releases did (the exception is the 2026 post-shutdown delay, which is a
     verified entry). Verified entries, including shutdown holdbacks, override it.

Run from the package root:  python -m src.experiments.release_timing_v2
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/experiments'
SAME_RELEASE = ['ny_ur', 'nj_ur', 'ct_ur', 'pa_ur']
NEW_LAG, NEW_JANUARY_LAG = 28, 59


def build():
    rules = pd.read_csv(ROOT / 'data/processed/release_rules.csv')
    calendar = pd.read_csv(ROOT / 'data/processed/release_calendar.csv')

    payroll = calendar[(calendar.variable == 'ny_payrolls')
                       & calendar.status.eq('verified_public_by')
                       & calendar.source_url.str.contains('bls.gov/news.release', na=False)]
    added = []
    for variable in SAME_RELEASE:
        have = set(calendar.loc[calendar.variable == variable, 'reference_date'])
        rows = payroll[~payroll.reference_date.isin(have)].copy()
        rows['variable'] = variable
        rows['evidence'] = ('Inherited from the verified NY payroll date: same BLS State Employment and '
                            'Unemployment release (experiment release_timing_v2).')
        added.append(rows)
    added = pd.concat(added, ignore_index=True)
    calendar_v2 = pd.concat([calendar, added], ignore_index=True)

    rules_v2 = rules.copy()
    rows = rules_v2.variable.isin(SAME_RELEASE + ['ny_payrolls'])
    rules_v2.loc[rows, 'lag_days'] = NEW_LAG
    rules_v2.loc[rows, 'january_lag_days'] = NEW_JANUARY_LAG
    rules_v2.loc[rows, 'rule_status'] = (f'experiment release_timing_v2: estimated month end + {NEW_LAG} days '
                                         f'(January + {NEW_JANUARY_LAG}); verified calendar entries override')

    OUT.mkdir(parents=True, exist_ok=True)
    rules_v2.to_csv(OUT / 'release_rules_v2.csv', index=False)
    calendar_v2.to_csv(OUT / 'release_calendar_v2.csv', index=False)
    print(f'Added {len(added)} inherited calendar rows '
          f'({added.groupby("variable").size().to_dict()}); fallback lag {NEW_LAG}/{NEW_JANUARY_LAG} days')
    return rules_v2, calendar_v2


if __name__ == '__main__':
    build()

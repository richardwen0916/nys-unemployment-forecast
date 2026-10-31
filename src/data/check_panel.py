"""Sanity-check data/processed/monthly_panel.csv and plot every column.

Prints PASS/FAIL for each check and exits with an error if anything fails.
Saves one chart per column to data/processed/panel_plots.png; look at it to
catch problems the checks can't (wrong units, odd spikes, a flat line).

Run from the repo root, after build_panel.py:  python src/data/check_panel.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Values confirmed on the FRED website.
KNOWN_VALUES = {
    ("ny_ur", "2026-08-01"): 4.3,
    ("ny_ur", "2026-07-01"): 4.4,
}

# First month each column should have data (its source start, or the panel
# start in 1976 if the source is older).
EXPECTED_START = {
    "ny_ur": "1976-01-01",
    "ny_claims": "1986-02-01",
    "ny_claims_avg": "1986-03-01",     # Feb 1986 has under 4 weeks of data
    "ny_claims_avg_sa": "1986-03-01",
    "ny_payrolls": "1990-01-01",
    "ny_coincident": "1979-01-01",
    "us_ur": "1976-01-01",
    "us_claims": "1976-01-01",
    "spread_10y3m": "1982-01-01",
    "fedfunds": "1976-01-01",
    "nj_ur": "1976-01-01",
    "ct_ur": "1976-01-01",
    "pa_ur": "1976-01-01",
    "sp500": "1976-01-01",
    "empire_general": "2001-07-01",
    "indeed_postings": "2020-02-01",
}

# Months that are allowed to be missing in the middle of a series.
EXPECTED_GAPS = {
    # Oct 2025: BLS household survey not collected during the federal shutdown.
    "2025-10-01": ["ny_ur", "us_ur", "nj_ur", "ct_ur", "pa_ur"],
}

# Columns that can legitimately go negative (an inverted yield curve; a
# diffusion index below zero means more firms report worse conditions).
CAN_BE_NEGATIVE = ["spread_10y3m", "empire_general"]

REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = REPO_ROOT / "data" / "processed" / "monthly_panel.csv"
PLOT_PATH = REPO_ROOT / "data" / "processed" / "panel_plots.png"
DICTIONARY_PATH = REPO_ROOT / "data" / "processed" / "data_dictionary.csv"

panel = pd.read_csv(PANEL_PATH, parse_dates=["date"]).set_index("date")
failures = []


def check(passed, message):
    print(f"{'PASS' if passed else 'FAIL'}  {message}")
    if not passed:
        failures.append(message)


# 1. Dates: unique, first of the month, no skipped months.
check(not panel.index.duplicated().any(), "no duplicate dates")
check((panel.index.day == 1).all(), "every date is the 1st of the month")
all_months = pd.date_range(panel.index.min(), panel.index.max(), freq="MS")
check(len(panel) == len(all_months), f"no skipped months ({len(panel)} rows)")

# 2. Every expected column is present.
missing_columns = [c for c in EXPECTED_START if c not in panel.columns]
check(not missing_columns, f"all expected columns present {missing_columns or ''}")

# 3. Values that match the source website.
for (column, month), expected in KNOWN_VALUES.items():
    actual = panel.loc[month, column]
    check(actual == expected, f"{column} {month[:7]} = {expected} (got {actual})")

# 4. Each column starts where expected and is reasonably up to date
#    (state data comes out about a month after national data).
latest_allowed = panel.index.max() - pd.DateOffset(months=2)
for column, start in EXPECTED_START.items():
    if column not in panel.columns:
        continue
    values = panel[column]
    first, last = values.first_valid_index(), values.last_valid_index()
    check(first == pd.Timestamp(start), f"{column} starts {start[:7]} (got {first:%Y-%m})")
    check(last >= latest_allowed, f"{column} is current (last value {last:%Y-%m})")

# 5. No unexpected gaps between a column's first and last value.
for column in panel.columns:
    values = panel[column]
    inside = values[values.first_valid_index():values.last_valid_index()]
    gaps = [d.strftime("%Y-%m-%d") for d in inside[inside.isna()].index]
    unexpected = [d for d in gaps if column not in EXPECTED_GAPS.get(d, [])]
    check(not unexpected, f"{column} has no unexpected gaps {unexpected or ''}")

# 6. No negative values where they're impossible.
for column in panel.columns:
    if column in CAN_BE_NEGATIVE:
        continue
    check((panel[column].dropna() >= 0).all(), f"{column} has no negative values")

# 7. Every panel column is documented in the data dictionary.
dictionary = pd.read_csv(DICTIONARY_PATH)
documented = set(dictionary.loc[dictionary["file"] == "monthly_panel.csv", "name"])
undocumented = [c for c in panel.columns if c not in documented]
check(not undocumented, f"every column is in data_dictionary.csv {undocumented or ''}")

# Plot every column on its own small chart.
n_cols = 3
n_rows = -(-len(panel.columns) // n_cols)  # round up
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
for ax, column in zip(axes.flat, panel.columns):
    panel[column].plot(ax=ax, linewidth=0.8)
    ax.set_title(column)
    ax.set_xlabel("")
for ax in list(axes.flat)[len(panel.columns):]:
    ax.set_visible(False)
fig.tight_layout()
fig.savefig(PLOT_PATH, dpi=100)
print(f"\nSaved plots to {PLOT_PATH}")

if failures:
    sys.exit(f"\n{len(failures)} check(s) failed.")
print("\nAll checks passed.")

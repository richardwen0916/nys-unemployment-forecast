"""Build the monthly panel from the untouched files in data/raw/.

Every series is converted to one value per month and joined on `date`:
  - monthly series: used as-is
  - weekly claims:  the week containing the 12th (the unemployment survey reference week)
  - daily series:   monthly average
NY claims also get a monthly-average version (`ny_claims_avg`) and an STL
seasonally adjusted version of it (`ny_claims_avg_sa`). See WEEKLY_AVERAGE_SERIES.

Rules: rows are dated by reference month (not release date), values are levels,
and missing values stay NaN (no filling).

Run from the repo root:  python src/data/build_panel.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

# Raw file name (in data/raw/, without .csv) -> panel column name, grouped by
# how each series is converted to monthly.
MONTHLY_SERIES = {
    "fred_NYUR": "ny_ur",
    "fred_NYNA": "ny_payrolls",
    "fred_NYPHCI": "ny_coincident",
    "fred_UNRATE": "us_ur",
    "fred_FEDFUNDS": "fedfunds",
    "fred_NJUR": "nj_ur",
    "fred_CTUR": "ct_ur",
    "fred_PAUR": "pa_ur",
    "fred_GACDISA066MSFRBNY": "empire_general",
}
WEEKLY_SERIES = {
    "fred_NYICLAIMS": "ny_claims",
    "fred_ICSA": "us_claims",
}
# NY claims are NSA with a short spike right after July 4. Whether the 12th's
# week lands inside that spike depends on the year's calendar, which breaks
# seasonal adjustment of the reference-week series. The average of all weeks in
# the month always includes the spike, so it adjusts cleanly.
WEEKLY_AVERAGE_SERIES = {
    "fred_NYICLAIMS": "ny_claims_avg",
}
DAILY_SERIES = {
    "fred_T10Y3M": "spread_10y3m",
    "yf_GSPC": "sp500",
    "indeed_NY": "indeed_postings",
}
# NSA columns that get an STL-adjusted "<name>_sa" column.
SEASONALLY_ADJUST = ["ny_claims_avg"]

# Final column order in the panel.
COLUMN_ORDER = [
    "ny_ur", "ny_claims", "ny_claims_avg", "ny_claims_avg_sa", "ny_payrolls", "ny_coincident",
    "us_ur", "us_claims", "spread_10y3m", "fedfunds",
    "nj_ur", "ct_ur", "pa_ur", "sp500",
    "empire_general", "indeed_postings",
]

# Panel starts with the target's history (NYUR begins Jan 1976) and ends at the
# last complete calendar month, so daily averages never cover a partial month.
PANEL_START = pd.Timestamp("1976-01-01")
PANEL_END = (pd.Timestamp.today().to_period("M") - 1).to_timestamp()

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
OUT_PATH = REPO_ROOT / "data" / "processed" / "monthly_panel.csv"


def read_raw(name):
    """Read a raw file as a Series of values indexed by date."""
    df = pd.read_csv(RAW_DIR / f"{name}.csv", parse_dates=["date"])
    return df.set_index("date").iloc[:, 0]


def weekly_to_monthly(series):
    """Keep the week containing the 12th of each month.

    The unemployment rate comes from a household survey that asks about the
    week containing the 12th, so this lines claims up with the target's
    reference period. Used for seasonally adjusted weekly series (ICSA); NY
    claims also get a monthly average, see WEEKLY_AVERAGE_SERIES.

    FRED dates weekly claims by the Saturday the week ends on. The week that
    contains the 12th is the one whose Saturday falls on the 12th-18th, and
    every month has exactly one such Saturday.
    """
    if not (series.index.dayofweek == 5).all():
        raise ValueError("Expected every weekly date to be a Saturday.")
    reference_weeks = series[(series.index.day >= 12) & (series.index.day <= 18)]
    reference_weeks.index = reference_weeks.index.to_period("M").to_timestamp()
    return reference_weeks


def weekly_to_monthly_average(series):
    """Average all weeks whose ending Saturday falls in the month.

    Every month has 4 or 5 Saturdays; months with fewer weeks of data (the
    first month of the series, or a month still being reported) become NaN.
    """
    monthly = series.resample("MS")
    return monthly.mean().where(monthly.count() >= 4)


def daily_to_monthly(series):
    """Average all days in each month (blank days, e.g. holidays, are skipped)."""
    return series.resample("MS").mean()


def stl_adjust(series):
    """Seasonally adjust a monthly series with STL.

    Claims seasonality scales with the level (a January spike is bigger when
    claims are high), so we adjust in logs: remove the seasonal part of
    log(claims), then convert back.
    """
    observed = series.dropna()
    expected_months = pd.date_range(observed.index.min(), observed.index.max(), freq="MS")
    if len(observed) != len(expected_months):
        raise ValueError(f"{series.name} has gaps; STL needs an unbroken monthly series.")

    log_values = np.log(observed)
    fit = STL(log_values, period=12, robust=True).fit()
    adjusted = np.exp(log_values - fit.seasonal)
    return adjusted.reindex(series.index)


# Convert every raw series to monthly.
columns = {}
for name, column in MONTHLY_SERIES.items():
    columns[column] = read_raw(name)
for name, column in WEEKLY_SERIES.items():
    columns[column] = weekly_to_monthly(read_raw(name))
for name, column in WEEKLY_AVERAGE_SERIES.items():
    columns[column] = weekly_to_monthly_average(read_raw(name))
for name, column in DAILY_SERIES.items():
    columns[column] = daily_to_monthly(read_raw(name))

# Join on date and keep exactly one row per month in the panel range.
# Months a series doesn't cover become NaN.
all_months = pd.date_range(PANEL_START, PANEL_END, freq="MS", name="date")
panel = pd.DataFrame(columns).reindex(all_months)

for column in SEASONALLY_ADJUST:
    panel[f"{column}_sa"] = stl_adjust(panel[column])

panel = panel[COLUMN_ORDER]

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
panel.to_csv(OUT_PATH, date_format="%Y-%m-%d")

# Summary: where each column starts and ends, and how many months are missing.
print(f"Wrote {OUT_PATH.name}: {len(panel)} rows, {PANEL_START:%Y-%m} to {PANEL_END:%Y-%m}\n")
print(f"{'column':<18}{'first':>10}{'last':>10}{'NaN':>6}")
for column in panel.columns:
    values = panel[column]
    first = f"{values.first_valid_index():%Y-%m}"
    last = f"{values.last_valid_index():%Y-%m}"
    print(f"{column:<18}{first:>10}{last:>10}{values.isna().sum():>6}")

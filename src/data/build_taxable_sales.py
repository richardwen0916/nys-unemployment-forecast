"""Build the quarterly NYS taxable sales file from data/raw/taxable_sales_NY.csv.

Kept separate from the monthly panel: the data are quarterly, and spreading
them across months would invent values.

Steps:
  1. Sum all industry groups into one statewide total per quarter. A few small
     industries are suppressed (blank) for confidentiality; they count as zero.
  2. Date each quarter by its first month. NY's sales tax year runs March to
     February, so quarters are Mar-May, Jun-Aug, Sep-Nov, Dec-Feb (not calendar
     quarters). E.g. sales tax year 2026-2027, quarter 1 -> 2026-03-01.
  3. Add an STL seasonally adjusted column (the source is NSA).

Run from the repo root:  python src/data/build_taxable_sales.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = REPO_ROOT / "data" / "raw" / "taxable_sales_NY.csv"
OUT_PATH = REPO_ROOT / "data" / "processed" / "quarterly_taxable_sales.csv"

raw = pd.read_csv(RAW_PATH)

# Sales tax year "2026 - 2027", quarter 1 -> first month 2026-03.
start_year = raw["sales_tax_year"].str[:4].astype(int)
start_month = 3 + 3 * (raw["sales_tax_quarter"] - 1)
raw["date"] = pd.to_datetime(
    start_year.astype(str) + "-" + start_month.astype(str).str.zfill(2) + "-01"
)

# One row per quarter. A quarter is preliminary (P) if any of its rows are.
quarterly = raw.groupby("date").agg(
    selling_period=("selling_period", "first"),
    status=("status", lambda s: "P" if (s == "P").any() else "F"),
    taxable_sales=("taxable_sales_and_purchases", "sum"),
    suppressed_industries=("taxable_sales_and_purchases", lambda s: s.isna().sum()),
)

# STL needs an unbroken series: every quarter from first to last must be present.
expected_quarters = pd.date_range(quarterly.index.min(), quarterly.index.max(), freq="3MS")
if len(quarterly) != len(expected_quarters):
    raise ValueError("Missing quarters; STL needs an unbroken quarterly series.")

# Seasonal adjustment in logs (seasonality scales with the level), period = 4 quarters.
# seasonal=13 keeps the seasonal pattern stable across years. With the default (7)
# and only 4 points a year, the 2020 COVID crash bent the Mar-May factor to 0.82
# (vs ~1.00 every other year), hiding a real drop inside the "seasonal" part.
log_sales = np.log(quarterly["taxable_sales"])
fit = STL(log_sales, period=4, seasonal=13, robust=True).fit()
quarterly["taxable_sales_sa"] = np.exp(log_sales - fit.seasonal).round(0)

quarterly = quarterly[
    ["selling_period", "status", "taxable_sales", "taxable_sales_sa", "suppressed_industries"]
]
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
quarterly.to_csv(OUT_PATH, date_format="%Y-%m-%d")

print(
    f"Wrote {OUT_PATH.name}: {len(quarterly)} quarters, "
    f"{quarterly.index.min():%Y-%m} to {quarterly.index.max():%Y-%m} "
    f"({(quarterly['status'] == 'P').sum()} preliminary)\n"
)
print(f"{'quarter':<10}{'months':<22}{'status':<8}{'$bn':>8}{'$bn SA':>9}")
for date, row in quarterly.tail(6).iterrows():
    print(
        f"{date:%Y-%m}".ljust(10) + f"{row['selling_period']:<22}{row['status']:<8}"
        f"{row['taxable_sales'] / 1e9:>8.1f}{row['taxable_sales_sa'] / 1e9:>9.1f}"
    )

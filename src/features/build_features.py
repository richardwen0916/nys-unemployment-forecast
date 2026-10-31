"""Turn the monthly panel into model-ready features for the time series models (Step 2).

Reads data/processed/monthly_panel.csv and quarterly_taxable_sales.csv, writes:
  - data/features/model_features.csv     one row per month, every feature below
  - data/features/feature_dictionary.csv  what each column is and when it is published

What it does:
  1. Fills the October 2025 gap in the unemployment rates (shutdown) by linear
     interpolation, and flags it in `ur_imputed`. `ny_ur_raw` keeps the gap as
     NaN for SARIMAX, which skips missing values on its own (no look-ahead);
     the interpolated columns are for VAR, which cannot take NaN.
  2. Makes every series stationary: differences for rates, 100 x log differences
     (monthly growth in %) for trending levels. empire_general is already
     stationary, so it stays in levels.
  3. Adds year-over-year claims growth on the NSA series. A 12-month log
     difference removes seasonality without STL, so it has no look-ahead.
  4. Adds lags 1-6 and 12 of the main regressors. ARIMAX uses direct
     forecasting: the model for horizon h only uses `_lagK` columns with K >= h,
     so every input is known when the forecast is made.
  5. Adds a COVID flag, sample flags (long / Empire / Indeed samples) and a
     `split` column (train / validation / test) that avoids COVID and the
     October 2025 gap. See SPLITS.
  6. Puts quarterly taxable sales on the monthly grid only from the month it
     would have been published, held constant until the next release. No
     values are invented between quarters.

Rows are still dated by reference month. `pub_lag_months` in the feature
dictionary says how many months after the reference month a value is
published, so backtests can drop anything not yet known at the forecast date.

Run from the repo root:  python src/features/build_features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PANEL = Path("data/processed/monthly_panel.csv")
TAXABLE = Path("data/processed/quarterly_taxable_sales.csv")
OUT_DIR = Path("data/features")

UR_COLUMNS = ["ny_ur", "us_ur", "nj_ur", "ct_ur", "pa_ur"]

# Panel column -> transform. "diff": first difference (percentage points).
# "growth": 100 x log difference (monthly % growth). "log": natural log.
TRANSFORMS = {
    "ny_ur": ["diff"],
    "us_ur": ["diff"],
    "nj_ur": ["diff"],
    "ct_ur": ["diff"],
    "pa_ur": ["diff"],
    "fedfunds": ["diff"],
    "spread_10y3m": ["diff"],
    "ny_payrolls": ["growth"],
    "ny_coincident": ["growth"],
    "sp500": ["growth"],
    "indeed_postings": ["growth"],
    "ny_claims_avg_sa": ["growth"],
    "us_claims": ["growth"],
}

# Regressors that get lagged copies, and which lags.
LAGS = [1, 2, 3, 4, 5, 6, 12]
LAGGED = [
    "d_ny_ur",
    "g_ny_claims_avg_sa", "yoy_ny_claims",
    "g_ny_payrolls", "g_ny_coincident",
    "empire_general", "d_us_ur", "g_us_claims",
    "d_fedfunds", "spread_10y3m", "g_sp500",
    "d_nj_ur", "d_ct_ur", "d_pa_ur",
]

COVID_START, COVID_END = "2020-03-01", "2021-12-01"

# Train / validation / test windows (inclusive). All three avoid the COVID
# shock (2020-03 to 2021-12, flagged "excluded") and the October 2025 gap, and
# start when every core regressor exists (1990-02). Rows after the test window
# ("live") are only for refitting and producing the real forecast.
SPLITS = {
    "train": ("1990-02-01", "2016-12-01"),
    "validation": ("2017-01-01", "2020-02-01"),
    "test": ("2022-01-01", "2025-09-01"),
}

# Months after the reference month until the value is out (from the
# typical_release_lag column of data/processed/data_dictionary.csv).
# 0 = known within the reference month or in the first days of the next one,
# before NYUR for that month comes out.
PUB_LAG = {
    "ny_ur": 1, "nj_ur": 1, "ct_ur": 1, "pa_ur": 1, "ny_payrolls": 1, "ny_coincident": 1,
    "us_ur": 0, "ny_claims_avg_sa": 0, "ny_claims_avg": 0, "us_claims": 0,
    "fedfunds": 0, "spread_10y3m": 0, "sp500": 0, "empire_general": 0, "indeed_postings": 0,
}
# Taxable sales come out about 3 months after the quarter ends.
TAXABLE_RELEASE_LAG = 3

panel = pd.read_csv(PANEL, parse_dates=["date"], index_col="date").asfreq("MS")
taxable = pd.read_csv(TAXABLE, parse_dates=["date"], index_col="date")

# 1. October 2025 gap in the unemployment rates
out = pd.DataFrame(index=panel.index)
out["ur_imputed"] = panel["ny_ur"].isna() & panel["ny_ur"].ffill().notna() & panel["ny_ur"].bfill().notna()
out["ur_imputed"] = out["ur_imputed"].astype(int)
out["ny_ur_raw"] = panel["ny_ur"]
for col in UR_COLUMNS:
    panel[col] = panel[col].interpolate(limit_area="inside")

# Levels, kept for SARIMA (which differences inside the model) and for VAR in levels
level_columns = UR_COLUMNS + ["fedfunds", "spread_10y3m", "empire_general", "indeed_postings"]
for col in level_columns:
    out[col] = panel[col]

# 2. Stationary transforms
dictionary = []
for col, kinds in TRANSFORMS.items():
    for kind in kinds:
        if kind == "diff":
            name = f"d_{col}"
            out[name] = panel[col].diff()
            desc = f"first difference of {col}"
        elif kind == "log":
            name = f"log_{col}"
            out[name] = np.log(panel[col])
            desc = f"natural log of {col}"
        else:
            name = f"g_{col}"
            out[name] = 100 * np.log(panel[col]).diff()
            desc = f"100 x log difference of {col} (monthly growth, %)"
        dictionary.append((name, col, desc))

# 3. Year-over-year NY claims growth on the NSA monthly average: no STL, no look-ahead
out["yoy_ny_claims"] = 100 * np.log(panel["ny_claims_avg"]).diff(12)
dictionary.append(("yoy_ny_claims", "ny_claims_avg",
                   "100 x 12-month log difference of NSA ny_claims_avg (removes seasonality without STL look-ahead)"))

# 4. Lags
lagged = {}
for col in LAGGED:
    for lag in LAGS:
        name = f"{col}_lag{lag}"
        lagged[name] = out[col].shift(lag)
        source = next((s for n, s, _ in dictionary if n == col), col)
        dictionary.append((name, source, f"{col} lagged {lag} month(s)"))
out = pd.concat([out, pd.DataFrame(lagged)], axis=1)

# 5. COVID dummies and sample flags
out = out.copy()  # defragment after adding many columns one by one
out["covid"] = ((out.index >= COVID_START) & (out.index <= COVID_END)).astype(int)
core = ["d_ny_ur", "g_ny_claims_avg_sa", "g_ny_payrolls", "g_ny_coincident", "d_us_ur", "g_us_claims"]
out["sample_long"] = out[core].notna().all(axis=1).astype(int)
out["sample_empire"] = (out["sample_long"].astype(bool) & out["empire_general"].notna()).astype(int)
out["sample_indeed"] = (out["sample_long"].astype(bool) & out["g_indeed_postings"].notna()).astype(int)
out["split"] = "before"
out.loc[out.index > SPLITS["test"][1], "split"] = "live"
out.loc[(out.index >= COVID_START) & (out.index <= COVID_END), "split"] = "excluded"
for name, (start, end) in SPLITS.items():
    out.loc[start:end, "split"] = name

# 6. Taxable sales on the monthly grid, from the month it is published
taxable["g_yoy_taxable_sales"] = 100 * np.log(taxable["taxable_sales"]).diff(4)
taxable["g_taxable_sales_sa"] = 100 * np.log(taxable["taxable_sales_sa"]).diff()
# Quarter starts on `date` and covers 3 months; published TAXABLE_RELEASE_LAG months after it ends.
release_month = taxable.index + pd.DateOffset(months=3 + TAXABLE_RELEASE_LAG)
released = taxable[["g_yoy_taxable_sales", "g_taxable_sales_sa"]].set_axis(release_month)
released = released.reindex(out.index).ffill()
# Stop carrying the last release forward past the next expected release
last_release = release_month.max()
released.loc[released.index >= last_release + pd.DateOffset(months=3)] = np.nan
out["taxable_yoy_latest"] = released["g_yoy_taxable_sales"]
out["taxable_sa_growth_latest"] = released["g_taxable_sales_sa"]

out.index.name = "date"
OUT_DIR.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT_DIR / "model_features.csv", float_format="%.6g")

# Feature dictionary
base_rows = [
    ("ur_imputed", "", "1 if the month's unemployment rates were linearly interpolated (Oct 2025, shutdown)"),
    ("covid", "", f"1 from {COVID_START[:7]} to {COVID_END[:7]}"),
    ("ny_ur_raw", "ny_ur", "ny_ur with the Oct 2025 gap left as NaN (for SARIMAX)"),
    ("split", "", "train 1990-02..2016-12 / validation 2017-01..2020-02 / excluded = COVID 2020-03..2021-12 / "
                  "test 2022-01..2025-09 / live = after the test window (contains the Oct 2025 gap) / before = pre-1990-02"),
    ("sample_long", "", "1 when all core regressors exist (NY claims, payrolls, coincident, US UR/claims): 1990-02 on"),
    ("sample_empire", "", "sample_long and empire_general exist: 2001-07 on"),
    ("sample_indeed", "", "sample_long and indeed_postings growth exist: 2020-03 on"),
    ("taxable_yoy_latest", "taxable_sales",
     "latest published NSA taxable sales growth vs same quarter last year (%), placed from its release month"),
    ("taxable_sa_growth_latest", "taxable_sales_sa",
     "latest published SA taxable sales growth vs previous quarter (%), placed from its release month"),
] + [(c, c, f"{c} level from monthly_panel.csv" + (" (Oct 2025 interpolated)" if c in UR_COLUMNS else ""))
     for c in level_columns]

rows = []
for name, source, desc in base_rows + dictionary:
    lag = int(name.rsplit("lag", 1)[1]) if "_lag" in name else 0
    pub = PUB_LAG.get(source, 0 if source == "" else None)
    if source.startswith("taxable"):
        pub = 0  # already placed at its release month
    first = out[name].first_valid_index()
    rows.append(dict(name=name, source_column=source, description=desc,
                     pub_lag_months=None if pub is None else max(pub - lag, 0) if lag else pub,
                     start=first.strftime("%Y-%m") if first is not None else ""))
fd = pd.DataFrame(rows).set_index("name").loc[out.columns]
fd.index.name = "name"
fd.to_csv(OUT_DIR / "feature_dictionary.csv")

print(f"Wrote {OUT_DIR / 'model_features.csv'}: {out.shape[0]} rows x {out.shape[1]} columns, "
      f"{out.index.min():%Y-%m} to {out.index.max():%Y-%m}")

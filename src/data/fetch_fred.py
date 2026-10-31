"""Download every FRED series used in the panel and save each one untouched to data/raw/.

No transformations happen here: each file keeps FRED's own dates and frequency
(weekly stays weekly, daily stays daily). build_panel.py does all conversion.

Run from the repo root:  python src/data/fetch_fred.py
"""

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

# FRED series ID -> panel column name. Add new FRED series here.
FRED_SERIES = {
    "NYUR": "ny_ur",               # NY unemployment rate (target), monthly, SA
    "NYICLAIMS": "ny_claims",      # NY initial claims, weekly, NSA
    "NYNA": "ny_payrolls",         # NY total nonfarm employees, monthly, SA, thousands
    "NYPHCI": "ny_coincident",     # NY coincident economic activity index, monthly, SA
    "UNRATE": "us_ur",             # US unemployment rate, monthly, SA
    "ICSA": "us_claims",           # US initial claims, weekly, SA
    "T10Y3M": "spread_10y3m",      # 10-year minus 3-month Treasury spread, daily
    "FEDFUNDS": "fedfunds",        # Effective federal funds rate, monthly
    "NJUR": "nj_ur",               # NJ unemployment rate, monthly, SA
    "CTUR": "ct_ur",               # CT unemployment rate, monthly, SA
    "PAUR": "pa_ur",               # PA unemployment rate, monthly, SA
    "GACDISA066MSFRBNY": "empire_general",  # NY Fed Empire State survey, general business conditions, monthly, SA
}

# Repo root is two folders up from this file (src/data/fetch_fred.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"

load_dotenv(REPO_ROOT / ".env")
api_key = os.getenv("FRED_API_KEY")
if not api_key:
    sys.exit("FRED_API_KEY is missing. Add it to .env (see .env.example).")

fred = Fred(api_key=api_key)
RAW_DIR.mkdir(parents=True, exist_ok=True)

for series_id, column in FRED_SERIES.items():
    # get_series returns the full history as a pandas Series indexed by date.
    series = fred.get_series(series_id)

    raw = pd.DataFrame({"date": series.index, "value": series.values})
    out_path = RAW_DIR / f"fred_{series_id}.csv"
    raw.to_csv(out_path, index=False)

    print(
        f"{series_id:<10} -> {out_path.name:<20} "
        f"{len(raw):>6} rows  {raw['date'].min():%Y-%m-%d} to {raw['date'].max():%Y-%m-%d}"
    )

print(f"\nSaved {len(FRED_SERIES)} FRED series to {RAW_DIR}")

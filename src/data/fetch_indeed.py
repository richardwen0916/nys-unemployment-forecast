"""Download the Indeed Hiring Lab job postings index for New York and save it to data/raw/.

Source: github.com/hiring-lab/job_postings_tracker (US/state_job_postings_us.csv),
licensed CC BY 4.0. The index is daily, already seasonally adjusted, and smoothed
with a 7-day trailing average; Feb 1, 2020 = 100. Indeed revises history, so
rerunning can change past values.

The source file covers every state. We keep only New York's rows and drop the
state column; the values themselves are untouched. build_panel.py converts the
daily index to monthly averages.

Run from the repo root:  python src/data/fetch_indeed.py
"""

from pathlib import Path

import pandas as pd

SOURCE_URL = (
    "https://raw.githubusercontent.com/hiring-lab/job_postings_tracker/"
    "master/US/state_job_postings_us.csv"
)
STATE = "ny"

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
OUT_PATH = RAW_DIR / "indeed_NY.csv"

source = pd.read_csv(SOURCE_URL, parse_dates=["date"])
ny = source[source["state"] == STATE]
if ny.empty:
    raise RuntimeError(f"No rows for state '{STATE}'. Check the source file's state codes.")

raw = ny[["date", "indeed_job_postings_index"]]
RAW_DIR.mkdir(parents=True, exist_ok=True)
raw.to_csv(OUT_PATH, index=False)

print(
    f"Indeed {STATE.upper()} -> {OUT_PATH.name}  {len(raw)} rows  "
    f"{raw['date'].min():%Y-%m-%d} to {raw['date'].max():%Y-%m-%d}"
)

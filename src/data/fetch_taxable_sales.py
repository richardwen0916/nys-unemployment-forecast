"""Download NYS taxable sales and purchases (quarterly) and save them untouched to data/raw/.

Source: NY open data, "Taxable Sales And Purchases Quarterly Data: Beginning
Sales Tax Year 2013-2014" (data.ny.gov dataset ny73-2j3u). Values are dollars,
not seasonally adjusted, broken out by NAICS industry group.

The source covers every county. We keep only the statewide rows
(jurisdiction "NY STATE"); the values themselves are untouched.
build_taxable_sales.py sums the industries into a statewide total.

Run from the repo root:  python src/data/fetch_taxable_sales.py
"""

import io
from pathlib import Path

import pandas as pd
import requests

SOURCE_URL = "https://data.ny.gov/resource/ny73-2j3u.csv"
JURISDICTION = "NY STATE"
ROW_LIMIT = 100000  # the API returns at most this many rows per request

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
OUT_PATH = RAW_DIR / "taxable_sales_NY.csv"

response = requests.get(
    SOURCE_URL,
    params={"$where": f"jurisdiction='{JURISDICTION}'", "$limit": ROW_LIMIT},
    timeout=60,
)
response.raise_for_status()
raw = pd.read_csv(io.StringIO(response.text))

if raw.empty:
    raise RuntimeError(f"No rows for jurisdiction '{JURISDICTION}'.")
if len(raw) == ROW_LIMIT:
    raise RuntimeError("Hit the row limit; some rows may be missing. Raise ROW_LIMIT.")

RAW_DIR.mkdir(parents=True, exist_ok=True)
raw.to_csv(OUT_PATH, index=False)

years = sorted(raw["sales_tax_year"].unique())
print(f"Taxable sales -> {OUT_PATH.name}  {len(raw)} rows  sales tax years {years[0]} to {years[-1]}")

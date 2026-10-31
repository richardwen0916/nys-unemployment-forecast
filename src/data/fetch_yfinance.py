"""Download daily market data from Yahoo Finance and save it untouched to data/raw/.

Kept separate from fetch_fred.py so each fetch script talks to exactly one source.
build_panel.py converts the daily closes to monthly averages.

Run from the repo root:  python src/data/fetch_yfinance.py
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

# Yahoo ticker -> panel column name.
YF_TICKERS = {
    "^GSPC": "sp500",  # S&P 500 index, daily close
}

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

for ticker, column in YF_TICKERS.items():
    # period="max" pulls the full history. auto_adjust=False keeps the plain close price.
    history = yf.Ticker(ticker).history(period="max", auto_adjust=False)
    if history.empty:
        raise RuntimeError(f"No data returned for {ticker}. Check the ticker or try again later.")

    # Yahoo dates carry a time zone (e.g. 2026-08-03 00:00-04:00); keep just the calendar date.
    raw = pd.DataFrame({
        "date": history.index.tz_localize(None).normalize(),
        "close": history["Close"].values,
    })
    out_path = RAW_DIR / f"yf_{ticker.lstrip('^')}.csv"
    raw.to_csv(out_path, index=False)

    print(
        f"{ticker:<10} -> {out_path.name:<20} "
        f"{len(raw):>6} rows  {raw['date'].min():%Y-%m-%d} to {raw['date'].max():%Y-%m-%d}"
    )

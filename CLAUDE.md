# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Team project for MGMTMSA 437 (Forecasting and Time Series Analysis). Question: can economic indicators and alternative data improve short-term forecasts of the New York State unemployment rate (FRED `NYUR`)? The end product is a Streamlit + Plotly forecasting dashboard with a built-in Claude API assistant.

Work is split into steps owned by different teammates: (1) data pipeline, (2) time series models, (3) ML + validation, (4) dashboard, (5) AI assistant. Steps 2-4 consume the processed panel produced by step 1, so panel column names are a shared contract: do not rename or remove columns without telling the team.

## Setup and commands

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in `FRED_API_KEY`. `.env` is gitignored; never commit it.

Run the pipeline from the repo root, in order:

```
python src/data/fetch_fred.py
python src/data/fetch_yfinance.py
python src/data/fetch_indeed.py
python src/data/build_panel.py
python src/data/check_panel.py
python src/data/fetch_taxable_sales.py
python src/data/build_taxable_sales.py
```

There is no test suite or linter configured. Validate pipeline changes with sanity checks on the output (latest values match the source site, dates unique and monthly, each column starts near its source history start, NaNs only where expected) and by plotting each column.

## Data pipeline architecture

- `data/raw/`: untouched source data written by `src/data/fetch_*.py` scripts. Never edited by hand.
- `data/manual/`: files that must be downloaded by hand. Record the source URL and download date alongside each one.
- `data/processed/`: output of `src/data/build_panel.py`, which reads only from `data/raw/` and `data/manual/`. Main output is `monthly_panel.csv`; `data_dictionary.csv` documents each column (description, unit, source, series ID, original frequency, transformation, start date). The processed CSV is committed so the team has a frozen snapshot.

## Pipeline rules

- Scripts are plain `.py` (not notebooks), rerunnable top to bottom, and overwrite their outputs cleanly.
- Keep code simple and readable; every transformation must be explainable by its owner. No classes or abstractions the task doesn't need.
- Series lists live in a config dict at the top of each script (source series ID -> panel column name).
- Secrets come from `.env` via python-dotenv.
- Verify every series ID on the source site before relying on it.

### Panel format

- Wide, one row per month, `date` column formatted `YYYY-MM-01`.
- Rows are aligned to the **reference month, not the release date**. Modelers must handle publication lag themselves to avoid look-ahead in backtests.
- Levels only. Growth rates, differencing, and lags are the modelers' job.
- Missing values stay NaN. No forward-fill, back-fill, or zero-fill (e.g. Indeed postings are genuinely missing before 2020).

### Frequency conversion

- Weekly claims -> monthly: take the week containing the 12th (the unemployment survey reference week). FRED weeks end on Saturday, so use the week whose ending Saturday falls on the 12th-18th of the month.
- Daily series (e.g. T10Y3M, ^GSPC, Indeed) -> monthly: monthly average.
- Quarterly series (NYS taxable sales) go in a separate file and are never interpolated into the monthly panel.
- Not-seasonally-adjusted series (e.g. NYICLAIMS, taxable sales): keep the raw column and add an STL-adjusted `<name>_sa` column.
- Exception for NY claims: the reference-week value is distorted by a short post-July 4 spike that the 12th's week only sometimes catches, so STL on it is unreliable. The panel has `ny_claims` (reference week, NSA), `ny_claims_avg` (average of all weeks ending in the month, NSA, NaN if fewer than 4 weeks) and `ny_claims_avg_sa` (STL on `ny_claims_avg`). Model with `ny_claims_avg_sa`. There is deliberately no SA reference-week column.
- STL is two-sided (it uses later data to estimate each month's seasonal factor), so `_sa` columns contain mild look-ahead; strict backtests should re-run the adjustment on each training window.

## Git workflow

Never commit directly to `main`. Work on a feature branch, push, open a PR (`gh pr create`), get a teammate review, then merge. Keep commits small with clear messages. If `main` moves, `git pull origin main` into the branch.

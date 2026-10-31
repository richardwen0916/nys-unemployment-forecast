"""SUPERSEDED (v0 exploratory). Kept for the record; do not use these results.

It uses the full-sample STL claims column (look-ahead) and ignores publication
lags. The final models are in modeling/ (see docs/STEP2_HANDOFF.md).

Time series models for NYUR (Step 2), evaluated with a rolling-origin backtest.

Models
  naive       next months = last observed NYUR
  sarima      SARIMA on ny_ur_raw, order picked by AIC on the train split
  arx_ar      direct AR model on NYUR changes only (no regressors), one per horizon
  arx_econ    arx_ar + NY claims, NY payrolls, NY coincident index, US UR, US claims
  arx_empire  arx_econ + Empire survey (starts 2001-07, so a shorter sample)
  arx_full    arx_econ + fed funds, 10y-3m spread, S&P 500
  var         VAR in first differences of NY, NJ, CT, PA unemployment rates

Direct forecasting (arx_*): for horizon h the model is
  ny_ur(t) - ny_ur(t-h) = a + sum_k b_k d_ny_ur(t-h-k) + sum_j c_j x(t-h-j) + e
so every input is at least h months old: the _lagK >= h rule.

Backtest: for each target month in the validation and test splits and each
horizon h, the forecast origin is target - h. Models are refit on data up to
the origin only. COVID months (split == "excluded") are never used to fit:
SARIMA sees them as NaN, regression/VAR rows that touch them are dropped.
No imputed values are used anywhere (Oct 2025 stays NaN).

Writes
  results/forecasts.csv  date, origin, model, horizon, yhat, lower, upper, actual, split
  results/metrics.csv    MAE / RMSE by model, horizon and split

Run from the repo root:  python src/models/run_models.py
"""

import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

FEATURES = Path("data/features/model_features.csv")
PANEL = Path("data/processed/monthly_panel.csv")
OUT_DIR = Path("results/v0_exploratory")

HORIZONS = [1, 3, 6]
TRAIN_START = "1990-02-01"
COVID = ("2020-03-01", "2021-12-01")
Z95 = 1.96

OWN_LAGS = 6       # d_ny_ur lags h .. h+5
X_LAGS = 3         # regressor lags h .. h+2
ARX_SETS = {
    "arx_ar": [],
    "arx_econ": ["g_ny_claims_avg_sa", "g_ny_payrolls", "g_ny_coincident", "d_us_ur", "g_us_claims"],
    "arx_empire": ["g_ny_claims_avg_sa", "g_ny_payrolls", "g_ny_coincident", "d_us_ur", "g_us_claims",
                   "empire_general"],
    "arx_full": ["g_ny_claims_avg_sa", "g_ny_payrolls", "g_ny_coincident", "d_us_ur", "g_us_claims",
                 "d_fedfunds", "spread_10y3m", "g_sp500"],
}
VAR_COLUMNS = ["ny_ur", "nj_ur", "ct_ur", "pa_ur"]
VAR_LAGS = 3       # HQIC on the train split (AIC picks 5, BIC 2)
SARIMA_GRID = dict(p=range(0, 4), q=range(0, 4), P=[0, 1])

features = pd.read_csv(FEATURES, parse_dates=["date"], index_col="date").asfreq("MS")
panel = pd.read_csv(PANEL, parse_dates=["date"], index_col="date").asfreq("MS")
# Extend the calendar so live forecasts have rows for future target months
future = pd.date_range(features.index[0], features.index[-1] + pd.DateOffset(months=max(HORIZONS)), freq="MS")
features, panel = features.reindex(future), panel.reindex(future)

y = features["ny_ur_raw"].copy()                # NaN in Oct 2025, never imputed
in_covid = (y.index >= COVID[0]) & (y.index <= COVID[1])
y_fit = y.copy()
y_fit[in_covid] = np.nan                        # COVID months hidden from every fit
dy = y.diff()
split = features["split"]
covid_series = pd.Series(in_covid, index=y.index)

# Every forecast to make: (origin, horizon, target)
jobs = []
for target in split[split.isin(["validation", "test"])].index:
    for h in HORIZONS:
        jobs.append((target - pd.DateOffset(months=h), h, target))
live_origin = y.last_valid_index()
for h in HORIZONS:
    jobs.append((live_origin, h, live_origin + pd.DateOffset(months=h)))
origins = sorted({o for o, _, _ in jobs})
rows = []


def add(model, origin, h, target, yhat, sd):
    rows.append(dict(date=target, origin=origin, model=model, horizon=h, yhat=yhat,
                     lower=yhat - Z95 * sd, upper=yhat + Z95 * sd))


def touches_covid(t, back):
    """True if any month in [t - back, t] is a COVID month."""
    return covid_series.loc[t - pd.DateOffset(months=back):t].any()


# ---------------------------------------------------------------- naive
for origin, h, target in jobs:
    hist = dy.loc[TRAIN_START:origin]
    hist = hist[~covid_series.loc[hist.index]]
    # random walk: h-step error sd = sqrt(h) * sd of monthly changes
    add("naive", origin, h, target, y.loc[:origin].dropna().iloc[-1], np.sqrt(h) * hist.std())

# ---------------------------------------------------------------- SARIMA
train_y = y_fit.loc[TRAIN_START:"2016-12-01"]
best = None
for p, q, P in itertools.product(SARIMA_GRID["p"], SARIMA_GRID["q"], SARIMA_GRID["P"]):
    try:
        res = SARIMAX(train_y, order=(p, 1, q), seasonal_order=(P, 0, 0, 12) if P else (0, 0, 0, 0)).fit(disp=False)
    except Exception:
        continue
    if best is None or res.aic < best[0]:
        best = (res.aic, (p, 1, q), (P, 0, 0, 12) if P else (0, 0, 0, 0))
SARIMA_ORDER, SARIMA_SEASONAL = best[1], best[2]
print(f"SARIMA order {SARIMA_ORDER} seasonal {SARIMA_SEASONAL} (train AIC {best[0]:.1f})")

sarima_fc = {}
start_params = None
for origin in origins:
    res = SARIMAX(y_fit.loc[TRAIN_START:origin], order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL) \
        .fit(disp=False, start_params=start_params)
    start_params = res.params
    fc = res.get_forecast(max(HORIZONS))
    sarima_fc[origin] = (fc.predicted_mean.values, np.sqrt(fc.var_pred_mean.values))
for origin, h, target in jobs:
    mean, sd = sarima_fc[origin]
    add("sarima", origin, h, target, mean[h - 1], sd[h - 1])


# ---------------------------------------------------------------- direct ARX
def arx_design(h, regressors):
    """Target and inputs for the horizon-h direct model, indexed by target month."""
    X = pd.DataFrame(index=y.index)
    for k in range(OWN_LAGS):
        X[f"d_ny_ur_l{h + k}"] = dy.shift(h + k)
    for col in regressors:
        for j in range(X_LAGS):
            X[f"{col}_l{h + j}"] = features[col].shift(h + j)
    X.insert(0, "const", 1.0)
    target = y - y.shift(h)
    return target, X


for model, regressors in ARX_SETS.items():
    for h in HORIZONS:
        target_all, X_all = arx_design(h, regressors)
        back = h + max(OWN_LAGS, X_LAGS)
        usable = X_all.notna().all(axis=1) & target_all.notna()
        usable &= ~pd.Series([touches_covid(t, back) for t in y.index], index=y.index)
        usable &= y.index >= TRAIN_START
        for origin, hh, target in jobs:
            if hh != h:
                continue
            fit_rows = usable & (y.index <= origin)
            x_new = X_all.loc[target]
            if fit_rows.sum() < 60 or x_new.isna().any():
                continue
            beta, *_ = np.linalg.lstsq(X_all[fit_rows].values, target_all[fit_rows].values, rcond=None)
            resid = target_all[fit_rows].values - X_all[fit_rows].values @ beta
            sd = resid.std(ddof=X_all.shape[1])
            add(model, origin, h, target, y.loc[origin] + x_new.values @ beta, sd)

# ---------------------------------------------------------------- VAR in differences
levels = panel[VAR_COLUMNS]                       # raw, Oct 2025 stays NaN
dv = levels.diff()
lagged = pd.concat({k: dv.shift(k) for k in range(1, VAR_LAGS + 1)}, axis=1)
var_ok = dv.notna().all(axis=1) & lagged.notna().all(axis=1)
var_ok &= ~pd.Series([touches_covid(t, VAR_LAGS) for t in y.index], index=y.index)
var_ok &= y.index >= TRAIN_START
n = len(VAR_COLUMNS)

var_fc = {}
for origin in origins:
    rows_ok = var_ok & (y.index <= origin)
    Xv = np.column_stack([np.ones(rows_ok.sum()), lagged[rows_ok].values])
    Yv = dv[rows_ok].values
    B, *_ = np.linalg.lstsq(Xv, Yv, rcond=None)           # (1 + n*p) x n
    E = Yv - Xv @ B
    sigma = E.T @ E / (len(Yv) - Xv.shape[1])
    A = [B[1 + i * n:1 + (i + 1) * n].T for i in range(VAR_LAGS)]   # lag matrices
    hist = dv.loc[:origin].dropna().values[-VAR_LAGS:][::-1].tolist()  # most recent first
    if levels.loc[origin].isna().any():
        continue
    preds = []
    for _ in range(max(HORIZONS)):
        step = B[0] + sum(A[i] @ np.array(hist[i]) for i in range(VAR_LAGS))
        preds.append(step)
        hist = [step.tolist()] + hist[:-1]
    # MA weights -> forecast error variance of the cumulated (level) forecast
    psi = [np.eye(n)]
    for j in range(1, max(HORIZONS)):
        psi.append(sum(A[i] @ psi[j - 1 - i] for i in range(min(j, VAR_LAGS))))
    cum = np.cumsum(psi, axis=0)
    level_var = np.cumsum([c @ sigma @ c.T for c in cum], axis=0)
    var_fc[origin] = (levels.loc[origin, "ny_ur"] + np.cumsum(preds, axis=0)[:, 0],
                      np.sqrt(level_var[:, 0, 0]))
for origin, h, target in jobs:
    if origin in var_fc:
        mean, sd = var_fc[origin]
        add("var", origin, h, target, mean[h - 1], sd[h - 1])

# ---------------------------------------------------------------- outputs
fc = pd.DataFrame(rows)
fc["actual"] = fc["date"].map(y)
fc["split"] = fc["date"].map(split).fillna("live")
fc.loc[fc["origin"] == live_origin, "split"] = "live"
OUT_DIR.mkdir(exist_ok=True)
fc.sort_values(["model", "horizon", "date"]).to_csv(OUT_DIR / "forecasts.csv", index=False, float_format="%.4f")

scored = fc[fc["split"].isin(["validation", "test"]) & fc["actual"].notna()].copy()
scored["err"] = scored["yhat"] - scored["actual"]
scored["covered"] = (scored["actual"] >= scored["lower"]) & (scored["actual"] <= scored["upper"])
scored_late = scored[(scored["split"] == "test") & (scored["date"] >= "2023-01-01")].assign(split="test_2023on")
metrics = (pd.concat([scored, scored_late])
           .groupby(["split", "horizon", "model"])
           .agg(n=("err", "size"), mae=("err", lambda e: e.abs().mean()),
                rmse=("err", lambda e: np.sqrt((e ** 2).mean())), coverage95=("covered", "mean"))
           .reset_index())
naive = metrics[metrics["model"] == "naive"].set_index(["split", "horizon"])["rmse"]
metrics["rmse_vs_naive"] = metrics["rmse"] / metrics.set_index(["split", "horizon"]).index.map(naive)
metrics.to_csv(OUT_DIR / "metrics.csv", index=False, float_format="%.4f")

pd.set_option("display.width", 140)
for s in ["validation", "test", "test_2023on"]:
    print(f"\n=== {s} ===")
    print(metrics[metrics["split"] == s].pivot(index="model", columns="horizon",
                                               values=["mae", "rmse", "rmse_vs_naive", "coverage95"]).round(3))
print("\n=== live forecasts (origin", live_origin.strftime("%Y-%m"), ") ===")
print(fc[fc["split"] == "live"].pivot(index="model", columns="horizon", values="yhat").round(2))

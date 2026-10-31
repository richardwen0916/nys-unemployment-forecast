"""Sanity checks on data/features/model_features.csv.

  - every column is documented in feature_dictionary.csv
  - dates are unique and monthly
  - ADF + KPSS unit-root tests on every level and transformed series, on the
    full sample and pre-COVID (to 2019-12); saved to data/features/stationarity.csv
  - plot of every transformed series to data/features/feature_plots.png

Run from the repo root:  python src/features/check_features.py
"""

import warnings
from pathlib import Path

import matplotlib
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")  # KPSS warns when the p-value is outside its lookup table

OUT_DIR = Path("data/features")
features = pd.read_csv(OUT_DIR / "model_features.csv", parse_dates=["date"], index_col="date")
dictionary = pd.read_csv(OUT_DIR / "feature_dictionary.csv", index_col="name")

missing = set(features.columns) - set(dictionary.index)
assert not missing, f"columns missing from feature_dictionary.csv: {sorted(missing)}"
assert features.index.is_unique and pd.infer_freq(features.index) == "MS", "dates must be unique and monthly"

# Series to test: levels and transforms, not lags, dummies or flags
skip = ("_lag", "covid", "sample_", "ur_imputed", "split")
tested = [c for c in features.columns if not any(s in c for s in skip)]


def unit_root(x):
    x = x.dropna()
    adf_p = adfuller(x, autolag="AIC")[1]
    kpss_p = kpss(x, regression="c", nlags="auto")[1]
    # Stationary when ADF rejects a unit root and KPSS does not reject stationarity
    return adf_p, kpss_p, (adf_p < 0.05) and (kpss_p >= 0.05)


rows = []
for col in tested:
    full = unit_root(features[col])
    pre = unit_root(features.loc[:"2019-12", col]) if features.loc[:"2019-12", col].notna().sum() > 30 else (None, None, None)
    rows.append(dict(series=col, n=features[col].notna().sum(),
                     adf_p=full[0], kpss_p=full[1], stationary=full[2],
                     adf_p_pre2020=pre[0], kpss_p_pre2020=pre[1], stationary_pre2020=pre[2]))
table = pd.DataFrame(rows).set_index("series")
table.to_csv(OUT_DIR / "stationarity.csv", float_format="%.4f")
print(table[["n", "adf_p", "kpss_p", "stationary", "stationary_pre2020"]].to_string(float_format="%.3f"))

# Plot every tested series
n = len(tested)
fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(14, 1.9 * ((n + 1) // 2)))
for ax, col in zip(axes.flat, tested):
    features[col].plot(ax=ax, lw=0.8)
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-01"), color="grey", alpha=0.2)
    ax.set_title(col, fontsize=9, loc="left")
    ax.set_xlabel("")
for ax in list(axes.flat)[n:]:
    ax.set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "feature_plots.png", dpi=90)
print(f"Saved {OUT_DIR / 'stationarity.csv'} and {OUT_DIR / 'feature_plots.png'}")

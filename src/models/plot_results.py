"""Figures summarising the backtest in results/ (run after run_models.py).

  results/fig_scorecard.png      RMSE relative to naive, by model x horizon x period
  results/fig_rmse_by_horizon.png  absolute RMSE by horizon, one panel per period
  results/fig_cum_gain.png       cumulative squared-error gain over naive through the test period
  results/fig_live.png           live forecasts with 95% intervals

Run from the repo root:  python src/models/plot_results.py
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

OUT = Path("results/v0_exploratory")
metrics = pd.read_csv(OUT / "metrics.csv")
fc = pd.read_csv(OUT / "forecasts.csv", parse_dates=["date", "origin"])
y = pd.read_csv("data/features/model_features.csv", parse_dates=["date"], index_col="date")["ny_ur_raw"]

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MODELS = ["naive", "sarima", "arx_ar", "arx_econ", "arx_empire", "arx_full", "var"]
LABELS = {"naive": "Naive", "sarima": "SARIMA", "arx_ar": "ARX: own lags", "arx_econ": "ARX: + econ",
          "arx_empire": "ARX: + econ + Empire", "arx_full": "ARX: + econ + financial", "var": "VAR (4 states)"}
COLORS = dict(zip(MODELS, ["#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]))
PERIODS = {"validation": "Validation 2017-01 to 2020-02", "test": "Test 2022-01 to 2025-09",
           "test_2023on": "Test, calm part 2023-01 to 2025-09"}
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#1c5cab", "#86b6ef", "#f0efec", "#f2a3a2", "#c22f2e"])

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})

# 1. Scorecard: RMSE / naive RMSE
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
norm = TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=1.5)
for ax, (period, title) in zip(axes, PERIODS.items()):
    grid = metrics[metrics.split == period].pivot(index="model", columns="horizon", values="rmse_vs_naive").loc[MODELS]
    ax.imshow(grid.clip(0.5, 1.5).values, cmap=DIVERGING, norm=norm, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10,
                    color="white" if abs(v - 1) > 0.35 else INK, fontweight="bold" if i and v == grid.iloc[1:, j].min() else None)
    ax.set_xticks(range(3), [f"h = {h}" for h in grid.columns])
    ax.set_yticks(range(len(MODELS)), [LABELS[m] for m in MODELS])
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
fig.suptitle("RMSE relative to naive  (blue < 1 = beats naive, red > 1 = worse;  bold = best in column)",
             x=0.01, ha="left", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "fig_scorecard.png", dpi=120)
plt.close(fig)

# 2. Absolute RMSE by horizon
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
for ax, (period, title) in zip(axes, PERIODS.items()):
    sub = metrics[metrics.split == period]
    for m in MODELS:
        s = sub[sub.model == m].set_index("horizon")["rmse"]
        ax.plot(s.index, s.values, marker="o", ms=5, lw=2 if m == "naive" else 1.5,
                ls="--" if m == "naive" else "-", color=COLORS[m], label=LABELS[m])
    ax.set_xticks([1, 3, 6], ["h = 1", "h = 3", "h = 6"])
    ax.set_title(title, loc="left", fontsize=10)
    ax.grid(axis="y", color=GRID, lw=0.8)
axes[0].set_ylabel("RMSE (percentage points)")
axes[-1].legend(fontsize=8, frameon=False, loc="upper left")
fig.suptitle("Forecast error grows with horizon; naive (dashed) is hard to beat at 3-6 months in calm periods",
             x=0.01, ha="left", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "fig_rmse_by_horizon.png", dpi=120)
plt.close(fig)

# 3. Cumulative squared-error gain over naive through the test period
show = ["sarima", "arx_econ", "arx_empire", "var"]
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
test = fc[fc.split == "test"]
for ax, h in zip(axes, [1, 3, 6]):
    t = test[test.horizon == h]
    se = t.assign(se=(t.yhat - t.actual) ** 2).pivot(index="date", columns="model", values="se")
    for m in show:
        gain = (se["naive"] - se[m]).cumsum()
        ax.plot(gain.index, gain.values, color=COLORS[m], lw=1.8, label=LABELS[m])
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axvline(pd.Timestamp("2023-01-01"), color=GRID, lw=1, ls=":")
    ax.set_title(f"h = {h}", loc="left", fontsize=10)
    ax.grid(axis="y", color=GRID, lw=0.8)
axes[0].set_ylabel("cumulative (naive SE - model SE)")
axes[0].legend(fontsize=8, frameon=False)
fig.suptitle("Test period: models build their lead in 2022 (post-COVID decline), then give some of it back "
             "(line going down = losing to naive)", x=0.01, ha="left", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "fig_cum_gain.png", dpi=120)
plt.close(fig)

# 4. Live forecasts
live = fc[fc.split == "live"]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
for ax, h in zip(axes, [1, 3, 6]):
    sub = live[live.horizon == h].set_index("model").loc[MODELS]
    ypos = np.arange(len(MODELS))
    for i, m in enumerate(MODELS):
        r = sub.loc[m]
        ax.plot([r.lower, r.upper], [i, i], color=COLORS[m], lw=2, solid_capstyle="round")
        ax.plot(r.yhat, i, "o", ms=8, color=COLORS[m], mec=SURFACE, mew=2)
        ax.text(r.upper + 0.05, i, f"{r.yhat:.2f}", va="center", fontsize=9, color=INK2)
    ax.axvline(y.dropna().iloc[-1], color=INK2, lw=1, ls="--")
    ax.set_yticks(ypos, [LABELS[m] for m in MODELS])
    ax.invert_yaxis()
    ax.set_title(f"{sub['date'].iloc[0]:%b %Y}  (h = {h})", loc="left", fontsize=10)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_xlabel("NY unemployment rate, %")
fig.suptitle(f"Live forecasts from {y.dropna().index[-1]:%b %Y} (dashed = last actual {y.dropna().iloc[-1]:.1f}%); "
             "bars = 95% intervals", x=0.01, ha="left", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "fig_live.png", dpi=120)
plt.close(fig)
print("saved figures to results/")

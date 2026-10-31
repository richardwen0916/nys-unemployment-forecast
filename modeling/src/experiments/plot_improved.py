"""Scorecard figure for experiment improved_v1: RMSE / naive RMSE on common samples.

Naive = old_M0 for the old-timing baselines and M0 for everything under v2 timing,
so each block is compared with the naive forecast that had the same information.
Run from the package root:  python -m src.experiments.plot_improved
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from src.experiments.evaluate_improved import load, KEYS

EXP = Path(__file__).resolve().parents[2] / 'results/experiments/improved_v1'
ROWS = [('old_M1', 'old_M0', 'ARIMA (old timing)'), ('old_M3', 'old_M0', 'Macro ARX (old timing)'),
        ('M1', 'M0', 'ARIMA'), ('M3', 'M0', 'Macro ARX, COVID in training'),
        ('M3x', 'M0', 'Macro ARX, COVID excluded'), ('M4x', 'M0', '+ Empire, COVID excluded'),
        ('VAR', 'M0', 'VAR 4 states'), ('COMBO', 'M0', 'Combo: naive+ARIMA+M3x')]
PERIODS = [('validation', None, 'Validation 2016-19'), ('covid', None, 'COVID 2020-21'),
           ('post_covid', None, 'Post-COVID 2022-26'), ('post_covid', '2023-01-01', 'Calm 2023-26')]
DIVERGING = LinearSegmentedColormap.from_list('d', ['#1c5cab', '#86b6ef', '#f0efec', '#f2a3a2', '#c22f2e'])

f = load()
wide = f.pivot_table(index=KEYS + ['period'], columns='model', values='yhat', aggfunc='first')
wide = wide.join(f.groupby(KEYS + ['period']).actual.first())
models = [c for c in wide.columns if c != 'actual']
plt.rcParams.update({'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'font.size': 10})
fig, axes = plt.subplots(1, 4, figsize=(17, 5), sharey=True)
for ax, (period, start, title) in zip(axes, PERIODS):
    sub = wide[wide.index.get_level_values('period') == period]
    if start:
        sub = sub[sub.index.get_level_values('target_month') >= start]
    sub = sub.dropna(subset=models + ['actual'])
    grid = np.array([[np.sqrt(((s[m] - s.actual) ** 2).mean()) / np.sqrt(((s[b] - s.actual) ** 2).mean())
                      for _, s in sub.groupby(level='horizon')] for m, b, _ in ROWS])
    ax.imshow(np.clip(grid, 0.5, 1.5), cmap=DIVERGING, norm=TwoSlopeNorm(1, 0.5, 1.5), aspect='auto')
    for i in range(grid.shape[0]):
        for j in range(3):
            ax.text(j, i, f'{grid[i, j]:.2f}', ha='center', va='center',
                    color='white' if abs(grid[i, j] - 1) > 0.35 else '#0b0b0b')
    ax.axhline(1.5, color='#0b0b0b', lw=1)
    ax.set_xticks(range(3), ['h=1', 'h=3', 'h=6'])
    ax.set_yticks(range(len(ROWS)), [r[2] for r in ROWS])
    n = sub.groupby(level='horizon').size().tolist()
    ax.set_title(f'{title}  (n={n[0]}/{n[1]}/{n[2]})', loc='left', fontsize=10)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
fig.suptitle('RMSE / naive RMSE with the same information (blue < 1 beats naive, red > 1 loses). '
             'Top 2 rows: teammate baselines, old timing. Below the line: new timing.', x=0.01, ha='left', fontsize=12)
fig.tight_layout()
fig.savefig(EXP / 'scorecard.png', dpi=110)
print('saved', EXP / 'scorecard.png')

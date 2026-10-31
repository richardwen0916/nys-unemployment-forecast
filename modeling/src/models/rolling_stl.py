"""Re-estimate log-STL using release-filtered raw monthly claims only."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from .data_availability import available_panel


@dataclass
class STLResult:
    series: pd.Series
    nobs: int
    first_reference: pd.Timestamp
    last_reference: pd.Timestamp
    origin: pd.Timestamp
    status: str


def rolling_claims_stl(panel, origin, rules, *, releases=None, raw=None,
                       training_start=None, min_months=36, allow_estimated=True,
                       end_of_day=True):
    """Fit robust log-STL (period=12) entirely inside this origin's information set.

    Minimum history is 36 months by default, never less than 24. Short histories
    return NaN with insufficient_history status. Internal gaps raise, rather than
    interpolate. training_start can bound the window and must be month-start.
    This removes future-observation leakage, not revisions in snapshot values.
    """
    if not isinstance(min_months, int) or min_months < 24:
        raise ValueError("min_months must be an integer of at least 24")
    columns = (["date"] if "date" in panel.columns else []) + ["ny_claims_avg"]
    view = available_panel(panel[columns], origin, rules, releases=releases, raw=raw,
                           allow_estimated=allow_estimated, end_of_day=end_of_day)
    claims = view.data.ny_claims_avg
    if training_start is not None:
        start = pd.Timestamp(training_start)
        if start.tzinfo is not None or not start.is_month_start or start != start.normalize():
            raise ValueError("training_start must be a naive month-start date")
        claims = claims.loc[claims.index >= start]
    observed = claims.dropna()
    first = observed.index.min() if len(observed) else pd.NaT
    last = observed.index.max() if len(observed) else pd.NaT
    adjusted = pd.Series(np.nan, index=view.data.index, name="ny_claims_avg_sa_train")
    if len(observed):
        expected = pd.date_range(first, last, freq="MS")
        if not observed.index.equals(expected.rename(observed.index.name)):
            raise ValueError("Available training claims have internal gaps; no imputation is performed")
        if not np.isfinite(observed).all() or (observed <= 0).any():
            raise ValueError("Log-STL requires finite strictly positive claims")
    if len(observed) < min_months:
        return STLResult(adjusted, len(observed), first, last, view.origin, "insufficient_history")
    logs = np.log(observed)
    fit = STL(logs, period=12, robust=True).fit()
    adjusted.loc[observed.index] = np.exp(logs - fit.seasonal)
    return STLResult(adjusted, len(observed), first, last, view.origin, "ok")

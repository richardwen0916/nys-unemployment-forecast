"""Release-aware views of a frozen panel; no fetching, filling, or file writes.

These are pseudo-real-time views: the panel contains revised snapshot values.
Estimated rules are policies, not historical publication dates. Set
allow_estimated=False to require documented timing for every constituent.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar, GoodFriday
from pandas.tseries.offsets import CustomBusinessDay


TZ = "America/New_York"
VERIFIED = "verified_public_by"
MARKET_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())
KINDS = {"index", "month_end_lag", "month_day", "reference_week",
         "weekly_average", "daily_average", "blocked_stl"}


@dataclass
class AvailabilityResult:
    """Only data contains observations. Audit contains timing and reasons only."""

    data: pd.DataFrame
    audit: pd.DataFrame
    origin: pd.Timestamp


def _timestamp(value, *, end_of_day=False):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("A date cannot be missing")
    stamp = stamp.tz_localize(TZ) if stamp.tzinfo is None else stamp.tz_convert(TZ)
    if end_of_day:
        stamp = stamp.normalize() + pd.DateOffset(days=1) - pd.Timedelta(nanoseconds=1)
    return stamp


def forecast_origin(value, *, end_of_day=True):
    """Date-only origins mean New York calendar-day end by default.

    For intraday experiments pass end_of_day=False; naive times mean New York.
    A non-midnight time with end_of_day=True raises instead of silently moving it.
    """
    stamp = _timestamp(value)
    if end_of_day and stamp != stamp.normalize():
        raise ValueError("Explicit times require end_of_day=False")
    return _timestamp(stamp, end_of_day=end_of_day)


def _monthly_panel(panel):
    frame = panel.copy(deep=True)
    if "date" in frame:
        frame = frame.set_index("date")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index), name="date")
    if (frame.index.tz is not None or frame.index.hasnans
            or frame.index.has_duplicates or not frame.index.is_month_start.all()
            or not (frame.index == frame.index.normalize()).all()):
        raise ValueError("Panel dates must be unique, naive month-start dates")
    if frame.columns.has_duplicates:
        raise ValueError("Panel columns must be unique")
    return frame.sort_index()


def _rules_table(rules):
    table = rules.copy()
    needed = {"variable", "series_id", "rule", "lag_days", "january_lag_days",
              "release_day", "not_before", "official_source_url"}
    if not needed.issubset(table.columns) or table.variable.duplicated().any():
        raise ValueError("Rules need the documented schema and unique variables")
    if not set(table.rule).issubset(KINDS):
        raise ValueError("Unknown availability rule")
    for name in ("lag_days", "january_lag_days", "release_day"):
        numeric = pd.to_numeric(table[name], errors="raise")
        if ((numeric.dropna() < 0) | (numeric.dropna() % 1 != 0)).any():
            raise ValueError("Rule delays/days must be nonnegative integers")
        table[name] = numeric
    return table.set_index("variable")


def _release_lookup(releases):
    if releases is None:
        return {}
    required = {"variable", "reference_date", "available_at", "status", "source_url"}
    if not required.issubset(releases.columns):
        raise ValueError("Release calendar needs variable/reference_date/available_at/status/source_url")
    result = {}
    for row in releases.itertuples(index=False):
        reference = pd.Timestamp(row.reference_date)
        if reference.tzinfo is not None or reference != reference.normalize():
            raise ValueError("Calendar reference dates must be naive dates")
        key = (row.variable, reference)
        if key in result:
            raise ValueError(f"Duplicate release calendar entry: {key}")
        if row.status not in {VERIFIED, "estimated_holdback"} or not pd.notna(row.source_url):
            raise ValueError("Calendar entries need status and a source URL")
        # Dates alone are public-by day-end, never assumed available at midnight.
        date_only = len(str(row.available_at)) == 10
        stamp = _timestamp(row.available_at, end_of_day=date_only)
        if stamp.tz_localize(None).normalize() < reference:
            raise ValueError("Release precedes its reference date")
        result[key] = (stamp, row.status, row.source_url)
    return result


def _raw_series(raw, key):
    if raw is None or key not in raw:
        return None
    series = raw[key].copy()
    if not isinstance(series, pd.Series) or not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Raw inputs must be Series with a DatetimeIndex")
    if (series.index.tz is not None or series.index.has_duplicates or series.index.hasnans
            or not (series.index == series.index.normalize()).all()):
        raise ValueError("Raw dates must be unique naive dates")
    return series.sort_index()


def _estimate(reference, rule):
    kind = rule.rule
    lag = rule.lag_days
    if reference.month == 1 and pd.notna(rule.january_lag_days):
        lag = rule.january_lag_days
    if kind == "month_day":
        base = reference.replace(day=int(rule.release_day))
    elif kind in {"reference_week", "weekly_average", "daily_average"}:
        base = reference + pd.Timedelta(days=int(lag))
    else:
        base = reference + pd.offsets.MonthEnd(0) + pd.Timedelta(days=int(lag))
    return _timestamp(base, end_of_day=True), "estimated", rule.official_source_url


def _timing(variable, reference, rule, lookup):
    estimate = _estimate(reference, rule)
    stamp, status, url = lookup.get((variable, reference), estimate)
    if pd.notna(rule.not_before) and str(rule.not_before):
        gate = _timestamp(rule.not_before, end_of_day=True)
        if gate > stamp:
            stamp, status = gate, "estimated"
    return stamp, status, url


def available_panel(panel, origin, rules, *, releases=None, raw=None,
                    allow_estimated=True, end_of_day=True):
    """Mask unavailable cells and omit future reference rows; never mutate inputs.

    rules: release_rules.csv DataFrame. releases: optional release_calendar.csv.
    raw: mapping from source series_id to daily/weekly Series, required to verify
    aggregate completeness and equality to the stored panel. Without it affected
    cells stay missing. Strict mode requires verified dates for every component.
    """
    cutoff = forecast_origin(origin, end_of_day=end_of_day)
    frame = _monthly_panel(panel)
    frame = frame.loc[frame.index <= cutoff.tz_localize(None).normalize()]
    table = _rules_table(rules)
    missing = set(frame.columns) - set(table.index)
    if missing:
        raise ValueError(f"No release rules for: {sorted(missing)}")
    lookup = _release_lookup(releases)
    sources = {key: _raw_series(raw, key) for key in table.series_id.dropna().unique()}
    records = []
    for variable in frame.columns:
        rule = table.loc[variable]
        for month in frame.index:
            value = frame.at[month, variable]
            stamp, status, url = pd.NaT, "blocked", ""
            reason = "not_released"
            if rule.rule == "blocked_stl":
                reason = "full_sample_stl_blocked"
            elif pd.isna(value):
                reason = "source_missing"
            elif rule.rule in {"reference_week", "weekly_average", "daily_average"}:
                end = month + pd.offsets.MonthEnd(0)
                if rule.rule == "reference_week":
                    twelfth = month.replace(day=12)
                    dates = pd.DatetimeIndex([twelfth + pd.Timedelta(days=(5-twelfth.dayofweek) % 7)])
                elif rule.rule == "weekly_average":
                    dates = pd.date_range(month, end, freq="W-SAT")
                elif rule.series_id == "indeed_NY":
                    dates = pd.date_range(month, end, freq="D")
                else:
                    # Conservative coverage floor, not a full exchange calendar.
                    dates = pd.date_range(month, end, freq=MARKET_DAY)
                    dates = dates.difference(GoodFriday.dates(month, end))
                source = sources.get(rule.series_id)
                if source is None:
                    reason = "raw_constituents_required"
                elif not source.reindex(dates).notna().all():
                    reason = "incomplete_constituents"
                else:
                    components = source.reindex(dates)
                    if rule.rule == "daily_average":
                        components = source.loc[month:end].dropna()
                    if not np.isfinite(components.to_numpy(dtype=float)).all():
                        raise ValueError("Nonfinite raw observations")
                    if not np.isclose(float(value), components.mean(), rtol=1e-9, atol=1e-8):
                        reason = "panel_aggregate_mismatch"
                    else:
                        times = [_timing(rule.series_id, date, rule, lookup) for date in components.index]
                        stamp = max(t[0] for t in times)
                        status = VERIFIED if all(t[1] == VERIFIED for t in times) else "estimated"
                        url = ";".join(sorted({t[2] for t in times}))
                        if rule.rule != "reference_week":
                            stamp = max(stamp, _timestamp(end, end_of_day=True))
                        reason = "available"
            else:
                stamp, status, url = _timing(variable, month, rule, lookup)
                reason = "available"
            if reason == "available":
                if stamp > cutoff:
                    reason = "not_released"
                elif not allow_estimated and status != VERIFIED:
                    reason = "unverified_timing"
            if reason != "available":
                frame.at[month, variable] = np.nan
            records.append((month, variable, stamp, status, reason, url))
    audit = pd.DataFrame(records, columns=["date", "variable", "available_at", "status", "reason", "source_url"])
    audit = audit.set_index(["date", "variable"])
    return AvailabilityResult(frame, audit, cutoff)


def load_raw_inputs(root):
    """Read only the five named non-secret constituent CSVs; no fetch scripts."""
    root = Path(root)
    files = {"NYICLAIMS": "fred_NYICLAIMS", "ICSA": "fred_ICSA",
             "T10Y3M": "fred_T10Y3M", "^GSPC": "yf_GSPC", "indeed_NY": "indeed_NY"}
    result = {}
    for key, name in files.items():
        frame = pd.read_csv(root / "data" / "raw" / f"{name}.csv", parse_dates=["date"])
        result[key] = frame.set_index("date").iloc[:, 0]
    return result


def target_months(origin, horizons=(1, 3, 6)):
    """Reference months after the origin's calendar month, not last observed NYUR."""
    month = _timestamp(origin).tz_localize(None).to_period("M")
    if any(not isinstance(h, int) or h < 1 for h in horizons):
        raise ValueError("Horizons must be positive integer month offsets")
    return {h: (month + h).to_timestamp() for h in horizons}

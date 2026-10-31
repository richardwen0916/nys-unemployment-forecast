"""Offline tests: run python -m unittest discover -s tests -v from repo root."""

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from statsmodels.tsa.seasonal import STL

from src.models.data_availability import (
    available_panel, forecast_origin, target_months,
)
from src.models.rolling_stl import rolling_claims_stl

ROOT = Path(__file__).resolve().parents[1]


class AvailabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = pd.read_csv(ROOT / "data/processed/monthly_panel.csv", parse_dates=["date"]).set_index("date")
        cls.rules = pd.read_csv(ROOT / "data/processed/release_rules.csv")
        cls.calendar = pd.read_csv(ROOT / "data/processed/release_calendar.csv")
        cls.raw = {"NYICLAIMS": pd.read_csv(ROOT / "data/raw/fred_NYICLAIMS.csv", parse_dates=["date"]).set_index("date").iloc[:, 0]}

    def view(self, origin, panel=None, **kwargs):
        return available_panel(self.panel if panel is None else panel, origin, self.rules,
                               releases=kwargs.pop("releases", self.calendar),
                               raw=kwargs.pop("raw", self.raw), **kwargs)

    def stl(self, origin, panel=None, **kwargs):
        return rolling_claims_stl(self.panel if panel is None else panel, origin, self.rules,
                                  releases=self.calendar, raw=kwargs.pop("raw", self.raw), **kwargs)


    def test_rule_coverage(self):
        self.assertEqual(set(self.rules.variable), {"date", *self.panel.columns})
        self.assertFalse(self.rules.variable.duplicated().any())

    def test_historical_origins_obey_all_release_cutoffs(self):
        for origin in ["2019-12-31", "2023-12-31", "2026-10-19"]:
            with self.subTest(origin=origin):
                result = self.view(origin)
                self.assertTrue((result.data.index <= pd.Timestamp(origin)).all())
                for variable in result.data:
                    dates = result.data[variable].dropna().index
                    audit = result.audit.xs(variable, level="variable").loc[dates]
                    self.assertTrue((audit.reason == "available").all())
                    self.assertTrue((audit.available_at <= result.origin).all())
                expected = self.panel.reindex(result.data.index)
                self.assertFalse((expected.isna() & result.data.notna()).any().any())
                self.assertTrue(result.data.ny_claims_avg_sa.isna().all())

    def test_nyur_payroll_verified_release_boundaries(self):
        subset = self.panel.loc["2023-12-01":"2023-12-01", ["ny_ur", "ny_payrolls"]]
        before = self.view("2024-01-17", subset)
        after = self.view("2024-01-18", subset)
        self.assertTrue(before.data.isna().all().all())
        assert_frame_equal(after.data, subset)
        self.assertTrue((after.audit.status == "verified_public_by").all())

    def test_bls_exact_timestamp_and_timezone(self):
        subset = self.panel.loc["2019-12-01":"2019-12-01", ["ny_ur", "ny_payrolls"]]
        before = self.view("2020-01-24T09:59:59-05:00", subset, end_of_day=False)
        after = self.view("2020-01-24T15:00:00+00:00", subset, end_of_day=False)
        self.assertTrue(before.data.isna().all().all())
        assert_frame_equal(after.data, subset)

    def test_date_only_calendar_is_not_available_at_midnight(self):
        subset = self.panel.loc["2023-12-01":"2023-12-01", ["ny_ur"]]
        result = self.view("2024-01-18T12:00:00", subset, end_of_day=False)
        self.assertTrue(result.data.isna().all().all())

    def test_national_unemployment_before_and_at_release(self):
        subset = self.panel.loc["2023-12-01":"2023-12-01", ["us_ur"]]
        self.assertTrue(self.view("2024-01-05T08:29:59", subset, end_of_day=False).data.isna().all().all())
        assert_frame_equal(self.view("2024-01-05T08:30:00", subset, end_of_day=False).data, subset)

    def test_ny_weekly_claims_conservative_delay(self):
        # December 2023 reference week ends Saturday Dec 16; policy adds 14 days.
        subset = self.panel.loc["2023-12-01":"2023-12-01", ["ny_claims"]]
        self.assertTrue(self.view("2023-12-29", subset).data.isna().all().all())
        after = self.view("2023-12-30", subset)
        assert_frame_equal(after.data, subset)
        self.assertEqual(after.audit.iloc[0].status, "estimated")


    def test_completed_claims_average_waits_for_final_week(self):
        # Five Saturdays, last Dec 30; final NY week held until Jan 13.
        subset = self.panel.loc["2023-12-01":"2023-12-01", ["ny_claims_avg"]]
        for origin in ["2023-12-31", "2024-01-12"]:
            self.assertTrue(self.view(origin, subset).data.isna().all().all())
        assert_frame_equal(self.view("2024-01-13", subset).data, subset)

    def test_four_weeks_cannot_stand_in_for_five(self):
        raw = dict(self.raw)
        raw["NYICLAIMS"] = raw["NYICLAIMS"].drop(pd.Timestamp("2023-12-30"))
        subset = self.panel.loc["2023-12-01":"2023-12-01", ["ny_claims_avg"]].copy()
        subset.iloc[0, 0] = raw["NYICLAIMS"].loc["2023-12"].mean()
        result = self.view("2024-02-01", subset, raw=raw)
        self.assertTrue(result.data.isna().all().all())
        self.assertEqual(result.audit.iloc[0].reason, "incomplete_constituents")

    def test_last_week_verified_delay_cannot_be_bypassed(self):
        delayed = pd.DataFrame([dict(variable="NYICLAIMS", reference_date="2023-12-30",
            available_at="2024-02-01", status="verified_public_by", source_url="https://example.invalid/test")])
        subset = self.panel.loc["2023-12-01":"2023-12-01", ["ny_claims_avg"]]
        result = self.view("2024-01-31", subset, releases=delayed)
        self.assertTrue(result.data.isna().all().all())

    def test_strict_average_requires_every_constituent_verified(self):
        final_week_only = pd.DataFrame([dict(variable="NYICLAIMS", reference_date="2023-12-30",
            available_at="2024-01-13", status="verified_public_by", source_url="https://example.invalid/test")])
        subset = self.panel.loc["2023-12-01":"2023-12-01", ["ny_claims_avg"]]
        result = self.view("2024-02-01", subset, releases=final_week_only, allow_estimated=False)
        self.assertTrue(result.data.isna().all().all())
        self.assertEqual(result.audit.iloc[0].reason, "unverified_timing")



    def test_missing_raw_inputs_and_mismatched_aggregate_fail_closed(self):
        subset = self.panel.loc["2023-12-01":"2023-12-01", ["ny_claims_avg"]]
        self.assertTrue(self.view("2024-02-01", subset, raw=None).data.isna().all().all())
        changed = subset * 2
        result = self.view("2024-02-01", changed)
        self.assertTrue(result.data.isna().all().all())
        self.assertTrue((result.audit.reason == "panel_aggregate_mismatch").all())

    def test_october_2025_gap_is_preserved(self):
        columns = ["ny_ur", "us_ur", "nj_ur", "ct_ur", "pa_ur"]
        subset = self.panel.loc["2025-10-01":"2025-10-01", columns]
        self.assertTrue(subset.isna().all().all())
        result = self.view("2026-10-19", subset)
        self.assertTrue(result.data.isna().all().all())
        self.assertTrue((result.audit.reason == "source_missing").all())

    def test_shutdown_exceptions_and_january_benchmark_holdback(self):
        subset = self.panel.loc["2025-09-01":"2025-09-01", ["ny_ur", "ny_payrolls", "us_ur"]]
        self.assertTrue(self.view("2025-11-19", subset).data.isna().all().all())
        at_national = self.view("2025-11-20", subset).data
        self.assertTrue(at_national[["ny_ur", "ny_payrolls"]].isna().all().all())
        self.assertTrue(at_national.us_ur.notna().all())
        assert_frame_equal(self.view("2025-12-11", subset).data, subset)
        january = self.panel.loc["2023-01-01":"2023-01-01", ["ny_ur", "ny_coincident"]]
        self.assertTrue(self.view("2023-02-28", january).data.isna().all().all())

    def test_strict_mode_and_unknown_columns(self):
        subset = self.panel.loc["2023-11-01":"2023-12-01", ["ny_ur", "fedfunds"]]
        result = self.view("2023-12-31", subset, allow_estimated=False)
        self.assertTrue(result.data.fedfunds.isna().all())
        self.assertTrue(pd.notna(result.data.loc["2023-11-01", "ny_ur"]))
        with self.assertRaisesRegex(ValueError, "No release rules"):
            self.view("2023-12-31", subset.assign(unknown_feature=1))

    def test_backfilled_series_not_usable_before_publication_gate(self):
        empire = self.panel.loc["2001-07-01":"2001-07-01", ["empire_general"]]
        self.assertTrue(self.view("2001-12-31", empire).data.isna().all().all())
        indeed = self.panel.loc["2020-02-01":"2020-02-01", ["indeed_postings"]]
        self.assertTrue(self.view("2020-12-31", indeed).data.isna().all().all())

    def test_input_objects_unchanged(self):
        panel = self.panel.loc["2023-12-01":"2023-12-01"].copy()
        before = panel.copy(deep=True)
        rules_before = self.rules.copy(deep=True)
        raw_before = self.raw["NYICLAIMS"].copy(deep=True)
        self.view("2023-12-31", panel)
        assert_frame_equal(panel, before)
        assert_frame_equal(self.rules, rules_before)
        assert_series_equal(self.raw["NYICLAIMS"], raw_before)

    def test_stl_matches_only_available_training_window(self):
        result = self.stl("2023-12-31", training_start="2018-01-01")
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.last_reference, pd.Timestamp("2023-11-01"))
        self.assertEqual(result.nobs, 71)
        observed = self.panel.loc["2018-01-01":"2023-11-01", "ny_claims_avg"]
        logs = np.log(observed)
        expected = np.exp(logs - STL(logs, period=12, robust=True).fit().seasonal)
        np.testing.assert_allclose(result.series.loc[observed.index], expected)
        self.assertTrue(result.series.loc["2023-12-01":].isna().all())

    def test_future_mutations_do_not_affect_stl(self):
        before = self.stl("2023-12-31", training_start="2018-01-01")
        changed = self.panel.copy()
        changed.loc["2023-12-01":, "ny_claims_avg"] = 1e12
        changed["ny_claims_avg_sa"] = -99999
        raw = dict(self.raw)
        raw["NYICLAIMS"] = raw["NYICLAIMS"].copy()
        raw["NYICLAIMS"].loc["2023-12-01":] = 1e12
        after = self.stl("2023-12-31", changed, raw=raw, training_start="2018-01-01")
        assert_series_equal(before.series, after.series)

    def test_stl_short_history_and_internal_gaps(self):
        short = self.stl("1987-01-31")
        self.assertEqual(short.status, "insufficient_history")
        self.assertTrue(short.series.isna().all())
        changed = self.panel.copy()
        changed.loc["2022-07-01", "ny_claims_avg"] = np.nan
        with self.assertRaisesRegex(ValueError, "internal gaps"):
            self.stl("2023-12-31", changed, training_start="2020-01-01")

    def test_dates_and_configuration_are_validated(self):
        self.assertEqual(target_months("2019-12-31")[6], pd.Timestamp("2020-06-01"))
        self.assertEqual(target_months("2026-10-19")[1], pd.Timestamp("2026-11-01"))
        with self.assertRaises(ValueError):
            forecast_origin("2024-01-05T08:30:00")
        duplicated = self.panel.iloc[[0, 0]]
        with self.assertRaisesRegex(ValueError, "unique"):
            self.view("2023-12-31", duplicated)
        duplicate_calendar = pd.concat([self.calendar, self.calendar.iloc[:1]])
        with self.assertRaisesRegex(ValueError, "Duplicate release"):
            self.view("2023-12-31", self.panel.iloc[:1], releases=duplicate_calendar)


if __name__ == "__main__":
    unittest.main()

"""Historical release regressions for the matching SA total nonfarm series."""
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.models.data_availability import available_panel
from src.models.feature_engineering import build_feature_window, load_core_inputs

ROOT = Path(__file__).resolve().parents[1]


class PayrollReleaseAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel, cls.rules, cls.calendar, cls.raw = load_core_inputs(ROOT)
        cls.payroll = cls.panel.set_index('date')[['ny_payrolls']]

    def view(self, reference, origin, **kwargs):
        return available_panel(self.payroll.loc[[pd.Timestamp(reference)]], origin,
            self.rules, releases=self.calendar, **kwargs)

    def test_all_validation_months_have_verified_matching_series_dates(self):
        records = self.calendar.loc[self.calendar.variable.eq('ny_payrolls')].set_index('reference_date')
        expected = pd.date_range('2016-01-01', '2019-12-01', freq='MS').strftime('%Y-%m-%d')
        self.assertTrue(records.loc[expected].status.eq('verified_public_by').all())
        self.assertTrue(records.loc[expected].source_url.str.startswith('https://www.bls.gov/news.release/archives/laus_').all())
        self.assertFalse(self.calendar.duplicated(['variable','reference_date']).any())

    def test_every_validation_bls_release_is_hidden_before_and_visible_at_timestamp(self):
        records = self.calendar.loc[self.calendar.variable.eq('ny_payrolls') &
            self.calendar.reference_date.between('2016-01-01','2019-12-01')]
        for row in records.itertuples():
            with self.subTest(reference=row.reference_date):
                stamp = pd.Timestamp(row.available_at)
                before = self.view(row.reference_date, stamp-pd.Timedelta(seconds=1),
                                   end_of_day=False, allow_estimated=False)
                after = self.view(row.reference_date, stamp, end_of_day=False, allow_estimated=False)
                self.assertTrue(before.data.isna().all().all())
                assert_frame_equal(after.data, self.payroll.loc[[pd.Timestamp(row.reference_date)]])

    def test_annual_benchmark_january_is_not_exposed_in_february(self):
        for year, day in [(2016,14),(2017,13),(2018,12),(2019,11)]:
            with self.subTest(year=year):
                reference = f'{year}-01-01'
                end_february = pd.Timestamp(f'{year}-02-01')+pd.offsets.MonthEnd(0)
                self.assertTrue(self.view(reference, end_february).data.isna().all().all())
                self.assertTrue(self.view(reference, f'{year}-03-{day-1:02d}').data.isna().all().all())
                self.assertTrue(self.view(reference, f'{year}-03-{day:02d}').data.notna().all().all())

    def test_november_2023_was_public_before_december_month_end(self):
        self.assertTrue(self.view('2023-11-01','2023-12-20').data.isna().all().all())
        # Dated NYS report contains the matching SA total nonfarm series on page 2;
        # BLS independently confirms it on December 22. Unknown NYS time means EOD.
        self.assertTrue(self.view('2023-11-01','2023-12-21T12:00:00',end_of_day=False).data.isna().all().all())
        for day in ['2023-12-21','2023-12-22','2023-12-31']:
            self.assertTrue(self.view('2023-11-01',day,allow_estimated=False).data.notna().all().all())

    def test_2026_irregular_dates_use_observation_specific_state_publication(self):
        for reference, previous, release in [('2026-01-01','2026-04-01','2026-04-02'),
                                             ('2026-02-01','2026-04-15','2026-04-16'),
                                             ('2026-03-01','2026-04-29','2026-04-30')]:
            with self.subTest(reference=reference):
                self.assertTrue(self.view(reference,previous).data.isna().all().all())
                self.assertTrue(self.view(reference,release,allow_estimated=False).data.notna().all().all())
        self.assertTrue(self.view('2026-01-01','2026-03-31').data.isna().all().all())

    def test_shutdown_delays_and_missing_source_values_remain_protected(self):
        self.assertTrue(self.view('2013-09-01','2013-11-21').data.isna().all().all())
        self.assertTrue(self.view('2013-09-01','2013-11-22').data.notna().all().all())
        self.assertTrue(self.view('2025-09-01','2025-12-10').data.isna().all().all())
        self.assertTrue(self.view('2025-09-01','2025-12-11').data.notna().all().all())
        self.assertTrue(self.view('2025-10-01','2026-01-31',allow_estimated=False).data.isna().all().all())
        missing = self.payroll.loc[['2019-11-01']].copy()*np.nan
        result = available_panel(missing,'2019-12-31',self.rules,releases=self.calendar)
        self.assertTrue(result.data.isna().all().all())

    def test_unverified_fallback_keeps_exact_45_and_75_day_calculation(self):
        for reference, lag in [('1991-07-01',45),('1991-01-01',75)]:
            stamp = pd.Timestamp(reference)+pd.offsets.MonthEnd(0)+pd.Timedelta(days=lag)
            self.assertTrue(self.view(reference,stamp-pd.Timedelta(days=1)).data.isna().all().all())
            result = self.view(reference,stamp)
            self.assertTrue(result.data.notna().all().all())
            self.assertEqual(result.audit.iloc[0].status,'estimated')
            self.assertTrue(self.view(reference,stamp,allow_estimated=False).data.isna().all().all())




if __name__ == '__main__':
    unittest.main()

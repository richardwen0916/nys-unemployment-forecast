"""Feature and horizon leakage tests; no forecasting models are fitted."""
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from src.models.feature_engineering import (
    CORE, TRANSFORMS, align_training_examples, build_feature_window,
    build_training_examples, load_core_inputs, split_for_target,
)

ROOT = Path(__file__).resolve().parents[1]


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel, cls.rules, cls.calendar, cls.raw = load_core_inputs(ROOT)

    def window(self, origin, panel=None, raw=None, **kwargs):
        return build_feature_window(self.panel if panel is None else panel, origin, self.rules,
            releases=self.calendar, raw=self.raw if raw is None else raw, **kwargs)


    def test_only_core_variables_and_seven_transforms(self):
        panel = self.panel.assign(unrelated_indicator=999, ny_claims_avg_sa=-999)
        result = self.window('2019-12-31', panel)
        self.assertEqual(set(result.history), set(TRANSFORMS))
        self.assertEqual(set(result.audit.variable), set(CORE))
        self.assertEqual(len(result.features), 21)

    def test_transformation_formulas_and_target_levels(self):
        result = self.window('2019-12-31')
        source = self.panel.set_index('date')
        date, previous = pd.Timestamp('2019-11-01'), pd.Timestamp('2019-10-01')
        self.assertEqual(result.history.at[date, 'ny_ur_level'], source.at[date, 'ny_ur'])
        self.assertAlmostEqual(result.history.at[date, 'ny_ur_diff_pp'], source.at[date, 'ny_ur']-source.at[previous, 'ny_ur'])
        expected = 100*np.log(source.at[date, 'ny_payrolls']/source.at[previous, 'ny_payrolls'])
        self.assertAlmostEqual(result.history.at[date, 'ny_payrolls_log_growth_pct'], expected)
        self.assertAlmostEqual(result.history.at[date, 'ny_claims_sa_log_diff'],
            result.history.at[date, 'ny_claims_sa_log']-result.history.at[previous, 'ny_claims_sa_log'])
        self.assertEqual(result.target_levels.at[date], source.at[date, 'ny_ur'])

    def test_lags_are_calendar_offsets_from_origin(self):
        result = self.window('2023-12-31', lags=(0, 1, 2, 3))
        self.assertTrue(pd.isna(result.features['ny_ur_level_lag0']))
        self.assertEqual(result.audit.at['ny_ur_level_lag1', 'reference_month'], pd.Timestamp('2023-11-01'))
        self.assertEqual(result.features['ny_ur_level_lag2'], result.history.at[pd.Timestamp('2023-10-01'), 'ny_ur_level'])
        for feature, value in result.features.items():
            if pd.notna(value):
                self.assertLessEqual(result.audit.at[feature, 'available_at'], result.origin)

    def test_future_raw_and_full_sample_adjustment_cannot_change_features(self):
        before = self.window('2015-12-31')
        changed = self.panel.copy()
        changed.loc[changed.date >= '2015-12-01', list(CORE)] = 1e12
        changed['ny_claims_avg_sa'] = -1e18
        raw = {'NYICLAIMS': self.raw['NYICLAIMS'].copy()}
        raw['NYICLAIMS'].loc['2015-12-01':] = 1e12
        after = self.window('2015-12-31', changed, raw)
        assert_series_equal(before.features, after.features)
        assert_frame_equal(before.history, after.history)
        self.assertLess(before.stl_last_reference, pd.Timestamp('2015-12-01'))

    def test_unreleased_monthly_value_mutation_has_no_effect(self):
        before = self.window('2015-12-31')
        changed = self.panel.copy()
        # December state values are unpublished at the December 31 origin.
        changed.loc[changed.date == '2015-12-01', ['ny_ur', 'ny_payrolls']] = 1e10
        after = self.window('2015-12-31', changed)
        assert_series_equal(before.features, after.features)

    def test_october_gap_and_following_difference_remain_missing(self):
        result = self.window('2026-01-31')
        for name in ['ny_ur_level', 'ny_ur_diff_pp', 'us_ur_level', 'us_ur_diff_pp']:
            self.assertTrue(pd.isna(result.history.at[pd.Timestamp('2025-10-01'), name]))
        self.assertTrue(pd.isna(result.history.at[pd.Timestamp('2025-11-01'), 'ny_ur_diff_pp']))
        self.assertTrue(pd.isna(result.history.at[pd.Timestamp('2025-11-01'), 'us_ur_diff_pp']))

    def test_missing_calendar_row_is_not_bridged(self):
        panel = self.panel.loc[self.panel.date != '2014-06-01']
        # Claims cannot be refit across the resulting gap; fail rather than fill.
        with self.assertRaisesRegex(ValueError, 'internal gaps'):
            self.window('2015-12-31', panel)

    def test_origin_must_be_month_end(self):
        with self.assertRaisesRegex(ValueError, 'last calendar day'):
            self.window('2026-10-19')
        with self.assertRaisesRegex(ValueError, 'Lags'):
            self.window('2019-12-31', lags=(-1,))

    def test_horizon_alignment_and_identical_origin_predictors(self):
        window = self.window('2019-12-31')
        examples = align_training_examples(self.panel, [window], '2026-10-19', self.rules, releases=self.calendar)
        origin = pd.Timestamp('2019-12-31')
        for h, expected in [(1,'2020-01-01'),(3,'2020-03-01'),(6,'2020-06-01')]:
            self.assertEqual(examples.labels.loc[(origin,h), 'target_reference'], pd.Timestamp(expected))
            self.assertEqual(examples.labels.loc[(origin,h), 'split'], 'covid_test')
            np.testing.assert_allclose(examples.X.loc[(origin,h)], window.features, equal_nan=True)
        self.assertNotIn('target_ny_ur_level', examples.X)

    def test_training_cutoff_censors_future_and_unreleased_labels(self):
        examples = build_training_examples(self.panel, ['2015-09-30','2015-12-31'], '2015-12-31',
            self.rules, releases=self.calendar, raw=self.raw)
        origin = pd.Timestamp('2015-12-31')
        self.assertFalse(examples.labels.loc[origin].label_known.any())
        X, y = examples.eligible('initial_train', columns=['ny_ur_level_lag3'])
        self.assertTrue(y.notna().all())
        self.assertTrue((examples.labels.loc[X.index, 'label_available_at'] <= examples.as_of).all())
        self.assertTrue((examples.labels.loc[X.index, 'target_reference'] <= pd.Timestamp('2015-12-01')).all())
        with self.assertRaisesRegex(ValueError, 'later than'):
            build_training_examples(self.panel, ['2016-01-31'], '2015-12-31', self.rules,
                                    releases=self.calendar, raw=self.raw)

    def test_year_boundary_split_purging_and_missing_evaluation_labels(self):
        windows = [self.window('2015-12-31'), self.window('2025-09-30')]
        examples = align_training_examples(self.panel, windows, '2026-10-19', self.rules, releases=self.calendar)
        self.assertEqual(examples.labels.loc[(pd.Timestamp('2015-12-31'),1),'split'], 'validation')
        self.assertFalse(examples.labels.loc[(pd.Timestamp('2025-09-30'),1),'label_known'])
        self.assertTrue(pd.isna(examples.labels.loc[(pd.Timestamp('2025-09-30'),1),'target_ny_ur_level']))
        self.assertEqual(split_for_target('2022-01-01'), 'post_covid_test')

    def test_strict_mode_does_not_synthesize_claims_stl(self):
        result = self.window('2023-12-31', allow_estimated=False)
        self.assertTrue(result.history.ny_claims_sa_log.isna().all())
        self.assertEqual(result.stl_status, 'insufficient_history')



    def test_feature_inputs_unchanged_and_early_stl_missing(self):
        before = self.panel.copy(deep=True)
        result = self.window('1991-12-31')
        assert_frame_equal(before,self.panel)
        self.assertEqual(result.stl_status, 'insufficient_history')
        self.assertTrue(result.history.ny_claims_sa_log.isna().all())


if __name__ == '__main__':
    unittest.main()

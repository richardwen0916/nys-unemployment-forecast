"""Portable-package checks supplement the retained baseline regression tests."""
from pathlib import Path
import ast
import importlib
import json
import unittest

import numpy as np
import pandas as pd

from src.handoff import verify_saved

ROOT = Path(__file__).resolve().parents[1]


class HandoffIntegrityTests(unittest.TestCase):
    def test_all_included_hashes_specs_and_metrics(self):
        verify_saved()

    def test_actual_import_graph_is_complete(self):
        for path in sorted((ROOT / 'src').rglob('*.py')):
            module = '.'.join(path.relative_to(ROOT).with_suffix('').parts)
            imported = importlib.import_module(module)
            self.assertTrue(Path(imported.__file__).resolve().is_relative_to(ROOT))

    def test_no_private_paths_credentials_or_external_symlinks(self):
        inventory = pd.read_csv(ROOT / 'FILE_INVENTORY.csv')
        for name in inventory.path:
            path = ROOT / name
            self.assertFalse(path.is_symlink())
            self.assertFalse(any(p.startswith('.') for p in Path(name).parts))
            self.assertFalse(any(s in path.name.lower() for s in ['credential', 'api_key', '.env']))
            if path.suffix in ('.py', '.json', '.csv', '.md', '.txt'):
                text = path.read_text()
                self.assertNotIn('/' + 'Users' + '/', text)
                self.assertNotIn('fan' + 'qingyang', text)
        for path in (ROOT / 'src').rglob('*.py'):
            ast.parse(path.read_text())

    def test_shock_example_and_documented_rmse_match_canonical_files(self):
        f = pd.read_csv(ROOT / 'results/modeling/test_forecasts.csv')
        shock = f.loc[f.forecast_origin.eq('2020-06-30') & f.horizon.eq(6) & f.model.eq('M2')].iloc[0]
        self.assertAlmostEqual(shock.yhat, 71.428, places=3)
        self.assertAlmostEqual(shock.actual, 8.7)
        validation = pd.read_csv(ROOT / 'results/modeling/validation_metrics.csv')
        expected = {'M0':[.1620,.2358,.3192], 'M1':[.1387,.2200,.3070],
                    'M2':[.1647,.2839,.4460], 'M3':[.1604,.2767,.4283]}
        for model, values in expected.items():
            np.testing.assert_allclose(validation.loc[validation.model.eq(model)].sort_values('horizon').rmse, values, atol=.00005)
        test = pd.read_csv(ROOT / 'results/modeling/test_metrics.csv')
        expected = {'M0':[.286,.502,.881], 'M1':[.315,.537,.916],
                    'M2':[1.021,.983,1.240], 'M3':[.778,.713,1.118]}
        for model, values in expected.items():
            mask = test.model.eq(model) & test.period.eq('post_covid') & test.evaluation_view.eq('requested_full_period')
            np.testing.assert_allclose(test.loc[mask].sort_values('horizon').rmse, values, atol=.0005)

    def test_core_input_closure_is_explicit(self):
        from src.models.feature_engineering import load_core_inputs
        panel, rules, calendar, raw = load_core_inputs(ROOT)
        self.assertEqual(set(panel.columns), {'date','ny_ur','ny_claims_avg','ny_payrolls','us_ur'})
        self.assertEqual(set(raw), {'NYICLAIMS'})
        self.assertFalse(calendar.duplicated(['variable','reference_date']).any())
        self.assertEqual(len(list((ROOT / 'data').rglob('*.csv'))), 5)


if __name__ == '__main__':
    unittest.main()

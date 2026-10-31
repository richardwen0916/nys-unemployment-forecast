"""Forecast alignment, publication gating, immutable X and calendar gap tests."""
from pathlib import Path
import json
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from src.models.data_availability import forecast_origin
from src.models.feature_engineering import build_feature_window, load_core_inputs
from src.models.rolling_validation import (
    predefined_specs, validation_plan, training_sample, direct_forecast,
    released_nyur, arima_forecasts, origin_row, scored_metrics,
)

ROOT=Path(__file__).resolve().parents[1]


class RollingValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel,cls.rules,cls.calendar,cls.raw=load_core_inputs(ROOT)
        cls.window=build_feature_window(cls.panel,'2016-03-31',cls.rules,releases=cls.calendar,raw=cls.raw)

    def synthetic_bank(self):
        dates=pd.date_range('2014-01-31','2014-05-31',freq='ME',name='forecast_origin')
        features=pd.DataFrame({'x':np.arange(5,dtype=float),'latest_nyur_level':5.},index=dates)
        rows=[]
        for origin in dates:
            target=(origin.to_period('M')+1).to_timestamp()
            release=target+pd.offsets.MonthEnd(0)+pd.Timedelta(days=45)
            rows.append(dict(forecast_origin=origin,horizon=1,target_reference=target,
                             target_ny_ur_level=5.+origin.month/10,
                             label_available_at=forecast_origin(release).tz_convert('UTC')))
        return features,pd.DataFrame(rows).set_index(['forecast_origin','horizon'])

    def test_target_reference_alignment_and_reserved_period_guard(self):
        plan=validation_plan()
        self.assertEqual(len(plan),144)
        self.assertTrue(plan.groupby('horizon').size().eq(48).all())
        for row in plan.itertuples():
            self.assertEqual(row.forecast_origin.to_period('M')+row.horizon,row.target_month.to_period('M'))
        first=plan.loc[plan.target_month.eq('2016-01-01')].set_index('horizon')
        self.assertEqual(first.at[6,'forecast_origin'],pd.Timestamp('2015-07-31'))
        self.assertEqual(first.at[1,'forecast_origin'],pd.Timestamp('2015-12-31'))
        with self.assertRaisesRegex(ValueError,'reserves'):
            validation_plan(end='2020-01-01')

    def test_predefined_grid_and_no_payroll_lag1(self):
        specs=predefined_specs()
        self.assertEqual(sum(s.model=='M1' for s in specs),18)
        self.assertEqual(sum(s.model=='M2' for s in specs),2)
        self.assertEqual(sum(s.model=='M3' for s in specs),4)
        for spec in specs:
            self.assertFalse(any(c=='ny_payrolls_log_growth_pct_lag1' for c in spec.features))
            self.assertFalse(any('ny_claims_avg_sa' in c for c in spec.features))

    def test_labels_become_eligible_only_after_publication(self):
        features,labels=self.synthetic_bank()
        X,y,meta,reason=training_sample(features,labels,'2014-05-31',1,('x',))
        self.assertEqual(list(X.index),list(features.index[:2]))
        self.assertEqual(meta['excluded_unpublished_labels'],2)
        self.assertLessEqual(meta['max_train_label_available_at'],forecast_origin('2014-05-31'))
        labels.loc[(features.index[0],1),'label_available_at']=pd.Timestamp('2014-06-01',tz='America/New_York').tz_convert('UTC')
        delayed,_,_,_=training_sample(features,labels,'2014-05-31',1,('x',))
        self.assertEqual(list(delayed.index),[features.index[1]])

    def test_future_labels_and_future_rows_do_not_change_training_sample(self):
        features,labels=self.synthetic_bank()
        before_X,before_y,_,_=training_sample(features,labels,'2014-05-31',1,('x',))
        changed=labels.copy()
        changed.loc[(features.index[2:],1),'target_ny_ur_level']=1e12
        future=features.copy();future.loc['2014-05-31','x']=1e15
        after_X,after_y,_,_=training_sample(future,changed,'2014-05-31',1,('x',))
        assert_frame_equal(before_X,after_X);assert_series_equal(before_y,after_y)

    def test_missing_training_features_and_actuals_are_explicit_exclusions(self):
        features,labels=self.synthetic_bank()
        features.loc[features.index[0],'x']=np.nan
        labels.loc[(features.index[1],1),'target_ny_ur_level']=np.nan
        X,y,meta,_=training_sample(features,labels,'2014-05-31',1,('x',))
        self.assertTrue(X.empty);self.assertTrue(y.empty)
        self.assertEqual(meta['excluded_missing_predictors'],1)
        self.assertEqual(meta['excluded_missing_actuals'],1)
        self.assertEqual(json.loads(meta['missing_feature_counts']),{'x':1})

    def test_direct_level_reconstruction_and_fit_only_scaling(self):
        X=pd.DataFrame({'x':np.linspace(-2,2,80)})
        y=1.5+.2*X.x
        current=pd.Series({'x':.5,'latest_nyur_level':8.})
        yhat,info,parameters=direct_forecast(X,y,current)
        self.assertAlmostEqual(yhat,9.6)
        self.assertEqual(info['status'],'ok')
        self.assertAlmostEqual(parameters[1]['train_mean'],X.x.mean())
        self.assertAlmostEqual(parameters[1]['train_scale'],X.x.std(ddof=0))
        missing=current.copy();missing['x']=np.nan
        self.assertTrue(np.isnan(direct_forecast(X,y,missing)[0]))
        self.assertEqual(direct_forecast(X,y,missing)[1]['status'],'missing_current_predictor')

    def test_arima_random_walk_advances_through_reporting_delay_and_preserves_gap(self):
        dates=pd.date_range('2000-01-01',periods=100,freq='MS')
        series=pd.Series(5.+.01*np.arange(100)+.1*np.sin(np.arange(100)),index=dates)
        series.iloc[40]=np.nan
        original=series.copy()
        origin=(dates[-1].to_period('M')+2).to_timestamp()+pd.offsets.MonthEnd(0)
        targets={h:(origin.to_period('M')+h).to_timestamp() for h in (1,3,6)}
        predictions,info,_=arima_forecasts(series,origin,(0,1,0),targets)
        self.assertEqual(info['status'],'ok')
        self.assertEqual(info['internal_missing'],1)
        self.assertEqual(info['n_calendar'],100)
        self.assertEqual(json.loads(info['forecast_steps']),{'1':3,'3':5,'6':8})
        for value in predictions.values():
            self.assertAlmostEqual(value,series.iloc[-1],places=6)
        assert_series_equal(series,original)

    def test_released_nyur_keeps_internal_months_and_hides_future_values(self):
        panel=self.panel.copy()
        panel.loc[panel.date.eq('2014-06-01'),'ny_ur']=np.nan
        before,_=released_nyur(panel,'2015-12-31',self.rules,self.calendar)
        panel.loc[panel.date.ge('2015-11-01'),'ny_ur']=1e12
        after,_=released_nyur(panel,'2015-12-31',self.rules,self.calendar)
        assert_series_equal(before,after)
        self.assertTrue(pd.isna(before.loc['2014-06-01']))
        self.assertEqual(before.index[-1],pd.Timestamp('2015-10-01'))

    def test_latest_information_is_actual_published_month_not_fixed_lag2(self):
        values,meta=origin_row(self.window)
        # March 2016 origin: Jan waits to Apr 15 under unchanged NYUR policy;
        # Feb waits to Apr 14, so December is the latest NYUR actually available.
        self.assertEqual(meta['latest_nyur_reference'],pd.Timestamp('2015-12-01'))
        self.assertEqual(values.nyur_age_months,3)
        self.assertEqual(values.latest_nyur_level,self.window.target_levels.loc['2015-12-01'])
        self.assertLessEqual(meta['stl_last_reference'],pd.Timestamp('2016-02-01'))

    def test_historical_snapshots_are_unchanged_by_later_information(self):
        before,_=origin_row(self.window)
        changed=self.panel.copy()
        changed.loc[changed.date.ge('2016-04-01'),['ny_ur','ny_claims_avg','ny_payrolls','us_ur']]=1e9
        raw={'NYICLAIMS':self.raw['NYICLAIMS'].copy()}
        raw['NYICLAIMS'].loc['2016-04-01':]=1e9
        window=build_feature_window(changed,'2016-03-31',self.rules,releases=self.calendar,raw=raw)
        after,_=origin_row(window)
        assert_series_equal(before,after)

    def test_common_evaluation_excludes_missing_actuals_and_missing_predictions(self):
        rows=[]
        for i,target in enumerate(pd.date_range('2016-01-01',periods=3,freq='MS')):
            origin=target-pd.Timedelta(days=1)
            for spec,model in [('naive','M0'),('core','M2')]:
                rows.append(dict(forecast_origin=origin,target_month=target,horizon=1,spec_id=spec,model=model,
                    status='ok',actual=5. if i<2 else np.nan,
                    yhat=6. if model=='M0' else 5.5 if i==0 else np.nan))
        metrics=scored_metrics(pd.DataFrame(rows),common=True).set_index('model')
        self.assertTrue(metrics.n_scored.eq(1).all())
        self.assertAlmostEqual(metrics.at['M2','rmse_improvement_pct'],50.)

    def test_reserved_labels_are_excluded_even_when_published(self):
        dates=pd.date_range('2019-09-30','2020-01-31',freq='ME',name='forecast_origin')
        features=pd.DataFrame({'x':np.arange(5.),'latest_nyur_level':5.},index=dates)
        rows=[]
        for date in dates:
            target=(date.to_period('M')+1).to_timestamp()
            release=target+pd.offsets.MonthEnd(0)+pd.Timedelta(days=45)
            rows.append(dict(forecast_origin=date,horizon=1,target_reference=target,
                             target_ny_ur_level=5.,label_available_at=forecast_origin(release).tz_convert('UTC')))
        labels=pd.DataFrame(rows).set_index(['forecast_origin','horizon'])
        X,_,meta,_=training_sample(features,labels,'2020-05-31',1,('x',))
        self.assertEqual(len(X),3)
        self.assertEqual(meta['excluded_outside_target_period'],2)
        self.assertLessEqual(meta['training_last_target'],pd.Timestamp('2019-12-01'))

    def test_selection_rejects_reserved_targets(self):
        from src.models.run_validation import select_specifications
        candidate=pd.DataFrame({'target_month':pd.to_datetime(['2020-01-01'])})
        with self.assertRaisesRegex(ValueError,'only 2016-2019'):
            select_specifications(candidate,predefined_specs())

    def test_saved_forecasts_have_common_targets_and_released_training_labels(self):
        path=ROOT/'results/modeling/validation_forecasts.csv'
        if not path.exists():
            self.skipTest('Run the offline validation runner to create integration artifacts')
        frame=pd.read_csv(path,parse_dates=['forecast_origin','target_month'])
        self.assertEqual(len(frame),576)
        self.assertTrue(frame.target_month.between('2016-01-01','2019-12-01').all())
        self.assertTrue(frame.groupby(['model','horizon']).size().eq(48).all())
        released=pd.to_datetime(frame.max_train_label_available_at,utc=True,format='mixed')
        cutoff=pd.to_datetime(frame.fit_as_of,utc=True,format='mixed')
        self.assertTrue(released.le(cutoff).all())
        for row in frame.itertuples():
            self.assertEqual(row.forecast_origin.to_period('M')+row.horizon,row.target_month.to_period('M'))

    def test_october_2025_target_remains_missing(self):
        series,_=released_nyur(self.panel,'2026-01-31',self.rules,self.calendar)
        self.assertTrue(pd.isna(series.loc['2025-10-01']))


if __name__=='__main__':
    unittest.main()

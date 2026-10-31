"""Frozen specification, expanding publication gating, and shutdown regressions."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal,assert_series_equal

from src.models.data_availability import available_panel,forecast_origin
from src.models.feature_engineering import load_core_inputs,build_feature_window
from src.models.rolling_validation import training_sample,origin_row,released_nyur,arima_forecasts,direct_forecast
from src.models.out_of_sample import freeze_specs,test_plan as make_test_plan,expanding_training_sample,evaluate_metrics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/modeling'


class OutOfSampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel,cls.rules,cls.calendar,cls.raw=load_core_inputs(ROOT)
        cls.january=build_feature_window(cls.panel,'2026-01-31',cls.rules,releases=cls.calendar,raw=cls.raw)
        cls.december=build_feature_window(cls.panel,'2025-12-31',cls.rules,releases=cls.calendar,raw=cls.raw)

    def bank(self,start='2019-09-30',end='2020-05-31'):
        dates=pd.date_range(start,end,freq='ME',name='forecast_origin')
        features=pd.DataFrame({'x':np.arange(len(dates),dtype=float),'latest_nyur_level':5.},index=dates)
        rows=[]
        for date in dates:
            reference=(date.to_period('M')+1).to_timestamp()
            release=reference+pd.offsets.MonthEnd(0)+pd.Timedelta(days=45)
            rows.append(dict(forecast_origin=date,horizon=1,target_reference=reference,
                target_ny_ur_level=5.+date.month/10,label_available_at=forecast_origin(release).tz_convert('UTC')))
        return features,pd.DataFrame(rows).set_index(['forecast_origin','horizon'])

    def test_saved_specifications_are_exactly_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            specs,cutoff=freeze_specs(ROOT,Path(tmp))
            self.assertEqual(cutoff,'2020-01-31')
            self.assertEqual(len(specs),12)
            for h in (1,3,6):
                self.assertEqual(specs[(h,'M1')].order,(0,1,2))
                self.assertEqual(specs[(h,'M2')].spec_id,'M2_payroll_lag2')
                self.assertFalse(any('payrolls_log_growth_pct_lag1' in c for c in specs[(h,'M2')].features))
            self.assertEqual(specs[(1,'M3')].spec_id,'M3_payroll_lag3_us_diff_pp')
            self.assertEqual(specs[(3,'M3')].spec_id,'M3_payroll_lag3_us_level')
            self.assertEqual(specs[(6,'M3')].spec_id,'M3_payroll_lag2_us_diff_pp')

    def test_lock_rejects_test_driven_feature_or_order_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);freeze_specs(ROOT,path)
            marker=path/'test_frozen_specifications.json'
            before=marker.read_bytes();freeze_specs(ROOT,path)
            self.assertEqual(marker.read_bytes(),before)
            data=json.loads(before);data['specifications'][1]['order']=[2,1,2]
            marker.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'mismatch'):
                freeze_specs(ROOT,path)

    def test_target_plan_preserves_missing_month_and_chronology_flags(self):
        plan=make_test_plan('2026-08-01','2020-01-31')
        self.assertEqual(len(plan),240)
        self.assertEqual(plan.target_month.nunique(),80)
        self.assertEqual(len(plan.loc[plan.target_month.eq('2025-10-01')]),3)
        for row in plan.itertuples():
            self.assertEqual(row.forecast_origin.to_period('M')+row.horizon,row.target_month.to_period('M'))
        covid=plan.loc[plan.test_period.eq('covid')]
        for h,count in [(1,23),(3,21),(6,18)]:
            self.assertEqual(covid.loc[covid.horizon.eq(h)].origin_after_validation_cutoff.sum(),count)

    def test_expanding_adapter_matches_original_helper_before_2020(self):
        features,labels=self.bank('2019-01-31','2019-09-30')
        old_X,old_y,_,_=training_sample(features,labels,'2019-08-31',1,('x',))
        new_X,new_y,_,_=expanding_training_sample(features,labels,'2019-08-31',1,('x',))
        assert_frame_equal(old_X,new_X);assert_series_equal(old_y,new_y)

    def test_later_test_labels_enter_only_after_publication(self):
        features,labels=self.bank()
        X,y,info,_=expanding_training_sample(features,labels,'2020-05-31',1,('x',))
        self.assertEqual(info['training_last_target'],pd.Timestamp('2020-03-01'))
        self.assertTrue((X.index<pd.Timestamp('2020-05-31')).all())
        self.assertLessEqual(info['max_train_label_available_at'],forecast_origin('2020-05-31'))
        self.assertIn(pd.Timestamp('2020-02-29'),X.index)
        old,_,_,_=training_sample(features,labels,'2020-05-31',1,('x',))
        self.assertLess(len(old),len(X))
        changed=labels.copy()
        changed.loc[(pd.Timestamp('2020-02-29'),1),'label_available_at']=pd.Timestamp('2020-06-01',tz='America/New_York').tz_convert('UTC')
        delayed,_,_,_=expanding_training_sample(features,changed,'2020-05-31',1,('x',))
        self.assertNotIn(pd.Timestamp('2020-02-29'),delayed.index)

    def test_unpublished_target_values_and_future_X_cannot_change_fit_sample(self):
        features,labels=self.bank()
        before_X,before_y,_,_=expanding_training_sample(features,labels,'2020-05-31',1,('x',))
        future=labels.copy();future.loc[future.target_reference.ge('2020-04-01'),'target_ny_ur_level']=1e15
        changed=features.copy();changed.loc['2020-05-31','x']=1e20
        after_X,after_y,_,_=expanding_training_sample(changed,future,'2020-05-31',1,('x',))
        assert_frame_equal(before_X,after_X);assert_series_equal(before_y,after_y)

    def test_original_october_2025_is_nan(self):
        row=self.panel.loc[self.panel.date.eq('2025-10-01'),'ny_ur']
        self.assertEqual(len(row),1);self.assertTrue(row.isna().all())
        self.assertTrue(pd.isna(self.january.history.at[pd.Timestamp('2025-10-01'),'ny_ur_level']))

    def test_november_differences_do_not_bridge_october(self):
        for column in ['ny_ur_diff_pp','us_ur_diff_pp']:
            self.assertTrue(pd.isna(self.january.history.at[pd.Timestamp('2025-11-01'),column]))
        values,meta=origin_row(self.january)
        self.assertEqual(meta['latest_nyur_reference'],pd.Timestamp('2025-11-01'))
        self.assertTrue(pd.isna(values.latest_nyur_diff_pp))
        december,_=origin_row(self.december)
        self.assertTrue(pd.isna(december.latest_usur_diff_pp))

    def test_direct_model_missing_current_difference_has_no_fallback(self):
        values,_=origin_row(self.january)
        specs,_=freeze_specs()
        for h in (1,3,6):
            for model in ('M2','M3'):
                columns=list(specs[(h,model)].features)
                X=pd.DataFrame(np.random.default_rng(10).normal(size=(80,len(columns))),columns=columns)
                y=pd.Series(np.linspace(-.5,.5,80))
                forecast,info,parameters=direct_forecast(X,y,values)
                self.assertTrue(np.isnan(forecast))
                self.assertEqual(info['status'],'missing_current_predictor')
                self.assertEqual(parameters,[])

    def test_state_space_retains_gap_when_subsequent_releases_arrive(self):
        before,_=released_nyur(self.panel,'2025-12-31',self.rules,self.calendar)
        january,_=released_nyur(self.panel,'2026-01-31',self.rules,self.calendar)
        february,_=released_nyur(self.panel,'2026-02-28',self.rules,self.calendar)
        self.assertEqual(before.index[-1],pd.Timestamp('2025-09-01'))
        self.assertEqual(january.index[-1],pd.Timestamp('2025-11-01'))
        self.assertEqual(len(january)-len(before),2)
        self.assertEqual(january.notna().sum()-before.notna().sum(),1)
        for series,origin in [(january,'2026-01-31'),(february,'2026-02-28')]:
            self.assertTrue(pd.isna(series.loc['2025-10-01']))
            self.assertEqual(series.isna().sum(),1)
            self.assertTrue(series.index.equals(pd.date_range(series.index[0],series.index[-1],freq='MS')))
            targets={h:(pd.Timestamp(origin).to_period('M')+h).to_timestamp() for h in (1,3,6)}
            prediction,info,_=arima_forecasts(series,origin,(0,1,2),targets)
            self.assertEqual(info['status'],'ok');self.assertEqual(info['internal_missing'],1)
            self.assertTrue(np.isfinite(list(prediction.values())).all())

    def test_common_metrics_exclude_missing_actual_and_missing_predictions(self):
        rows=[]
        for i,target in enumerate(pd.date_range('2025-09-01',periods=3,freq='MS')):
            for model in ('M0','M1','M2','M3'):
                rows.append(dict(forecast_origin=target-pd.Timedelta(days=1),target_month=target,horizon=1,
                    test_period='post_covid',origin_after_validation_cutoff=True,model=model,spec_id=model,
                    status='missing_current_predictor' if i==2 and model=='M2' else 'ok',
                    yhat=np.nan if i==2 and model=='M2' else 5.5,actual=np.nan if i==1 else 5.))
        frame=pd.DataFrame(rows);before=frame.copy()
        metrics=evaluate_metrics(frame)
        post=metrics.loc[metrics.period.eq('post_covid')].set_index('model')
        self.assertTrue(post.n_scored.eq(1).all());self.assertTrue(post.n_actual.eq(2).all())
        self.assertEqual(post.at['M0','n_forecasts'],3);self.assertEqual(post.at['M2','n_forecasts'],2)
        self.assertAlmostEqual(post.at['M0','bias'],.5)
        assert_frame_equal(frame,before)

    def test_saved_october_forecasts_are_kept_but_never_scored(self):
        path=OUT/'test_forecasts.csv'
        if not path.exists():
            self.skipTest('Run the frozen evaluator for integration artifacts')
        f=pd.read_csv(path,parse_dates=['forecast_origin','target_month'])
        self.assertEqual(len(f),960)
        gap=f.loc[f.target_month.eq('2025-10-01')]
        self.assertEqual(len(gap),12);self.assertTrue(gap.actual.isna().all())
        self.assertTrue(gap.yhat.notna().all());self.assertFalse(gap.common_scored.any())
        self.assertTrue(gap.score_exclusion_reason.eq('missing_actual').all())
        maximum=pd.to_datetime(f.max_train_label_available_at,utc=True,format='mixed')
        cutoff=pd.to_datetime(f.fit_as_of,utc=True,format='mixed')
        self.assertTrue(maximum.le(cutoff).all())




if __name__=='__main__':
    unittest.main()

"""Focused tests for Step 3 temporal eligibility, preprocessing and metrics."""
import unittest
import numpy as np
import pandas as pd

from src.models.data_availability import available_panel
from src.experiments.step3_ml.features import ROOT, regional_row, AR_FEATURES
from src.experiments.step3_ml.models import fit_predict, sample
from src.experiments.step3_ml.report import metrics, ALL_MODELS


class Step3Tests(unittest.TestCase):
    def test_scaling_uses_training_only(self):
        X=pd.DataFrame({'latest_nyur_level':np.linspace(3,5,70),
                        'latest_nyur_diff_pp':np.sin(np.arange(70)),
                        'nyur_age_months':np.ones(70)})
        y=pd.Series(np.linspace(-.2,.2,70))
        current=pd.Series({'latest_nyur_level':100.,'latest_nyur_diff_pp':100.,'nyur_age_months':1.})
        prediction,status,model=fit_predict('ridge',{'alpha':10.},X,y,current)
        self.assertEqual(status,'ok')
        np.testing.assert_allclose(model.steps[0][1].mean_,X.mean())
        self.assertTrue(np.isfinite(prediction))

    def test_missing_current_input_is_not_imputed(self):
        X=pd.DataFrame(np.ones((70,3)),columns=AR_FEATURES)
        current=pd.Series({AR_FEATURES[0]:4.,AR_FEATURES[1]:np.nan,AR_FEATURES[2]:1.})
        _,status,_=fit_predict('ridge',{'alpha':1.},X,pd.Series(np.ones(70)),current)
        self.assertEqual(status,'missing_current_predictor')

    def test_unreleased_label_and_missing_target_are_excluded(self):
        idx=pd.date_range('2010-01-31',periods=90,freq='ME',name='forecast_origin')
        features=pd.DataFrame({c:1. for c in AR_FEATURES},index=idx)
        target=(idx.to_period('M')+1).to_timestamp()
        labels=pd.DataFrame({'target_reference':target,'target_ny_ur_level':4.,
                             'label_available_at':pd.to_datetime(target,utc=True)+pd.Timedelta(days=45)},index=idx)
        labels.loc[idx[30],'target_ny_ur_level']=np.nan
        labels.loc[idx[31],'label_available_at']=pd.Timestamp('2099-01-01',tz='UTC')
        labels['horizon']=1
        labels=labels.set_index('horizon',append=True)
        X,_,_=sample(features,labels,idx[-1],1,AR_FEATURES)
        self.assertNotIn(idx[30],X.index)
        self.assertNotIn(idx[31],X.index)
        self.assertLess(X.index.max(),idx[-1])

    def test_regional_features_ignore_future_values(self):
        panel=pd.read_csv(ROOT/'data/processed/monthly_panel.csv',parse_dates=['date'])
        rules=pd.read_csv(ROOT/'data/experiments/release_rules_v2.csv')
        calendar=pd.read_csv(ROOT/'data/experiments/release_calendar_v2.csv')
        before,_=regional_row(panel,'2023-12-31',rules,calendar)
        panel.loc[panel.date.gt('2023-12-31'),['nj_ur','ct_ur','pa_ur','empire_general']]=999999.
        after,_=regional_row(panel,'2023-12-31',rules,calendar)
        pd.testing.assert_series_equal(pd.Series(before),pd.Series(after))

    def test_rmse_and_common_sample(self):
        rows=[]
        for model in ALL_MODELS:
            for date,error in [('2023-01-01',3.),('2023-02-01',4.)]:
                rows.append(dict(origin=pd.Timestamp(date)-pd.offsets.MonthEnd(1),date=pd.Timestamp(date),
                                 period='post_covid',horizon=1,model=model,actual=4.,yhat=4.+error,status='ok'))
        table,coverage=metrics(pd.DataFrame(rows))
        selected=table[(table.view=='post_covid')&(table['sample']=='common_all_models')]
        np.testing.assert_allclose(selected.rmse,np.sqrt(12.5))
        self.assertTrue(selected.n.eq(2).all())
        rows[-1]['yhat']=np.nan
        table,_=metrics(pd.DataFrame(rows))
        selected=table[(table.view=='post_covid')&(table['sample']=='common_all_models')]
        self.assertTrue(selected.n.eq(1).all())


if __name__ == '__main__':
    unittest.main()

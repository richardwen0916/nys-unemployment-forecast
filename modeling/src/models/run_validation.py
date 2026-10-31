"""Offline Phase 1 validation; run python -m src.models.run_validation.

Only 2016-2019 targets are evaluated. No 2020+ target or predictor enters a fit,
selection criterion or diagnostic. Existing datasets/release files are read-only.
"""
from pathlib import Path
from dataclasses import asdict
import hashlib
import inspect
import json
import platform
import warnings

import numpy as np
import pandas as pd
import statsmodels

from .data_availability import available_panel, forecast_origin, target_months
from .feature_engineering import load_core_inputs
from .rolling_validation import (
    HORIZONS, predefined_specs, origin_row, build_origin_bank, label_bank,
    training_sample, direct_forecast, released_nyur, arima_forecasts,
    validation_plan, scored_metrics,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/modeling'
EVALUATION_CUTOFF = '2020-01-31'  # outcomes only, never a fitting cutoff


def snapshot_fingerprint(root):
    names = ['data/processed/monthly_panel.csv','data/raw/fred_NYICLAIMS.csv',
             'data/processed/release_rules.csv','data/processed/release_calendar.csv',
             'src/models/data_availability.py','src/models/rolling_stl.py','src/models/feature_engineering.py']
    hashes = {n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}
    hashes['origin_row_function']=hashlib.sha256(inspect.getsource(origin_row).encode()).hexdigest()
    return hashes


def snapshots(panel, rules, calendar, raw, last_origin):
    """Disk cache only for origin-specific snapshots with identical input hashes."""
    fingerprint=snapshot_fingerprint(ROOT)
    marker=OUT/'snapshot_manifest.json'
    files=['historical_features.csv','historical_feature_metadata.csv','historical_labels.csv','historical_feature_audit.csv']
    if marker.exists() and all((OUT/name).exists() for name in files):
        saved=json.loads(marker.read_text())
        if saved['inputs']==fingerprint and saved['last_origin']==str(last_origin.date()):
            if all(hashlib.sha256((OUT/name).read_bytes()).hexdigest()==saved['output_sha256'][name] for name in files):
                print('Reusing verified per-origin snapshots; no old STL is refitted at a later origin.',flush=True)
                features=pd.read_csv(OUT/files[0],parse_dates=['forecast_origin']).set_index('forecast_origin')
                meta=pd.read_csv(OUT/files[1],parse_dates=['forecast_origin','latest_nyur_reference','latest_usur_reference','stl_first_reference','stl_last_reference']).set_index('forecast_origin')
                labels=pd.read_csv(OUT/files[2],parse_dates=['forecast_origin','target_reference']).set_index(['forecast_origin','horizon'])
                labels['label_available_at']=pd.to_datetime(labels.label_available_at,utc=True,format='mixed')
                return features,meta,labels
    origins=pd.date_range('1990-01-31',last_origin,freq='ME')
    features,meta,windows=build_origin_bank(panel,origins,rules,calendar,raw,progress=lambda s:print(s,flush=True))
    labels=label_bank(panel,windows,EVALUATION_CUTOFF,rules,calendar)
    features.to_csv(OUT/files[0]);meta.to_csv(OUT/files[1]);labels.to_csv(OUT/files[2])
    audit=pd.concat([w.audit.reset_index().assign(forecast_origin=o) for o,w in zip(origins,windows)],ignore_index=True)
    audit.to_csv(OUT/files[3],index=False)
    prior=pd.read_csv(ROOT/'results/payroll_release_audit/origin_features.csv',parse_dates=['origin']).set_index('origin')
    np.testing.assert_allclose(features[prior.columns].to_numpy(),prior.reindex(origins).to_numpy(),equal_nan=True,rtol=1e-10,atol=1e-10)
    saved=dict(inputs=fingerprint,last_origin=str(last_origin.date()),origins=len(origins),
               stl_policy='Each original historical origin, expanding start 1990-01, robust log-STL, minimum 36 months',
               prior_audited_snapshot_reproduction='all original 21 features match within 1e-10',
               output_sha256={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in files})
    marker.write_text(json.dumps(saved,indent=2)+'\n')
    return features,meta,labels


def select_specifications(candidates, specs):
    """Require full planned coverage before ranking validation RMSE.

    This predeclared rule prevents numerical failures or missing predictions from
    improving a candidate's apparent score by excluding hard targets.
    """
    if not candidates.target_month.between('2016-01-01','2019-12-01').all():
        raise ValueError('Selection may use only 2016-2019 validation targets')
    own=scored_metrics(candidates,common=False)
    selected=[]
    for h in HORIZONS:
        for family in ('M0','M1','M2','M3'):
            subset=own.loc[own.horizon.eq(h)&own.model.eq(family)&own.n_scored.eq(48)].copy()
            if subset.empty:
                raise RuntimeError(f'No full-coverage candidate for {family}, horizon {h}; inspect exclusions')
            subset['complexity']=subset.spec_id.map({s.spec_id:sum(s.order) if s.order else len(s.features) for s in specs})
            best=subset.sort_values(['rmse','mae','complexity','spec_id']).iloc[0]
            selected.append(dict(horizon=h,model=family,spec_id=best.spec_id,
                                 selection_rmse=best.rmse,selection_n=int(best.n_scored)))
    return own,pd.DataFrame(selected)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    plan=validation_plan()
    plan.to_csv(OUT/'validation_plan.csv',index=False)
    specs=predefined_specs()
    declared=[]
    for s in specs:
        record=asdict(s)
        record['features']=';'.join(s.features)
        record['order']=','.join(map(str,s.order))
        record['response']='NYUR level' if s.model in ('M0','M1') else 'NYUR target minus latest published NYUR at original feature origin'
        record['trend']='constant' if s.order and s.order[1]==0 else ('none' if s.order else 'OLS intercept' if s.features else 'none')
        record['minimum_training_observations']=60 if s.model!='M0' else 1
        declared.append(record)
    pd.DataFrame(declared).to_csv(OUT/'model_specifications.csv',index=False)
    panel,rules,calendar,raw=load_core_inputs(ROOT)
    # Reserve all 2020+ values from preprocessing, model fitting and scoring.
    panel=panel.loc[panel.date.between('1990-01-01','2019-12-01')].copy()
    raw={'NYICLAIMS':raw['NYICLAIMS'].loc[:'2019-11-30'].copy()}
    features,meta,labels=snapshots(panel,rules,calendar,raw,plan.forecast_origin.max())
    actual_view=available_panel(panel[['date','ny_ur']],EVALUATION_CUTOFF,rules,releases=calendar)
    actuals=actual_view.data.ny_ur
    forecasts,samples,coefficients,notes=[],[],[],[]
    for i,(origin,tasks) in enumerate(plan.groupby('forecast_origin',sort=True)):
        current=features.loc[origin]
        series,max_release=released_nyur(panel,origin,rules,calendar)
        horizons=tasks.horizon.tolist()
        targets={int(r.horizon):r.target_month for r in tasks.itertuples()}
        naive=float(series.dropna().iloc[-1])
        assert np.isclose(naive,current.latest_nyur_level)
        base=dict(forecast_origin=origin,fit_as_of=forecast_origin(origin),
                  latest_nyur_reference=series.index[-1],latest_nyur_level=naive,
                  nyur_age_months=int(current.nyur_age_months),
                  max_predictor_information_at=forecast_origin(origin),
                  stl_last_reference=meta.at[origin,'stl_last_reference'],
                  availability_policy='calendar_first_with_estimated_fallbacks',pseudo_real_time=True)
        for spec in specs:
            if spec.model in ('M0','M1'):
                if spec.model=='M0':
                    predictions={h:naive for h in horizons}
                    info=dict(status='ok',n_train=int(series.notna().sum()),n_calendar=len(series),
                              internal_missing=int(series.isna().sum()),converged=True,fit_attempts=0)
                    caught=[]
                else:
                    predictions,info,caught=arima_forecasts(series,origin,spec.order,targets)
                for warning in caught:
                    notes.append(dict(forecast_origin=origin,spec_id=spec.spec_id,model=spec.model,**warning))
                for h in horizons:
                    target=targets[h]
                    record=dict(base,model=spec.model,spec_id=spec.spec_id,target_month=target,horizon=h,
                                yhat=predictions.get(h,np.nan),actual=actuals.get(target,np.nan),**info)
                    record['forecast_steps']=target.to_period('M').ordinal-series.index[-1].to_period('M').ordinal
                    record['max_train_label_available_at']=max_release
                    record['training_first_target']=series.index[0]
                    record['training_last_target']=series.index[-1]
                    forecasts.append(record)
            else:
                for h in horizons:
                    X,y,sample,reason=training_sample(features,labels,origin,h,spec.features)
                    yhat,fit_info,params=direct_forecast(X,y,current)
                    samples.append(dict(forecast_origin=origin,horizon=h,model=spec.model,spec_id=spec.spec_id,**sample,**fit_info))
                    coefficients += [dict(forecast_origin=origin,horizon=h,spec_id=spec.spec_id,**p) for p in params]
                    forecasts.append(dict(base,model=spec.model,spec_id=spec.spec_id,target_month=targets[h],horizon=h,
                                          yhat=yhat,actual=actuals.get(targets[h],np.nan),**sample,**fit_info))
        if (i+1)%4==0:
            print(f'Rolling fitting: {i+1}/{plan.forecast_origin.nunique()} distinct origins',flush=True)
    candidates=pd.DataFrame(forecasts)
    candidates['actual_available_at']=candidates.target_month.map(actual_view.audit.xs('ny_ur',level='variable').available_at)
    assert candidates.target_month.between('2016-01-01','2019-12-01').all()
    assert not candidates.duplicated(['forecast_origin','horizon','spec_id']).any()
    candidate_metrics,selection=select_specifications(candidates,specs)
    candidates=candidates.merge(selection[['horizon','spec_id']].assign(selected=True),on=['horizon','spec_id'],how='left',validate='many_to_one')
    candidates['selected']=candidates.selected.eq(True)
    chosen=candidates.loc[candidates.selected].copy()
    metrics=scored_metrics(chosen,common=True)
    metrics['sample']='common_selected_four_models'
    forecasts_path=OUT/'validation_forecasts.csv'
    candidates.to_csv(OUT/'candidate_forecasts.csv',index=False)
    chosen.to_csv(forecasts_path,index=False)
    candidate_metrics['selection_eligible']=candidate_metrics.n_scored.eq(48)
    candidate_metrics.to_csv(OUT/'candidate_metrics.csv',index=False)
    selection.to_csv(OUT/'selected_specifications.csv',index=False)
    metrics.to_csv(OUT/'validation_metrics.csv',index=False)
    pd.DataFrame(samples).to_csv(OUT/'training_sample_coverage.csv',index=False)
    pd.DataFrame(coefficients).to_csv(OUT/'direct_coefficients.csv',index=False)
    pd.DataFrame(notes,columns=['forecast_origin','spec_id','model','attempt','optimizer','warning']).to_csv(OUT/'fit_warnings.csv',index=False)
    candidates.loc[candidates.status.ne('ok')].to_csv(OUT/'forecast_exclusions.csv',index=False)
    cover=candidates.groupby(['horizon','model','spec_id']).agg(planned=('yhat','size'),
        forecasts=('yhat','count'),available_actuals=('actual','count'),
        minimum_training_n=('n_train','min'),maximum_training_n=('n_train','max')).reset_index()
    cover.to_csv(OUT/'forecast_coverage.csv',index=False)
    matched=[]
    for spec in specs:
        if spec.model=='M3':
            for h in HORIZONS:
                macro=candidate_metrics.loc[candidate_metrics.spec_id.eq(spec.spec_id)&candidate_metrics.horizon.eq(h)].iloc[0]
                core=candidate_metrics.loc[candidate_metrics.spec_id.eq(spec.core_spec)&candidate_metrics.horizon.eq(h)].iloc[0]
                matched.append(dict(horizon=h,macro_spec=spec.spec_id,matched_core_spec=spec.core_spec,
                    n_macro=int(macro.n_scored),n_core=int(core.n_scored),core_rmse=core.rmse,macro_rmse=macro.rmse,
                    rmse_reduction_pct=100*(1-macro.rmse/core.rmse)))
    pd.DataFrame(matched).to_csv(OUT/'matched_usur_comparisons.csv',index=False)
    provenance=dict(python=platform.python_version(),pandas=pd.__version__,numpy=np.__version__,statsmodels=statsmodels.__version__,
        horizons=HORIZONS,validation_target_start='2016-01-01',validation_target_end='2019-12-01',
        first_forecast_origin=str(plan.forecast_origin.min().date()),last_forecast_origin=str(plan.forecast_origin.max().date()),
        training_start='1990-01-01',evaluation_label_cutoff=EVALUATION_CUTOFF,
        test_periods_used_for_fitting_or_selection=False,candidates=len(specs),candidate_forecasts=len(candidates),
        selected_forecasts=len(chosen),selection='Full 48/48 coverage required; minimum validation RMSE per family/horizon; ties MAE then complexity then spec ID',
        target='NYUR level in percent; scoring revised snapshot; pseudo-real-time',
        direct_target='Change relative to latest published NYUR at each original row origin; add that origin baseline back',
        arima_policy='State-space ARIMA; c for d=0, no drift for d=1; internal NaNs retained; calendar steps include reporting delay',
        data_sources_expanded=False)
    (OUT/'run_metadata.json').write_text(json.dumps(provenance,indent=2)+'\n')
    from .validation_report import write_outputs
    write_outputs(OUT,chosen,metrics,selection,candidate_metrics,pd.DataFrame(samples),pd.DataFrame(matched))
    print(metrics[['horizon','model','spec_id','n_scored','mae','rmse','rmse_improvement_pct']].to_string(index=False),flush=True)


if __name__=='__main__':
    run()

"""Final frozen-specification, expanding-window pseudo-real-time test evaluation.

Run: python -m src.models.out_of_sample
Existing source, release metadata and validation artifacts are never overwritten.
"""
from pathlib import Path
from dataclasses import asdict
import hashlib
import json

import numpy as np
import pandas as pd

from .data_availability import available_panel, forecast_origin
from .feature_engineering import build_feature_window, load_core_inputs
from .rolling_validation import (
    ModelSpec, HORIZONS, origin_month, origin_row, build_origin_bank,
    released_nyur, arima_forecasts, direct_forecast,
)
from .run_validation import snapshot_fingerprint

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/modeling'
AS_OF='2026-10-19'
MODELS=('M0','M1','M2','M3')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_specs(root=ROOT, out=OUT):
    """Load the saved validation choices; reject any subsequent lock mismatch."""
    source_files=['selected_specifications.csv','model_specifications.csv','run_metadata.json']
    source_hashes={n:sha(root/'results/modeling'/n) for n in source_files}
    selected=pd.read_csv(root/'results/modeling/selected_specifications.csv')
    declared=pd.read_csv(root/'results/modeling/model_specifications.csv',keep_default_na=False)
    merged=selected.merge(declared,on=['spec_id','model'],validate='many_to_one')
    if len(merged)!=12 or merged.duplicated(['horizon','model']).any():
        raise ValueError('Need exactly one saved specification per model and horizon')
    if set(map(tuple,merged[['horizon','model']].to_numpy()))!={(h,m) for h in HORIZONS for m in MODELS}:
        raise ValueError('Saved selection does not cover the required models/horizons')
    specs={}
    for row in merged.itertuples():
        features=tuple(row.features.split(';')) if row.features else ()
        order=tuple(map(int,row.order.split(','))) if row.order else ()
        specs[(int(row.horizon),row.model)]=ModelSpec(row.spec_id,row.model,features,order,row.core_spec)
    cutoff=json.loads((root/'results/modeling/run_metadata.json').read_text())['evaluation_label_cutoff']
    payload=dict(locked_as_of=AS_OF,source_sha256=source_hashes,
                 validation_information_cutoff=cutoff,
                 forecasting_function_sha256=sha(root/'src/models/rolling_validation.py'),
                 feature_function_sha256=sha(root/'src/models/feature_engineering.py'),
                 stl_function_sha256=sha(root/'src/models/rolling_stl.py'),
                 specifications=[dict(horizon=h,**asdict(specs[(h,m)])) for h in HORIZONS for m in MODELS],
                 policy='Frozen orders, features, lags, transformations, trend, OLS scaling, min_nobs and optimizer policy; coefficients refitted only on released expanding history')
    marker=out/'test_frozen_specifications.json'
    if marker.exists():
        if json.loads(marker.read_text())!=json.loads(json.dumps(payload)):
            raise ValueError('Frozen specification/source hash mismatch; no test-driven replacement permitted')
    else:
        marker.write_text(json.dumps(payload,indent=2)+'\n')
        merged.to_csv(out/'test_frozen_specifications.csv',index=False)
    return specs,cutoff


def test_plan(last_target, selection_cutoff):
    rows=[]
    for target in pd.date_range('2020-01-01',last_target,freq='MS'):
        for h in HORIZONS:
            origin=(target.to_period('M')-h).to_timestamp()+pd.offsets.MonthEnd(0)
            rows.append(dict(forecast_origin=origin,target_month=target,horizon=h,
                test_period='covid' if target.year<=2021 else 'post_covid',
                origin_after_validation_cutoff=forecast_origin(origin)>=forecast_origin(selection_cutoff)))
    return pd.DataFrame(rows).sort_values(['forecast_origin','horizon']).reset_index(drop=True)


def make_label_bank(features,panel,actual_view):
    """Separate target values/timestamps; each fit gates them at its own origin."""
    source=actual_view.data.ny_ur
    timing=actual_view.audit.xs('ny_ur',level='variable').available_at
    rows=[]
    for origin in features.index:
        for h in HORIZONS:
            reference=(origin.to_period('M')+h).to_timestamp()
            rows.append(dict(forecast_origin=origin,horizon=h,target_reference=reference,
                target_ny_ur_level=source.get(reference,np.nan),label_available_at=timing.get(reference,pd.NaT)))
    labels=pd.DataFrame(rows).set_index(['forecast_origin','horizon'])
    labels['label_available_at']=pd.to_datetime(labels.label_available_at,utc=True,format='mixed')
    return labels


def expanding_training_sample(features,labels,origin,horizon,columns):
    """Same direct response/complete cases as Phase 1; allow later released labels.

    The old helper's 2019 ceiling is a validation boundary, not a frozen model
    parameter. No response after this fitting cutoff can become eligible.
    """
    origin=origin_month(origin);cutoff=forecast_origin(origin).tz_convert('UTC')
    if features.index.has_duplicates or labels.index.has_duplicates:
        raise ValueError('Duplicate historical feature/label keys')
    target=labels.xs(horizon,level='horizon').reindex(features.index)
    prior=features.index<origin
    in_history=target.target_reference.ge(pd.Timestamp('1990-01-01'))
    observed=target.target_ny_ur_level.notna()
    published=target.label_available_at.notna()&target.label_available_at.le(cutoff)
    complete=features.loc[:,list(columns)].notna().all(axis=1)&features.latest_nyur_level.notna()
    released=prior&in_history&observed&published
    keep=released&complete
    X=features.loc[keep,list(columns)].copy()
    y=target.loc[keep,'target_ny_ur_level']-features.loc[keep,'latest_nyur_level']
    reason=pd.Series('eligible',index=features.index)
    reason.loc[~complete]='missing_predictor';reason.loc[~published]='label_not_published'
    reason.loc[~observed]='missing_actual';reason.loc[~in_history]='outside_training_history';reason.loc[~prior]='not_prior_origin'
    info=dict(prior_rows=int(prior.sum()),labels_released=int(released.sum()),n_train=len(X),
        excluded_missing_predictors=int((released&~complete).sum()),
        excluded_unpublished_labels=int((prior&in_history&observed&~published).sum()),
        excluded_missing_actuals=int((prior&in_history&~observed).sum()),
        training_first_origin=X.index.min(),training_last_origin=X.index.max(),
        training_first_target=target.loc[keep,'target_reference'].min(),training_last_target=target.loc[keep,'target_reference'].max(),
        max_train_label_available_at=target.loc[keep,'label_available_at'].max(),
        missing_feature_counts=json.dumps({c:int(features.loc[released,c].isna().sum()) for c in columns},sort_keys=True),
        membership_sha256=hashlib.sha256('|'.join(str(d.date()) for d in X.index).encode()).hexdigest())
    if len(X):
        assert X.index.max()<origin and info['max_train_label_available_at']<=cutoff
    return X,y,info,reason


def origin_snapshots(panel,rules,calendar,raw,last_origin):
    """Preserve the verified Phase 1 prefix and extend at original origins only."""
    fingerprint=snapshot_fingerprint(ROOT)
    files=['test_historical_features.csv','test_historical_feature_metadata.csv','test_historical_feature_audit.csv']
    marker=OUT/'test_snapshot_manifest.json'
    if marker.exists():
        saved=json.loads(marker.read_text())
        if saved['inputs']==fingerprint and saved['last_origin']==str(last_origin.date()) and all(
            (OUT/n).exists() and sha(OUT/n)==saved['output_sha256'][n] for n in files):
            features=pd.read_csv(OUT/files[0],parse_dates=['forecast_origin']).set_index('forecast_origin')
            meta=pd.read_csv(OUT/files[1],parse_dates=['forecast_origin','latest_nyur_reference','latest_usur_reference','stl_first_reference','stl_last_reference']).set_index('forecast_origin')
            audit=pd.read_csv(OUT/files[2])
            print('Reusing hash-verified test-origin snapshots.',flush=True)
            return features,meta,audit
    old=json.loads((OUT/'snapshot_manifest.json').read_text())
    if old['inputs']!=fingerprint:
        raise ValueError('Phase 1 snapshot provenance mismatch; preserve originals and investigate')
    for name,value in old['output_sha256'].items():
        if sha(OUT/name)!=value:
            raise ValueError('Phase 1 snapshot content hash mismatch')
    prefix=pd.read_csv(OUT/'historical_features.csv',parse_dates=['forecast_origin']).set_index('forecast_origin')
    old_meta=pd.read_csv(OUT/'historical_feature_metadata.csv',parse_dates=['forecast_origin','latest_nyur_reference','latest_usur_reference','stl_first_reference','stl_last_reference']).set_index('forecast_origin')
    old_audit=pd.read_csv(OUT/'historical_feature_audit.csv',parse_dates=['forecast_origin'])
    origins=pd.date_range(prefix.index.max()+pd.offsets.MonthEnd(1),last_origin,freq='ME')
    extension,new_meta,windows=build_origin_bank(panel,origins,rules,calendar,raw,progress=lambda s:print(s,flush=True))
    features=pd.concat([prefix,extension]);meta=pd.concat([old_meta,new_meta])
    new_audit=pd.concat([w.audit.reset_index().assign(forecast_origin=o) for o,w in zip(origins,windows)],ignore_index=True)
    audit=pd.concat([old_audit,new_audit],ignore_index=True)
    pd.testing.assert_frame_equal(features.loc[prefix.index],prefix)
    features.to_csv(OUT/files[0]);meta.to_csv(OUT/files[1]);audit.to_csv(OUT/files[2],index=False)
    # Retain only available reference values in the shutdown audit, never a filled gap.
    reference_rows=[]
    for origin,window in zip(origins,windows):
        if pd.Timestamp('2025-09-30')<=origin<=pd.Timestamp('2026-01-31'):
            history=window.history.loc['2025-08-01':,['ny_ur_level','ny_ur_diff_pp','us_ur_level','us_ur_diff_pp']]
            reference_rows.append(history.reset_index().assign(forecast_origin=origin))
    pd.concat(reference_rows,ignore_index=True).to_csv(OUT/'test_shutdown_reference_audit.csv',index=False)
    marker.write_text(json.dumps(dict(inputs=fingerprint,last_origin=str(last_origin.date()),
        preserved_prefix_sha256=old['output_sha256'],prefix_origins=len(prefix),new_origins=len(extension),
        policy='Reuse original historical prefix; new robust STL per original origin; unchanged transformations',
        output_sha256={n:sha(OUT/n) for n in files}),indent=2)+'\n')
    return features,meta,audit


def evaluate_metrics(forecasts,*,common=True,post_selection_only=False):
    """Common observed targets per period/horizon; coverage is reported separately."""
    records=[]
    for period in ('covid','post_covid','combined'):
        part=forecasts if period=='combined' else forecasts.loc[forecasts.test_period.eq(period)]
        if post_selection_only:
            part=part.loc[part.origin_after_validation_cutoff]
        for h,sub in part.groupby('horizon'):
            keys=['forecast_origin','target_month']
            valid=sub.loc[sub.status.eq('ok')&np.isfinite(sub.yhat)&np.isfinite(sub.actual)]
            pivot=valid.pivot(index=keys,columns='model',values='yhat').reindex(columns=MODELS)
            shared=pivot.dropna().index
            for model in MODELS:
                group=sub.loc[sub.model.eq(model)]
                if group.empty:
                    continue
                own=group.loc[group.status.eq('ok')&np.isfinite(group.yhat)&np.isfinite(group.actual)]
                scored=own.set_index(keys).reindex(shared).reset_index() if common else own.copy()
                error=scored.yhat-scored.actual
                references=sub.loc[sub.model.isin(['M0','M1'])].pivot(index=keys,columns='model',values='yhat')
                comparable=scored.merge(references,on=keys,how='left',validate='one_to_one')
                rmse=float(np.sqrt(np.mean(error**2))) if len(error) else np.nan
                baseline={m:float(np.sqrt(np.mean((comparable[m]-comparable.actual)**2))) if len(error) else np.nan for m in ['M0','M1']}
                n_forecast=int((group.status.eq('ok')&np.isfinite(group.yhat)).sum())
                n_actual=int(np.isfinite(group.actual).sum())
                records.append(dict(period=period,horizon=int(h),model=model,spec_id=group.spec_id.iloc[0],
                    evaluation_view='post_selection_origins' if post_selection_only else 'requested_full_period',
                    sample='common_four_models' if common else 'own_available',n_planned=len(group),
                    n_actual=n_actual,n_forecasts=n_forecast,n_own_scored=len(own),n_scored=len(scored),
                    forecast_coverage=n_forecast/len(group),scored_coverage_of_available_actuals=len(scored)/n_actual if n_actual else np.nan,
                    mae=float(abs(error).mean()) if len(error) else np.nan,rmse=rmse,
                    bias=float(error.mean()) if len(error) else np.nan,
                    naive_rmse_same_rows=baseline['M0'],arima_rmse_same_rows=baseline['M1'],
                    rmse_improvement_vs_naive_pct=100*(1-rmse/baseline['M0']) if baseline['M0']>0 else np.nan,
                    rmse_improvement_vs_arima_pct=100*(1-rmse/baseline['M1']) if baseline['M1']>0 else np.nan))
    return pd.DataFrame(records)


def main():
    specs,selection_cutoff=freeze_specs()
    panel,rules,calendar,raw=load_core_inputs(ROOT)
    view=available_panel(panel[['date','ny_ur']],AS_OF,rules,releases=calendar)
    last_target=view.data.ny_ur.dropna().index.max()
    plan=test_plan(last_target,selection_cutoff)
    plan.to_csv(OUT/'test_plan.csv',index=False)
    features,meta,feature_audit=origin_snapshots(panel,rules,calendar,raw,plan.forecast_origin.max())
    labels=make_label_bank(features,panel,view)
    labels.to_csv(OUT/'test_historical_labels.csv')
    rows,samples,parameters,warnings,arima_audit,missing_refs=[],[],[],[],[],[]
    for i,(origin,tasks) in enumerate(plan.groupby('forecast_origin',sort=True)):
        current=features.loc[origin]
        series,max_release=released_nyur(panel,origin,rules,calendar)
        latest=series.dropna().iloc[-1]
        assert np.isclose(latest,current.latest_nyur_level)
        targets={int(t.horizon):t.target_month for t in tasks.itertuples()}
        order=specs[(int(tasks.horizon.iloc[0]),'M1')].order
        if any(specs[(int(h),'M1')].order!=order for h in tasks.horizon):
            raise ValueError('Different frozen ARIMA orders require independent fits')
        predictions,arima_info,caught=arima_forecasts(series,origin,order,targets)
        warnings += [dict(forecast_origin=origin,**w) for w in caught]
        arima_audit.append(dict(forecast_origin=origin,first_reference=series.index[0],last_reference=series.index[-1],
            n_calendar=len(series),n_observed=int(series.notna().sum()),internal_missing=int(series.isna().sum()),
            monthly_index_preserved=series.index.equals(pd.date_range(series.index[0],series.index[-1],freq='MS')),
            october_2025_present=pd.Timestamp('2025-10-01') in series.index,
            october_2025_value=series.get(pd.Timestamp('2025-10-01'),np.nan),status=arima_info['status']))
        missing_refs += [dict(forecast_origin=origin,missing_reference=d) for d in series.index[series.isna()]]
        for task in tasks.itertuples():
            h=int(task.horizon);target=task.target_month
            base=dict(forecast_origin=origin,target_month=target,horizon=h,test_period=task.test_period,
                origin_after_validation_cutoff=task.origin_after_validation_cutoff,
                fit_as_of=forecast_origin(origin),latest_nyur_reference=series.index[-1],latest_nyur_level=latest,
                nyur_age_months=int(current.nyur_age_months),latest_usur_reference=meta.at[origin,'latest_usur_reference'],
                stl_last_reference=meta.at[origin,'stl_last_reference'],stl_nobs=meta.at[origin,'stl_nobs'],
                max_predictor_information_at=forecast_origin(origin),actual=view.data.ny_ur.get(target,np.nan),
                actual_available_at=view.audit.xs('ny_ur',level='variable').available_at.get(target,pd.NaT),
                frozen_specification=True,pseudo_real_time=True)
            for model in MODELS:
                spec=specs[(h,model)]
                if model in ('M0','M1'):
                    info=(dict(status='ok',n_train=int(series.notna().sum()),internal_missing=int(series.isna().sum())) if model=='M0' else arima_info.copy())
                    info['max_train_label_available_at']=max_release
                    info['training_first_target']=series.index[0];info['training_last_target']=series.index[-1]
                    info['forecast_steps']=target.to_period('M').ordinal-series.index[-1].to_period('M').ordinal
                    yhat=float(latest) if model=='M0' else predictions.get(h,np.nan)
                    missing=[]
                else:
                    X,y,info,_=expanding_training_sample(features,labels,origin,h,spec.features)
                    yhat,fit_info,coef=direct_forecast(X,y,current)
                    info.update(fit_info)
                    missing=[c for c in spec.features if pd.isna(current[c])]
                    samples.append(dict(forecast_origin=origin,horizon=h,model=model,spec_id=spec.spec_id,**info))
                    parameters += [dict(forecast_origin=origin,horizon=h,model=model,spec_id=spec.spec_id,**p) for p in coef]
                rows.append(dict(base,model=model,spec_id=spec.spec_id,yhat=yhat,
                                 missing_current_features=json.dumps(missing),**info))
        if (i+1)%12==0:
            print(f'Frozen model refits: {i+1}/{plan.forecast_origin.nunique()} origins',flush=True)
    forecasts=pd.DataFrame(rows)
    forecasts['actual_observed']=np.isfinite(forecasts.actual)
    forecasts['own_scored']=forecasts.status.eq('ok')&np.isfinite(forecasts.yhat)&forecasts.actual_observed
    keys=['forecast_origin','target_month','horizon']
    shared=forecasts.groupby(keys).own_scored.transform('all')
    forecasts['common_scored']=shared
    forecasts['score_exclusion_reason']=np.where(~forecasts.actual_observed,'missing_actual',
        np.where(forecasts.status.ne('ok'),forecasts.status,np.where(~shared,'another_model_unavailable','scored')))
    forecasts.to_csv(OUT/'test_forecasts.csv',index=False)
    metrics=pd.concat([evaluate_metrics(forecasts),evaluate_metrics(forecasts,post_selection_only=True)],ignore_index=True)
    metrics.to_csv(OUT/'test_metrics.csv',index=False)
    evaluate_metrics(forecasts,common=False).to_csv(OUT/'test_own_sample_metrics.csv',index=False)
    pd.DataFrame(samples).to_csv(OUT/'test_training_sample_coverage.csv',index=False)
    pd.DataFrame(parameters).to_csv(OUT/'test_direct_coefficients.csv',index=False)
    pd.DataFrame(warnings,columns=['forecast_origin','attempt','optimizer','warning']).to_csv(OUT/'test_fit_warnings.csv',index=False)
    pd.DataFrame(arima_audit).to_csv(OUT/'test_arima_calendar_audit.csv',index=False)
    pd.DataFrame(missing_refs,columns=['forecast_origin','missing_reference']).to_csv(OUT/'test_arima_missing_reference_months.csv',index=False)
    forecasts.loc[forecasts.score_exclusion_reason.ne('scored')].to_csv(OUT/'test_exclusions.csv',index=False)
    shutdown=features.loc['2025-09-30':'2026-01-31'].join(meta,rsuffix='_metadata')
    shutdown.to_csv(OUT/'test_shutdown_feature_audit.csv')
    forecasts.loc[forecasts.forecast_origin.between('2025-09-30','2026-01-31')].to_csv(OUT/'test_shutdown_forecasts.csv',index=False)
    forecasts.loc[forecasts.target_month.eq('2025-10-01')].to_csv(OUT/'test_october_2025_forecasts.csv',index=False)
    run=dict(as_of=AS_OF,latest_available_target=str(last_target.date()),target_start='2020-01-01',
        planned_target_months=plan.target_month.nunique(),origins=plan.forecast_origin.nunique(),
        forecast_records=len(forecasts),new_model_selection=False,validation_information_cutoff=selection_cutoff,
        boundary_policy='Full requested test retained; supplementary origins on/after validation-information cutoff; no retuning',
        training_policy='Expanding original-origin snapshots; earlier test labels enter only after publication',
        missing_policy='No interpolation, clipping or fallback forecast; incomplete direct predictors excluded explicitly',
        original_code_modified=False,release_metadata_modified=False)
    (OUT/'test_run_metadata.json').write_text(json.dumps(run,indent=2)+'\n')
    from .out_of_sample_report import write_report
    write_report(OUT,forecasts,metrics)
    print(metrics.loc[metrics.evaluation_view.eq('requested_full_period'),['period','horizon','model','n_scored','rmse','bias','forecast_coverage']].to_string(index=False),flush=True)


if __name__=='__main__':
    main()

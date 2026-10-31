"""Small forecasting primitives for expanding-window pseudo-real-time validation.

Historical X is an immutable origin snapshot. Labels are separate and publication
filtered at each fit. Calendar gaps are never compressed, interpolated or filled.
"""
from dataclasses import dataclass
from itertools import product
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from .data_availability import available_panel, forecast_origin, target_months
from .feature_engineering import build_feature_window, align_training_examples

HORIZONS = (1, 3, 6)
BASE_FEATURES = ('latest_nyur_level', 'latest_nyur_diff_pp', 'nyur_age_months')


@dataclass(frozen=True)
class ModelSpec:
    spec_id: str
    model: str
    features: tuple = ()
    order: tuple = ()
    core_spec: str = ''


def predefined_specs():
    """Grid declared before scores: 18 ARIMA, 2 core, 4 macro, one Naive."""
    specs = [ModelSpec('M0_naive', 'M0')]
    specs += [ModelSpec(f'M1_{p}{d}{q}', 'M1', order=(p,d,q))
              for p,d,q in product(range(3), range(2), range(3))]
    for lag in (2,3):
        core_id = f'M2_payroll_lag{lag}'
        features = BASE_FEATURES + ('ny_claims_sa_log_diff_lag1', f'ny_payrolls_log_growth_pct_lag{lag}')
        specs.append(ModelSpec(core_id, 'M2', features=features, core_spec=core_id))
        for transform in ('level','diff_pp'):
            specs.append(ModelSpec(f'M3_payroll_lag{lag}_us_{transform}', 'M3',
                                   features=features+(f'latest_usur_{transform}',), core_spec=core_id))
    return specs


def origin_month(value):
    cutoff = forecast_origin(value)
    if not cutoff.is_month_end:
        raise ValueError('Forecast origin must be the last calendar day')
    return cutoff.tz_localize(None).normalize()


def released_nyur(panel, origin, rules, releases):
    columns = ['date','ny_ur'] if 'date' in panel else ['ny_ur']
    view = available_panel(panel[columns], origin, rules, releases=releases)
    series = view.data.ny_ur.loc['1990-01-01':]
    observed = series.dropna()
    if observed.empty:
        raise ValueError('No NYUR published by forecast origin')
    # Trim unavailable endpoints only; retain every internal reference month.
    series = series.loc[observed.index.min():observed.index.max()].asfreq('MS')
    stamps = view.audit.xs('ny_ur',level='variable').loc[observed.index,'available_at']
    return series, stamps.max()


def origin_row(window):
    """Derive latest published information from this window, not a later fit."""
    values = window.features.copy()
    ny = window.target_levels.dropna()
    us = window.history.us_ur_level.dropna()
    ny_ref = ny.index.max() if len(ny) else pd.NaT
    us_ref = us.index.max() if len(us) else pd.NaT
    values['latest_nyur_level'] = ny.iloc[-1] if len(ny) else np.nan
    values['latest_nyur_diff_pp'] = window.history.at[ny_ref,'ny_ur_diff_pp'] if len(ny) else np.nan
    values['nyur_age_months'] = window.origin.tz_localize(None).to_period('M').ordinal-ny_ref.to_period('M').ordinal if len(ny) else np.nan
    values['latest_usur_level'] = us.iloc[-1] if len(us) else np.nan
    values['latest_usur_diff_pp'] = window.history.at[us_ref,'us_ur_diff_pp'] if len(us) else np.nan
    return values, dict(latest_nyur_reference=ny_ref, latest_usur_reference=us_ref,
                        stl_first_reference=window.stl_first_reference,
                        stl_last_reference=window.stl_last_reference, stl_nobs=window.stl_nobs,
                        stl_status=window.stl_status)


def build_origin_bank(panel, origins, rules, releases, raw, *, progress=None):
    """One expanding STL per original feature origin; never refit old predictors."""
    windows, rows, metadata = [], [], []
    dates = pd.DatetimeIndex([origin_month(o) for o in origins],name='forecast_origin')
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError('Historical origins must be unique and increasing')
    for i,origin in enumerate(dates):
        window = build_feature_window(panel,origin,rules,releases=releases,raw=raw,lags=(1,2,3))
        row, details = origin_row(window)
        windows.append(window); rows.append(row); metadata.append(details)
        if progress and ((i+1)%24==0 or i+1==len(dates)):
            progress(f'Historical origin snapshots: {i+1}/{len(dates)}')
    features = pd.DataFrame(rows,index=dates)
    meta = pd.DataFrame(metadata,index=dates)
    return features, meta, windows


def label_bank(panel, windows, evaluation_cutoff, rules, releases):
    """Attach separate labels with publication timestamps; not a training sample."""
    examples = align_training_examples(panel,windows,evaluation_cutoff,rules,releases=releases)
    labels = examples.labels.rename_axis(index={'origin':'forecast_origin'}).copy()
    labels['label_available_at'] = pd.to_datetime(labels.label_available_at,utc=True,format='mixed')
    return labels


def training_sample(features, labels, origin, horizon, columns):
    """Explicitly partition every prior row; target release precedes fitting."""
    origin = origin_month(origin)
    cutoff = forecast_origin(origin).tz_convert('UTC')
    if features.index.has_duplicates or labels.index.has_duplicates:
        raise ValueError('Duplicate historical feature or label keys')
    target = labels.xs(horizon,level='horizon').reindex(features.index)
    prior = features.index < origin
    in_period = target.target_reference.between(pd.Timestamp('1990-01-01'),pd.Timestamp('2019-12-01'))
    response_present = target.target_ny_ur_level.notna()
    published = target.label_available_at.notna() & target.label_available_at.le(cutoff)
    complete = features.loc[:,list(columns)].notna().all(axis=1) & features.latest_nyur_level.notna()
    reason = pd.Series('eligible',index=features.index)
    reason.loc[~complete] = 'missing_predictor'
    reason.loc[~published] = 'label_not_published'
    reason.loc[~response_present] = 'missing_actual'
    reason.loc[~in_period] = 'outside_training_target_period'
    reason.loc[~prior] = 'not_prior_origin'
    eligible = reason.eq('eligible')
    X = features.loc[eligible,list(columns)].copy()
    y = target.loc[eligible,'target_ny_ur_level']-features.loc[eligible,'latest_nyur_level']
    released_rows = prior & in_period & response_present & published
    metadata = dict(prior_rows=int(prior.sum()), labels_released=int(released_rows.sum()),
        n_train=len(X), excluded_missing_predictors=int((released_rows & ~complete).sum()),
        excluded_unpublished_labels=int((prior & in_period & response_present & ~published).sum()),
        excluded_missing_actuals=int((prior & in_period & ~response_present).sum()),
        excluded_outside_target_period=int((prior & ~in_period).sum()),
        training_first_origin=X.index.min(), training_last_origin=X.index.max(),
        training_first_target=target.loc[eligible,'target_reference'].min(),
        training_last_target=target.loc[eligible,'target_reference'].max(),
        max_train_label_available_at=target.loc[eligible,'label_available_at'].max(),
        missing_feature_counts=json.dumps({c:int(features.loc[released_rows,c].isna().sum()) for c in columns},sort_keys=True),
        membership_sha256=hashlib.sha256('|'.join(str(d.date()) for d in X.index).encode()).hexdigest())
    if len(X):
        assert metadata['max_train_label_available_at'] <= cutoff
        assert X.index.max()<origin
    return X,y,metadata,reason


def direct_forecast(X, y, current, *, min_nobs=60):
    """OLS of target change; fit-only standardization, no imputation or clipping."""
    if not X.index.equals(y.index):
        raise ValueError('Predictors and targets are not aligned')
    info = dict(status='ok',rank=np.nan,condition_number=np.nan)
    parameters = []
    if len(X)<max(min_nobs,len(X.columns)+2):
        return np.nan,{**info,'status':'insufficient_training'},parameters
    if current.reindex(X.columns).isna().any():
        return np.nan,{**info,'status':'missing_current_predictor'},parameters
    if not np.isfinite(X.to_numpy()).all() or not np.isfinite(y.to_numpy()).all():
        raise ValueError('Nonfinite training values')
    means = X.mean(); scales = X.std(ddof=0).replace(0,1)
    design = np.column_stack([np.ones(len(X)),((X-means)/scales).to_numpy()])
    beta,_,rank,_ = np.linalg.lstsq(design,y.to_numpy(),rcond=None)
    info.update(rank=int(rank),condition_number=float(np.linalg.cond(design)))
    if rank<design.shape[1]:
        return np.nan,{**info,'status':'rank_deficient'},parameters
    vector = np.r_[1.,((current.reindex(X.columns)-means)/scales).to_numpy(dtype=float)]
    prediction = float(current['latest_nyur_level']+vector@beta)
    parameters.append(dict(feature='intercept',coefficient=float(beta[0]),train_mean=np.nan,train_scale=np.nan))
    for i,column in enumerate(X):
        parameters.append(dict(feature=column,coefficient=float(beta[i+1]),train_mean=float(means[column]),train_scale=float(scales[column])))
    return prediction,info,parameters


def arima_forecasts(series, origin, order, targets, *, min_nobs=60):
    """Fit on monthly calendar with internal NaNs; advance through reporting gap."""
    origin = origin_month(origin)
    series = series.copy().asfreq('MS')
    observed = series.dropna()
    info = dict(status='ok',n_train=len(observed),n_calendar=len(series),
                internal_missing=int(series.isna().sum()),converged=False,fit_attempts=0)
    if len(observed)<min_nobs:
        return {},{**info,'status':'insufficient_training'},[]
    if series.index[-1]!=observed.index[-1] or series.index[-1]>origin:
        raise ValueError('ARIMA needs endpoint at latest released reference month')
    steps = {h:pd.Timestamp(t).to_period('M').ordinal-series.index[-1].to_period('M').ordinal for h,t in targets.items()}
    if min(steps.values())<=0:
        raise ValueError('Forecast target must follow last observed reference month')
    notes, fit = [], None
    start = None
    # A fixed optimizer retry policy; no validation errors enter estimation.
    for attempt,(optimizer,maxiter) in enumerate([('lbfgs',200),('powell',300),('lbfgs',400)],start=1):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model = ARIMA(series,order=order,trend='c' if order[1]==0 else 'n',
                              enforce_stationarity=True,enforce_invertibility=True,missing='none')
                fit = model.fit(start_params=start,method='statespace',
                                method_kwargs={'method':optimizer,'maxiter':maxiter,'disp':False})
            notes += [dict(attempt=attempt,optimizer=optimizer,warning=str(w.message).replace('\n',' ')) for w in caught]
            info['fit_attempts']=attempt
            if fit.mle_retvals.get('converged',False) and np.isfinite(fit.params).all():
                info.update(converged=True,aic=float(fit.aic),bic=float(fit.bic))
                break
            start = fit.params if np.isfinite(fit.params).all() else None
        except (ValueError,np.linalg.LinAlgError,RuntimeError) as exc:
            notes.append(dict(attempt=attempt,optimizer=optimizer,warning=f'{type(exc).__name__}: {exc}'))
            start = None
    if not info['converged']:
        return {},{**info,'status':'fit_failed_or_not_converged'},notes
    prediction = fit.forecast(steps=max(steps.values()))
    values = {h:float(prediction.iloc[n-1]) for h,n in steps.items()}
    if not np.isfinite(list(values.values())).all():
        return {},{**info,'status':'nonfinite_forecast'},notes
    info['forecast_steps']=json.dumps(steps,sort_keys=True)
    return values,info,notes


def validation_plan(start='2016-01-01',end='2019-12-01',horizons=HORIZONS):
    """Split by target reference month; horizon-6 starts at July 2015 origin."""
    if pd.Timestamp(start)<pd.Timestamp('2016-01-01') or pd.Timestamp(end)>pd.Timestamp('2019-12-01'):
        raise ValueError('This runner reserves 2020 onward and validates only 2016-2019')
    rows=[]
    for target in pd.date_range(start,end,freq='MS'):
        for h in horizons:
            if type(h) is not int or h<1:
                raise ValueError('Horizons must be positive integers')
            month=target.to_period('M')-h
            rows.append(dict(forecast_origin=month.to_timestamp()+pd.offsets.MonthEnd(0),target_month=target,horizon=h))
    return pd.DataFrame(rows).sort_values(['forecast_origin','horizon']).reset_index(drop=True)


def scored_metrics(forecasts, *, common=True):
    """Compare on identical origin/target/horizon keys; never score missing NYUR."""
    records=[]
    for horizon,sub in forecasts.groupby('horizon'):
        keys=['forecast_origin','target_month']
        valid=sub.loc[sub.yhat.notna() & sub.actual.notna() & sub.status.eq('ok')]
        pivot=valid.pivot(index=keys,columns='spec_id',values='yhat')
        all_specs=sub.spec_id.unique()
        common_keys=pivot.reindex(columns=all_specs).dropna().index
        for spec,group in sub.groupby('spec_id'):
            usable=group.loc[group.yhat.notna() & group.actual.notna() & group.status.eq('ok')]
            if common:
                usable=usable.set_index(keys).reindex(common_keys).reset_index()
            base=sub.loc[sub.model.eq('M0'),keys+['yhat']].rename(columns={'yhat':'naive_yhat'})
            usable=usable.merge(base,on=keys,how='left',validate='one_to_one')
            error=usable.yhat-usable.actual
            rmse=float(np.sqrt(np.mean(error**2))) if len(error) else np.nan
            naive=float(np.sqrt(np.mean((usable.naive_yhat-usable.actual)**2))) if len(error) else np.nan
            records.append(dict(horizon=int(horizon),model=group.model.iloc[0],spec_id=spec,n_scored=len(error),
                n_planned=len(group),coverage=len(error)/len(group),mae=float(np.mean(abs(error))) if len(error) else np.nan,
                rmse=rmse,naive_rmse_on_same_rows=naive,
                rmse_improvement_pct=100*(1-rmse/naive) if naive>0 else np.nan,
                sample='common_all_candidates' if common else 'own_available'))
    return pd.DataFrame(records)

"""Run Step 3: python -m src.experiments.step3_ml.run (from modeling/).

    tuning -> frozen validation-only selections -> retrospective test -> live

Use --phase tune/test/all. Test never searches parameters. An existing lock
cannot be overwritten with different choices without a new experiment name.
"""
import argparse
import json
import platform
import subprocess
from importlib.metadata import version

import numpy as np
import pandas as pd

from src.models.rolling_validation import validation_plan
from src.models.out_of_sample import test_plan
from .features import ROOT, OUT, AS_OF, load, sha
from .models import GRIDS, MODELS, columns_for, sample, fit_predict, importance

SELECTION_CUTOFF = pd.Timestamp('2020-01-31')
LIVE_ORIGINS = pd.to_datetime(['2026-08-31', '2026-09-30'])


def forecast_plan(features, labels, plan, choices, validation=False, explain=False):
    rows, diagnostics = [], []
    for task in plan.itertuples():
        origin, h, target = task.forecast_origin, int(task.horizon), task.target_month
        current = features.loc[origin]
        for model, params in choices[h].items():
            columns = columns_for(model)
            X, y, info = sample(features, labels, origin, h, columns, validation)
            yhat, status, fitted = fit_predict(model, params, X, y, current)
            actual = labels.loc[(origin, h), 'target_ny_ur_level']
            rows.append(dict(
                origin=origin, date=target, horizon=h, model=model,
                spec_id=f'step3_{model}_h{h}', yhat=yhat, actual=actual,
                period='validation' if validation else getattr(task, 'test_period', 'live'),
                status=status, n_train=info['n_train'],
                excluded_covid=info['excluded_covid'],
                excluded_unpublished_labels=info['excluded_unpublished_labels'],
                training_first_origin=info['training_first_origin'],
                training_last_origin=info['training_last_origin'],
                max_train_label_available_at=info['max_train_label_available_at'],
                latest_nyur_level=current.latest_nyur_level,
                nyur_age_months=current.nyur_age_months,
                selection_known_at_origin=origin >= SELECTION_CUTOFF,
                lower=np.nan, upper=np.nan,
            ))
            if explain and fitted is not None and origin == LIVE_ORIGINS[-1]:
                detail = importance(fitted, columns).assign(model=model, horizon=h, origin=origin)
                diagnostics.append(detail)
    return pd.DataFrame(rows), diagnostics


def source_hashes():
    files = [ROOT/'src/experiments/step3_ml'/n for n in ('features.py','models.py','run.py')]
    files += [ROOT/'results/experiments/step3_ml/feature_manifest.json',
              ROOT.parent/'results/forecasts.csv']
    return {str(p.relative_to(ROOT.parent)): sha(p) for p in files}


def tune(features, labels):
    plan = validation_plan()
    # Grid declaration is saved before any tuning scores are calculated.
    protocol = dict(as_of=AS_OF, grid=GRIDS, horizons=[1,3,6], min_train=60,
                    feature_groups={m:columns_for(m) for m in MODELS},
                    selection_period='2016-01 through 2019-12 targets',
                    selection_information_cutoff=str(SELECTION_CUTOFF.date()),
                    selection_metric='common-candidate validation RMSE',
                    training='published-label expanding window, Step 2 COVID exclusion',
                    evaluation='retrospective; 2020-2026 previously inspected by team')
    protocol_path = OUT/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Protocol changed: use a new experiment directory')
    protocol_path.write_text(json.dumps(protocol, indent=2)+'\n')
    scores, selected, selected_rows = [], {}, []
    for h in (1,3,6):
        selected[str(h)] = {}
        for model in MODELS:
            predictions = []
            for candidate, params in enumerate(GRIDS[model]):
                part, _ = forecast_plan(features, labels, plan[plan.horizon.eq(h)],
                                         {h:{model:params}}, validation=True)
                predictions.append(part.assign(candidate=candidate))
            candidate_frame = pd.concat(predictions, ignore_index=True)
            pivot = candidate_frame.pivot(index=['origin','date'], columns='candidate', values='yhat')
            valid = pivot.reindex(columns=range(len(GRIDS[model]))).notna().all(axis=1)
            actual = candidate_frame.groupby(['origin','date']).actual.first()
            shared = pivot.index[valid & actual.reindex(pivot.index).notna()]
            if len(shared) < 24:
                raise ValueError(f'Too few comparable validation forecasts for {model} h={h}')
            model_scores = []
            for candidate, params in enumerate(GRIDS[model]):
                group = candidate_frame[candidate_frame.candidate.eq(candidate)].set_index(['origin','date']).loc[shared]
                e = group.yhat - group.actual
                record = dict(model=model, horizon=h, candidate=candidate, params=json.dumps(params,sort_keys=True),
                              n=len(e), rmse=float(np.sqrt(np.mean(e**2))), mae=float(abs(e).mean()))
                scores.append(record)
                model_scores.append(record)
            best = min(model_scores, key=lambda r:(r['rmse'],r['candidate']))
            selected[str(h)][model] = GRIDS[model][best['candidate']]
            selected_rows.append(candidate_frame[candidate_frame.candidate.eq(best['candidate'])].drop(columns='candidate'))
            print(f'Tuned h={h} {model}: RMSE={best["rmse"]:.4f}, {best["params"]}',flush=True)
    locked = dict(protocol=protocol, selected=selected, sources=source_hashes())
    lock = OUT/'selected_models.json'
    if lock.exists() and json.loads(lock.read_text()) != locked:
        raise ValueError('Existing model lock differs; preserve it and name a new experiment')
    lock.write_text(json.dumps(locked,indent=2)+'\n')
    pd.DataFrame(scores).to_csv(OUT/'tuning_scores.csv',index=False)
    pd.concat(selected_rows).to_csv(OUT/'validation_forecasts.csv',index=False)


def evaluate(features, labels):
    lock = json.loads((OUT/'selected_models.json').read_text())
    if lock['sources'] != source_hashes():
        raise ValueError('Model sources or inputs changed since selection')
    choices = {int(h):v for h,v in lock['selected'].items()}
    last_target = labels.loc[labels.target_ny_ur_level.notna(),'target_reference'].max()
    plan = test_plan(last_target, SELECTION_CUTOFF)
    # Keep the never-published October 2025 target as an explicit unscored row.
    tests = []
    for origin, group in plan.groupby('forecast_origin'):
        frame, _ = forecast_plan(features, labels, group, choices)
        tests.append(frame)
        if origin.month == 12:
            print(f'Test through origin {origin.date()}',flush=True)
    live_plan = pd.DataFrame([dict(forecast_origin=o, horizon=h,
                         target_month=(o.to_period('M')+h).to_timestamp())
                         for o in LIVE_ORIGINS for h in (1,3,6)])
    live, diagnostics = forecast_plan(features, labels, live_plan, choices, explain=True)
    validation = pd.read_csv(OUT/'validation_forecasts.csv',parse_dates=['origin','date'])
    for (m,h), group in validation.groupby(['model','horizon']):
        errors = (group.actual-group.yhat).dropna()
        lo, hi = errors.quantile([0.025,0.975])
        mask = live.model.eq(m) & live.horizon.eq(h) & live.status.eq('ok')
        live.loc[mask,'lower'] = live.loc[mask,'yhat']+lo
        live.loc[mask,'upper'] = live.loc[mask,'yhat']+hi
    frame = pd.concat([validation,*tests,live],ignore_index=True)
    frame.sort_values(['period','model','horizon','date']).to_csv(OUT/'forecasts.csv',index=False)
    if diagnostics:
        pd.concat(diagnostics).to_csv(OUT/'feature_importance.csv',index=False)
    # Select a single headline ML model per horizon on identical validation rows.
    wide = validation.pivot(index=['origin','date','horizon'],columns='model',values='yhat')
    actual = validation.groupby(['origin','date','horizon']).actual.first()
    winners = {}
    for h in (1,3,6):
        subset = wide.xs(h,level='horizon').dropna()
        truth = actual.xs(h,level='horizon').reindex(subset.index)
        eligible = [m for m in MODELS if m != 'ridge_ar_only']
        scores = {m:float(np.sqrt(np.mean((subset[m]-truth)**2))) for m in eligible}
        winners[str(h)] = min(scores,key=scores.get)
    (OUT/'recommended_models.json').write_text(json.dumps(winners,indent=2)+'\n')
    metadata = dict(python=platform.python_version(),
                    packages={p:version(p) for p in ['numpy','pandas','scipy','scikit-learn','statsmodels','matplotlib']},
                    git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
                    sources=source_hashes(), selected_lock_sha256=sha(OUT/'selected_models.json'),
                    forecasts=len(frame), recommended=winners,
                    limitations=['Revised snapshot; not historical vintages',
                                 'Release timing includes estimated rules',
                                 'Test history already inspected; retrospective benchmarking',
                                 'Live intervals use validation residuals only; not shock-calibrated',
                                 'Model selection unknown at early COVID origins; flagged in forecasts'])
    (OUT/'run_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Saved forecasts, selection lock, diagnostics and provenance.',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['all','tune','test'],default='all')
    args = parser.parse_args()
    features, labels = load()
    if args.phase in ('all','tune'):
        tune(features, labels)
    if args.phase in ('all','test'):
        evaluate(features, labels)
        from .report import main as report
        report()


if __name__ == '__main__':
    main()

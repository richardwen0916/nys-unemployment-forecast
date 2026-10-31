"""Small declared grids: interpretable linear shrinkage and shallow tree ensembles."""
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.models.rolling_validation import training_sample
from src.models.out_of_sample import expanding_training_sample
from src.experiments.improved_models import covid_free_rows
from .features import AR_FEATURES, CORE_FEATURES, EXTRA_FEATURES

SEED = 437
MIN_TRAIN = 60
GRIDS = {
    'ridge': [{'alpha': a} for a in (1.0, 10.0, 100.0)],
    'lasso': [{'alpha': a} for a in (0.001, 0.01, 0.1)],
    'random_forest': [{'max_depth': 3, 'min_samples_leaf': 8},
                      {'max_depth': None, 'min_samples_leaf': 16}],
    'gradient_boosting': [{'max_depth': 1, 'min_samples_leaf': 10},
                          {'max_depth': 2, 'min_samples_leaf': 10}],
    'ridge_ar_only': [{'alpha': a} for a in (1.0, 10.0, 100.0)],
    'ridge_regional': [{'alpha': a} for a in (1.0, 10.0, 100.0)],
}
MODELS = tuple(GRIDS)


def columns_for(model):
    if model == 'ridge_ar_only':
        return AR_FEATURES
    if model == 'ridge_regional':
        return CORE_FEATURES + EXTRA_FEATURES
    return CORE_FEATURES


def estimator(model, params):
    if model.startswith('ridge'):
        return make_pipeline(StandardScaler(), Ridge(**params))
    if model == 'lasso':
        return make_pipeline(StandardScaler(), Lasso(**params, max_iter=20000, tol=1e-6))
    if model == 'random_forest':
        return RandomForestRegressor(n_estimators=80, max_features=0.8, random_state=SEED,
                                     n_jobs=1, **params)
    if model == 'gradient_boosting':
        return GradientBoostingRegressor(n_estimators=80, learning_rate=0.04,
                                         loss='huber', random_state=SEED, **params)
    raise ValueError(model)


def sample(features, labels, origin, horizon, columns, validation=False):
    sampler = training_sample if validation else expanding_training_sample
    X, y, info, _ = sampler(features, labels, origin, horizon, columns)
    before = len(X)
    targets = labels.xs(horizon, level='horizon').reindex(X.index).target_reference
    keep = covid_free_rows(X, targets)
    X, y = X.loc[keep], y.loc[keep]
    info['excluded_covid'] = before - len(X)
    info['n_train'] = len(X)
    info['training_first_origin'] = X.index.min()
    info['training_last_origin'] = X.index.max()
    return X, y, info


def fit_predict(model, params, X, y, current):
    """Predict target change, then add latest published NYUR; no target filling.

    Current features are clipped to the eligible training range, as in Step 2.
    Scaling is fitted only here, never on the full feature bank.
    """
    if not X.index.equals(y.index):
        raise ValueError('Misaligned training predictors and responses')
    if len(X) < MIN_TRAIN:
        return np.nan, 'insufficient_training', None
    x_now = current.reindex(X.columns).astype(float)
    if not np.isfinite(x_now.to_numpy()).all() or not np.isfinite(current.latest_nyur_level):
        return np.nan, 'missing_current_predictor', None
    if not np.isfinite(X.to_numpy()).all() or not np.isfinite(y.to_numpy()).all():
        raise ValueError('Nonfinite eligible training sample')
    fitted = estimator(model, params)
    fitted.fit(X, y)
    current_frame = x_now.clip(X.min(), X.max()).to_frame().T
    yhat = float(current.latest_nyur_level + fitted.predict(current_frame)[0])
    return yhat, 'ok' if np.isfinite(yhat) else 'nonfinite_forecast', fitted


def importance(fitted, columns):
    """Training-fit diagnostics, not causal effects or held-out importance."""
    if hasattr(fitted, 'steps'):
        model = fitted.steps[-1][1]
        return pd.DataFrame({'feature': columns, 'value': model.coef_,
                             'kind': 'coefficient_per_training_sd'})
    return pd.DataFrame({'feature': columns, 'value': fitted.feature_importances_,
                         'kind': 'training_impurity_importance'})

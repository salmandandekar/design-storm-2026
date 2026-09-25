"""Model registry: small, honest models suited to ~900 usable rows.

Regressors are trained on the delta target (y(T+h) - y(T)) in the lab-known
scenario and on the level in the lab-blind scenario; predictions are always
converted back to level space before scoring. Hyperparameters come from a
small grid searched with TimeSeriesSplit inside the training window only.
No deep learning: the sample size does not support it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
CV_SPLITS = 3


def _grid(estimator, param_grid):
    return GridSearchCV(
        estimator, param_grid,
        cv=TimeSeriesSplit(n_splits=CV_SPLITS),
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )


def make_regressor(name: str):
    if name == "ridge":
        pipe = Pipeline([("scale", StandardScaler()), ("m", Ridge(random_state=SEED))])
        return _grid(pipe, {"m__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]})
    if name == "rf":
        est = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
        return _grid(est, {"max_depth": [4, None], "min_samples_leaf": [5, 20]})
    if name == "hgb":
        est = HistGradientBoostingRegressor(random_state=SEED)
        return _grid(est, {"learning_rate": [0.05, 0.1], "max_depth": [3, None],
                           "min_samples_leaf": [20]})
    raise KeyError(name)


REGRESSORS = ["ridge", "rf", "hgb"]


def make_classifier(name: str):
    if name == "logit":
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("m", LogisticRegression(class_weight="balanced", max_iter=5000,
                                     random_state=SEED)),
        ])
        return _grid(pipe, {"m__C": [0.01, 0.1, 1.0, 10.0]})
    if name == "rfc":
        est = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     random_state=SEED, n_jobs=-1)
        return _grid(est, {"max_depth": [4, None], "min_samples_leaf": [5, 20]})
    raise KeyError(name)


CLASSIFIERS = ["logit", "rfc"]


def fit_predict_level(model_name: str, X_train: pd.DataFrame, X_test: pd.DataFrame,
                      lab_train: pd.DataFrame, lab_test: pd.DataFrame,
                      scenario: str) -> pd.Series:
    """Train on delta (lab-known) or level (lab-blind); return level predictions."""
    gs = make_regressor(model_name)
    if scenario == "lab_known":
        gs.fit(X_train, lab_train["y_delta"])
        pred = lab_test["y_now"].to_numpy() + gs.predict(X_test)
    elif scenario == "lab_blind":
        gs.fit(X_train, lab_train["y_future"])
        pred = gs.predict(X_test)
    else:
        raise KeyError(scenario)
    return pd.Series(np.asarray(pred, dtype=float), index=X_test.index, name="pred")

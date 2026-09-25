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
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

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


def _clean_number(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def fit_operational_ols(
    X: pd.DataFrame,
    y: pd.Series,
    test_fraction: float = 0.2,
    cv_folds: int = 5,
    persistence: pd.Series | None = None,
) -> tuple[object, dict]:
    """Fit an OLS level model and report chronological validation metrics."""
    if not 0 < test_fraction < 0.5:
        raise ValueError("model.test_fraction must be in (0, 0.5)")
    if len(X) < max(30, X.shape[1] * 10):
        raise ValueError(
            f"insufficient training rows: {len(X)} for {X.shape[1]} predictors")
    cut = max(1, int(len(X) * (1 - test_fraction)))
    X_train, X_test = X.iloc[:cut], X.iloc[cut:]
    y_train, y_test = y.iloc[:cut], y.iloc[cut:]
    baseline_value = float(y_train.mean())
    baseline_pred = np.full(len(y_test), baseline_value)
    fitted = sm.OLS(y_train, sm.add_constant(X_train, has_constant="add")).fit()
    pred = fitted.predict(sm.add_constant(X_test, has_constant="add"))

    splits = min(cv_folds, max(2, len(X_train) // 20))
    cv_errors = []
    for train_idx, test_idx in TimeSeriesSplit(n_splits=splits).split(X_train):
        fold = sm.OLS(
            y_train.iloc[train_idx],
            sm.add_constant(X_train.iloc[train_idx], has_constant="add"),
        ).fit()
        fold_pred = fold.predict(
            sm.add_constant(X_train.iloc[test_idx], has_constant="add"))
        cv_errors.extend(np.abs(fold_pred.to_numpy() - y_train.iloc[test_idx].to_numpy()))

    mae = mean_absolute_error(y_test, pred)
    holdout_r2 = r2_score(y_test, pred)
    predictor_count = X.shape[1]
    adjusted_r2 = (
        1 - (1 - holdout_r2) * (len(y_test) - 1)
        / (len(y_test) - predictor_count - 1)
        if len(y_test) > predictor_count + 1 else np.nan
    )
    metrics = {
        "n": int(len(X)),
        "train_n": int(len(X_train)),
        "test_n": int(len(X_test)),
        "train_start": X.index.min().date().isoformat(),
        "train_end": X.index.max().date().isoformat(),
        "holdout_mae": _clean_number(mae),
        "holdout_rmse": _clean_number(mean_squared_error(y_test, pred) ** 0.5),
        "holdout_r2": _clean_number(holdout_r2),
        "holdout_adjusted_r2": _clean_number(adjusted_r2),
        "train_adjusted_r2": _clean_number(fitted.rsquared_adj),
        "cv_mae": _clean_number(np.mean(cv_errors)),
        "baseline": "training_mean",
        "baseline_mae": _clean_number(mean_absolute_error(y_test, baseline_pred)),
        "beats_baseline": bool(mae < mean_absolute_error(y_test, baseline_pred)),
        "durbin_watson": _clean_number(
            sm.stats.stattools.durbin_watson(fitted.resid)),
        "coefficients": {
            key: _clean_number(value) for key, value in fitted.params.items()
        },
        "pvalues": {
            key: _clean_number(value) for key, value in fitted.pvalues.items()
        },
    }
    vif_frame = sm.add_constant(X, has_constant="add")
    metrics["vif"] = {
        column: _clean_number(variance_inflation_factor(vif_frame.to_numpy(), i))
        for i, column in enumerate(vif_frame.columns)
        if column != "const"
    }
    if persistence is not None:
        available = persistence.reindex(y_test.index).dropna()
        if len(available):
            common = available.index.intersection(y_test.index)
            persistence_mae = mean_absolute_error(
                y_test.loc[common], available.loc[common])
            metrics["persistence_mae"] = _clean_number(persistence_mae)
            metrics["beats_persistence"] = bool(
                mean_absolute_error(y_test.loc[common], pred.loc[common])
                < persistence_mae)
    final = sm.OLS(y, sm.add_constant(X, has_constant="add")).fit()
    return final, metrics


def fit_operational_ridge(
    X: pd.DataFrame,
    y: pd.Series,
    test_fraction: float = 0.2,
    cv_folds: int = 5,
    persistence: pd.Series | None = None,
) -> tuple[Pipeline, dict]:
    """Fit the regularized diagnostic model used for per-gauge features."""
    if len(X) < max(30, X.shape[1] * 10):
        raise ValueError(
            f"insufficient training rows: {len(X)} for {X.shape[1]} predictors")
    cut = max(1, int(len(X) * (1 - test_fraction)))
    X_train, X_test = X.iloc[:cut], X.iloc[cut:]
    y_train, y_test = y.iloc[:cut], y.iloc[cut:]

    def estimator() -> Pipeline:
        return Pipeline([
            ("scale", StandardScaler()),
            ("m", RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])),
        ])

    fitted = estimator().fit(X_train, y_train)
    pred = pd.Series(fitted.predict(X_test), index=X_test.index)
    splits = min(cv_folds, max(2, len(X_train) // 20))
    cv_errors = []
    for train_idx, test_idx in TimeSeriesSplit(n_splits=splits).split(X_train):
        fold = estimator().fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        fold_pred = fold.predict(X_train.iloc[test_idx])
        cv_errors.extend(np.abs(fold_pred - y_train.iloc[test_idx].to_numpy()))
    baseline_pred = np.full(len(y_test), float(y_train.mean()))
    mae = mean_absolute_error(y_test, pred)
    holdout_r2 = r2_score(y_test, pred)
    adjusted_r2 = (
        1 - (1 - holdout_r2) * (len(y_test) - 1)
        / (len(y_test) - X.shape[1] - 1)
        if len(y_test) > X.shape[1] + 1 else np.nan
    )
    vif_frame = sm.add_constant(X, has_constant="add")
    metrics = {
        "n": int(len(X)),
        "train_n": int(len(X_train)),
        "test_n": int(len(X_test)),
        "train_start": X.index.min().date().isoformat(),
        "train_end": X.index.max().date().isoformat(),
        "holdout_mae": _clean_number(mae),
        "holdout_rmse": _clean_number(mean_squared_error(y_test, pred) ** 0.5),
        "holdout_r2": _clean_number(holdout_r2),
        "holdout_adjusted_r2": _clean_number(adjusted_r2),
        "cv_mae": _clean_number(np.mean(cv_errors)),
        "baseline": "training_mean",
        "baseline_mae": _clean_number(mean_absolute_error(y_test, baseline_pred)),
        "beats_baseline": bool(mae < mean_absolute_error(y_test, baseline_pred)),
        "ridge_alpha": _clean_number(fitted.named_steps["m"].alpha_),
        "standardized_coefficients": {
            column: _clean_number(value)
            for column, value in zip(X.columns, fitted.named_steps["m"].coef_)
        },
        "vif": {
            column: _clean_number(variance_inflation_factor(vif_frame.to_numpy(), i))
            for i, column in enumerate(vif_frame.columns)
            if column != "const"
        },
    }
    if persistence is not None:
        available = persistence.reindex(y_test.index).dropna()
        if len(available):
            common = available.index.intersection(y_test.index)
            persistence_mae = mean_absolute_error(
                y_test.loc[common], available.loc[common])
            metrics["persistence_mae"] = _clean_number(persistence_mae)
            metrics["beats_persistence"] = bool(
                mean_absolute_error(y_test.loc[common], pred.loc[common])
                < persistence_mae)
    final = estimator().fit(X, y)
    residual = y.to_numpy() - final.predict(X)
    final.prediction_interval_half_width_ = float(
        np.quantile(np.abs(residual), 0.95))
    return final, metrics


def predict_operational(
    fitted: object, X: pd.DataFrame, alpha: float = 0.05
) -> pd.DataFrame:
    if not hasattr(fitted, "get_prediction"):
        prediction = np.asarray(fitted.predict(X), dtype=float)
        half_width = float(fitted.prediction_interval_half_width_)
        return pd.DataFrame({
            "prediction": prediction,
            "lower": prediction - half_width,
            "upper": prediction + half_width,
        }, index=X.index)
    frame = sm.add_constant(X, has_constant="add")
    expected = list(fitted.model.exog_names)
    missing = set(expected) - set(frame.columns)
    if missing:
        raise ValueError(f"prediction features missing: {sorted(missing)}")
    frame = frame.reindex(columns=expected)
    summary = fitted.get_prediction(frame).summary_frame(alpha=alpha)
    return pd.DataFrame({
        "prediction": summary["mean"],
        "lower": summary["obs_ci_lower"],
        "upper": summary["obs_ci_upper"],
    }, index=X.index)

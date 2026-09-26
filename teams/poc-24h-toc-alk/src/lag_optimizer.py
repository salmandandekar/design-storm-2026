"""Time-ordered empirical lag scanning."""
from __future__ import annotations

import pandas as pd

from .combine import combine
from .config import Config
from .models import fit_operational_ols
from .network_features import align_training, build_feature_frame
from .time_shift import shift_frame


def scan_lag(
    rain: pd.DataFrame,
    targets: pd.DataFrame,
    weights: pd.Series,
    cfg: Config,
) -> pd.DataFrame:
    start, end = [pd.Timedelta(value)
                  for value in cfg.travel_time.get("empirical_lag_range", ["0D", "7D"])]
    step = pd.Timedelta(cfg.travel_time.get("empirical_lag_step", "1D"))
    if step <= pd.Timedelta(0) or end < start:
        raise ValueError("invalid empirical lag range")
    rows = []
    lag = start
    while lag <= end:
        taus = {column: lag for column in rain.columns}
        effective, coverage = combine(shift_frame(rain, taus), weights, cfg)
        features = build_feature_frame(effective, coverage, cfg)
        scores = []
        for target in ("toc", "alk"):
            X, y = align_training(features, targets[[target]], "arrival_nowcast")
            _, metrics = fit_operational_ols(
                X, y[target],
                test_fraction=float(cfg.model.get("test_fraction", 0.2)),
                cv_folds=int(cfg.model.get("cv_folds", 5)),
            )
            scores.append(metrics["cv_mae"])
        rows.append({
            "lag": str(lag),
            "lag_seconds": lag.total_seconds(),
            "mean_cv_mae": sum(scores) / len(scores),
        })
        lag += step
    return pd.DataFrame(rows)

"""Leakage-safe rainfall-window features for operational models."""
from __future__ import annotations

import math

import pandas as pd

from .config import Config


def _expected_points(window: pd.Timedelta, interval: pd.Timedelta) -> int:
    return max(1, int(math.ceil(window / interval)))


def build_feature_frame(
    rainfall: pd.Series | pd.DataFrame,
    coverage: pd.Series,
    cfg: Config,
) -> pd.DataFrame:
    """Build trailing sums over (t-window, t], never using future rainfall."""
    interval = pd.Timedelta(cfg.data.get("resample_interval", "1D"))
    source = rainfall.to_frame() if isinstance(rainfall, pd.Series) else rainfall
    out = pd.DataFrame(index=source.index)
    for column in source.columns:
        for raw_window in cfg.features["windows"]:
            window = pd.Timedelta(raw_window)
            name = f"rain_{column}_{str(raw_window).lower()}"
            out[name] = source[column].rolling(
                window, min_periods=_expected_points(window, interval),
                closed="right",
            ).sum()
    if cfg.features.get("include_coverage_fraction", False):
        out["coverage_fraction"] = coverage.reindex(out.index)
    return out


def align_training(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    product: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if product == "arrival_nowcast":
        labels = targets.reindex(features.index)
    elif product == "forecast_24h":
        requested = features.index + pd.Timedelta(days=1)
        labels = targets.reindex(requested)
        labels.index = features.index
    else:
        raise ValueError(f"unknown prediction product: {product}")
    joined = features.join(labels.add_prefix("target_"), how="left").dropna()
    target_columns = [column for column in joined if column.startswith("target_")]
    return joined[features.columns], joined[target_columns].rename(
        columns=lambda value: value.removeprefix("target_"))

"""Naive reference forecasts every model must beat.

All baselines emit level-space predictions for y(T+h) so that every
comparison (models predict delta, then reconstruct level) happens in one
space. Skill is reported against these, not raw R^2: with day-to-day
persistence R^2 ~ 0.96, R^2 on levels flatters any forecaster.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D

CLIMATOLOGY_DOY_WINDOW = 7  # +/- days pooled around each day-of-year
DRIFT_MAX_GAP_DAYS = 7      # how far back the drift baseline may look


def persistence(lab: pd.DataFrame) -> pd.Series:
    """y_hat(T+h) = y(T): the operator's 'no change' expectation."""
    return lab["y_now"].rename("pred")


def climatology(lab: pd.DataFrame, target: str, train_end: pd.Timestamp) -> pd.Series:
    """Mean of training-period values within +/-7 days of day-of-year."""
    col = {"toc": "TOC_mg_L", "alk": "Alk_mg_L"}[target]
    hist = D.load_foothills()[col]
    hist = hist[hist.index <= train_end]
    hist_doy = hist.index.dayofyear.to_numpy()
    preds = []
    for t in lab.index:
        doy = (t + pd.Timedelta(days=0)).dayofyear
        dist = np.minimum(np.abs(hist_doy - doy), 365 - np.abs(hist_doy - doy))
        vals = hist[dist <= CLIMATOLOGY_DOY_WINDOW]
        preds.append(vals.mean() if len(vals) else hist.mean())
    return pd.Series(preds, index=lab.index, name="pred")


def persistence_drift(lab: pd.DataFrame, target: str, lead: int) -> pd.Series:
    """y_hat(T+h) = y(T) + h * recent slope, from the last lab pair <=7d apart."""
    col = {"toc": "TOC_mg_L", "alk": "Alk_mg_L"}[target]
    y = D.load_foothills()[col]
    idx = y.index
    preds = []
    for t, y_now in lab["y_now"].items():
        pos = idx.searchsorted(t)
        slope = 0.0
        if pos >= 1 and idx[pos] == t and pos >= 1:
            prev = idx[pos - 1]
            gap = (t - prev).days
            if 1 <= gap <= DRIFT_MAX_GAP_DAYS:
                slope = (y_now - y.loc[prev]) / gap
        preds.append(y_now + lead * slope)
    return pd.Series(preds, index=lab.index, name="pred")

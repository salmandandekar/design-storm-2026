"""Shared fold/data plumbing for the analysis scripts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import evaluate as E
from . import features as F


def anchored_frame(feat: pd.DataFrame, target: str, lead: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Anchored rows: labels exist at T and T+lead AND every set-D feature is
    present at T. All ablation sets are scored on exactly these days."""
    lab = F.make_labels(feat, target, lead)
    X_full = feat.loc[lab.index, F.FEATURE_SETS_LAB_KNOWN["ABCD"]].dropna()
    lab = lab.loc[X_full.index]
    return X_full, lab


def rolling_fold_slices(index: pd.DatetimeIndex):
    """Yield (fold_name, train_mask, test_mask) for the rolling-origin folds."""
    for fold in E.ROLLING_FOLDS:
        train_end = pd.Timestamp(fold["train_end"])
        y0, y1 = fold["test"]
        train = (index <= train_end).to_numpy() if hasattr(index <= train_end, "to_numpy") else (index <= train_end)
        test = np.isin(index.year, range(y0, y1 + 1))
        if test.sum() == 0 or train.sum() < 100:
            continue
        yield fold["name"], np.asarray(train), np.asarray(test)


def regime_fold_slices(index: pd.DatetimeIndex):
    for fold in E.REGIME_FOLDS:
        train = np.isin(index.year, fold["train_years"])
        test = np.isin(index.year, fold["test_years"])
        yield fold["name"], train, test

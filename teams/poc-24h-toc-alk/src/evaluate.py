"""Evaluation harness: folds, error metrics, uncertainty, novelty guardrail.

Statistical choices (pre-registered in results/decision_rule.json before any
model result was produced):
- Rolling-origin folds by season; no shuffling ever.
- Moving-block bootstrap (block = 14 days, B = 2000) for CIs on MAE and on
  skill vs persistence, respecting autocorrelation of daily errors.
- Diebold-Mariano test with Newey-West variance and the Harvey-Leybourne-
  Newbold small-sample correction on |error| differentials vs persistence.
- Wilson score intervals for event precision/recall.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

BOOT_BLOCK_DAYS = 14
BOOT_N = 2000
SEED = 42

# Rolling-origin folds: train on all seasons strictly before the test year.
ROLLING_FOLDS = [
    {"name": "test2023", "train_end": "2022-12-31", "test": (2023, 2023)},
    {"name": "test2024", "train_end": "2023-12-31", "test": (2024, 2024)},
    {"name": "test2025", "train_end": "2024-12-31", "test": (2025, 2025)},
    {"name": "test2026", "train_end": "2025-12-31", "test": (2026, 2026)},
]
# Regime folds: wet/high-TOC seasons (2023-24) vs dry seasons (2025-26).
REGIME_FOLDS = [
    {"name": "wet_to_dry", "train_years": (2022, 2023, 2024), "test_years": (2025, 2026)},
    {"name": "dry_to_wet", "train_years": (2022, 2025, 2026), "test_years": (2023, 2024)},
]


def year_mask(index: pd.DatetimeIndex, years) -> np.ndarray:
    return np.isin(index.year, list(years))


def mae(err: np.ndarray) -> float:
    return float(np.mean(np.abs(err)))


def skill(err_model: np.ndarray, err_ref: np.ndarray) -> float:
    """1 - MAE_model / MAE_ref. Positive = beats the reference."""
    return 1.0 - mae(err_model) / mae(err_ref)


def block_bootstrap_ci(values: np.ndarray, statistic, n_boot: int = BOOT_N,
                       block: int = BOOT_BLOCK_DAYS, seed: int = SEED,
                       paired: np.ndarray | None = None) -> tuple[float, float]:
    """Moving-block bootstrap CI. If `paired` given, statistic(v_blocks, p_blocks)."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n < block + 1:
        return (float("nan"), float("nan"))
    starts_max = n - block
    n_blocks = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, starts_max + 1, n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        if paired is None:
            out[b] = statistic(values[idx])
        else:
            out[b] = statistic(values[idx], paired[idx])
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def dm_test(err_model: np.ndarray, err_ref: np.ndarray, h: int = 1) -> tuple[float, float]:
    """Diebold-Mariano on absolute-error differentials, HLN-corrected.

    Returns (statistic, two-sided p). Negative statistic = model better.
    """
    d = np.abs(err_model) - np.abs(err_ref)
    n = len(d)
    if n < 10 or np.allclose(d, d[0]):
        return (float("nan"), float("nan"))
    dbar = d.mean()
    lag = max(h - 1, int(np.floor(1.5 * n ** (1 / 3))))
    dc = d - dbar
    var = dc @ dc / n
    for k in range(1, min(lag, n - 1) + 1):
        w = 1 - k / (lag + 1)
        var += 2 * w * (dc[:-k] @ dc[k:]) / n
    if var <= 0:
        return (float("nan"), float("nan"))
    dm = dbar / np.sqrt(var / n)
    hln = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(np.abs(hln), df=n - 1)
    return (float(hln), float(p))


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float, float]:
    """Wilson score interval: (point, lo, hi). NaNs when n == 0."""
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def novelty_count(X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.Series:
    """Per test row: how many features sit outside the train 5th-95th pct."""
    lo = X_train.quantile(0.05)
    hi = X_train.quantile(0.95)
    outside = (X_test.lt(lo) | X_test.gt(hi))
    return outside.sum(axis=1).rename("novelty")


def summarize_level_preds(y_true: pd.Series, pred_model: pd.Series,
                          pred_persist: pd.Series, lead: int = 1) -> dict:
    """Pooled metrics + uncertainty for one model against persistence."""
    e_m = (pred_model - y_true).to_numpy()
    e_p = (pred_persist - y_true).to_numpy()
    sk = skill(e_m, e_p)
    sk_lo, sk_hi = block_bootstrap_ci(
        e_m, lambda em, ep: skill(em, ep), paired=e_p)
    mae_lo, mae_hi = block_bootstrap_ci(e_m, mae)
    dm_stat, dm_p = dm_test(e_m, e_p, h=lead)
    return {
        "n": len(e_m),
        "mae": mae(e_m),
        "mae_lo": mae_lo, "mae_hi": mae_hi,
        "rmse": float(np.sqrt(np.mean(e_m**2))),
        "mae_persistence": mae(e_p),
        "skill_vs_persistence": sk,
        "skill_lo": sk_lo, "skill_hi": sk_hi,
        "dm_stat": dm_stat, "dm_p": dm_p,
    }

"""Coverage-aware shift-then-combine."""
from __future__ import annotations

import pandas as pd

from .config import Config


def combine(
    shifted_wide: pd.DataFrame, weights: pd.Series, cfg: Config
) -> tuple[pd.Series | pd.DataFrame, pd.Series]:
    missing = set(shifted_wide.columns) - set(weights.index)
    if missing:
        raise ValueError(f"weights missing for gauges: {sorted(missing)}")
    aligned = weights.reindex(shifted_wide.columns)
    present = shifted_wide.notna()
    coverage = present.mul(aligned, axis=1).sum(axis=1).rename("coverage_fraction")
    if cfg.combine.get("method") == "per_gauge":
        return shifted_wide.copy(), coverage
    policy = cfg.combine.get("missing_gauge_policy", "reweight")
    if policy == "drop":
        effective = shifted_wide.mul(aligned, axis=1).sum(axis=1, min_count=len(aligned))
    elif policy == "reweight":
        numerator = shifted_wide.mul(aligned, axis=1).sum(axis=1, min_count=1)
        effective = numerator / coverage.replace(0, pd.NA)
        effective = effective.where(
            coverage >= float(cfg.combine.get("min_coverage", 0.7)))
    else:
        raise ValueError(f"unsupported missing_gauge_policy: {policy}")
    return effective.rename("effective_rainfall_in"), coverage

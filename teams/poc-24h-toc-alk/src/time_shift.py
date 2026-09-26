"""Pure per-gauge rainfall time shifting."""
from __future__ import annotations

import pandas as pd


def shift_series(series: pd.Series, tau: pd.Timedelta) -> pd.Series:
    shifted = series.copy()
    shifted.index = shifted.index + tau
    return shifted


def shift_frame(
    rain_wide: pd.DataFrame, taus: dict[str, pd.Timedelta]
) -> pd.DataFrame:
    unknown = set(rain_wide.columns) - set(taus)
    if unknown:
        raise ValueError(f"travel times missing for gauges: {sorted(unknown)}")
    shifted = [shift_series(rain_wide[col], taus[col]).rename(col)
               for col in rain_wide.columns]
    return pd.concat(shifted, axis=1).sort_index()

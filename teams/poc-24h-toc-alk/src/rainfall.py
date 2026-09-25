"""Rainfall-network and TOC ingestion without modifying source files."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, Gauge


def _local_day(value: object, timezone: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(
            timezone, ambiguous="raise", nonexistent="raise")
    return timestamp.tz_convert("UTC").tz_localize(None).normalize()


def load_rainfall(
    cfg: Config, gauges: list[Gauge], path: str | Path | None = None
) -> pd.DataFrame:
    """Load long-format or NOAA-style rainfall into a daily wide frame."""
    source = Path(path or cfg.data["rainfall_csv"])
    frame = pd.read_csv(source)
    timestamp_col = cfg.data.get("timestamp_col", "timestamp")
    gauge_col = cfg.data.get("gauge_col", "gauge_id")
    rainfall_col = cfg.data.get("rainfall_col", "rainfall_in")
    if gauge_col not in frame and "STATION" in frame:
        gauge_col = "STATION"
    required = {timestamp_col, gauge_col, rainfall_col}
    if not required.issubset(frame.columns):
        raise ValueError(f"{source} requires columns {sorted(required)}")

    registry = {g.gauge_id: g for g in gauges}
    observed = set(frame[gauge_col].astype(str))
    unknown = observed - set(registry)
    if unknown:
        raise ValueError(f"rainfall contains gauges absent from registry: {sorted(unknown)}")
    frame = frame[[timestamp_col, gauge_col, rainfall_col]].copy()
    frame[gauge_col] = frame[gauge_col].astype(str)
    frame[rainfall_col] = pd.to_numeric(frame[rainfall_col], errors="raise")
    if (frame[rainfall_col] < 0).any():
        raise ValueError("rainfall contains negative values")
    frame["_day"] = [
        _local_day(ts, registry[gauge_id].timezone
                   or cfg.data.get("default_timezone", "UTC"))
        for ts, gauge_id in zip(frame[timestamp_col], frame[gauge_col])
    ]
    duplicate = frame.duplicated(["_day", gauge_col], keep=False)
    if duplicate.any():
        groups = frame.loc[duplicate].groupby(["_day", gauge_col])[rainfall_col]
        if groups.nunique().gt(1).any():
            raise ValueError("conflicting duplicate rainfall rows")
        frame = frame.drop_duplicates(["_day", gauge_col])

    pieces = []
    interval = cfg.data.get("resample_interval", "1D")
    for gauge_id, sub in frame.groupby(gauge_col):
        gauge = registry[gauge_id]
        series = sub.set_index("_day")[rainfall_col].sort_index()
        if gauge.is_cumulative:
            incremental = series.diff()
            series = incremental.where(incremental >= 0)
        series = series.resample(interval).sum(min_count=1)
        if gauge.active_from is not None:
            series = series.where(series.index >= gauge.active_from.normalize())
        if gauge.active_to is not None:
            series = series.where(series.index <= gauge.active_to.normalize())
        pieces.append(series.rename(gauge_id))
    wide = pd.concat(pieces, axis=1).sort_index()
    return wide.reindex(columns=[g.gauge_id for g in gauges])


def load_toc(cfg: Config) -> pd.DataFrame:
    frame = pd.read_csv(cfg.data["toc_csv"])
    timestamp_col = cfg.data.get("toc_timestamp_col", "timestamp")
    target_cols = cfg.data.get(
        "toc_columns", {"toc": "toc_mg_per_L", "alk": "alk_mg_per_L"})
    required = {timestamp_col, *target_cols.values()}
    if not required.issubset(frame.columns):
        raise ValueError(f"TOC file requires columns {sorted(required)}")
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], errors="raise")
    if frame[timestamp_col].duplicated().any():
        raise ValueError("TOC file contains duplicate timestamps")
    out = frame.set_index(timestamp_col)[list(target_cols.values())].sort_index()
    out = out.rename(columns={value: key for key, value in target_cols.items()})
    return out.apply(pd.to_numeric, errors="coerce")


def gauge_qc(rain: pd.DataFrame, gauges: list[Gauge]) -> pd.DataFrame:
    rows = []
    for gauge in gauges:
        series = rain[gauge.gauge_id]
        available = series.notna()
        rows.append({
            "gauge_id": gauge.gauge_id,
            "rows": int(available.sum()),
            "gaps": int((~available).sum()),
            "start": series.index[available].min() if available.any() else pd.NaT,
            "end": series.index[available].max() if available.any() else pd.NaT,
            "zero_fraction": float((series[available] == 0).mean()) if available.any()
            else np.nan,
        })
    return pd.DataFrame(rows)

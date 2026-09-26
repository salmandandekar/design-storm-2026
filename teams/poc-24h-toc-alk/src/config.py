"""Typed configuration for the operational rainfall-network pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass(frozen=True)
class Gauge:
    gauge_id: str
    weight: float | None = None
    x: float | None = None
    y: float | None = None
    crs: str | None = None
    flow_path_ft: float | None = None
    velocity_ft_per_s: float | None = None
    sub_area_acres: float | None = None
    is_cumulative: bool = False
    timezone: str | None = None
    active_from: pd.Timestamp | None = None
    active_to: pd.Timestamp | None = None


@dataclass(frozen=True)
class Config:
    path: Path
    raw: dict[str, Any]
    data: dict[str, Any]
    travel_time: dict[str, Any]
    combine: dict[str, Any]
    features: dict[str, Any]
    model: dict[str, Any]
    live: dict[str, Any]
    output_dir: Path
    gauges_path: Path


def _resolve(base: Path, value: str | None) -> Path | None:
    return None if value is None else (base / value).resolve()


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    for section in ("data", "travel_time", "combine", "features", "model"):
        if not isinstance(raw.get(section), dict):
            raise ValueError(f"config section {section!r} is required")
    base = path.parent
    data = dict(raw["data"])
    for key in ("rainfall_csv", "gauges_csv", "toc_csv", "reaches_csv",
                "catchment_geojson"):
        if data.get(key):
            data[key] = str(_resolve(base, data[key]))
    windows = raw["features"].get("windows", [])
    if not windows:
        raise ValueError("features.windows must not be empty")
    for window in windows:
        if pd.Timedelta(window) <= pd.Timedelta(0):
            raise ValueError(f"feature window must be positive: {window}")
    combine = dict(raw["combine"])
    coverage = float(combine.get("min_coverage", 0.7))
    if not 0 < coverage <= 1:
        raise ValueError("combine.min_coverage must be in (0, 1]")
    output_dir = _resolve(base, raw.get("output_dir", "outputs"))
    assert output_dir is not None
    gauges_path = Path(data["gauges_csv"])
    return Config(
        path=path,
        raw=raw,
        data=data,
        travel_time=dict(raw["travel_time"]),
        combine=combine,
        features=dict(raw["features"]),
        model=dict(raw["model"]),
        live=dict(raw.get("live", {})),
        output_dir=output_dir,
        gauges_path=gauges_path,
    )


def _optional_number(row: pd.Series, name: str) -> float | None:
    value = row.get(name)
    return None if pd.isna(value) or value == "" else float(value)


def _optional_time(row: pd.Series, name: str) -> pd.Timestamp | None:
    value = row.get(name)
    return None if pd.isna(value) or value == "" else pd.Timestamp(value)


def load_gauges(cfg: Config) -> list[Gauge]:
    frame = pd.read_csv(cfg.gauges_path)
    if "gauge_id" not in frame:
        raise ValueError("gauges.csv requires gauge_id")
    if frame["gauge_id"].duplicated().any():
        duplicates = frame.loc[frame["gauge_id"].duplicated(), "gauge_id"].tolist()
        raise ValueError(f"duplicate gauge IDs: {duplicates}")
    gauges = []
    for _, row in frame.iterrows():
        included = str(row.get("include", "true")).lower() not in {"false", "0", "no"}
        if not included:
            continue
        gauges.append(Gauge(
            gauge_id=str(row["gauge_id"]),
            weight=_optional_number(row, "weight"),
            x=_optional_number(row, "x"),
            y=_optional_number(row, "y"),
            crs=None if pd.isna(row.get("crs")) else str(row.get("crs")),
            flow_path_ft=_optional_number(row, "flow_path_ft"),
            velocity_ft_per_s=_optional_number(row, "velocity_ft_per_s"),
            sub_area_acres=_optional_number(row, "sub_area_acres"),
            is_cumulative=str(row.get("is_cumulative", "false")).lower()
            in {"true", "1", "yes"},
            timezone=None if pd.isna(row.get("timezone")) else str(row.get("timezone")),
            active_from=_optional_time(row, "active_from"),
            active_to=_optional_time(row, "active_to"),
        ))
    if not gauges:
        raise ValueError("gauge registry contains no included gauges")
    return gauges

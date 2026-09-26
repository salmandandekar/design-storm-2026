"""Static gauge weighting methods."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, Gauge


def _normalize(values: pd.Series) -> pd.Series:
    if values.isna().any() or (values < 0).any() or values.sum() <= 0:
        raise ValueError("gauge weights must be non-negative, present, and sum above zero")
    return values / values.sum()


def _coordinates(gauges: list[Gauge]) -> np.ndarray:
    if any(g.x is None or g.y is None for g in gauges):
        raise ValueError("selected spatial weighting method requires x and y for every gauge")
    return np.asarray([(g.x, g.y) for g in gauges], dtype=float)


def _catchment(cfg: Config):
    path = cfg.data.get("catchment_geojson")
    if not path:
        raise ValueError("selected spatial weighting method requires catchment_geojson")
    try:
        from shapely.geometry import shape
    except ImportError as exc:
        raise RuntimeError("install shapely to use catchment weighting") from exc
    with Path(path).open(encoding="utf-8") as fh:
        obj = json.load(fh)
    if obj.get("type") == "FeatureCollection":
        features = obj.get("features", [])
        if len(features) != 1:
            raise ValueError("catchment GeoJSON must contain exactly one polygon")
        obj = features[0]["geometry"]
    elif obj.get("type") == "Feature":
        obj = obj["geometry"]
    polygon = shape(obj)
    if polygon.is_empty or polygon.area <= 0:
        raise ValueError("catchment polygon is empty")
    return polygon


def compute_weights(cfg: Config, gauges: list[Gauge]) -> pd.Series:
    method = cfg.combine.get("method", "mean")
    index = [g.gauge_id for g in gauges]
    if method == "per_gauge":
        return pd.Series(1 / len(gauges), index=index, name="weight")
    if method == "mean":
        return pd.Series(1 / len(gauges), index=index, name="weight")
    if method == "explicit":
        values = pd.Series([g.weight for g in gauges], index=index,
                           dtype=float, name="weight")
        if values.isna().any() or (values < 0).any():
            raise ValueError("explicit weights must be present and non-negative")
        if not np.isclose(values.sum(), 1.0):
            raise ValueError("explicit weights must sum to 1")
        return values
    if method == "area":
        return _normalize(pd.Series([g.sub_area_acres for g in gauges], index=index,
                                    dtype=float, name="weight"))
    xy = _coordinates(gauges)
    catchment = _catchment(cfg)
    if method == "idw":
        target = catchment.centroid
        distances = np.hypot(xy[:, 0] - target.x, xy[:, 1] - target.y)
        if np.any(distances == 0):
            values = (distances == 0).astype(float)
        else:
            values = distances ** -float(cfg.combine.get("idw_power", 2))
        return _normalize(pd.Series(values, index=index, name="weight"))
    if method == "thiessen":
        try:
            from shapely import MultiPoint, Point, voronoi_polygons
        except ImportError as exc:
            raise RuntimeError("shapely>=2 is required for Thiessen weights") from exc
        if len(gauges) == 1:
            return pd.Series([1.0], index=index, name="weight")
        cells = voronoi_polygons(MultiPoint(xy), extend_to=catchment)
        areas = []
        for point in map(Point, xy):
            matches = [cell for cell in cells.geoms if cell.covers(point)]
            if not matches:
                raise ValueError("could not assign a Thiessen cell to every gauge")
            areas.append(matches[0].intersection(catchment).area)
        return _normalize(pd.Series(areas, index=index, name="weight"))
    raise ValueError(f"unsupported combine.method: {method}")

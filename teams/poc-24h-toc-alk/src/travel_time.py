"""Per-gauge runoff travel-time calculations."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Config, Gauge


def _round_to_grid(value: pd.Timedelta, grid: pd.Timedelta) -> pd.Timedelta:
    return pd.Timedelta(round(value / grid) * grid)


def _reach_taus(cfg: Config, gauges: list[Gauge]) -> dict[str, pd.Timedelta]:
    reaches_path = cfg.data.get("reaches_csv")
    if not reaches_path:
        raise ValueError("travel_time.method=reach_sum requires data.reaches_csv")
    reaches = pd.read_csv(Path(reaches_path)).set_index("reach_id")
    required = {"length_ft", "velocity_ft_per_s", "downstream_reach_id"}
    if not required.issubset(reaches.columns):
        raise ValueError(f"reaches.csv requires {sorted(required)}")
    registry = pd.read_csv(cfg.gauges_path).set_index("gauge_id")
    if "reach_ids" not in registry:
        raise ValueError("reach_sum requires reach_ids in gauges.csv")
    out: dict[str, pd.Timedelta] = {}
    for gauge in gauges:
        ids = [item.strip() for item in str(registry.loc[gauge.gauge_id, "reach_ids"]).split("|")]
        seconds = 0.0
        for reach_id in ids:
            if reach_id not in reaches.index:
                raise ValueError(f"{gauge.gauge_id}: unknown reach {reach_id}")
            reach = reaches.loc[reach_id]
            velocity = float(reach["velocity_ft_per_s"])
            if velocity <= 0:
                raise ValueError(f"{reach_id}: velocity must be positive")
            seconds += float(reach["length_ft"]) / velocity
        out[gauge.gauge_id] = pd.Timedelta(seconds=seconds)
    return out


def compute_taus(cfg: Config, gauges: list[Gauge]) -> tuple[dict[str, pd.Timedelta], str]:
    """Return grid-rounded travel times and their provenance."""
    method = cfg.travel_time.get("method", "direct")
    direct = cfg.travel_time.get("tau_direct", {}) or {}
    grid = pd.Timedelta(cfg.data.get("resample_interval", "1D"))
    if method == "empirical":
        return {g.gauge_id: pd.Timedelta(0) for g in gauges}, "empirical"
    if method == "reach_sum":
        raw = _reach_taus(cfg, gauges)
    else:
        raw = {}
        default_velocity = cfg.travel_time.get("velocity_ft_per_s")
        for gauge in gauges:
            if gauge.gauge_id in direct:
                tau = pd.Timedelta(direct[gauge.gauge_id])
            elif method == "fixed_velocity":
                velocity = gauge.velocity_ft_per_s or default_velocity
                if gauge.flow_path_ft is None or velocity is None or float(velocity) <= 0:
                    raise ValueError(
                        f"{gauge.gauge_id}: fixed_velocity requires positive "
                        "flow_path_ft and velocity_ft_per_s"
                    )
                tau = pd.Timedelta(seconds=gauge.flow_path_ft / float(velocity))
            else:
                raise ValueError(f"unsupported travel-time method: {method}")
            if tau < pd.Timedelta(0):
                raise ValueError(f"{gauge.gauge_id}: travel time must not be negative")
            raw[gauge.gauge_id] = tau
    return {key: _round_to_grid(value, grid) for key, value in raw.items()}, method

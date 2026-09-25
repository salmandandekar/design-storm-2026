from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import Config, Gauge
from src.rainfall import load_rainfall
from src.live_data import merge_rainfall


def make_cfg(tmp_path: Path, rain_path: Path) -> Config:
    data = {
        "rainfall_csv": str(rain_path),
        "timestamp_col": "timestamp",
        "gauge_col": "gauge_id",
        "rainfall_col": "rainfall_in",
        "default_timezone": "UTC",
        "resample_interval": "1D",
    }
    return Config(
        path=tmp_path / "config.yaml", raw={}, data=data, travel_time={},
        combine={}, features={"windows": ["1D"]}, model={}, live={},
        output_dir=tmp_path / "out", gauges_path=tmp_path / "gauges.csv",
    )


def test_per_gauge_timezone_and_unknown_gauge(tmp_path):
    path = tmp_path / "rain.csv"
    pd.DataFrame([
        {"timestamp": "2026-01-01 23:30", "gauge_id": "mountain", "rainfall_in": 1},
        {"timestamp": "2026-01-02 01:30", "gauge_id": "utc", "rainfall_in": 2},
    ]).to_csv(path, index=False)
    gauges = [Gauge("mountain", timezone="America/Denver"), Gauge("utc", timezone="UTC")]
    rain = load_rainfall(make_cfg(tmp_path, path), gauges)
    assert rain.loc["2026-01-02", "mountain"] == 1
    assert rain.loc["2026-01-02", "utc"] == 2
    with pytest.raises(ValueError, match="absent from registry"):
        load_rainfall(make_cfg(tmp_path, path), [gauges[0]])


def test_cumulative_reset_becomes_gap(tmp_path):
    path = tmp_path / "rain.csv"
    pd.DataFrame([
        {"timestamp": "2026-01-01", "gauge_id": "g", "rainfall_in": 1},
        {"timestamp": "2026-01-02", "gauge_id": "g", "rainfall_in": 2},
        {"timestamp": "2026-01-03", "gauge_id": "g", "rainfall_in": 0},
    ]).to_csv(path, index=False)
    rain = load_rainfall(
        make_cfg(tmp_path, path), [Gauge("g", is_cumulative=True, timezone="UTC")])
    assert rain.loc["2026-01-02", "g"] == 1
    assert pd.isna(rain.loc["2026-01-03", "g"])


def test_live_merge_does_not_erase_other_gauges():
    index = pd.to_datetime(["2026-01-01", "2026-01-02"])
    committed = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}, index=index)
    latest = pd.DataFrame({"a": [9.0], "b": [float("nan")]}, index=index[-1:])
    merged = merge_rainfall(committed, latest)
    assert merged.loc["2026-01-02", "a"] == 9
    assert merged.loc["2026-01-02", "b"] == 4

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.combine import combine
from src.config import Config, Gauge
from src.models import (
    fit_operational_ols,
    fit_operational_ridge,
    predict_operational,
)
from src.network_features import align_training, build_feature_frame
from src.spatial import compute_weights
from src.time_shift import shift_frame, shift_series


def cfg(tmp_path: Path, combine_method: str = "mean") -> Config:
    raw = {
        "data": {"resample_interval": "1D"},
        "travel_time": {},
        "combine": {
            "method": combine_method,
            "missing_gauge_policy": "reweight",
            "min_coverage": 0.6,
            "idw_power": 2,
        },
        "features": {"windows": ["1D", "3D"]},
        "model": {"test_fraction": 0.2, "cv_folds": 3},
    }
    return Config(
        path=tmp_path / "config.yaml",
        raw=raw,
        data=raw["data"],
        travel_time=raw["travel_time"],
        combine=raw["combine"],
        features=raw["features"],
        model=raw["model"],
        live={},
        output_dir=tmp_path / "outputs",
        gauges_path=tmp_path / "gauges.csv",
    )


def test_each_gauge_is_shifted_by_its_own_tau():
    index = pd.date_range("2026-01-01", periods=3, freq="D")
    rain = pd.DataFrame({"near": [1.0, 0, 0], "far": [2.0, 0, 0]}, index=index)
    shifted = shift_frame(
        rain, {"near": pd.Timedelta("1D"), "far": pd.Timedelta("2D")})
    assert shifted.loc["2026-01-02", "near"] == 1
    assert shifted.loc["2026-01-03", "far"] == 2
    assert shift_series(shift_series(rain["near"], pd.Timedelta("1D")),
                        pd.Timedelta("-1D")).equals(rain["near"])


def test_reweight_and_low_coverage(tmp_path):
    config = cfg(tmp_path)
    rain = pd.DataFrame(
        {"a": [1.0, np.nan], "b": [3.0, 3.0]},
        index=pd.date_range("2026-01-01", periods=2, freq="D"),
    )
    effective, coverage = combine(rain, pd.Series({"a": 0.5, "b": 0.5}), config)
    assert effective.iloc[0] == 2
    assert pd.isna(effective.iloc[1])
    assert coverage.iloc[1] == 0.5


def test_per_gauge_coverage_is_a_fraction(tmp_path):
    config = cfg(tmp_path, "per_gauge")
    rain = pd.DataFrame(
        {"a": [1.0], "b": [2.0]},
        index=pd.date_range("2026-01-01", periods=1, freq="D"),
    )
    weights = compute_weights(config, [Gauge("a"), Gauge("b")])
    result, coverage = combine(rain, weights, config)
    assert result.equals(rain)
    assert coverage.iloc[0] == 1


def test_window_features_do_not_look_ahead(tmp_path):
    config = cfg(tmp_path)
    index = pd.date_range("2026-01-01", periods=5, freq="D")
    rain = pd.Series([1, 2, 4, 8, 16], index=index, name="effective")
    features = build_feature_frame(rain, pd.Series(1.0, index=index), config)
    assert features.loc["2026-01-03", "rain_effective_1d"] == 4
    assert features.loc["2026-01-03", "rain_effective_3d"] == 7


def test_24h_label_requires_exact_next_calendar_day():
    index = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])
    features = pd.DataFrame({"rain": [1, 2, 3]}, index=index)
    targets = pd.DataFrame(
        {"toc": [2.0, 9.0]},
        index=pd.to_datetime(["2026-01-02", "2026-02-20"]),
    )
    X, y = align_training(features, targets, "forecast_24h")
    assert list(X.index) == [pd.Timestamp("2026-01-01")]
    assert y.iloc[0]["toc"] == 2


def test_weighting_methods_and_thiessen(tmp_path):
    gauges = [
        Gauge("a", weight=0.25, x=2, y=5, sub_area_acres=25),
        Gauge("b", weight=0.75, x=8, y=5, sub_area_acres=75),
    ]
    assert compute_weights(cfg(tmp_path, "mean"), gauges).tolist() == [0.5, 0.5]
    assert compute_weights(cfg(tmp_path, "explicit"), gauges).tolist() == [0.25, 0.75]
    assert compute_weights(cfg(tmp_path, "area"), gauges).tolist() == [0.25, 0.75]
    catchment = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
    }
    path = tmp_path / "catchment.geojson"
    path.write_text(json.dumps(catchment), encoding="utf-8")
    config = cfg(tmp_path, "thiessen")
    config.data["catchment_geojson"] = str(path)
    weights = compute_weights(config, gauges)
    assert weights.sum() == pytest.approx(1)
    assert weights["a"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="sum to 1"):
        compute_weights(
            cfg(tmp_path, "explicit"),
            [Gauge("a", weight=1), Gauge("b", weight=1)],
        )


def test_operational_model_round_trip_prediction(tmp_path):
    index = pd.date_range("2024-01-01", periods=100, freq="D")
    X = pd.DataFrame({"rain": np.linspace(0, 5, 100)}, index=index)
    y = pd.Series(2 + 0.4 * X["rain"], index=index)
    model, metrics = fit_operational_ols(X, y, test_fraction=0.2, cv_folds=3)
    prediction = predict_operational(model, X.iloc[[-1]])
    assert prediction.iloc[0]["prediction"] == pytest.approx(4.0)
    assert metrics["holdout_mae"] < 1e-10
    assert metrics["beats_baseline"]


def test_per_gauge_ridge_prediction_interval():
    index = pd.date_range("2024-01-01", periods=100, freq="D")
    X = pd.DataFrame({
        "near": np.linspace(0, 5, 100),
        "far": np.linspace(0, 5, 100) * 0.9 + np.sin(np.arange(100)),
    }, index=index)
    y = pd.Series(2 + 0.4 * X["near"], index=index)
    model, metrics = fit_operational_ridge(X, y, test_fraction=0.2, cv_folds=3)
    prediction = predict_operational(model, X.iloc[[-1]])
    assert prediction.iloc[0]["lower"] <= prediction.iloc[0]["prediction"]
    assert prediction.iloc[0]["upper"] >= prediction.iloc[0]["prediction"]
    assert "vif" in metrics

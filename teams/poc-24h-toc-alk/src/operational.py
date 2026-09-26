"""End-to-end training and inference for rainfall-network models."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from .combine import combine
from .config import Config, load_gauges
from .lag_optimizer import scan_lag
from .models import fit_operational_ols, fit_operational_ridge, predict_operational
from .network_features import align_training, build_feature_frame
from .rainfall import gauge_qc, load_rainfall, load_toc
from .report import build_report
from .spatial import compute_weights
from .time_shift import shift_frame
from .travel_time import compute_taus

PRODUCTS = ("arrival_nowcast", "forecast_24h")
TARGETS = ("toc", "alk")
PROVISIONAL_NOTE = (
    "Source readings are provisional and subject to revision. "
    "Denver Water data terms apply."
)


def _json_default(value: Any):
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if pd.isna(value):
        return None
    raise TypeError(f"cannot encode {type(value).__name__}")


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _prepare_outputs(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    terms = cfg.path.parent / "TERMS.md"
    if terms.exists():
        shutil.copyfile(terms, cfg.output_dir / "TERMS.md")


def _registry_fingerprint(cfg: Config) -> str:
    return hashlib.sha256(cfg.gauges_path.read_bytes()).hexdigest()


def _preprocessing_config(cfg: Config) -> dict:
    return {
        "resample_interval": cfg.data.get("resample_interval", "1D"),
        "travel_time": cfg.travel_time,
        "combine": cfg.combine,
        "features": cfg.features,
    }


def _fingerprint(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def prepare_network(
    cfg: Config,
    rainfall_path: str | Path | None = None,
    fitted_taus: dict[str, pd.Timedelta] | None = None,
) -> tuple[pd.DataFrame, pd.Series | pd.DataFrame, pd.Series, pd.Series,
           dict[str, pd.Timedelta], str, pd.DataFrame]:
    gauges = load_gauges(cfg)
    rain = load_rainfall(cfg, gauges, rainfall_path)
    weights = compute_weights(cfg, gauges)
    if fitted_taus is None:
        taus, provenance = compute_taus(cfg, gauges)
    else:
        taus, provenance = fitted_taus, "stored_model"
    shifted = shift_frame(rain, taus)
    effective, coverage = combine(shifted, weights, cfg)
    qc = gauge_qc(rain, gauges)
    return rain, effective, coverage, weights, taus, provenance, qc


def _select_empirical_taus(
    cfg: Config,
    rain: pd.DataFrame,
    targets: pd.DataFrame,
    weights: pd.Series,
) -> tuple[dict[str, pd.Timedelta], pd.DataFrame]:
    scores = scan_lag(rain, targets, weights, cfg)
    best = scores.loc[scores["mean_cv_mae"].idxmin()]
    lag = pd.Timedelta(seconds=float(best["lag_seconds"]))
    return {column: lag for column in rain.columns}, scores


def train(cfg: Config) -> dict:
    _prepare_outputs(cfg)
    gauges = load_gauges(cfg)
    rain = load_rainfall(cfg, gauges)
    targets = load_toc(cfg)
    weights = compute_weights(cfg, gauges)
    taus, provenance = compute_taus(cfg, gauges)
    lag_scores = None
    if provenance == "empirical":
        taus, lag_scores = _select_empirical_taus(cfg, rain, targets, weights)
        provenance = "empirical_time_ordered_cv_non_physical"
        lag_scores.to_csv(cfg.output_dir / "lag_scan.csv", index=False)

    shifted = shift_frame(rain, taus)
    effective, coverage = combine(shifted, weights, cfg)
    features = build_feature_frame(effective, coverage, cfg)
    qc = gauge_qc(rain, gauges)
    effective_out = (
        effective.copy() if isinstance(effective, pd.DataFrame)
        else effective.to_frame()
    )
    effective_out["coverage_fraction"] = coverage
    effective_out.to_csv(cfg.output_dir / "effective_rainfall.csv")
    qc.to_csv(cfg.output_dir / "gauge_qc.csv", index=False)

    models: dict[str, object] = {}
    metrics: dict[str, dict] = {}
    aligned_parts = []
    for product in PRODUCTS:
        for target in TARGETS:
            X, y_frame = align_training(features, targets[[target]], product)
            persistence = targets[target].reindex(X.index) if product == "forecast_24h" else None
            fit_function = (
                fit_operational_ridge
                if cfg.model.get("type") == "ridge"
                or cfg.combine.get("method") == "per_gauge"
                else fit_operational_ols
            )
            fitted, result = fit_function(
                X, y_frame[target],
                test_fraction=float(cfg.model.get("test_fraction", 0.2)),
                cv_folds=int(cfg.model.get("cv_folds", 5)),
                persistence=persistence,
            )
            key = f"{product}:{target}"
            models[key] = fitted
            metrics[key] = result
            aligned = X.copy()
            aligned["target"] = y_frame[target]
            aligned["product"] = product
            aligned["target_name"] = target
            aligned_parts.append(aligned.reset_index(names="timestamp"))
    pd.concat(aligned_parts, ignore_index=True).to_csv(
        cfg.output_dir / "aligned_dataset.csv", index=False)

    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gauge_ids": [g.gauge_id for g in gauges],
        "gauge_registry_sha256": _registry_fingerprint(cfg),
        "preprocessing": _preprocessing_config(cfg),
        "preprocessing_sha256": _fingerprint(_preprocessing_config(cfg)),
        "travel_times": {key: str(value) for key, value in taus.items()},
        "travel_time_provenance": provenance,
        "weights": weights.to_dict(),
        "combine_method": cfg.combine.get("method"),
        "missing_gauge_policy": cfg.combine.get("missing_gauge_policy"),
        "min_coverage": float(cfg.combine.get("min_coverage", 0.7)),
        "feature_names": list(features.columns),
        "windows": list(cfg.features["windows"]),
        "training_rainfall_range": [
            rain.index.min().date().isoformat(),
            rain.index.max().date().isoformat(),
        ],
        "training_target_range": [
            targets.index.min().date().isoformat(),
            targets.index.max().date().isoformat(),
        ],
        "source_latency": "NOAA GHCN daily data may publish about two days late",
        "provisional_notice": PROVISIONAL_NOTE,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    bundle = {"models": models, "metadata": metadata, "metrics": metrics}
    joblib.dump(bundle, cfg.output_dir / "model.joblib")
    _atomic_json(cfg.output_dir / "model_meta.json", metadata)
    _atomic_json(cfg.output_dir / "metrics.json", metrics)
    build_report(metrics, cfg.output_dir, lag_scores)
    payload = predict_latest(cfg, rain=rain, bundle=bundle, source="committed")
    write_ui_payload(cfg, payload)
    return payload


def load_bundle(cfg: Config) -> dict:
    path = cfg.output_dir / "model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"model not trained: {path}")
    bundle = joblib.load(path)
    expected = _registry_fingerprint(cfg)
    actual = bundle["metadata"].get("gauge_registry_sha256")
    if actual != expected:
        raise ValueError("gauge registry differs from the registry used to train the model")
    current_preprocessing = _fingerprint(_preprocessing_config(cfg))
    if bundle["metadata"].get("preprocessing_sha256") != current_preprocessing:
        raise ValueError(
            "preprocessing configuration differs from training; retrain the model")
    return bundle


def predict_latest(
    cfg: Config,
    rain: pd.DataFrame | None = None,
    bundle: dict | None = None,
    source: str = "input",
    retrieved_at: str | None = None,
) -> dict:
    bundle = bundle or load_bundle(cfg)
    gauges = load_gauges(cfg)
    if rain is None:
        rain = load_rainfall(cfg, gauges)
    trained = bundle["metadata"]["gauge_ids"]
    unknown = set(rain.columns) - set(trained)
    if unknown:
        raise ValueError(f"unknown gauges at prediction time: {sorted(unknown)}")
    missing = set(trained) - set(rain.columns)
    if missing:
        raise ValueError(f"trained gauges absent at prediction time: {sorted(missing)}")
    rain = rain.reindex(columns=trained)
    taus = {
        key: pd.Timedelta(value)
        for key, value in bundle["metadata"]["travel_times"].items()
    }
    weights = pd.Series(bundle["metadata"]["weights"], dtype=float).reindex(trained)
    effective, coverage = combine(shift_frame(rain, taus), weights, cfg)
    features = build_feature_frame(effective, coverage, cfg)
    expected = bundle["metadata"]["feature_names"]
    valid = features.reindex(columns=expected).dropna()
    if valid.empty:
        raise ValueError("no prediction row has complete rainfall windows and coverage")
    minimum_coverage = float(bundle["metadata"]["min_coverage"])
    eligible = coverage[coverage >= minimum_coverage]
    if eligible.empty:
        raise ValueError("no prediction timestamp meets minimum gauge coverage")
    issue_time = eligible.index.max()
    if issue_time not in valid.index:
        raise ValueError(
            f"latest issue time {issue_time.date()} lacks a complete rainfall "
            "history; refusing to substitute an older prediction"
        )
    row = valid.loc[[issue_time]]
    products: dict[str, dict] = {}
    for product in PRODUCTS:
        products[product] = {}
        for target in TARGETS:
            key = f"{product}:{target}"
            prediction = predict_operational(bundle["models"][key], row).iloc[0]
            target_time = issue_time + (
                pd.Timedelta(days=1) if product == "forecast_24h"
                else pd.Timedelta(0)
            )
            products[product][target] = {
                "value": float(prediction["prediction"]),
                "lower": float(prediction["lower"]),
                "upper": float(prediction["upper"]),
                "target_time": target_time.date().isoformat(),
                "metrics": bundle["metrics"][key],
            }
    latest_coverage = float(coverage.reindex([issue_time]).iloc[0])
    payload = {
        "status": "ok",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "retrieved_at": retrieved_at,
        "source": source,
        "latest_rainfall_observation": rain.dropna(how="all").index.max().date().isoformat(),
        "issue_time": issue_time.date().isoformat(),
        "gauge_count": len(trained),
        "coverage_fraction": latest_coverage,
        "travel_time_provenance": bundle["metadata"]["travel_time_provenance"],
        "travel_times": bundle["metadata"]["travel_times"],
        "products": products,
        "warnings": [
            PROVISIONAL_NOTE,
            bundle["metadata"]["source_latency"],
            "This adapter uses one rain gauge and is not a spatially representative network.",
            "Research model: predictions are shown even when validation does not beat a baseline.",
        ],
    }
    if "empirical" in bundle["metadata"]["travel_time_provenance"]:
        payload["warnings"].append(
            "Travel time was selected empirically and is not a physical measurement.")
    return payload


def write_ui_payload(cfg: Config, payload: dict) -> Path:
    path = cfg.path.parents[1] / "explainable-viz" / "viz-data" / "latest-prediction.json"
    _atomic_json(path, payload)
    return path

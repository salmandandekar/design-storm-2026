"""Latest NOAA Daily Summaries adapter with explicit cache/error behavior."""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .config import Config, load_gauges
from .operational import predict_latest, write_ui_payload
from .rainfall import load_rainfall

NOAA_URL = "https://www.ncei.noaa.gov/access/services/data/v1"


def fetch_noaa(cfg: Config) -> tuple[pd.DataFrame, str]:
    station = cfg.live.get("station")
    if not station:
        raise ValueError("live.station is required")
    end = date.today()
    start = end - timedelta(days=int(cfg.live.get("lookback_days", 60)))
    query = urlencode({
        "dataset": "daily-summaries",
        "stations": station,
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
        "format": "json",
        "units": "standard",
        "includeAttributes": "false",
    })
    headers = {"User-Agent": "design-storm-rain-toc/1.0"}
    token = os.environ.get("NOAA_TOKEN")
    if token:
        headers["token"] = token
    request = Request(f"{NOAA_URL}?{query}", headers=headers)
    timeout = float(cfg.live.get("timeout_seconds", 20))
    with urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"NOAA request failed with HTTP {response.status}")
        payload = json.load(response)
    if not isinstance(payload, list) or not payload:
        raise ValueError("NOAA returned no daily observations")
    frame = pd.DataFrame(payload)
    if not {"STATION", "DATE", "PRCP"}.issubset(frame.columns):
        raise ValueError("NOAA response lacks STATION, DATE, or PRCP")
    frame = frame[["STATION", "DATE", "PRCP"]]
    frame["PRCP"] = pd.to_numeric(frame["PRCP"], errors="raise")
    retrieved_at = datetime.now(timezone.utc).isoformat()
    cache = cfg.output_dir / "latest_noaa.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache, index=False)
    return frame, retrieved_at


def refresh(cfg: Config) -> dict:
    gauges = load_gauges(cfg)
    committed = load_rainfall(cfg, gauges)
    frame, retrieved_at = fetch_noaa(cfg)
    temporary = cfg.output_dir / ".latest_noaa_input.csv"
    frame.to_csv(temporary, index=False)
    try:
        latest = load_rainfall(cfg, gauges, temporary)
    finally:
        temporary.unlink(missing_ok=True)
    rain = merge_rainfall(committed, latest)
    payload = predict_latest(
        cfg, rain=rain, source="NOAA Daily Summaries API",
        retrieved_at=retrieved_at)
    write_ui_payload(cfg, payload)
    return payload


def merge_rainfall(committed: pd.DataFrame, latest: pd.DataFrame) -> pd.DataFrame:
    """Prefer non-null live cells without erasing other gauges or dates."""
    return latest.combine_first(committed).sort_index()

#!/usr/bin/env python3
"""Build deterministic synthetic data for exercising the 3D prediction UI.

The values are intentionally invented for interface testing and must never be
presented as Denver Water measurements or model output.
"""
from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[2] / "water-system-3d" / "prediction-demo.json"


def rainfall(day: int) -> float:
    storms = (
        0.85 * math.exp(-((day - 6) / 1.3) ** 2)
        + 1.25 * math.exp(-((day - 17) / 1.8) ** 2)
        + 0.55 * math.exp(-((day - 25) / 1.1) ** 2)
    )
    return round(max(0, storms - 0.04), 3)


def main() -> None:
    start = date(2026, 8, 20)
    rain = [rainfall(day) for day in range(30)]
    frames = []
    for day in range(30):
        lagged_3 = sum(rain[max(0, day - 6):max(0, day - 3)])
        lagged_7 = sum(rain[max(0, day - 10):max(0, day - 3)])
        season = math.sin(day / 5) * 0.08
        toc = 2.15 + 0.52 * lagged_3 + 0.18 * lagged_7 + season
        alk = 68.0 - 3.8 * lagged_3 - 1.4 * lagged_7 - season * 4
        toc_width = 0.28 + 0.12 * min(2, lagged_3)
        alk_width = 4.2 + 0.8 * min(2, lagged_3)
        frames.append({
            "date": (start + timedelta(days=day)).isoformat(),
            "rainfall_in": rain[day],
            "coverage_fraction": 1.0 if day not in (12, 13) else 0.67,
            "toc": {
                "value": round(toc, 3),
                "lower": round(toc - toc_width, 3),
                "upper": round(toc + toc_width, 3),
            },
            "alk": {
                "value": round(alk, 2),
                "lower": round(alk - alk_width, 2),
                "upper": round(alk + alk_width, 2),
            },
        })

    toc_values = [frame["toc"]["value"] for frame in frames]
    alk_values = [frame["alk"]["value"] for frame in frames]
    payload = {
        "schema_version": 1,
        "synthetic": True,
        "label": "SYNTHETIC DEMO — NOT OBSERVED OR PREDICTED WATER QUALITY",
        "purpose": "Exercise the 3D prediction timeline and statistics UI.",
        "station": {
            "id": "plant-foothills",
            "name": "Foothills Treatment Plant",
            "lat": 39.46637,
            "lon": -105.06154,
        },
        "thresholds": {"toc_watch_mg_L": 3.0, "alk_watch_mg_L": 60.0},
        "statistics": {
            "frames": len(frames),
            "rainfall_total_in": round(sum(rain), 2),
            "toc_mean_mg_L": round(sum(toc_values) / len(toc_values), 2),
            "toc_peak_mg_L": round(max(toc_values), 2),
            "alk_mean_mg_L": round(sum(alk_values) / len(alk_values), 2),
            "alk_min_mg_L": round(min(alk_values), 2),
            "toc_watch_days": sum(value >= 3 for value in toc_values),
            "alk_watch_days": sum(value <= 60 for value in alk_values),
        },
        "model_status": {
            "product": "forecast_24h",
            "gauge_count": 3,
            "travel_time_provenance": "synthetic_configured_demo",
            "toc_holdout_mae": 0.31,
            "alk_holdout_mae": 4.7,
            "beats_persistence": False,
        },
        "frames": frames,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(frames)} synthetic frames)")


if __name__ == "__main__":
    main()

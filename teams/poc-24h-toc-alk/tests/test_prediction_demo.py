from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_prediction_demo_is_explicitly_synthetic_and_well_formed():
    payload = json.loads(
        (ROOT / "water-system-3d" / "prediction-demo.json").read_text(
            encoding="utf-8"))
    assert payload["synthetic"] is True
    assert "SYNTHETIC" in payload["label"]
    assert len(payload["frames"]) == 30
    dates = [frame["date"] for frame in payload["frames"]]
    assert dates == sorted(dates)
    for frame in payload["frames"]:
        assert 0 <= frame["coverage_fraction"] <= 1
        for target in ("toc", "alk"):
            prediction = frame[target]
            assert prediction["lower"] <= prediction["value"] <= prediction["upper"]


def test_3d_page_keeps_synthetic_warning_visible():
    html = (ROOT / "design-storm-water-system-3d.html").read_text(encoding="utf-8")
    assert "Prediction terrain demo" in html
    assert "Every value in this replay is deterministic dummy data" in html
    assert 'location.hash === "#prediction-demo"' in html

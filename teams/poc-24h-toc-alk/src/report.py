"""Metric-driven operational report artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def build_report(
    metrics: dict[str, dict],
    output_dir: Path,
    lag_scores: pd.DataFrame | None = None,
) -> None:
    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        key: {
            "holdout_mae": value["holdout_mae"],
            "baseline_mae": value["baseline_mae"],
            "beats_baseline": value["beats_baseline"],
            "beats_persistence": value.get("beats_persistence"),
        }
        for key, value in metrics.items()
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    labels = list(metrics)
    model_mae = [metrics[key]["holdout_mae"] for key in labels]
    baseline_mae = [metrics[key]["baseline_mae"] for key in labels]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([i - 0.18 for i in x], model_mae, width=0.36, label="rainfall model")
    ax.bar([i + 0.18 for i in x], baseline_mae, width=0.36, label="training mean")
    ax.set_xticks(list(x), [label.replace(":", "\n") for label in labels])
    ax.set_ylabel("Hold-out MAE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(report_dir / "holdout_mae.png", dpi=150)
    plt.close(fig)

    if lag_scores is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(lag_scores["lag_seconds"] / 86400, lag_scores["mean_cv_mae"],
                marker="o")
        ax.set_xlabel("Common empirical lag (days)")
        ax.set_ylabel("Mean time-ordered CV MAE")
        ax.set_title("Empirical lag scan (non-physical)")
        fig.tight_layout()
        fig.savefig(report_dir / "lag_scan.png", dpi=150)
        plt.close(fig)

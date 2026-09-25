#!/usr/bin/env python3
"""CLI for multi-gauge rainfall modeling and latest-data predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_config, load_gauges
from src.lag_optimizer import scan_lag
from src.live_data import refresh
from src.operational import (
    load_bundle,
    predict_latest,
    prepare_network,
    train,
    write_ui_payload,
)
from src.rainfall import load_rainfall, load_toc
from src.server import serve
from src.spatial import compute_weights


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    def command(name: str) -> argparse.ArgumentParser:
        item = sub.add_parser(name)
        item.add_argument("--config", default="config.yaml")
        return item

    command("train")
    command("check-gauges")
    command("scan-lag")
    predict = command("predict")
    predict.add_argument("--input", required=True)
    command("refresh")
    server = command("serve")
    server.add_argument("--port", type=int, default=8765)
    return result


def main() -> int:
    args = parser().parse_args()
    cfg = load_config(args.config)
    if args.command == "train":
        payload = train(cfg)
        print(json.dumps(payload, indent=2))
    elif args.command == "check-gauges":
        _, _, coverage, weights, taus, provenance, qc = prepare_network(cfg)
        print(qc.to_string(index=False))
        print("\nweights:", weights.to_dict())
        print("travel_times:", {key: str(value) for key, value in taus.items()})
        print("travel_time_provenance:", provenance)
        print("minimum coverage:", float(coverage.min()))
    elif args.command == "scan-lag":
        gauges = load_gauges(cfg)
        scores = scan_lag(
            load_rainfall(cfg, gauges), load_toc(cfg),
            compute_weights(cfg, gauges), cfg)
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        scores.to_csv(cfg.output_dir / "lag_scan.csv", index=False)
        print(scores.to_string(index=False))
    elif args.command == "predict":
        gauges = load_gauges(cfg)
        rain = load_rainfall(cfg, gauges, Path(args.input))
        payload = predict_latest(cfg, rain=rain, bundle=load_bundle(cfg),
                                 source=str(Path(args.input)))
        write_ui_payload(cfg, payload)
        print(json.dumps(payload, indent=2))
    elif args.command == "refresh":
        print(json.dumps(refresh(cfg), indent=2))
    elif args.command == "serve":
        serve(cfg, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

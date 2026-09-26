# Open the visualization

[Open the Prediction UI in this Codespace](https://friendly-space-happiness-7w795w97g7rhxgpr-8765.app.github.dev/prediction)

[Open the 3D Prediction Terrain Demo](https://friendly-space-happiness-7w795w97g7rhxgpr-8765.app.github.dev/prediction-3d)

[Open the Terrain Flow Simulation and 12-hour feasibility page](https://friendly-space-happiness-7w795w97g7rhxgpr-8765.app.github.dev/teams/terrain-flow-sim/)
— serve with `python3 serve.py` from the repository root; the written
assessment is `teams/terrain-flow-sim/FEASIBILITY.md`.

## Prediction outputs

Start the prediction-enabled server first:

```bash
cd teams/poc-24h-toc-alk
python3 main.py serve --config config.yaml
```

The Prediction tab shows separate rainfall-arrival and 24-hour estimates for
TOC and alkalinity. Its Refresh button requests the latest available NOAA daily
data. NOAA can lag by about two days, and the current adapter uses one gauge, so
the page labels the output as provisional and research-only.

The 3D demo animates deterministic dummy rainfall, TOC, and alkalinity values
over the existing terrain. It is an interface test only: all values and
statistics are prominently labeled synthetic and are not Denver Water readings
or model predictions. Regenerate its feed with:

```bash
python3 teams/poc-24h-toc-alk/analysis/build_prediction_demo.py
```

Historical proof-of-concept outputs:

- [Alkalinity 24-hour prediction and warning chart](https://friendly-space-happiness-7w795w97g7rhxgpr-8765.app.github.dev/teams/poc-24h-toc-alk/figures/event_timeline_alk.png)
- [TOC 24-hour prediction chart](https://friendly-space-happiness-7w795w97g7rhxgpr-8765.app.github.dev/teams/poc-24h-toc-alk/figures/event_timeline_toc.png)
- [All saved 24-hour predictions (CSV)](https://friendly-space-happiness-7w795w97g7rhxgpr-8765.app.github.dev/teams/poc-24h-toc-alk/results/predictions_lead1.csv)

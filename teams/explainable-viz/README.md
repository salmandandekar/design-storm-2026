# Explainable visualizations: 2D suite + 3D map extensions

Team workspace for the explainable 2D and 3D visualizations of the Design Storm
data, with sub-basin behaviour. Everything here is derived from the committed
files in `../../data/`, `../../strontia-brief/`, and `../../water-system-3d/`;
nothing is invented, and every view says what it shows and why that data was
chosen.

Two things live here:

1. **The 2D suite** (`index.html`): six linked views of the challenge data —
   system timeline, Strontia sonde depth–time heatmap, storm anatomy, seasonal
   rhythm, a sub-basin behaviour board, and the research prediction panel. The
   first five views remain zero-dependency and fully offline; the prediction
   panel can show its last generated JSON offline but needs the PoC server for
   a live NOAA refresh.
2. **Data bundles** (`viz-data/`) that also power the sub-basin choropleth,
   water-year replay, and reservoir depth column added to the repo's 3D map
   (`../../design-storm-water-system-3d.html`). The 3D page degrades gracefully
   if this folder is absent.

## Viewing

```
python3 serve.py                                  # from the repo root
open http://localhost:8765/teams/explainable-viz/  # 2D suite
open http://localhost:8765/design-storm-water-system-3d  # 3D map
```

For the Prediction tab and its manual Refresh button:

```
cd teams/poc-24h-toc-alk
python3 main.py train --config config.yaml
python3 main.py serve --config config.yaml
open http://localhost:8765/teams/explainable-viz/#prediction
```

The historical views need no network. The prediction refresh uses NOAA Daily
Summaries, which can lag by about two days, and the 3D map streams satellite
imagery, terrain, and live station charts as before.

## What is here

| Path | What it is |
|------|-----------|
| `index.html` | The 2D suite. Hand-maintained; no build step. |
| `build_viz_data.py` | Regenerates everything in `viz-data/` except `basin-flows.json`, from the seven CSVs in `../../data/`, the sonde xlsx, the basin polygons, and the committed SNOTEL metadata/history. Offline. |
| `fetch_basin_flows.py` | One-time fetch (network: USGS daily-values API) of outlet-gage daily flow for the Blue, Fraser, Williams Fork, and Trumbull basins into `viz-data/basin-flows.json`. Not yet run — the shipped CSVs cover only the Strontia mainstem, and both pages show a note until this exists. |
| `viz-data/daily-series.json` | Generated. The seven daily CSVs as one aligned calendar (Apr 2022–Aug 2026): TOC, alkalinity, USGS river quality, flow, gage height, SWE from both snow stations, NOAA precipitation/snow/temperature. |
| `viz-data/sonde.json` | Generated. The 16,093-reading Strontia sonde file as per-day, per-metre-depth grids for seven parameters, plus surface/bottom bands and the stratification index. |
| `viz-data/water-years.json` | Generated. SWE, flow, TOC, alkalinity and turbidity folded per water year (Oct 1 = day 0) with medians, the shape the 3D map's overlays already use. |
| `viz-data/storms.json` | Generated. The top NOAA precipitation events with their flow/turbidity peak offsets, plus a pointer to the committed Aug 14–15 15-minute series. |
| `viz-data/basins.json` | Generated. Per sub-basin: polygon reference, outlet gage, member SNOTEL stations (same point-in-polygon assignment as `build_system.py`), delivery path, behaviour blurb, basin-average SWE folds, and melt markers per water year. |
| `viz-data/basin-flows.json` | *Not committed yet.* Produced by `fetch_basin_flows.py`; see above. |
| `TERMS.md` | Denver Water's data terms. They travel with every derived dataset here (also copied inside `viz-data/`). |

## Data hygiene baked into the bundles

Applied at generation time and disclosed in the pages themselves:

- The USGS dissolved-oxygen −999999 sentinel family is masked.
- Michigan Creek SWE May 12–15 2026 is masked (known NRCS feed error: 9.0
  between zero readings) — including inside the basin averages built from
  `../../water-system-3d/snotel-history.json`.
- The DWR telemetry `Precip` column is excluded (cumulative, reset-prone
  counter); NOAA `PRCP` is the precipitation record used.
- NOAA GHCN publishes about two days behind real time; labelled where shown.
- Everything is stamped provisional: USGS, NRCS, NOAA and DWR revise readings
  after publication.

## Why this data (summary)

| Data | Reason |
|---|---|
| `FoothillsInfluent.csv` | The target and the "why anyone cares" anchor of every view. |
| `USGS_South_Platte.csv` | The only continuous upstream water-quality record; turbidity is the #1 early warning. |
| `SouthPlatteFlow/Telemetry.csv` | Flow scales loading; gage height feeds the storm view. |
| `HoosierPass.csv` (not Michigan Creek) | Jake's replacement station; snowpack sets melt timing. |
| `USC00058022.csv` (not DWR precip) | The clean precipitation record. |
| `Strontia 0407_0819.xlsx` | The only depth-resolved dataset — the reservoir-mixing question. |
| `strontia-brief/` basins & series | Committed USGS geometry and the 15-minute storm, zero new fetching. |
| `water-system-3d/` histories | Folded per-water-year context (medians, "vs normal") for free. |
| New west-slope gage flows | The one gap: each basin's outlet hydrograph; one-time committed fetch. |

## Honest limitations

The pre-registered study in `../poc-24h-toc-alk/` found that at 24 hours
"tomorrow looks like today" is essentially unbeatable with this data, and that
upstream data's value plausibly lives at multi-day horizons. These views build
that multi-day intuition; they do not claim predictive skill. The sonde covers
one location and one partial season. Both pages carry this framing in their
footers and caveats.

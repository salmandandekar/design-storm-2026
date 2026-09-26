# Can Foothills get a 12-hour warning of TOC and alkalinity?

**Short answer: not with the measurements that exist today, and not because the
physics forbids it.** The river physically delivers 11–16 hours of warning for
anything that passes Trumbull, and 2–4 days for anything that starts in the
headwaters. What is missing is an instrument at the right place measuring the
right thing, and a target measured more than once a day so that a 12-hour
forecast could be scored at all.

Every number below is traceable to a file in `sim-data/` or `results/`, or to
the pre-registered study in `teams/poc-24h-toc-alk/`. Each is tagged:

- **[computed]** — derived from the committed data by a script in this folder
- **[assumed]** — a literature-range constant we chose (listed in
  `build_channels.py` `ASSUMED`); change it and rerun
- **[general]** — general hydrology / climate knowledge, not derived from these
  files
- **[Denver Water]** — stated by Denver Water in the materials they sent

All readings are provisional; USGS, NRCS, NOAA and DWR revise after
publication. The interactive version of this memo is the **12-hour
feasibility** tab of `index.html`.

---

## 1. Why 12-hour prediction is not currently feasible

In order of severity.

1. **The target is measured once a day.** [computed] Foothills influent TOC and
   alkalinity are single daily lab values dated by sample day, not hour
   (`sim-data/forcings.json` `series.toc`, `series.alk`). A 12-hour forecast
   has nothing to be scored against. The PoC scoped 12 h out for exactly this
   reason (`teams/poc-24h-toc-alk/README.md`, "12-hour prediction is
   untestable from daily lab data").

2. **Persistence is very hard to beat even at 24 h.** [computed, PoC] Day-to-day
   R² ≈ 0.96 for both targets; median 24-hour change 0.03 mg/L TOC and
   1.1 mg/L alkalinity. Best pooled model skill vs persistence at 24 h: TOC
   −0.09 (95% CI −0.19 to −0.03), alkalinity +0.02 (CI −0.01 to +0.04).
   Upstream sensors without today's lab value are 6× (TOC) and 3× (Alk) worse
   than persistence. A 12-hour forecast would face the same wall with half the
   movement to predict.

3. **Nothing upstream measures dissolved organic carbon.** [computed] The
   sentinel sonde (USGS 06707525) reports turbidity, specific conductance, pH,
   temperature and dissolved oxygen (`sim-data/storm-windows.json`
   parameters 63680, 00095). Turbidity is sediment; conductance is salt.
   Neither is TOC, and the PoC's sonde study found them weak proxies.

4. **Rain is measured at the outlet, a day late, as one number.** [computed]
   The only precipitation record is NOAA COOP station USC00058022,
   "STRONTIA SPRINGS DAM, CO US" at −105.121, 39.434
   (`sim-data/noaa-station.json`): a manual 8-inch can read once a day at the
   dam, published with a ~2-day lag. It sits at the bottom of the basin the
   simulation routes, not where the snow or the convective storms are. Of the
   eight biggest gauge days in the record, three left no turbidity response at
   all and the river's daily-mean peak arrived 0–6 days later
   (`teams/explainable-viz/viz-data/storms.json`).

5. **Side-canyon storms arrive in under four hours.** [computed + Denver Water]
   The reach between Trumbull (06701900) and the sentinel gage has no gauge of
   any kind and drains straight into Strontia. In the Aug 14–15 2026 storm
   Trumbull rose only 12 cfs while sentinel turbidity spiked to 329 FNU
   (`results/travel_time_empirical.csv`, row 2026-08-14): the sediment came
   from the side canyons, not down the mainstem. Denver Water says water at the
   sentinel gage reaches the intake in about four hours
   (`water-system-3d/system.json`). No sensor placement on the mainstem
   changes this class of event.

6. **Provisional data.** [general + computed] Every input series is subject to
   later revision; a forecast trained on revised values and run on unrevised
   ones inherits the difference. `data/MichiganCreek.csv` already contains one
   known feed error (SWE 9.0 on 12–15 May 2026 between zero readings).

7. **Five partial seasons, no winter, one sonde season.** [computed] The
   influent record runs 2022-04-01 to 2026-08-19 (1,602 days,
   `sim-data/forcings.json`) with no January–March data; TOC events cluster in
   2023–24; the sonde covers one partial season. Any model will meet
   conditions it has never seen.

## 2. What the physics allows: the travel-time budget

Manning's equation, applied per reach of the 120-reach, 182 km NHDPlus
mainstem with slopes sampled from the terrain, gives the time for water (and
anything dissolved in it) to reach the sentinel gage
(`sim-data/channels.json` `mainstem.travel`; [assumed] n = 0.045, width 18 m;
[computed] slopes, distances, velocities). Add Denver Water's ~4 h from the
gage to the intake for the time to the plant.

| Flow tier | cfs | Trumbull → gage (27.6 km) | Cheesman → gage (38.8 km) | Headwaters → gage | Trumbull → plant |
|---|---|---|---|---|---|
| low | 202 | 10.6 h | 14.5 h | 3.6 d | ≈ 14.6 h |
| median | 348 | 8.6 h | 11.7 h | 2.9 d | ≈ 12.6 h |
| high | 703 | 6.6 h | 9.0 h | 2.2 d | ≈ 10.6 h |
| Aug 2026 storm peak | 147 | 12.0 h | 16.4 h | 4.1 d | ≈ 16.0 h |

**Empirical check.** [computed] Cross-correlating 15-minute Trumbull discharge
against sentinel turbidity in 13 storm windows (`analysis/01_travel_time.py`,
`results/travel_time_summary.json`) gave only 3 usable windows (Trumbull rose
≥ 20 cfs and r ≥ 0.3): lags of 3, 13 and 5 h (median 5 h) against Manning
estimates of 8.7, 7.0 and 11.5 h at the same flows. Same order of magnitude;
Manning with our assumed roughness runs slow by roughly a factor of two, which
is within the uncertainty of an assumed n and a 355 m terrain grid. The other
ten windows had lags near zero or negative: the turbidity did not come from
Trumbull.

**Reading.** A signal measured at Trumbull reaches the plant in roughly 11–16
hours at the flows on record, so a 12-hour warning is *physically available*
for anything coming down the mainstem past Trumbull, and comfortably available
(13–20 h) from Cheesman's outlet. It is *physically unavailable* for anything
generated in the last 28 km, which is where the largest turbidity spike in the
record came from.

**Reservoir.** [computed] Strontia's nominal residence time V/Q at median
storage 6,966 af is 5 d (high flow) to 17 d (low flow)
(`channels.json` `strontia.residence_days_at_tier`), but [Denver Water] the
raw-water group says gage water reaches the intake in ~4 h and the plant record
tracks the gage 1–2 days later. The reservoir short-circuits along the old
channel; [general] the nominal residence time describes dilution of the bulk,
not arrival of the front. This is why the sonde at the outlet gives hours, not
days.

**Tunnels and conduits.** [computed, assumed] Bernoulli between free surfaces
with Darcy–Weisbach friction (f = 0.02, assumed diameters) gives lower-bound
travel times: Roberts Tunnel 38.2 km, 132.6 m head, ≥ 3.3 h; Moffat 10.0 km,
48 m, ≥ 0.75 h; Conduit 26 6.7 km, 63 m, ≥ 0.4 h
(`channels.json` `conduits`). Valves throttle real flow, so real times are
longer. Blue River water via Roberts joins the North Fork at Grant and reaches
the mainstem confluence at 06707000; no North Fork flowline is committed, so
that leg is drawn as a straight line and its travel time is not computed.

## 3. What would make a 12-hour forecast feasible

Sites are numbered as on the Feasibility tab of the map
(`app.js` `proposedSites()`); leads come from the table above.

| # | Where | What | Lead it buys | Effort |
|---|---|---|---|---|
| 1 | Foothills influent | Online TOC (or UV254/fDOM surrogate) + online alkalinity, ≤ 15-min | 0 h — this is the **target**. Without it 12-hour skill cannot be measured, let alone claimed. | Essential |
| 2 | Sentinel gage 06707525 | Add fDOM/UV254 to the existing sonde | ≈ 4 h to the intake [Denver Water]; 1–2 d in the plant record | High value, low cost: power and telemetry already there |
| 3 | Trumbull 06701900 | Turbidity, SC, fDOM sonde | 6.6–12 h to the gage, ≈ 11–16 h to the plant [computed] | High: the one site that turns mainstem events into a 12-hour warning |
| 4 | Cheesman outlet | Outlet water quality paired with the release schedule | 9–16 h to the gage, 13–20 h to the plant [computed]; [general] summer mainstem flow is largely Cheesman release, and releases are scheduled, so this is the longest *deterministic* lead on the mainstem | Medium |
| 5, 6 | Two side canyons between Trumbull and the gage | Tipping-bucket rain gauges, telemetered | 1–4 h [computed from the ≤ 4 h reach] — a nowcast, not a forecast, but it is the only warning this class of event can give | High for storms, low cost |
| 7 | — | Ingest NEXRAD KFTG radar QPE (already streamed by the main map's storm replay) | 0–6 h nowcast over the whole basin, hourly, not one point a day late | Software only |
| 8 | North Fork at Grant 06702500 | Commit a flowline; add turbidity/SC | unknown until routed; Roberts Tunnel water enters here | Medium |
| 9 | Existing SNOTEL pillows | Ingest the PREC (accumulated precipitation) element, not just SWE | 2–4 d via the river [computed]; fills the "rain where the water is" gap for melt-season storms | Software only |

Beyond hardware:

- **Sample the target sub-daily first.** Nothing else on this list can be
  evaluated until item 1 exists. Even a 6-hourly grab sample for one season
  would let a 12-hour forecast be scored.
- **Score events, not days.** [computed, PoC] Persistence wins on the ~96 % of
  days when nothing changes. Model the *onset* of the rare excursions
  (`teams/poc-24h-toc-alk/results/events_onset.csv`) conditional on an
  upstream trigger, and report hit/miss/false-alarm against an operator cost
  matrix rather than MAE.
- **Reframe to multi-day where the data already help.** [computed, PoC] The
  lead sweep shows upstream information closing on persistence as lead grows
  (persistence MAE triples from lead 1 to 7). 3–7 days is where snowpack and
  flow plausibly win; that is also the headwaters travel time (2–4 d) above.
- **Keep the physics as a constraint, not a model.** The Manning budget says
  which sensor can *in principle* see an event 12 h ahead; it does not say the
  event will move TOC. Only site 1 can answer that.

## 4. El Niño and warming: what the long records say

Both were tested against the multi-decade records already in the repository
rather than asserted (`analysis/02_enso.py`, `analysis/03_trends.py`).

**ENSO.** [general] ENSO shifts the winter storm track; central Colorado sits
between the El Niño-wet southern Rockies and the La Niña-wet northern Rockies,
so its signal there is weak and inconsistent. [computed] Nov–Jan mean ONI
against mean peak SWE over the mapped SNOTEL pillows, 46 water years
(1981–2026): Spearman ρ = 0.018, permutation p = 0.91. Mean peak SWE by phase:
El Niño 17.5 in (n = 15), Neutral 17.0 in (n = 14), La Niña 16.7 in (n = 17).
Of 26 station/reservoir tests in `results/enso_correlations.csv`, none reached
p < 0.05. On this range, in these years, the ENSO phase tells you almost
nothing about the snowpack — and [general] ENSO is at best a seasonal-odds
predictor, not a 12-hour one.

**Warming.** [general] Colorado has warmed roughly 1 °C over the past century
and western snowmelt runoff has shifted earlier by days to weeks (Colorado
Climate Center, USGS assessments; not derived here). [computed] In the mapped
pillows, per-station anomalies pooled 1981–2026 (n = 46): peak SWE Theil–Sen
−0.2 in/decade (Mann–Kendall p = 0.64); peak-SWE day −1.0 d/decade (p = 0.47);
melt-out day +0.01 d/decade (p = 0.96); 1 April SWE +0.04 in/decade
(p = 0.94) — no significant trend (`results/trends.csv`, `sim-data/trends.json`).
A naive pooled mean *did* show −1.3 in/decade, an artefact of low-elevation
pillows joining the network in 1999; the anomaly method removes it. The NOAA
TMAX/TMIN record here is 4.5 years long and says nothing about trends.

[general] The relevant warming mechanism for TOC is not the snowpack total but
wildfire and earlier, hotter dry seasons that prime side canyons to shed
sediment and carbon in the first monsoon storms — which is the ≤ 4 h class of
event in section 2. That argues for sites 5–7 (storm nowcasting) more than for
any seasonal-climate input.

## 5. Bottom line

- **Today:** a 12-hour forecast can be neither built nor scored. Persistence
  is the forecast, and the PoC shows nothing in the existing data beats it.
- **Physically:** 11–16 hours of warning exists on the mainstem past Trumbull
  and 13–20 hours from Cheesman; under four hours for side-canyon storms, and
  no sensor changes that.
- **To make it feasible, in order:** sub-daily TOC/alkalinity at the plant
  (1); fDOM at the sentinel (2) and Trumbull (3); side-canyon rain gauges and
  radar QPE (5–7) for the storm class that a river sensor cannot warn of; then
  event-conditional scoring against an operator cost matrix.
- **ENSO and warming** are context for the season, not inputs to a 12-hour
  forecast; in these records neither shows a usable signal.

---

*Denver Water's data terms travel with this memo (`TERMS.md`). Readings are
provisional. Generated numbers come from the scripts listed in `README.md`;
rerun them rather than editing values here.*

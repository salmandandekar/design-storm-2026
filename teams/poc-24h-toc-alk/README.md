# Can Foothills see TOC and alkalinity changes coming 24 hours ahead?

A statistically defensible proof of concept testing whether Denver Water can
predict an **actionable TOC or alkalinity change** at the Foothills WTP
influent **~24 hours in advance** from the data in this repository. The
hypothesis was tested to be falsified, not confirmed; the decision rule was
written and committed (`results/decision_rule.json`, commit `483cad2`) before
any model result existed (first results commit `e45ebf2`).

**Denver Water's data terms apply to everything here** (see `TERMS.md`, also
copied into `derived/`). All readings are provisional — USGS revises after
publication — and no number below should be treated as settled.

## Verdict

| Question (pre-registered endpoint) | TOC | Alkalinity |
|---|---|---|
| 24 h change skill: does any model beat the persistence forecast "tomorrow = today"? | **Not supported.** Best pooled skill −0.09 (95% CI −0.19 to −0.03): significantly *worse* than persistence. | **Not supported.** Best pooled skill +0.02 (95% CI −0.01 to +0.04, DM p = 0.10): indistinguishable from persistence. |
| 24 h event-onset warning (entering TOC > 3 / Alk < 60 when not there today) | **Insufficient evidence.** Only 4 pooled test onsets (< 20 required by the pre-registered rule). | **Supported, with a false-alarm cost.** Balanced RF: recall 0.87 (CI 0.70–0.95), precision 0.33 (CI 0.23–0.43) vs base rate 0.11 — both criteria met, but 2 of 3 warnings are false alarms. |
| Can upstream sensors alone (no day-T lab value) match what the operator already knows? | **No.** Soft-sensor MAE 0.44–0.46 mg/L vs persistence 0.073 (≈6× worse). | **No.** Soft-sensor MAE 5.3 vs persistence 1.57 (≈3× worse). |

**Plain-language summary.** At 24 hours, both targets are so persistent
(day-to-day R² ≈ 0.96, median 24 h change 0.03 mg/L TOC / 1.1 mg/L Alk) that
"tomorrow will look like today" is close to unbeatable with this data, and
every model we or Jake's feature sets could build confirms it. The genuinely
useful 24 h signal found is a **low-alkalinity onset flag** that catches
~87% of entries below 60 mg/L at triple the base-rate precision — usable if
operators tolerate two false alarms per real warning. Upstream data does
*not* become competitive with persistence anywhere out to 7 days, but the
gap closes steadily with lead (persistence MAE grows ~3× from lead 1 to 7
while the soft-sensor error stays flat), which is consistent with Jake's
view that this data's value lives at multi-day, not 24-hour, horizons.
**12-hour prediction is untestable from daily lab data** and was scoped out.

## What was tested

- **H1:** upstream watershed/hydrologic/reservoir signals available at issue
  time T give statistically significant skill beyond naive baselines for an
  actionable TOC/alkalinity change at T+24 h.
- **H0:** at 24 h the targets are persistence-dominated and actionable
  changes are not predictable from this data.

Jake's soft sensors answer a different question (estimate today's value
without lab history at 2–4-day lags); his R² = 0.66–0.74 is context, not the
bar. The bar here is the operator's own "no change" expectation.

## Method in one paragraph

One leakage-safe daily feature frame (`src/features.py`): every column at
date T is computable from data timestamped ≤ T (NOAA shifted 2 days for its
publication latency; join/window/label spot checks in
`analysis/02_build_datasets.py`). Anchored evaluation: all feature sets are
scored on the same 835 days (lab at T and T+1 plus full set-D features
present). Rolling-origin folds by season (train < test year; test 2023,
2024, 2025, 2026; never shuffled), pooled with moving-block bootstrap CIs
(14-day blocks), Diebold–Mariano tests (HLN-corrected) against persistence,
Wilson intervals on event metrics, and a novelty guardrail (features outside
the train 5th–95th percentiles). Models: Ridge / random forest / gradient
boosting on the 24 h *change* (lab-known scenario) or level (lab-blind),
grid-searched with `TimeSeriesSplit` inside the training window only. Events
use Jake's operational thresholds (TOC > 3, Alk < 60 mg/L — round numbers to
confirm with operators).

## The numbers

### Baselines (pooled over test seasons, lead 1; `results/baselines_lead1.csv`)

| Baseline | TOC MAE (mg/L) | Alk MAE (mg/L) |
|---|---|---|
| Persistence ŷ(T+1)=y(T) | **0.073** | **1.570** |
| Persistence + 3-day drift | 0.119 | 2.394 |
| Seasonal climatology | 0.725 | 8.739 |

### Ablation ladder at lead 1 (best model per set; skill vs persistence, `results/ablation_lead1.csv`)

| Set (cumulative) | TOC skill [95% CI] | Alk skill [95% CI] |
|---|---|---|
| A lab history only | −0.09 [−0.13, −0.04] | +0.01 [−0.02, +0.03] |
| A+B +USGS gage | −0.10 [−0.21, −0.03] | +0.02 [−0.01, +0.04] |
| A+B+C +DWR flow | −0.12 [−0.26, −0.04] | **+0.02 [−0.01, +0.04]** (p = 0.10) |
| A+B+C+D +snow/weather/season | **−0.09 [−0.19, −0.03]** | +0.02 [−0.01, +0.04] |
| Lab-blind best (B/BC/BCD) | −5.3 [−7.5, −3.6] | −2.3 [−3.2, −1.7] |
| Jake-equivalent feature set (RF, lag 1) | −5.0 [−7.0, −3.5] | −2.9 [−3.8, −2.3] |

No set at any rung beats persistence; adding upstream data never helps at
24 h. Regime folds (wet→dry / dry→wet, `results/regime_lead1.csv`) agree:
alkalinity's best combo is +0.03 in one direction, −0.02 in the other.

### Lead-time sweep (best model per lead; `results/lead_sweep.csv`, figure below)

Persistence MAE grows with lead (TOC 0.073 → 0.233; Alk 1.57 → 4.12 mg/L at
lead 7) while lab-blind soft-sensor MAE stays roughly flat (TOC ≈ 0.45–0.47;
Alk ≈ 5.2–5.4). Lab-known model skill is never significantly positive at any
lead 1–7 (best points: TOC +0.04 at lead 7, p = 0.65; Alk +0.05 at lead 3,
p = 0.14). The crossover where upstream data would beat persistence lies
beyond 7 days for both targets in this record.

### Event onsets at 24 h (pooled test seasons; `results/events_onset.csv`)

| Target | Method | Eligible days | Onsets | Recall [CI] | Precision [CI] | Base rate |
|---|---|---|---|---|---|---|
| TOC > 3 | best (RF regressor thresholded) | 516 | **4** | 0.75 [0.30, 0.95] | 0.25 [0.09, 0.53] | 0.008 |
| Alk < 60 | balanced RF classifier | 271 | 30 | **0.87 [0.70, 0.95]** | **0.33 [0.23, 0.43]** | 0.11 |

TOC: 4 onsets < 20 → **insufficient evidence** by the pre-registered rule
(point estimates shown for transparency only). Alkalinity: both secondary
criteria met (Wilson lower bound 0.23 > base rate 0.11; recall ≥ 0.5), at
the cost of 80 flags for 26 true onsets. Episode view
(`results/episodes_*.csv`): 4 of 7 TOC episodes and 31 of 38 alkalinity
episodes in the test seasons got a warning within 2 days of onset — but the
big 2023/2024 TOC episodes were flagged only on their onset day (lead 0 on
top of the built-in 24 h), and 12 TOC / 33 Alk flagged days were false.

### Novelty guardrail (`results/novelty.csv`)

TOC error rises with novelty (MAE 0.059 → 0.104 from 0 to >5 novel
features; Spearman ρ = 0.11, p = 0.006): the model degrades exactly
where warnings matter most (unseen hydrologic corners). Alkalinity shows no
significant trend (p = 0.44).

### Strontia sonde addendum (2026 overlap; `results/sonde_study.csv`, `results/sonde_correlations.csv`)

On the 104-day 2026 overlap (expanding split within 2026, anchored days):
adding daily depth-band sonde features to set D changes nothing (TOC skill
−0.003 → −0.003; Alk −0.03 → −0.03). The sonde correlates with the influent
(bottom-water temperature vs TOC ρ ≈ −0.62 at lag 0; signals persist to
lag 5) but is redundant with what the gage and season already encode, on
this sample. **The sonde's event-warning value cannot be established yet**:
2026 has zero TOC > 3 days and one partial season. Worth revisiting after a
full spring flush and at least one high-TOC season of casts; the
2026-onward record is the right foundation for the multi-day reservoir
mixing question, where this analysis says the real predictive opportunity
lives.

## Honest limitations

- **Winter gap is structural**: no Jan–Mar influent data; all claims are
  April–December only.
- **Five partial seasons, one regime each**: TOC events cluster in 2023–24;
  none in 2026. Event statistics carry the wide CIs shown.
- **Provisional data**: a fresh API pull can differ from these CSVs.
- **Thresholds are round numbers** (TOC 3, Alk 60): confirm with operators
  what change magnitude is actually actionable; the alkalinity flag's value
  depends entirely on the real cost ratio of false alarms to misses.
- **Single-point proxies**: one SNOTEL pillow, one NOAA station, one gage;
  DWR's precip counter excluded per Jake's guidance.
- **12/24 h framing**: daily grab samples cannot test sub-daily leads, and
  the recorded lab timestamp is the sample date, not collection time.

## What would change the answer

1. **Sub-daily influent measurement** (even 2–3 samples/day, or an online
   TOC/alkalinity analyzer) — the only way to test 12–24 h dynamics
   properly. 2. **More seasons of sonde casts** spanning a high-TOC year.
3. **Storm-resolved sampling**: the rare large 24 h changes are storm-driven;
   15-minute USGS pulls around events (grabber notebooks in `scripts/`)
   would support an event-conditional model. 4. **Operator thresholds**: a
   cost matrix for false alarm vs miss would turn the alkalinity flag into a
   tunable, defensible tool. 5. **Aim at multi-day leads**: the lead sweep
   shows upstream data approaches persistence as lead grows — 3–7 day
   warnings (Jake's regime) are where this data plausibly wins.

## Reproduce

```bash
pip install -r requirements.txt   # from teams/poc-24h-toc-alk/
python3 analysis/01_audit.py          # audit table + coverage figure
python3 analysis/02_build_datasets.py # derived frames + leakage spot checks
python3 analysis/03_baselines.py      # the bars (validates the harness)
python3 analysis/04_ablation.py       # ~15 min: ablation, regimes, predictions
python3 analysis/05_lead_sweep.py     # ~15 min: leads 1,2,3,4,7
python3 analysis/06_events.py         # onsets, episodes, novelty, timelines
python3 analysis/07_sonde.py          # 2026 sonde study + lag correlations
```

Deterministic (seed 42 throughout); originals in `data/` are never written.
Layout: `src/` (loaders, features, baselines, models, evaluation, events),
`analysis/` (numbered scripts), `results/` (every number above),
`figures/` (`audit_coverage.png`, `ablation_lead1.png`, `lead_sweep.png`,
`event_timeline_{toc,alk}.png`), `derived/` (feature frames + terms).

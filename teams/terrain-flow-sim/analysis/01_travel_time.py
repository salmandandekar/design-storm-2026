#!/usr/bin/env python3
"""Empirical Trumbull -> Strontia travel time from 15-minute storm windows,
compared with the Manning estimate in sim-data/channels.json. Offline.

    python3 analysis/01_travel_time.py        # from teams/terrain-flow-sim/

Reads sim-data/storm-windows.json (fetch_storm_windows.py) and
sim-data/channels.json (build_channels.py). Writes
results/travel_time_empirical.csv (one row per storm window) and
results/travel_time_summary.json.

Two estimates per window:
  peak_lag_h   time from the Trumbull discharge peak to the sentinel turbidity
               peak (and to the conductance minimum, which dilution produces)
  xcorr_lag_h  lag maximising the cross-correlation of the hourly first
               differences of Trumbull discharge and sentinel turbidity within
               +/- 36 h; sign positive means the sentinel follows Trumbull.
Both are descriptive. In monsoon storms sediment often enters from side
canyons between the two gauges (Denver Water's note on the Aug 2026 storm in
system.json), so a short or negative lag says "the turbidity did not come
from Trumbull", not "water travelled instantly".
"""

import csv
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM = os.path.join(HERE, "sim-data")
RES = os.path.join(HERE, "results")
MAX_LAG_H = 36


def to_series(pairs):
    if not pairs:
        return None
    s = pd.Series([v for _, v in pairs], index=pd.to_datetime([t for t, _ in pairs]))
    s = s[~s.index.duplicated()].sort_index()
    return s.resample("15min").mean().interpolate(limit=8)


def xcorr_lag(a, b, max_lag_h):
    """Lag (hours) of b relative to a maximising correlation of hourly diffs."""
    ah = a.resample("1h").mean().diff().dropna()
    bh = b.resample("1h").mean().diff().dropna()
    idx = ah.index.intersection(bh.index)
    if len(idx) < 24:
        return None, None
    ah, bh = ah.loc[idx], bh.loc[idx]
    best, best_r = None, -2
    for lag in range(-max_lag_h, max_lag_h + 1):
        bs = bh.shift(-lag)  # b at t+lag
        m = ah.notna() & bs.notna()
        if m.sum() < 24 or ah[m].std() == 0 or bs[m].std() == 0:
            continue
        r = np.corrcoef(ah[m], bs[m])[0, 1]
        if r > best_r:
            best_r, best = r, lag
    return best, (round(float(best_r), 3) if best is not None else None)


def manning_hours_for_flow(channels, q_cfs):
    """Interpolate the Manning Trumbull->gage hours across the flow tiers."""
    pts = sorted((t["flow_cfs"], t["hours_landmark_to_gage"].get("gage-06701900"))
                 for t in channels["mainstem"]["travel"].values()
                 if t["hours_landmark_to_gage"].get("gage-06701900") is not None)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return float(np.interp(q_cfs, xs, ys))


def main():
    sw = json.load(open(os.path.join(SIM, "storm-windows.json")))
    ch = json.load(open(os.path.join(SIM, "channels.json")))
    rows = []
    for w in sw["windows"]:
        s = w["series"]
        q = to_series(s.get("06701900:discharge_cfs"))
        turb = to_series(s.get("06707525:turbidity_fnu"))
        cond = to_series(s.get("06707525:spec_cond_uscm"))
        row = {"date": w["date"], "why": w["why"]}
        if q is None:
            rows.append(row)
            continue
        row["q_peak_cfs"] = round(float(q.max()), 1)
        row["q_peak_time"] = q.idxmax().isoformat(timespec="minutes")
        row["q_rise_cfs"] = round(float(q.max() - q.iloc[:max(4, len(q) // 6)].median()), 1)
        row["manning_hours_at_peak"] = round(manning_hours_for_flow(ch, float(q.max())), 1)
        if turb is not None and turb.notna().sum() > 48:
            row["turb_peak_fnu"] = round(float(turb.max()), 1)
            row["turb_peak_time"] = turb.idxmax().isoformat(timespec="minutes")
            row["peak_lag_turb_h"] = round((turb.idxmax() - q.idxmax()).total_seconds() / 3600, 1)
            lag, r = xcorr_lag(q, turb, MAX_LAG_H)
            row["xcorr_lag_turb_h"], row["xcorr_r_turb"] = lag, r
        if cond is not None and cond.notna().sum() > 48:
            row["cond_min_time"] = cond.idxmin().isoformat(timespec="minutes")
            row["peak_lag_cond_h"] = round((cond.idxmin() - q.idxmax()).total_seconds() / 3600, 1)
            lag, r = xcorr_lag(q, -cond, MAX_LAG_H)
            row["xcorr_lag_cond_h"], row["xcorr_r_cond"] = lag, r
        rows.append(row)

    os.makedirs(RES, exist_ok=True)
    cols = ["date", "why", "q_peak_cfs", "q_rise_cfs", "q_peak_time", "manning_hours_at_peak",
            "turb_peak_fnu", "turb_peak_time", "peak_lag_turb_h", "xcorr_lag_turb_h", "xcorr_r_turb",
            "cond_min_time", "peak_lag_cond_h", "xcorr_lag_cond_h", "xcorr_r_cond"]
    with open(os.path.join(RES, "travel_time_empirical.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow({c: r.get(c, "") for c in cols})

    # Summary: windows where Trumbull actually rose (a real upstream pulse) and the
    # turbidity cross-correlation is meaningful (r >= 0.3).
    usable = [r for r in rows if r.get("q_rise_cfs", 0) >= 20 and (r.get("xcorr_r_turb") or 0) >= 0.3]
    lags = [r["xcorr_lag_turb_h"] for r in usable]
    manning = [r["manning_hours_at_peak"] for r in usable]
    summary = {
        "_comment": "Generated by analysis/01_travel_time.py. Do not edit.",
        "n_windows": len(rows),
        "n_usable": len(usable),
        "usable_rule": "Trumbull discharge rose >= 20 cfs within the window and turbidity cross-correlation r >= 0.3",
        "xcorr_lag_turb_h": {"values": lags, "median": float(np.median(lags)) if lags else None,
                             "min": min(lags) if lags else None, "max": max(lags) if lags else None},
        "manning_hours_at_same_flows": {"values": manning, "median": float(np.median(manning)) if manning else None},
        "denver_water_quote_gage_to_intake_h": 4,
        "reading": ("If the empirical lag sits near the Manning figure, the sentinel is seeing Trumbull water and the "
                    "channel physics is a fair travel-time model. Lags near zero or negative mean the turbidity was "
                    "generated between the gauges (side canyons), which no upstream gauge can warn of."),
        "windows": [{k: r.get(k) for k in ("date", "q_rise_cfs", "manning_hours_at_peak", "peak_lag_turb_h",
                                             "xcorr_lag_turb_h", "xcorr_r_turb", "xcorr_lag_cond_h", "xcorr_r_cond")}
                    for r in rows],
    }
    with open(os.path.join(RES, "travel_time_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"{len(rows)} windows, {len(usable)} usable; empirical median lag "
          f"{summary['xcorr_lag_turb_h']['median']} h vs Manning median {summary['manning_hours_at_same_flows']['median']} h")
    for r in rows:
        print(f"  {r['date']} rise {r.get('q_rise_cfs')} cfs  manning {r.get('manning_hours_at_peak')} h  "
              f"peak-lag turb {r.get('peak_lag_turb_h')} h  xcorr {r.get('xcorr_lag_turb_h')} h (r={r.get('xcorr_r_turb')})  "
              f"cond xcorr {r.get('xcorr_lag_cond_h')} h (r={r.get('xcorr_r_cond')})")


if __name__ == "__main__":
    main()

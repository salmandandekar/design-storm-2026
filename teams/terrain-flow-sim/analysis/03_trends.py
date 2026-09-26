#!/usr/bin/env python3
"""Long-record trends: is the snowpack and its timing changing? Offline.

    python3 analysis/03_trends.py        # from teams/terrain-flow-sim/

Reads the folded SNOTEL history in ../../water-system-3d/snotel-history.json
(back to water year 1980 for the oldest pillows), the DWR storage history,
and the NOAA daily temperatures in ../explainable-viz/viz-data/
daily-series.json. Writes results/trends.csv and sim-data/trends.json.

For each station and metric (peak SWE, day of peak, melt-out day, April 1
SWE) it reports the Theil-Sen slope per decade and the Mann-Kendall test
(two-sided, normal approximation with tie correction), n years, and the
first/last year. These are standard non-parametric trend tools (general
knowledge); the numbers are computed from the committed records.

The NOAA air-temperature record here spans only 2022-2026: the script prints
its mean and range for completeness and labels it far too short for any
climate statement. Regional warming (about +1 C over the last century in
Colorado, earlier melt across the West) is general knowledge from NOAA/
Colorado Climate Center assessments and is stated as such in the page.
"""

import csv
import json
import math
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W3D = os.path.join(HERE, "..", "..", "water-system-3d")
VIZ = os.path.join(HERE, "..", "explainable-viz", "viz-data")
SIM = os.path.join(HERE, "sim-data")
RES = os.path.join(HERE, "results")

APRIL1_DOY = 182  # Oct 1 = 0 (non-leap count; Apr 1 is day 182 or 183)


def mann_kendall(y):
    n = len(y)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += (y[j] > y[i]) - (y[j] < y[i])
    # tie correction
    counts = {}
    for v in y:
        counts[v] = counts.get(v, 0) + 1
    ties = sum(t * (t - 1) * (2 * t + 5) for t in counts.values() if t > 1)
    var = (n * (n - 1) * (2 * n + 5) - ties) / 18
    if var <= 0:
        return 0.0, 1.0
    z = (s - 1) / math.sqrt(var) if s > 0 else (s + 1) / math.sqrt(var) if s < 0 else 0.0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return round(z, 2), round(p, 4)


def theil_sen(x, y):
    slopes = []
    for i in range(len(x) - 1):
        for j in range(i + 1, len(x)):
            if x[j] != x[i]:
                slopes.append((y[j] - y[i]) / (x[j] - x[i]))
    slopes.sort()
    m = len(slopes)
    return slopes[m // 2] if m % 2 else (slopes[m // 2 - 1] + slopes[m // 2]) / 2


def metrics_for(trace):
    vals = [(v, i) for i, v in enumerate(trace) if v is not None]
    if len(vals) < 250:
        return None
    peak, pi = max(vals)
    if peak <= 0:
        return None
    melt_out = next((i for v, i in vals if i > pi and v <= 0.1), None)
    apr1 = next((v for v, i in vals if i in (APRIL1_DOY, APRIL1_DOY + 1)), None)
    return {"peak_swe_in": peak, "peak_doy": pi, "melt_out_doy": melt_out, "april1_swe_in": apr1}


def trend_rows(label, series_by_year, metric_names, unit_scale=10):
    rows = []
    for metric in metric_names:
        pts = [(wy, m[metric]) for wy, m in sorted(series_by_year.items()) if m and m.get(metric) is not None]
        if len(pts) < 15:
            continue
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        z, p = mann_kendall(y)
        slope = theil_sen(x, y)
        rows.append({"series": label, "metric": metric, "n_years": len(pts), "first_wy": x[0], "last_wy": x[-1],
                     "theil_sen_per_decade": round(slope * unit_scale, 2), "mk_z": z, "mk_p": p,
                     "mean": round(sum(y) / len(y), 1)})
    return rows


def main():
    snotel = json.load(open(os.path.join(W3D, "snotel-history.json")))
    storage = json.load(open(os.path.join(W3D, "storage-history.json")))
    system = json.load(open(os.path.join(W3D, "system.json")))
    daily = json.load(open(os.path.join(VIZ, "daily-series.json")))
    names = {p["id"].replace("sntl-", "") + ":CO:SNTL": p["name"] for p in system["points"] if p["kind"] == "snotel"}

    rows = []
    per_station = {}
    for trip, st in snotel["stations"].items():
        by = {int(wy): metrics_for(tr) for wy, tr in st["years"].items()}
        by = {k: v for k, v in by.items() if v}
        per_station[trip] = by
        rows += trend_rows(names.get(trip, trip), by, ["peak_swe_in", "peak_doy", "melt_out_doy", "april1_swe_in"])
    # pooled: mean of per-station ANOMALIES (value minus that station's own
    # mean), so stations added later (lower, drier pillows from 1999 on) shift
    # the level but not the trend. A plain pooled mean would fake a decline.
    station_means = {}
    for trip, by in per_station.items():
        station_means[trip] = {}
        for k in ("peak_swe_in", "peak_doy", "melt_out_doy", "april1_swe_in"):
            vals = [m[k] for m in by.values() if m.get(k) is not None]
            station_means[trip][k] = sum(vals) / len(vals) if vals else None
    pooled = {}
    years = sorted({wy for by in per_station.values() for wy in by})
    for wy in years:
        ms = [(trip, by[wy]) for trip, by in per_station.items() if wy in by]
        if len(ms) >= 3:
            pooled[wy] = {}
            for k in ("peak_swe_in", "peak_doy", "melt_out_doy", "april1_swe_in"):
                an = [m[k] - station_means[trip][k] for trip, m in ms
                      if m.get(k) is not None and station_means[trip][k] is not None]
                pooled[wy][k] = sum(an) / len(an) if an else None
    rows += trend_rows("all mapped SNOTEL (mean anomaly)", pooled, ["peak_swe_in", "peak_doy", "melt_out_doy", "april1_swe_in"])
    # Also the long-record stations only (data from WY1981), plain mean, as a cross-check.
    long_trips = [t for t, by in per_station.items() if min(by) <= 1981]
    pooled_long = {}
    for wy in years:
        ms = [per_station[t][wy] for t in long_trips if wy in per_station[t]]
        if len(ms) >= max(3, len(long_trips) - 1):
            pooled_long[wy] = {k: (sum(m[k] for m in ms if m[k] is not None) / sum(1 for m in ms if m[k] is not None))
                               if any(m[k] is not None for m in ms) else None
                               for k in ("peak_swe_in", "peak_doy", "melt_out_doy", "april1_swe_in")}
    rows += trend_rows(f"{len(long_trips)} pillows with records from WY1981 (mean)", pooled_long,
                       ["peak_swe_in", "peak_doy", "melt_out_doy", "april1_swe_in"])

    for code, res in storage["reservoirs"].items():
        by = {}
        for wy, tr in res["years"].items():
            vals = [v for v in tr if v is not None]
            if len(vals) >= 250:
                by[int(wy)] = {"peak_storage_af": max(vals), "min_storage_af": min(vals)}
        rows += trend_rows(code, by, ["peak_storage_af", "min_storage_af"])

    tmax = [v for v in daily["series"]["tmax"] if v is not None]
    tmin = [v for v in daily["series"]["tmin"] if v is not None]
    noaa = {"station": "USC00058022", "years_covered": f"{daily['start'][:4]}-{daily['end'][:4]}",
            "n_days": len(tmax), "tmax_mean_f": round(sum(tmax) / len(tmax), 1), "tmin_mean_f": round(sum(tmin) / len(tmin), 1),
            "verdict": "4.5 years: far too short for any climate trend; shown only so nobody has to wonder whether it was checked."}

    os.makedirs(RES, exist_ok=True)
    cols = ["series", "metric", "n_years", "first_wy", "last_wy", "theil_sen_per_decade", "mk_z", "mk_p", "mean"]
    with open(os.path.join(RES, "trends.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    pooled_rows = [r for r in rows if r["series"].startswith("all mapped SNOTEL") or "from WY1981" in r["series"]]
    out = {"_comment": "Generated by analysis/03_trends.py. Theil-Sen slope per decade and Mann-Kendall test per station/metric. Do not edit.",
           "terms": "Public NRCS/DWR/NOAA records alongside Denver Water Design Storm data; Denver Water's terms travel with it (TERMS.md).",
           "pooled": pooled_rows, "rows": rows, "noaa_temperature": noaa,
           "pooled_series": {str(k): v for k, v in pooled.items()},
           "general_knowledge": [
               "Colorado has warmed roughly 1 C (2 F) over the past century and snowmelt runoff across the West has shifted earlier by days to weeks (Colorado Climate Center, USGS); these are external assessments, not results of this repository.",
               "Warmer springs mean more of the snowpack leaves as earlier melt and as sublimation, which changes when TOC-rich soil water is flushed, not whether it is.",
           ]}
    with open(os.path.join(SIM, "trends.json"), "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"{len(rows)} trend rows written")
    for r in pooled_rows:
        print(f"  {r['series']} {r['metric']}: {r['theil_sen_per_decade']} per decade, MK p={r['mk_p']}, n={r['n_years']} ({r['first_wy']}-{r['last_wy']})")
    sig = [r for r in rows if r["mk_p"] < 0.05]
    print(f"  {len(sig)} of {len(rows)} station/metric tests significant at 0.05")
    print(f"  NOAA temps: {noaa}")


if __name__ == "__main__":
    main()

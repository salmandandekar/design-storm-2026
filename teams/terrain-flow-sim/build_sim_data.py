#!/usr/bin/env python3
"""Build the daily forcing bundle behind the terrain flow simulation. Offline.

Reads only committed files: the aligned daily calendar in
../explainable-viz/viz-data/daily-series.json, the per-basin SNOTEL membership
in ../explainable-viz/viz-data/basins.json, and the folded SNOTEL history in
../../water-system-3d/snotel-history.json. Writes sim-data/forcings.json.

    python3 build_sim_data.py

What comes out (all generated, do not edit by hand):
    forcings.json
      dates                 2022-04-01 .. 2026-08-19, one entry per day
      series                the daily inputs the simulation is driven by:
                            prcp/snow/tmax/tmin (NOAA USC00058022), flow and
                            gage height (DWR PLASPLCO), turbidity/conductance
                            (USGS 06707525), TOC/alkalinity (Foothills)
      derived.degree_days_c positive-degree-days, ((tmax+tmin)/2 in C)+, one
                            value per day, from the NOAA station
      basins[id].swe_in     basin-mean SWE from the member SNOTEL pillows
      basins[id].melt_in    daily SWE loss (max(0, SWE[t-1]-SWE[t])), the
                            snowmelt input the simulation seeds particles from
      basins[id].input_in   melt + prcp (a single daily "liquid input" proxy;
                            the same NOAA gauge is used for every basin because
                            it is the only precipitation record in the data)
      melt_fit              degree-day melt factor fitted to Hoosier Pass:
                            melt_in_per_day = k * degree_days_c, least squares
                            through the origin over melt-season days, with n,
                            r and the residual spread. COMPUTED from these
                            files; the classic degree-day form itself is
                            general snow hydrology.

Hygiene inherited from build_viz_data.py: the DWR cumulative precip counter
is not used, the Michigan Creek May 12-15 2026 SWE error is masked inside the
basin averages, and every reading is provisional.
"""

import json
import math
import os
from datetime import date, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
VIZ = os.path.join(HERE, "..", "explainable-viz", "viz-data")
W3D = os.path.join(HERE, "..", "..", "water-system-3d")
OUT = os.path.join(HERE, "sim-data", "forcings.json")

TERMS_NOTE = ("Derived from Denver Water Design Storm data; Denver Water's data "
              "terms apply (TERMS.md in this folder). All readings are provisional "
              "and subject to revision by the source agencies.")

MICHIGAN_CREEK = "937:CO:SNTL"
MICHIGAN_MASK = {date(2026, 5, d).isoformat() for d in range(12, 16)}


def load(path):
    with open(path) as fh:
        return json.load(fh)


def water_year_and_day(d):
    wy = d.year + 1 if d.month >= 10 else d.year
    return wy, (d - date(wy - 1, 10, 1)).days


def station_swe_on(hist, triplet, d):
    st = hist["stations"].get(triplet)
    if not st:
        return None
    wy, i = water_year_and_day(d)
    years = st["years"].get(str(wy))
    if not years or i >= len(years):
        return None
    v = years[i]
    if triplet == MICHIGAN_CREEK and d.isoformat() in MICHIGAN_MASK:
        return None
    return v


def basin_mean_swe(hist, members, dates):
    out = []
    for d in dates:
        vals = [station_swe_on(hist, m["triplet"], d) for m in members]
        vals = [v for v in vals if v is not None]
        out.append(round(sum(vals) / len(vals), 2) if vals else None)
    return out


def melt_from_swe(swe):
    melt = [None]
    for prev, cur in zip(swe, swe[1:]):
        if prev is None or cur is None:
            melt.append(None)
        else:
            melt.append(round(max(0.0, prev - cur), 2))
    return melt


def degree_days(tmax, tmin):
    out = []
    for hi, lo in zip(tmax, tmin):
        if hi is None or lo is None:
            out.append(None)
        else:
            mean_c = ((hi + lo) / 2 - 32) * 5 / 9
            out.append(round(max(0.0, mean_c), 2))
    return out


def fit_melt_factor(melt, dd, swe):
    """Least-squares slope through the origin of melt on degree-days, on days
    with snow on the ground, decreasing SWE, and positive degree-days."""
    xs, ys = [], []
    for m, x, s in zip(melt, dd, swe):
        if m is None or x is None or s is None:
            continue
        if s <= 0 or m <= 0 or x <= 0:
            continue
        xs.append(x)
        ys.append(m)
    n = len(xs)
    if n < 10:
        return {"n": n, "note": "too few melt days to fit"}
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    k = sxy / sxx
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    r = cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else None
    resid = [y - k * x for x, y in zip(xs, ys)]
    rmse = math.sqrt(sum(e * e for e in resid) / n)
    return {
        "k_in_per_degree_day_c": round(k, 4),
        "k_mm_per_degree_day_c": round(k * 25.4, 2),
        "n_days": n,
        "pearson_r": round(r, 3) if r is not None else None,
        "rmse_in_per_day": round(rmse, 3),
        "station": "Hoosier Pass SNOTEL (531:CO:SNTL)",
        "air_temperature_from": "NOAA USC00058022 daily TMAX/TMIN, the only air record in the data; it sits far below the pillow, so k absorbs the lapse rate",
        "note": ("Fitted here from committed data. The degree-day form "
                 "melt = k * PDD is standard snow hydrology (general knowledge); "
                 "textbook k for open sites is roughly 3-6 mm per degree-day C."),
    }


def main():
    daily = load(os.path.join(VIZ, "daily-series.json"))
    basins = load(os.path.join(VIZ, "basins.json"))
    hist = load(os.path.join(W3D, "snotel-history.json"))

    dates_iso = daily["dates"]
    dates = [date.fromisoformat(s) for s in dates_iso]
    s = daily["series"]

    dd = degree_days(s["tmax"], s["tmin"])
    prcp = s["prcp"]

    out_basins = {}
    for b in basins["basins"]:
        swe = basin_mean_swe(hist, b["snotel_members"], dates)
        melt = melt_from_swe(swe)
        inp = [None if (m is None and p is None) else round((m or 0) + (p or 0), 2)
               for m, p in zip(melt, prcp)]
        out_basins[b["id"]] = {
            "name": b["name"],
            "outlet_gage": b["outlet_gage"],
            "n_snotel": len(b["snotel_members"]),
            "swe_in": swe,
            "melt_in": melt,
            "input_in": inp,
        }

    hoosier = [station_swe_on(hist, "531:CO:SNTL", d) for d in dates]
    melt_fit = fit_melt_factor(melt_from_swe(hoosier), dd, hoosier)

    bundle = {
        "_comment": "Generated by build_sim_data.py. Do not edit by hand.",
        "terms": TERMS_NOTE,
        "start": daily["start"],
        "end": daily["end"],
        "dates": dates_iso,
        "series": {k: s[k] for k in ("prcp", "snow", "tmax", "tmin", "flow",
                                      "gage_height", "turb", "cond", "temp",
                                      "toc", "alk", "swe_hoosier")},
        "labels": {k: daily["labels"][k] for k in daily["labels"]},
        "derived": {"degree_days_c": dd},
        "basins": out_basins,
        "melt_fit": melt_fit,
        "assumptions": [
            "One NOAA gauge (USC00058022) stands in for rainfall over every basin; "
            "there is no other precipitation record in the shipped data.",
            "Basin SWE is the plain mean of member pillows; no elevation weighting.",
            "Daily melt is SWE loss between consecutive readings; sublimation and "
            "rain-on-snow are folded into it.",
        ],
        "hygiene_notes": daily["hygiene_notes"],
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(bundle, fh, separators=(",", ":"))
    print(f"wrote {os.path.relpath(OUT, HERE)}: {len(dates_iso)} days, "
          f"{len(out_basins)} basins, melt k = {melt_fit.get('k_mm_per_degree_day_c')} mm/degC-day "
          f"(n={melt_fit.get('n_days')}, r={melt_fit.get('pearson_r')})")


if __name__ == "__main__":
    main()

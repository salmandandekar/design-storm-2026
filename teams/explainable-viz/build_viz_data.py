#!/usr/bin/env python3
"""Build the JSON bundles behind the explainable 2D/3D visualizations. Offline.

Reads only committed files: the seven daily CSVs and the Strontia sonde xlsx in
../../data/, the basin polygons in ../../strontia-brief/basins/, the SNOTEL
station metadata and folded SWE history in ../../water-system-3d/. Writes into
viz-data/. Every output carries Denver Water's data terms (TERMS.md beside it)
and a provisional-data note.

    python3 build_viz_data.py

Outputs (all generated, do not edit by hand):
    viz-data/daily-series.json   one row per day 2022-04-01..2026-08-19, every
                                 daily series the views draw, nulls for gaps
    viz-data/sonde.json          Strontia sonde depth-time grids (1 m x 1 day)
                                 per parameter, plus daily band aggregates and
                                 a stratification index
    viz-data/water-years.json    per-water-year folded traces (SWE, flow, TOC,
                                 alkalinity) with cross-year medians
    viz-data/storms.json         the top NOAA precipitation events, plus the
                                 committed Aug 14-15 storm reference
    viz-data/basins.json         per sub-basin: outlet gage, member SNOTELs
                                 (point-in-polygon), delivery path, blurbs,
                                 melt-season markers per water year

Data hygiene applied here, surfaced as notes in the JSON so the UI can show it:
  - USGS Dissolved_Oxygen_Min -999999 sentinel masked to null
  - Michigan Creek SWE 2026-05-12..15 masked (known NRCS feed error; Hoosier
    Pass is the station Jake's models use)
  - DWR cumulative Precip counter excluded entirely (dirty per Jake; NOAA
    PRCP is the precipitation record)
  - NOAA GHCN publishes ~2 days behind; noted, not shifted (these views show
    history, not forecasts)
"""

import csv
import json
import math
import os
from collections import defaultdict
from datetime import date, datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data")
BRIEF = os.path.join(HERE, "..", "..", "strontia-brief")
W3D = os.path.join(HERE, "..", "..", "water-system-3d")
OUT = os.path.join(HERE, "viz-data")

START, END = date(2022, 4, 1), date(2026, 8, 19)

TERMS_NOTE = ("Derived from Denver Water Design Storm data; Denver Water's "
              "data terms apply (TERMS.md in this folder). All readings are "
              "provisional and subject to revision by the source agencies.")

MICHIGAN_CREEK_MASK = {date(2026, 5, d).isoformat() for d in range(12, 16)}


def daterange():
    d = START
    while d <= END:
        yield d
        d += timedelta(days=1)


DATES = [d.isoformat() for d in daterange()]
DATE_INDEX = {d: i for i, d in enumerate(DATES)}


def parse_day(text):
    text = text.strip()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"unparseable date {text!r}")


def blank():
    return [None] * len(DATES)


def read_rows(filename):
    with open(os.path.join(DATA, filename)) as f:
        yield from csv.DictReader(f)


def fill(series, day, raw, transform=float):
    i = DATE_INDEX.get(day)
    if i is None or raw in (None, ""):
        return
    series[i] = transform(raw)


def round3(v):
    return None if v is None else round(v, 3)


# ---------------------------------------------------------------- daily series

def build_daily():
    s = {name: blank() for name in (
        "toc", "alk", "turb", "cond", "temp", "do", "ph",
        "flow", "gage_height", "swe_hoosier", "swe_michigan",
        "prcp", "snow", "tmax", "tmin")}
    notes = []

    for row in read_rows("FoothillsInfluent.csv"):
        day = parse_day(row["DATE"])
        fill(s["toc"], day, row["TOC_mg_L"])
        fill(s["alk"], day, row["Alk_mg_L"])

    do_masked = 0
    for row in read_rows("USGS_South_Platte.csv"):
        day = parse_day(row["Date"])
        fill(s["turb"], day, row["Turbidity_Median"])
        fill(s["cond"], day, row["Specific_Cond_Mean"])
        fill(s["temp"], day, row["Temp_C_Mean"])
        fill(s["ph"], day, row["pH_Median"])
        raw_do = row["Dissolved_Oxygen_Mean"]
        if raw_do and float(raw_do) < 0:  # -999999 sentinel family
            do_masked += 1
        else:
            fill(s["do"], day, raw_do)
    if do_masked:
        notes.append(f"USGS dissolved oxygen: {do_masked} negative sentinel "
                     "value(s) masked to null.")

    for row in read_rows("SouthPlatteTelemetry.csv"):
        day = parse_day(row["Date"])
        fill(s["flow"], day, row["Flow_CFS"])
        fill(s["gage_height"], day, row["GageHeight_ft"])
    notes.append("DWR telemetry Precip column excluded: it is a cumulative, "
                 "reset-prone counter (Jake's guidance). NOAA PRCP is the "
                 "precipitation record here.")

    for row in read_rows("HoosierPass.csv"):
        fill(s["swe_hoosier"], parse_day(row["DATE"]), row["SWE"])

    mc_masked = 0
    for row in read_rows("MichiganCreek.csv"):
        day = parse_day(row["DATE"])
        if day in MICHIGAN_CREEK_MASK:
            mc_masked += 1
            continue
        fill(s["swe_michigan"], day, row["SWE"])
    notes.append(f"Michigan Creek SWE 2026-05-12..15 masked ({mc_masked} "
                 "value(s)): known NRCS feed error (9.0 between zero "
                 "readings). Jake's models use Hoosier Pass instead.")

    for row in read_rows("USC00058022.csv"):
        day = parse_day(row["DATE"])
        fill(s["prcp"], day, row["PRCP"])
        fill(s["snow"], day, row["SNOW"])
        fill(s["tmax"], day, row["TMAX"])
        fill(s["tmin"], day, row["TMIN"])
    notes.append("NOAA GHCN (station USC00058022) publishes about 2 days "
                 "behind real time; a live pull will trail these files.")

    return {
        "_comment": "Generated by build_viz_data.py. Do not edit by hand.",
        "terms": TERMS_NOTE,
        "start": DATES[0], "end": DATES[-1],
        "dates": DATES,
        "series": {k: [round3(v) for v in vals] for k, vals in s.items()},
        "labels": {
            "toc": "TOC at Foothills influent, mg/L",
            "alk": "Alkalinity at Foothills influent, mg/L",
            "turb": "Turbidity (daily median) above Strontia, FNU",
            "cond": "Specific conductance (daily mean), uS/cm",
            "temp": "Water temperature (daily mean), C",
            "do": "Dissolved oxygen (daily mean), mg/L",
            "ph": "pH (daily median)",
            "flow": "South Platte flow at Waterton (DWR PLASPLCO), cfs",
            "gage_height": "Gage height, ft",
            "swe_hoosier": "Snow water equivalent, Hoosier Pass SNOTEL, in",
            "swe_michigan": "Snow water equivalent, Michigan Creek SNOTEL, in",
            "prcp": "Precipitation (NOAA USC00058022), in",
            "snow": "Snowfall (NOAA), in",
            "tmax": "Air temperature max (NOAA), F",
            "tmin": "Air temperature min (NOAA), F",
        },
        "hygiene_notes": notes,
    }, s


# ----------------------------------------------------------------------- sonde

SONDE_PARAMS = {
    "temp": ("Temp C", "Water temperature, C"),
    "cond": ("Conductivity", "Conductivity, uS/cm"),
    "ph": ("pH", "pH"),
    "turb": ("Turbidity NTU", "Turbidity, NTU"),
    "chl": ("Chl ug/L", "Chlorophyll, ug/L"),
    "phyco": ("Phycocyanin", "Phycocyanin, RFU"),
    "do": ("ODO mg/L", "Dissolved oxygen, mg/L"),
}
SURFACE_MAX_M = 5.0     # PoC-verified: low Vertical Position = warm surface
BOTTOM_MARGIN_M = 5.0   # bottom band = within 5 m of that day's deepest reading


def build_sonde():
    import pandas as pd

    df = pd.read_excel(os.path.join(DATA, "Strontia 0407_0819.xlsx"))
    df.columns = [c.strip() for c in df.columns]
    df["Time stamp"] = pd.to_datetime(df["Time stamp"])
    df["day"] = df["Time stamp"].dt.date.astype(str)
    df["depth_bin"] = df["Vertical Position"].round().astype(int)

    days = sorted(df["day"].unique())
    day_index = {d: i for i, d in enumerate(days)}
    max_depth = int(df["depth_bin"].max())
    depths = list(range(0, max_depth + 1))

    grids = {}
    for key, (col, _) in SONDE_PARAMS.items():
        grid = [[None] * len(days) for _ in depths]
        grouped = df.groupby(["day", "depth_bin"])[col].mean()
        for (day, depth), value in grouped.items():
            if not math.isnan(value):
                grid[depth][day_index[day]] = round(float(value), 3)
        grids[key] = grid

    bands = {"surface_temp": [], "bottom_temp": [], "strat": [],
             "max_depth": [], "readings": []}
    for day in days:
        sub = df[df["day"] == day]
        deepest = sub["Vertical Position"].max()
        surf = sub[sub["Vertical Position"] < SURFACE_MAX_M]["Temp C"].mean()
        bot = sub[sub["Vertical Position"] >
                  deepest - BOTTOM_MARGIN_M]["Temp C"].mean()
        surf = None if math.isnan(surf) else round(float(surf), 2)
        bot = None if math.isnan(bot) else round(float(bot), 2)
        bands["surface_temp"].append(surf)
        bands["bottom_temp"].append(bot)
        bands["strat"].append(
            None if surf is None or bot is None else round(surf - bot, 2))
        bands["max_depth"].append(round(float(deepest), 1))
        bands["readings"].append(int(len(sub)))

    return {
        "_comment": "Generated by build_viz_data.py from Strontia "
                    "0407_0819.xlsx. Grid cells are means of all readings in "
                    "a 1 m depth bin on a calendar day. Do not edit by hand.",
        "terms": TERMS_NOTE,
        "convention": "Vertical Position is depth in metres (low value = "
                      "warm surface water; verified in the poc-24h-toc-alk "
                      "audit). Surface band < 5 m; bottom band within 5 m of "
                      "the day's deepest reading.",
        "days": days,
        "depths": depths,
        "params": {k: {"label": label, "grid": grids[k]}
                   for k, (_, label) in SONDE_PARAMS.items()},
        "bands": bands,
        "total_readings": int(len(df)),
    }


# ----------------------------------------------------------------- water years

def water_year(day_iso):
    y, m, _ = (int(p) for p in day_iso.split("-"))
    return y + 1 if m >= 10 else y


def day_of_water_year(day_iso):
    y, m, d = (int(p) for p in day_iso.split("-"))
    start = date(y if m >= 10 else y - 1, 10, 1)
    return (date(y, m, d) - start).days


def fold(series):
    years = defaultdict(lambda: [None] * 366)
    for i, day in enumerate(DATES):
        v = series[i]
        if v is not None:
            years[water_year(day)][day_of_water_year(day)] = round3(v)
    return dict(sorted(years.items()))


def median_trace(years):
    trace = []
    for doy in range(366):
        vals = sorted(v[doy] for v in years.values() if v[doy] is not None)
        if len(vals) < 2:
            trace.append(None)
            continue
        mid = len(vals) // 2
        med = vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2
        trace.append(round3(med))
    return trace


def build_water_years(raw):
    out = {}
    for key in ("swe_hoosier", "flow", "toc", "alk", "turb"):
        years = fold(raw[key])
        out[key] = {"years": {str(y): t for y, t in years.items()},
                    "median": median_trace(years)}
    return {
        "_comment": "Generated by build_viz_data.py. Series folded by water "
                    "year (Oct 1 = day 0). Median needs >= 2 years with data "
                    "on that day. Do not edit by hand.",
        "terms": TERMS_NOTE,
        "series": out,
    }


# ---------------------------------------------------------------------- storms

def build_storms(raw):
    prcp, flow, turb = raw["prcp"], raw["flow"], raw["turb"]
    candidates = sorted(
        ((v, i) for i, v in enumerate(prcp) if v is not None and v > 0),
        reverse=True)
    picked = []
    for v, i in candidates:
        if len(picked) >= 8:
            break
        if all(abs(i - j) > 10 for _, j in picked):
            picked.append((v, i))
    picked.sort(key=lambda p: p[1])

    def peak_offset(series, i, lo, hi):
        window = [(series[j], j - i) for j in range(max(0, i + lo),
                                                    min(len(DATES), i + hi + 1))
                  if series[j] is not None]
        return max(window)[1] if window else None

    events = [{
        "date": DATES[i],
        "prcp_in": round3(v),
        "flow_peak_offset_days": peak_offset(flow, i, 0, 6),
        "turb_peak_offset_days": peak_offset(turb, i, 0, 6),
    } for v, i in picked]

    return {
        "_comment": "Generated by build_viz_data.py. The top NOAA "
                    "precipitation days (>= 10 days apart) with the offset, "
                    "in days, from rain to the local flow and turbidity "
                    "peaks. Do not edit by hand.",
        "terms": TERMS_NOTE,
        "note": "The NOAA station is one point in a 4,000 sq mi watershed: "
                "a storm can soak the upper basin and miss this gauge, and "
                "the reverse. Offsets are descriptive, not fitted lags.",
        "events": events,
        "reference_storm": {
            "label": "Aug 14-15, 2026 (15-minute record)",
            "date": "2026-08-14",
            "series": [
                "strontia-brief/series/06701900-discharge-aug14-15.json",
                "strontia-brief/series/06707525-turbidity-conductance-aug14-15.json",
            ],
            "note": "The committed 15-minute series behind the 3D map's "
                    "storm replay: evening monsoon rain became a 329 FNU "
                    "turbidity spike at the sentinel gage by 1:45 AM.",
        },
    }


# ---------------------------------------------------------------------- basins

BASINS = [
    {
        "id": "south-platte-above-strontia",
        "name": "South Platte above Strontia Springs",
        "file": "strontia-brief/basins/south-platte-above-strontia-06707525.json",
        "outlet_gage": {"site": "06707525",
                        "name": "Strontia Springs Sentinel Gauge"},
        "slope": "east",
        "delivery": "Gravity, down the mainstem South Platte into Strontia "
                    "Springs Reservoir.",
        "blurb": "The whole upstream catchment of the sentinel gage: "
                 "roughly 2,600 sq mi of South Park and the Platte canyons, "
                 "including Cheesman, Eleven Mile and Antero reservoirs. "
                 "Everything the other basins send east eventually flows "
                 "through here; about 80% of Denver's supply passes its "
                 "outlet.",
        "committed_flow": "flow",
    },
    {
        "id": "south-platte-above-trumbull",
        "name": "South Platte above Trumbull",
        "file": "strontia-brief/basins/south-platte-above-trumbull-06701900.json",
        "outlet_gage": {"site": "06701900",
                        "name": "South Platte Gauge near Trumbull"},
        "slope": "east",
        "nested_in": "south-platte-above-strontia",
        "delivery": "Gravity, mainstem South Platte; nested inside the "
                    "Strontia basin.",
        "blurb": "A nested sub-basin: the same river measured one gage "
                 "earlier. Comparing Trumbull with the sentinel gage "
                 "brackets the unmonitored side canyons between them, "
                 "where the August 2026 sediment pulse washed in.",
        "committed_flow": None,
    },
    {
        "id": "blue-river-below-dillon",
        "name": "Blue River below Dillon",
        "file": "strontia-brief/basins/blue-river-below-dillon-09050700.json",
        "outlet_gage": {"site": "09050700",
                        "name": "Blue River below Dillon (USGS 09050700)"},
        "slope": "west",
        "delivery": "Roberts Tunnel: 23 miles under the Continental Divide "
                    "from Dillon Reservoir to the North Fork at Grant.",
        "blurb": "West-slope snow country draining to Dillon Reservoir, "
                 "Denver Water's largest. Left alone this water would reach "
                 "the Pacific; the Roberts Tunnel sends it east instead. Its "
                 "snowpack matters to Foothills months before its water "
                 "does.",
        "committed_flow": None,
    },
    {
        "id": "fraser-below-moffat",
        "name": "Fraser River below Moffat Tunnel",
        "file": "strontia-brief/basins/fraser-below-moffat-09023562.json",
        "outlet_gage": {"site": "09023562",
                        "name": "Fraser River below Moffat Tunnel"},
        "slope": "west",
        "delivery": "Moffat Tunnel: 6.2 miles under the divide at Winter "
                    "Park into South Boulder Creek, bound for Gross "
                    "Reservoir and the north system.",
        "blurb": "The Fraser headwaters at Winter Park. Water collected "
                 "here supplies the north system (Gross Reservoir, Ralston, "
                 "Northwater), not Foothills, so its behaviour explains the "
                 "other half of Denver's supply.",
        "committed_flow": None,
    },
    {
        "id": "williams-fork-parshall",
        "name": "Williams Fork at Parshall",
        "file": "strontia-brief/basins/williams-fork-parshall-09037500.json",
        "outlet_gage": {"site": "09037500",
                        "name": "Williams Fork Gauge near Parshall"},
        "slope": "west",
        "delivery": "Stays west: Williams Fork Reservoir water repays the "
                    "west slope for water sent east.",
        "blurb": "The accounting basin. Its water never reaches a Denver "
                 "tap; it is released to meet downstream obligations so the "
                 "Roberts and Moffat tunnels can keep diverting. Snowpack "
                 "here is still supply, one step removed.",
        "committed_flow": None,
    },
]

BASIN_POLYGON_FILES = {
    b["id"]: os.path.join(BRIEF, "basins", os.path.basename(b["file"]))
    for b in BASINS
}


def polygon_rings(path):
    with open(path) as f:
        geometry = json.load(f)["features"][0]["geometry"]
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"][0]]
    return [poly[0] for poly in geometry["coordinates"]]


def point_in_ring(point, ring):
    x, y = point
    inside = False
    for i in range(len(ring)):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % len(ring)]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def melt_markers(trace):
    """Peak SWE, melt onset (last day at >=90% of peak) and melt-out."""
    peak_doy, peak = None, 0.0
    for doy, v in enumerate(trace):
        if v is not None and v > peak:
            peak, peak_doy = v, doy
    if peak_doy is None or peak <= 0:
        return None
    onset = peak_doy
    for doy in range(peak_doy, 366):
        v = trace[doy]
        if v is not None:
            if v >= 0.9 * peak:
                onset = doy
            else:
                break
    melt_out = None
    for doy in range(peak_doy, 366):
        v = trace[doy]
        if v is not None and v <= 0.1:
            melt_out = doy
            break
    return {"peak_doy": peak_doy, "peak_swe": round(peak, 1),
            "onset_doy": onset, "melt_out_doy": melt_out}


def build_basins():
    with open(os.path.join(W3D, "snotel-co-stations.json")) as f:
        stations = json.load(f)
    with open(os.path.join(W3D, "snotel-history.json")) as f:
        history = json.load(f)["stations"]

    # The Michigan Creek May 2026 artifact (9.0 SWE between zero readings,
    # a known NRCS feed error) is present in snotel-history.json as well;
    # mask it before averaging so basin SWE stays consistent with the
    # masked daily series.
    mc = history.get("937:CO:SNTL", {}).get("years", {}).get("2026")
    if mc:
        for day_iso in sorted(MICHIGAN_CREEK_MASK):
            doy = day_of_water_year(day_iso)
            if mc[doy] is not None and mc[doy] == 9.0:
                mc[doy] = None

    basins = []
    for spec in BASINS:
        rings = polygon_rings(BASIN_POLYGON_FILES[spec["id"]])
        members = []
        for st in stations:
            point = (st["longitude"], st["latitude"])
            if any(point_in_ring(point, ring) for ring in rings):
                members.append({"triplet": st["stationTriplet"],
                                "station_id": st["stationId"],
                                "name": st["name"],
                                "elevation_ft": st["elevation"]})
        members.sort(key=lambda m: m["name"])

        member_years = [history[m["triplet"]]["years"]
                        for m in members if m["triplet"] in history]
        swe_years, markers = {}, {}
        all_years = sorted({y for years in member_years for y in years})
        for wy in all_years:
            trace = []
            for doy in range(366):
                vals = [years[wy][doy] for years in member_years
                        if wy in years and years[wy][doy] is not None]
                trace.append(round(sum(vals) / len(vals), 2)
                             if vals else None)
            swe_years[wy] = trace
        swe_median = median_trace(swe_years) if swe_years else [None] * 366
        for wy in ("2022", "2023", "2024", "2025", "2026"):
            if wy in swe_years:
                m = melt_markers(swe_years[wy])
                if m:
                    markers[wy] = m

        basins.append({
            **{k: spec[k] for k in ("id", "name", "file", "outlet_gage",
                                    "slope", "delivery", "blurb")},
            **({"nested_in": spec["nested_in"]} if "nested_in" in spec else {}),
            "committed_flow": spec["committed_flow"],
            "snotel_members": members,
            "swe": {"years": {y: swe_years[y]
                              for y in ("2022", "2023", "2024", "2025", "2026")
                              if y in swe_years},
                    "median": swe_median},
            "melt_markers": markers,
        })

    return {
        "_comment": "Generated by build_viz_data.py. Basin-average SWE is "
                    "the plain mean over member SNOTEL stations "
                    "(point-in-polygon on the committed NLDI polygons), "
                    "folded by water year; median is across all years in "
                    "snotel-history.json. The Michigan Creek 2026-05-12..15 "
                    "NRCS artifact is masked before averaging. Do not edit "
                    "by hand.",
        "terms": TERMS_NOTE,
        "flow_note": "Outlet-gage daily flow for the west-slope and nested "
                     "basins comes from fetch_basin_flows.py (network; USGS "
                     "daily-values API) into basin-flows.json. The Strontia "
                     "basin's committed flow record is the DWR PLASPLCO "
                     "series in daily-series.json.",
        "basins": basins,
    }


# ------------------------------------------------------------------------ main

def write(name, payload):
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"wrote {path} ({os.path.getsize(path):,} bytes)")


def main():
    os.makedirs(OUT, exist_ok=True)
    daily, raw = build_daily()
    write("daily-series.json", daily)
    write("sonde.json", build_sonde())
    write("water-years.json", build_water_years(raw))
    write("storms.json", build_storms(raw))
    write("basins.json", build_basins())


if __name__ == "__main__":
    main()

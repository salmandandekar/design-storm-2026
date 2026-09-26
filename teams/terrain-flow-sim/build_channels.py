#!/usr/bin/env python3
"""Build the channel network and travel-time table behind the simulation. Offline.

Reads the committed USGS NLDI mainstem flowlines and the OSM conduit lines in
../../strontia-brief/basins/, the system points/lines in
../../water-system-3d/system.json, the flow record in
../explainable-viz/viz-data/daily-series.json, and the z12 channel-floor
elevations written by fetch_dem_d8.py (sim-data/reach-elevations.json; if
absent, slopes fall back to a single basin-average value and the output says
so). Writes sim-data/channels.json.

    python3 build_channels.py

Physics, and what is assumed versus computed
--------------------------------------------
Open channel (every river reach): continuity Q = v*A and Manning's equation
v = (1/n) R^(2/3) S^(1/2), with R = A/P the hydraulic radius. Q is the DWR
Waterton flow record (committed data). S is the reach slope from the sampled
elevations (computed). n and the channel width are NOT in the data and are
ASSUMED from standard tables (Chow 1959): n = 0.045 for a mountain stream
with cobbles and boulders, wide rectangular section of width B = 18 m on the
mainstem. Both are exposed in the output and can be edited; the travel time
scales roughly with n and B^(2/5).

Pressurised conduits and tunnels: Bernoulli between the upstream pool and
the downstream portal, z1 + p1/rho g + v1^2/2g = z2 + p2/rho g + v2^2/2g + h_f,
with both ends at atmospheric pressure and the Darcy-Weisbach loss
h_f = f (L/D) v^2/2g, gives v = sqrt(2 g dz / (1 + f L/D)). dz and L are
computed from the sampled elevations and portal coordinates; f = 0.02 and
the diameter D are assumed (Roberts Tunnel is publicly described as about
3 m in diameter; the Strontia-Foothills conduit is taken as 2.5 m). The
frictionless bound sqrt(2 g dz) is also reported. Actual conveyance is
valve-controlled and usually far below the hydraulic capacity, so these are
upper bounds on velocity, i.e. lower bounds on travel time.

Denver Water's own number: system.json quotes their raw-water group that
water at the sentinel gage reaches the plant intake in about four hours.
That is reported alongside the computed figure, not replaced by it.
"""

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BRIEF = os.path.join(HERE, "..", "..", "strontia-brief", "basins")
W3D = os.path.join(HERE, "..", "..", "water-system-3d")
VIZ = os.path.join(HERE, "..", "explainable-viz", "viz-data")
SIM = os.path.join(HERE, "sim-data")
OUT = os.path.join(SIM, "channels.json")

G = 9.81
ASSUMED = {
    "manning_n": 0.045,
    "channel_width_m": 18.0,
    "darcy_f": 0.02,
    "tunnel_diameter_m": {"roberts": 3.0, "moffat": 3.0, "conduit-26": 2.5, "conduit-20": 2.0},
    "min_slope": 0.0005,
    "note": ("ASSUMED, not from the data: Manning n and channel width from Chow (1959) tables for "
             "a cobble/boulder mountain stream; Darcy f and conduit diameters from typical values "
             "and public descriptions. Edit here and rerun to see the sensitivity."),
}

FLOW_TIERS = {"low": 10, "median": 50, "high": 90}
STORM_PEAK_CFS_KEY = "aug_2026_peak"


def load(path):
    with open(path) as fh:
        return json.load(fh)


def seg_len_m(a, b):
    dx = (b[0] - a[0]) * 111320 * math.cos(math.radians((a[1] + b[1]) / 2))
    dy = (b[1] - a[1]) * 110540
    return math.hypot(dx, dy)


def line_len_m(coords):
    return sum(seg_len_m(a, b) for a, b in zip(coords, coords[1:]))


def chain(features):
    """Order NLDI upstream-main flowlines from the outlet upward by matching endpoints."""
    by_end = {tuple(f["geometry"]["coordinates"][-1]): f for f in features}
    starts = {tuple(f["geometry"]["coordinates"][0]) for f in features}
    # The outlet reach is the one whose end matches no other reach's start.
    outlet = [f for f in features if tuple(f["geometry"]["coordinates"][-1]) not in starts]
    if len(outlet) != 1:
        outlet = [features[0]]
    ordered, cur = [], outlet[0]
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        ordered.append(cur)
        cur = by_end.get(tuple(cur["geometry"]["coordinates"][0]))
    return ordered


def manning_velocity(q_m3s, slope, n, width):
    """Solve normal depth for a wide rectangular channel by bisection, return (v, depth)."""
    slope = max(slope, ASSUMED["min_slope"])
    lo, hi = 1e-3, 20.0
    for _ in range(80):
        h = (lo + hi) / 2
        a = width * h
        r = a / (width + 2 * h)
        q = (1 / n) * a * r ** (2 / 3) * math.sqrt(slope)
        if q > q_m3s:
            hi = h
        else:
            lo = h
    h = (lo + hi) / 2
    return (q_m3s / (width * h), h)


def percentile(vals, p):
    v = sorted(x for x in vals if x is not None)
    if not v:
        return None
    k = (len(v) - 1) * p / 100
    f, c = math.floor(k), math.ceil(k)
    return v[f] if f == c else v[f] + (v[c] - v[f]) * (k - f)


def nearest_vertex_distance(coords_chain, lon, lat):
    """Cumulative along-chain distance (m) of the vertex nearest a point."""
    best, best_d, cum = None, float("inf"), 0.0
    for a, b in zip(coords_chain, coords_chain[1:]):
        d = seg_len_m(a, (lon, lat))
        if d < best_d:
            best_d, best = d, cum
        cum += seg_len_m(a, b)
    return best, best_d


def main():
    flows = load(os.path.join(BRIEF, "mainstem-flowlines-06707525.json"))["features"]
    down = load(os.path.join(BRIEF, "downstream-mainstem-06707525.json"))["features"]
    system = load(os.path.join(W3D, "system.json"))
    daily = load(os.path.join(VIZ, "daily-series.json"))
    elev_path = os.path.join(SIM, "reach-elevations.json")
    elev = load(elev_path) if os.path.exists(elev_path) else None

    pts = {p["id"]: p for p in system["points"]}
    lines = {l["id"]: l for l in system["lines"]}

    flow_cfs = daily["series"]["flow"]
    tiers = {k: percentile(flow_cfs, p) for k, p in FLOW_TIERS.items()}
    tiers[STORM_PEAK_CFS_KEY] = None
    try:
        storm = load(os.path.join(HERE, "..", "..", "strontia-brief", "series",
                                  "06701900-discharge-aug14-15.json"))
        vals = []
        for ts in storm.get("value", {}).get("timeSeries", []):
            for blk in ts.get("values", []):
                vals += [float(v["value"]) for v in blk.get("value", []) if v.get("value") not in (None, "-999999")]
        tiers[STORM_PEAK_CFS_KEY] = max(vals) if vals else None
    except Exception:
        pass
    tiers = {k: v for k, v in tiers.items() if v is not None}

    # --- mainstem above the sentinel gage, ordered outlet -> headwater
    ordered = chain(flows)
    reaches, cum = [], 0.0
    flat = []
    slope_fallback = None
    if elev:
        zs = [(elev["reaches"].get(f"mainstem:{f['properties']['nhdplus_comid']}"), line_len_m(f["geometry"]["coordinates"])) for f in ordered]
        tot_drop = sum(max(0.0, z["z_start_m"] - z["z_end_m"]) for z, _ in zs if z)
        tot_len = sum(L for z, L in zs if z)
        slope_fallback = tot_drop / tot_len if tot_len else 0.01
    for f in ordered:
        cs = f["geometry"]["coordinates"]
        L = line_len_m(cs)
        cid = str(f["properties"]["nhdplus_comid"])
        z = elev["reaches"].get(f"mainstem:{cid}") if elev else None
        if z and z["z_start_m"] is not None and z["z_end_m"] is not None:
            slope = (z["z_start_m"] - z["z_end_m"]) / L if L else 0
            slope_src = "sampled"
            if slope <= 0:
                slope, slope_src = slope_fallback, "fallback (sampled drop <= 0)"
        else:
            slope, slope_src = slope_fallback or 0.01, "fallback"
        reaches.append({"comid": cid, "length_m": round(L), "slope": round(slope, 5), "slope_source": slope_src,
                        "dist_to_gage_m": round(cum), "coords": [[round(x, 5), round(y, 5)] for x, y in cs],
                        "z_up_m": z["z_start_m"] if z else None, "z_down_m": z["z_end_m"] if z else None})
        cum += L
        flat += cs if not flat else cs[1:] if tuple(cs[0]) == tuple(flat[-1]) else cs
    # flowline chain runs downstream within each reach; reversed reaches are ordered outlet-first,
    # so `flat` is outlet -> headwater with per-reach downstream vertex order. Build a clean
    # gage->headwater vertex list by reversing each reach.
    flat = []
    for r in reaches:
        cs = list(reversed(r["coords"]))
        flat += cs if not flat else cs[1:]

    # Landmarks along the mainstem: gage upstream distances
    landmarks = {}
    for pid in ("gage-06701900", "gage-06707000", "gage-06707500", "res-cheesman"):
        p = pts.get(pid)
        if p:
            d, off = nearest_vertex_distance(flat, p["lon"], p["lat"])
            landmarks[pid] = {"name": p["name"], "dist_to_gage_m": round(d), "offset_from_line_m": round(off)}

    # Travel time per tier: sum over reaches of L / v(Q). Q attenuates upstream; we scale
    # Q by upstream fraction of accumulated length as a crude proxy (labelled).
    total_len = cum
    travel = {}
    for tier, q_cfs in tiers.items():
        q = q_cfs * 0.0283168
        t_cum, per_reach = 0.0, []
        for r in reaches:
            frac = max(0.15, 1 - r["dist_to_gage_m"] / total_len)
            v, h = manning_velocity(q * frac, r["slope"], ASSUMED["manning_n"], ASSUMED["channel_width_m"])
            t_cum += r["length_m"] / v
            per_reach.append({"comid": r["comid"], "v_m_s": round(v, 2), "depth_m": round(h, 2), "hours_to_gage": round(t_cum / 3600, 2)})
        lm = {}
        for pid, info in landmarks.items():
            # hours from landmark to gage = cumulative time at that distance
            h_at = next((pr["hours_to_gage"] for r, pr in zip(reaches, per_reach) if r["dist_to_gage_m"] + r["length_m"] >= info["dist_to_gage_m"]), None)
            lm[pid] = h_at
        travel[tier] = {"flow_cfs": round(q_cfs), "flow_m3s": round(q, 2), "per_reach": per_reach,
                        "hours_headwater_to_gage": round(t_cum / 3600, 1), "hours_landmark_to_gage": lm}

    # --- reservoir and conduits (Bernoulli / Darcy-Weisbach)
    def z_of(key):
        return elev["points"].get(key) if elev else None

    pool_z = z_of("gage-06707525")  # the gage sits at the reservoir's inflow; pool ~ 6,000 ft
    conduits = []
    for lid, up_key, down_key in (("roberts", "line:roberts:start", "line:roberts:end"),
                                  ("moffat", "line:moffat:start", "line:moffat:end"),
                                  ("conduit-26", None, "plant-foothills"),
                                  ("conduit-20", "line:conduit-20:start", "line:conduit-20:end")):
        ln = lines.get(lid)
        if not ln:
            continue
        L = line_len_m(ln["coords"])
        z1 = pool_z if up_key is None else z_of(up_key)
        z2 = z_of(down_key)
        D = ASSUMED["tunnel_diameter_m"].get(lid, 2.5)
        entry = {"id": lid, "name": ln["name"], "length_m": round(L), "z_up_m": z1, "z_down_m": z2, "diameter_m_assumed": D}
        if z1 is not None and z2 is not None and z1 > z2:
            dz = z1 - z2
            v_free = math.sqrt(2 * G * dz)
            v_fric = math.sqrt(2 * G * dz / (1 + ASSUMED["darcy_f"] * L / D))
            entry.update({"head_m": round(dz, 1), "v_frictionless_m_s": round(v_free, 1),
                          "v_darcy_m_s": round(v_fric, 2),
                          "hours_at_darcy_v": round(L / v_fric / 3600, 2),
                          "capacity_m3s_at_darcy_v": round(v_fric * math.pi * D * D / 4, 1),
                          "note": "Upper-bound velocity (valves throttle real flow); lower-bound travel time."})
        else:
            entry["note"] = "No positive head from sampled elevations; pumped or level conduit; Bernoulli bound not computed."
        conduits.append(entry)

    # Reservoir residence: storage / outflow (continuity), from committed storage history median
    strontia = None
    try:
        sh = load(os.path.join(W3D, "storage-history.json"))["reservoirs"]["STRRESCO"]
        med = [v for v in sh["median"] if v is not None]
        vol_af = sum(med) / len(med)
        vol_m3 = vol_af * 1233.48
        res = {}
        for tier, q_cfs in tiers.items():
            q = q_cfs * 0.0283168
            res[tier] = round(vol_m3 / q / 86400, 1)
        strontia = {"median_storage_af": round(vol_af), "median_storage_m3": round(vol_m3),
                    "residence_days_at_tier": res,
                    "note": ("Nominal residence time V/Q assumes full mixing. Denver Water's raw-water group says "
                             "gage water reaches the intake in about four hours (short-circuiting along the old "
                             "channel), while the plant record tracks the gage a day or two later. Both are quoted "
                             "in system.json; the physics here brackets them.")}
    except Exception:
        pass

    # Downstream mainstem (gage -> dam) for drawing
    down_coords = []
    for f in chain_downstream(down):
        cs = f["geometry"]["coordinates"]
        down_coords += cs if not down_coords else cs[1:]
    dam = pts.get("dam-strontia")
    gage_to_dam_m = nearest_vertex_distance(down_coords, dam["lon"], dam["lat"])[0] if dam else None

    out = {
        "_comment": "Generated by build_channels.py. Do not edit by hand.",
        "terms": ("Derived from Denver Water Design Storm data and public USGS/OSM geometry; Denver Water's "
                  "data terms apply (TERMS.md in this folder). All readings are provisional."),
        "assumed": ASSUMED,
        "elevation_source": elev["source"] if elev else "none (fetch_dem_d8.py not run); slopes are fallbacks",
        "flow_tiers_cfs": {k: round(v) for k, v in tiers.items()},
        "flow_tiers_note": ("low/median/high are the 10th/50th/90th percentiles of the DWR Waterton daily flow "
                            "record 2022-2026; aug_2026_peak is the 15-minute peak at Trumbull on Aug 14-15 2026."),
        "mainstem": {"total_length_m": round(total_len), "n_reaches": len(reaches),
                     "reaches": reaches, "landmarks": landmarks,
                     "denver_water_quote": "Water passing the sentinel gage reaches the plant intake in about four hours (Denver Water raw water group, quoted in system.json).",
                     "travel": travel},
        "gage_to_dam_m": round(gage_to_dam_m) if gage_to_dam_m else None,
        "downstream_coords": [[round(x, 5), round(y, 5)] for x, y in down_coords],
        "conduits": conduits,
        "strontia": strontia,
        "physics_labels": {
            "computed": ["reach lengths and slopes", "flow percentiles", "Manning velocities given n and B",
                         "conduit head and Bernoulli/Darcy bounds given f and D", "nominal residence time"],
            "assumed": ["Manning n", "channel width", "upstream flow attenuation proxy", "Darcy f", "conduit diameters"],
            "general_knowledge": ["Manning and Bernoulli equations themselves", "typical n/f values",
                                  "Roberts Tunnel ~3 m diameter"],
        },
    }
    os.makedirs(SIM, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"wrote {os.path.relpath(OUT, HERE)}: {len(reaches)} reaches, {total_len / 1000:.0f} km mainstem")
    for tier, t in travel.items():
        print(f"  {tier:>14} {t['flow_cfs']:>6} cfs: headwater->gage {t['hours_headwater_to_gage']} h; "
              f"Trumbull->gage {t['hours_landmark_to_gage'].get('gage-06701900')} h")
    for c in conduits:
        print(f"  {c['name']}: head {c.get('head_m')} m, Darcy v {c.get('v_darcy_m_s')} m/s, {c.get('hours_at_darcy_v')} h")
    if strontia:
        print(f"  Strontia residence days: {strontia['residence_days_at_tier']}")


def chain_downstream(features):
    by_start = {tuple(f["geometry"]["coordinates"][0]): f for f in features}
    ends = {tuple(f["geometry"]["coordinates"][-1]) for f in features}
    head = [f for f in features if tuple(f["geometry"]["coordinates"][0]) not in ends]
    cur = head[0] if head else features[0]
    ordered, seen = [], set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        ordered.append(cur)
        cur = by_start.get(tuple(cur["geometry"]["coordinates"][-1]))
    return ordered


if __name__ == "__main__":
    main()

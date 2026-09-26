#!/usr/bin/env python3
"""One-time fetch of terrain and derivation of a D8 flow-direction grid. Network.

Pulls Mapzen terrarium elevation tiles (AWS Open Data, the same source the
3D map streams) at zoom 10 over the four sub-basins, decodes them to metres,
block-averages to a coarse grid, fills pits (priority flood), computes D8
flow direction and flow accumulation, masks cells to the basin polygons, and
writes sim-data/d8-grid.json. Also samples channel-floor elevations at zoom 12 for the
mainstem flowlines and system points into sim-data/reach-elevations.json, which build_channels.py reads.

    python3 fetch_dem_d8.py             # ~25 tiles, a few seconds of network
    python3 fetch_dem_d8.py --factor 4  # coarser grid

Rerun rarely: terrain does not change. Terrain tiles: Mapzen via AWS
(https://registry.opendata.aws/terrain-tiles/), attribution as on the map.

Output d8-grid.json:
    bbox            [west, south, east, north] of the grid in degrees
    ncols, nrows    grid size; cell (r, c) centre is at
                    lon = west + (c + 0.5) * dlon, lat = north - (r + 0.5) * dlat
    cell_m          approximate cell size in metres at the grid's mid latitude
    elev_m          row-major Int16 elevations (metres), -32768 = nodata
    dir             row-major D8 direction codes 0..7 (E, SE, S, SW, W, NW, N,
                    NE), 255 = sink/nodata/outside basins
    accum           row-major upstream cell counts (int), 0 outside basins
    basin           row-major basin index into `basins` (255 = outside)
    basins          the polygon ids, in index order
This is general terrain analysis (D8 is O'Callaghan & Mark 1984); nothing in
it is a Denver Water reading, but it travels with the terms because the
simulation that uses it is driven by their data.
"""

import argparse
import heapq
import io
import json
import math
import os
import sys
import urllib.request

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
BRIEF = os.path.join(HERE, "..", "..", "strontia-brief", "basins")
OUT_GRID = os.path.join(HERE, "sim-data", "d8-grid.json")
OUT_REACH = os.path.join(HERE, "sim-data", "reach-elevations.json")

ZOOM = 10
TILE = 256
URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"

BASIN_FILES = [
    ("south-platte-above-strontia", "south-platte-above-strontia-06707525.json"),
    ("blue-river-below-dillon", "blue-river-below-dillon-09050700.json"),
    ("fraser-below-moffat", "fraser-below-moffat-09023562.json"),
    ("williams-fork-parshall", "williams-fork-parshall-09037500.json"),
]
# Nested inside Strontia's polygon; painted after it so its cells win.
NESTED = [("south-platte-above-trumbull", "south-platte-above-trumbull-06701900.json")]

# D8 neighbour offsets, code order E, SE, S, SW, W, NW, N, NE
D8 = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]


def lonlat_to_tile(lon, lat, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    lat_r = math.radians(lat)
    y = (1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n
    return x, y


def tile_to_lonlat(x, y, z):
    n = 2 ** z
    lon = x / n * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon, lat


def fetch_tile(x, y, z):
    req = urllib.request.Request(URL.format(z=z, x=x, y=y),
                                 headers={"User-Agent": "design-storm-2026 fetch_dem_d8"})
    with urllib.request.urlopen(req, timeout=60) as r:
        img = Image.open(io.BytesIO(r.read())).convert("RGB")
    a = np.asarray(img, dtype=np.float64)
    return a[:, :, 0] * 256 + a[:, :, 1] + a[:, :, 2] / 256 - 32768


def load_polygons():
    polys = []
    for pid, fn in BASIN_FILES + NESTED:
        g = json.load(open(os.path.join(BRIEF, fn)))
        for ft in g["features"]:
            if ft["geometry"]["type"] == "Polygon":
                polys.append((pid, np.array(ft["geometry"]["coordinates"][0])))
    return polys


def point_in_poly_mask(lons, lats, poly):
    """Vectorised ray casting. lons/lats are 2-D grids of cell centres."""
    x, y = lons, lats
    inside = np.zeros(x.shape, dtype=bool)
    px, py = poly[:, 0], poly[:, 1]
    n = len(px)
    j = n - 1
    for i in range(n):
        xi, yi, xj, yj = px[i], py[i], px[j], py[j]
        cond = (yi > y) != (yj > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (xj - xi) * (y - yi) / (yj - yi) + xi
        inside ^= cond & (x < xint)
        j = i
    return inside


def priority_flood(elev, valid):
    """Fill depressions so every valid cell drains to the grid edge or to a
    valid/invalid boundary. Returns the filled surface."""
    rows, cols = elev.shape
    filled = elev.copy()
    seen = np.zeros_like(valid)
    heap = []
    for r in range(rows):
        for c in range(cols):
            if not valid[r, c]:
                continue
            edge = r in (0, rows - 1) or c in (0, cols - 1)
            if not edge:
                for dr, dc in D8:
                    rr, cc = r + dr, c + dc
                    if not valid[rr, cc]:
                        edge = True
                        break
            if edge:
                heapq.heappush(heap, (filled[r, c], r, c))
                seen[r, c] = True
    while heap:
        z, r, c = heapq.heappop(heap)
        for dr, dc in D8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < rows and 0 <= cc < cols and valid[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                if filled[rr, cc] < z:
                    filled[rr, cc] = z + 1e-3
                heapq.heappush(heap, (filled[rr, cc], rr, cc))
    return filled


def d8_directions(filled, valid, dx_m, dy_m):
    rows, cols = filled.shape
    direction = np.full((rows, cols), 255, dtype=np.uint8)
    dist = [math.hypot(dr * dy_m, dc * dx_m) for dr, dc in D8]
    for r in range(rows):
        for c in range(cols):
            if not valid[r, c]:
                continue
            best, bcode = 0.0, 255
            for code, (dr, dc) in enumerate(D8):
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    drop = (filled[r, c] - filled[rr, cc]) / dist[code]
                    if drop > best:
                        best, bcode = drop, code
            direction[r, c] = bcode
    return direction


def flow_accumulation(direction, filled, valid):
    rows, cols = direction.shape
    accum = np.ones((rows, cols), dtype=np.int32)
    accum[~valid] = 0
    order = np.argsort(filled.ravel())[::-1]
    for idx in order:
        r, c = divmod(int(idx), cols)
        code = direction[r, c]
        if code == 255:
            continue
        dr, dc = D8[code]
        accum[r + dr, c + dc] += accum[r, c]
    return accum


def block_mean(a, f):
    rows, cols = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    return a[:rows, :cols].reshape(rows // f, f, cols // f, f).mean(axis=(1, 3))


POINT_ZOOM = 12  # ~30 m pixels: resolves canyon floors that z10 smooths away
_tile_cache = {}


def sample_point(lon, lat, z=POINT_ZOOM):
    """Elevation (m) at a point from a finer tile, minimum over a 3x3 pixel
    window so a coordinate a few metres off the channel still reads the floor."""
    x, y = lonlat_to_tile(lon, lat, z)
    key = (int(x), int(y), z)
    if key not in _tile_cache:
        _tile_cache[key] = fetch_tile(*key)
    t = _tile_cache[key]
    px, py = int((x % 1) * TILE), int((y % 1) * TILE)
    w = t[max(0, py - 1):py + 2, max(0, px - 1):px + 2]
    return round(float(w.min()), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", type=int, default=3, help="block-average factor")
    args = ap.parse_args()

    polys = load_polygons()
    allpts = np.vstack([p for _, p in polys])
    west, east = allpts[:, 0].min() - 0.02, allpts[:, 0].max() + 0.08
    south, north = allpts[:, 1].min() - 0.02, allpts[:, 1].max() + 0.02
    # Make sure the Strontia -> Foothills corridor is in the grid.
    east = max(east, -105.0)

    x0, y1 = lonlat_to_tile(west, south, ZOOM)
    x1, y0 = lonlat_to_tile(east, north, ZOOM)
    tx0, tx1, ty0, ty1 = int(x0), int(x1), int(y0), int(y1)
    ntx, nty = tx1 - tx0 + 1, ty1 - ty0 + 1
    print(f"fetching {ntx * nty} tiles at z{ZOOM}", file=sys.stderr)

    mosaic = np.zeros((nty * TILE, ntx * TILE))
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            t = fetch_tile(tx, ty, ZOOM)
            mosaic[(ty - ty0) * TILE:(ty - ty0 + 1) * TILE,
                   (tx - tx0) * TILE:(tx - tx0 + 1) * TILE] = t

    g_west, g_north = tile_to_lonlat(tx0, ty0, ZOOM)
    g_east, g_south = tile_to_lonlat(tx1 + 1, ty1 + 1, ZOOM)

    # Web-mercator tiles are not equal-degree in latitude; at this extent
    # (~1.3 deg) treating rows as equal-degree costs < 1% in cell placement,
    # acceptable for a coarse routing grid. Recorded in the output.
    f = args.factor
    elev = block_mean(mosaic, f)
    rows, cols = elev.shape
    dlon = (g_east - g_west) / cols
    dlat = (g_north - g_south) / rows
    mid_lat = (g_north + g_south) / 2
    dx_m = dlon * 111320 * math.cos(math.radians(mid_lat))
    dy_m = dlat * 110540

    lons = g_west + (np.arange(cols) + 0.5) * dlon
    lats = g_north - (np.arange(rows) + 0.5) * dlat
    LON, LAT = np.meshgrid(lons, lats)

    basin_ids = [pid for pid, _ in BASIN_FILES + NESTED]
    basin = np.full((rows, cols), 255, dtype=np.uint8)
    for i, (pid, poly) in enumerate(polys):
        m = point_in_poly_mask(LON, LAT, poly)
        basin[m] = i
    valid = basin != 255

    print(f"grid {rows}x{cols}, cell ~{dx_m:.0f} m, {int(valid.sum())} basin cells; filling",
          file=sys.stderr)
    filled = priority_flood(elev, valid)
    direction = d8_directions(filled, valid, dx_m, dy_m)
    accum = flow_accumulation(direction, filled, valid)

    grid = {
        "_comment": "Generated by fetch_dem_d8.py from Mapzen terrarium tiles (AWS Open Data). Do not edit.",
        "source": "Terrain: Mapzen via AWS terrain tiles, zoom 10, block-averaged",
        "method": "Priority-flood pit fill; D8 steepest-descent direction; upstream cell-count accumulation. Rows treated as equal-degree (<1% placement error at this extent).",
        "bbox": [g_west, g_south, g_east, g_north],
        "ncols": cols, "nrows": rows,
        "dlon": dlon, "dlat": dlat,
        "cell_m": round((dx_m + dy_m) / 2),
        "basins": basin_ids,
        "elev_m": [int(round(v)) for v in elev.ravel()],
        "dir": direction.ravel().tolist(),
        "accum": accum.ravel().tolist(),
        "basin": basin.ravel().tolist(),
    }
    os.makedirs(os.path.dirname(OUT_GRID), exist_ok=True)
    with open(OUT_GRID, "w") as fh:
        json.dump(grid, fh, separators=(",", ":"))
    print(f"wrote {os.path.relpath(OUT_GRID, HERE)} ({os.path.getsize(OUT_GRID) // 1024} KB)")

    # Reach elevations for build_channels.py: sample a finer tile (z12) at
    # the ends of every flowline and at each system point. The z10 grid
    # above smooths canyon floors by 100-200 m; z12 resolves them.
    reach = {"_comment": f"Generated by fetch_dem_d8.py. Elevations (m) sampled from Mapzen terrarium tiles at zoom {POINT_ZOOM} (3x3 pixel minimum) at flowline endpoints and system points. Do not edit.",
             "source": f"Mapzen terrarium via AWS, zoom {POINT_ZOOM}", "reaches": {}, "points": {}}
    for fn in ("mainstem-flowlines-06707525.json", "downstream-mainstem-06707525.json"):
        g = json.load(open(os.path.join(BRIEF, fn)))
        for ft in g["features"]:
            cid = str(ft["properties"].get("nhdplus_comid"))
            cs = ft["geometry"]["coordinates"]
            reach["reaches"][f"{fn.split('-')[0]}:{cid}"] = {
                "z_start_m": sample_point(*cs[0]),
                "z_end_m": sample_point(*cs[-1]),
            }
    sysjson = json.load(open(os.path.join(HERE, "..", "..", "water-system-3d", "system.json")))
    for p in sysjson["points"]:
        reach["points"][p["id"]] = sample_point(p["lon"], p["lat"])
    for ln in sysjson["lines"]:
        reach["points"][f"line:{ln['id']}:start"] = sample_point(*ln["coords"][0])
        reach["points"][f"line:{ln['id']}:end"] = sample_point(*ln["coords"][-1])
    print(f"sampled {len(_tile_cache)} z{POINT_ZOOM} tiles for point elevations", file=sys.stderr)
    with open(OUT_REACH, "w") as fh:
        json.dump(reach, fh, indent=1)
    print(f"wrote {os.path.relpath(OUT_REACH, HERE)}")


if __name__ == "__main__":
    main()

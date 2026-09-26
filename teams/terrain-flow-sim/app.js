/* Terrain flow simulation for the Explore DDD 2026 Design Storm (Denver Water).
   Hand-maintained; no build step. Served from teams/terrain-flow-sim/ by the
   repo's serve.py. Reads generated JSON from sim-data/ (see README.md), the
   committed geodata in ../../strontia-brief/, the system points in
   ../../water-system-3d/system.json and the aligned series in
   ../explainable-viz/viz-data/. Every optional bundle degrades to a note. */

"use strict";

const ROOT = "../../";
const VIZ = "../explainable-viz/viz-data/";
const SIM = "sim-data/";

const HOME = { center: [-105.55, 39.35], zoom: 8.1, pitch: 58, bearing: -14 };
const KIND_COLORS = { reservoir: "#3987e5", gage: "#d95926", snotel: "#ffffff", plant: "#199e70" };
const MODE_COLORS = { overland: "#a9d7ff", channel: "#4fb3ff", tunnel: "#ffd166", northfork: "#4fb3ff",
                      reservoir: "#7fb0ff", conduit: "#4cd7a5" };

const $ = (id) => document.getElementById(id);
const content = $("content");

const state = {
  data: {}, missing: [], mode: "flow",
  day: 0, playing: false, speed: 1, tick: 0, particles: [], arrivedToday: 0, arrivals: [],
  layersReady: false, tierByDay: null, channelCells: null, proposedShown: false,
  selectedStorm: null,
};

/* ------------------------------------------------------------------ map */

const map = new maplibregl.Map({
  container: "map",
  center: HOME.center, zoom: HOME.zoom, pitch: HOME.pitch, bearing: HOME.bearing,
  maxPitch: 72, attributionControl: false,
  style: {
    version: 8,
    glyphs: "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf",
    sources: {
      satellite: { type: "raster", tileSize: 256, maxzoom: 18,
        tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"] },
      dem: { type: "raster-dem", tileSize: 256, maxzoom: 13, encoding: "terrarium",
        tiles: ["https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"] },
    },
    layers: [{ id: "satellite", type: "raster", source: "satellite" }],
    sky: { "sky-color": "#88b8e0", "horizon-color": "#d8e8f0", "fog-color": "#c8d8e0",
           "sky-horizon-blend": 0.6, "horizon-fog-blend": 0.7 },
  },
});
map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "top-right");
map.on("load", () => { if (map.setTerrain) map.setTerrain({ source: "dem", exaggeration: 1.35 }); });

async function fetchJson(url, optional = false) {
  try {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${r.status} ${url}`);
    return await r.json();
  } catch (e) {
    if (!optional) throw e;
    state.missing.push(url);
    return null;
  }
}

function fmt(v, d = 1) { return v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(d); }
function esc(s) { return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c])); }
function badge(kind, text) { return `<span class="badge ${kind}">${text || kind}</span>`; }
function segLenM(a, b) {
  const dx = (b[0] - a[0]) * 111320 * Math.cos(((a[1] + b[1]) / 2) * Math.PI / 180);
  const dy = (b[1] - a[1]) * 110540;
  return Math.hypot(dx, dy);
}

/* ------------------------------------------------------------------ load */

async function loadEverything() {
  const d = state.data;
  [d.system, d.channels, d.forcings, d.d8] = await Promise.all([
    fetchJson(ROOT + "water-system-3d/system.json"),
    fetchJson(SIM + "channels.json", true),
    fetchJson(SIM + "forcings.json", true),
    fetchJson(SIM + "d8-grid.json", true),
  ]);
  [d.basinSP, d.basinBlue, d.basinFraser, d.basinWF, d.basinTrumbull, d.riversMain, d.riversDown,
   d.storms, d.stormWindows, d.enso, d.trends, d.station, d.travel, d.basinsMeta, d.basinFlows] = await Promise.all([
    fetchJson(ROOT + "strontia-brief/basins/south-platte-above-strontia-06707525.json", true),
    fetchJson(ROOT + "strontia-brief/basins/blue-river-below-dillon-09050700.json", true),
    fetchJson(ROOT + "strontia-brief/basins/fraser-below-moffat-09023562.json", true),
    fetchJson(ROOT + "strontia-brief/basins/williams-fork-parshall-09037500.json", true),
    fetchJson(ROOT + "strontia-brief/basins/south-platte-above-trumbull-06701900.json", true),
    fetchJson(ROOT + "strontia-brief/basins/mainstem-flowlines-06707525.json", true),
    fetchJson(ROOT + "strontia-brief/basins/downstream-mainstem-06707525.json", true),
    fetchJson(VIZ + "storms.json", true),
    fetchJson(SIM + "storm-windows.json", true),
    fetchJson(SIM + "enso-context.json", true),
    fetchJson(SIM + "trends.json", true),
    fetchJson(SIM + "noaa-station.json", true),
    fetchJson("results/travel_time_summary.json", true),
    fetchJson(VIZ + "basins.json", true),
    fetchJson(VIZ + "basin-flows.json", true),
  ]);
  d.byId = {};
  d.system.points.forEach((p) => { d.byId[p.id] = p; });
  d.system.lines.forEach((l) => { d.byId[l.id] = l; });

  if (map.loaded()) addLayers(); else map.once("load", addLayers);
}

function addLayers() {
  const d = state.data;
  const basinFiles = [
    ["south-platte-above-strontia", d.basinSP], ["blue-river-below-dillon", d.basinBlue],
    ["fraser-below-moffat", d.basinFraser], ["williams-fork-parshall", d.basinWF],
    ["south-platte-above-trumbull", d.basinTrumbull],
  ].filter(([, f]) => f);
  const basins = { type: "FeatureCollection", features: basinFiles.flatMap(([id, f], n) =>
    f.features.map((ft) => ({ ...ft, id: n, properties: { ...ft.properties, basin: id } }))) };
  d.basinIndex = Object.fromEntries(basinFiles.map(([id], n) => [id, n]));
  const rivers = { type: "FeatureCollection", features: [
    ...(d.riversMain ? d.riversMain.features : []), ...(d.riversDown ? d.riversDown.features : [])] };
  const tunnels = { type: "FeatureCollection", features: d.system.lines.map((l) => ({
    type: "Feature", properties: { id: l.id, name: l.name }, geometry: { type: "LineString", coordinates: l.coords } })) };
  const points = { type: "FeatureCollection", features: d.system.points.map((p) => ({
    type: "Feature", properties: { id: p.id, kind: p.kind, name: p.name }, geometry: { type: "Point", coordinates: [p.lon, p.lat] } })) };

  map.addSource("basins", { type: "geojson", data: basins });
  map.addSource("rivers", { type: "geojson", data: rivers });
  map.addSource("tunnels", { type: "geojson", data: tunnels });
  map.addSource("points", { type: "geojson", data: points });
  map.addSource("particles", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
  map.addSource("proposed", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
  map.addSource("northfork", { type: "geojson", data: northForkLine() });

  map.addLayer({ id: "basin-fill", type: "fill", source: "basins",
    paint: { "fill-color": ["coalesce", ["feature-state", "fill"], "#9db8d8"],
             "fill-opacity": ["coalesce", ["feature-state", "op"], 0.06] } });
  map.addLayer({ id: "basin-line", type: "line", source: "basins",
    paint: { "line-color": "rgba(255,255,255,0.55)", "line-width": 1.4, "line-dasharray": [1, 2] } });
  map.addLayer({ id: "river-line", type: "line", source: "rivers",
    paint: { "line-color": "#6fc3ff", "line-width": 2, "line-opacity": 0.9 } });
  map.addLayer({ id: "northfork-line", type: "line", source: "northfork",
    paint: { "line-color": "#6fc3ff", "line-width": 1.2, "line-opacity": 0.6, "line-dasharray": [1, 1.5] } });
  map.addLayer({ id: "tunnel-casing", type: "line", source: "tunnels",
    paint: { "line-color": "rgba(16,20,24,0.85)", "line-width": 6 } });
  map.addLayer({ id: "tunnel-line", type: "line", source: "tunnels",
    paint: { "line-color": "#ffd166", "line-width": 2.6, "line-dasharray": [2.2, 1.6] } });
  map.addLayer({ id: "particles", type: "circle", source: "particles",
    paint: { "circle-radius": ["interpolate", ["linear"], ["zoom"], 7, 1.8, 10, 3.2, 13, 5],
             "circle-color": ["get", "c"], "circle-opacity": 0.85,
             "circle-stroke-width": 0.4, "circle-stroke-color": "rgba(16,20,24,0.6)" } });
  map.addLayer({ id: "points", type: "circle", source: "points",
    paint: { "circle-radius": ["match", ["get", "kind"], "snotel", 3.5, 5],
             "circle-color": ["match", ["get", "kind"], "reservoir", KIND_COLORS.reservoir, "gage", KIND_COLORS.gage,
                              "plant", KIND_COLORS.plant, KIND_COLORS.snotel],
             "circle-stroke-width": 1.5, "circle-stroke-color": "rgba(16,20,24,0.9)" } });
  map.addLayer({ id: "point-labels", type: "symbol", source: "points",
    layout: { "text-field": ["step", ["zoom"], "", 9.4, ["get", "name"]], "text-font": ["Noto Sans Regular"],
              "text-size": 11, "text-offset": [0, 1.0], "text-anchor": "top", "text-optional": true },
    paint: { "text-color": "#fff", "text-halo-color": "rgba(16,20,24,0.9)", "text-halo-width": 1.4 } });
  map.addLayer({ id: "proposed", type: "circle", source: "proposed",
    paint: { "circle-radius": 7, "circle-color": "#ff7ad9", "circle-opacity": 0.9,
             "circle-stroke-width": 2, "circle-stroke-color": "#fff" }, layout: { visibility: "none" } });
  map.addLayer({ id: "proposed-labels", type: "symbol", source: "proposed",
    layout: { "text-field": ["get", "label"], "text-font": ["Noto Sans Regular"], "text-size": 11,
              "text-offset": [0, -1.3], "text-anchor": "bottom", "text-allow-overlap": true, visibility: "none" },
    paint: { "text-color": "#ff7ad9", "text-halo-color": "rgba(16,20,24,0.95)", "text-halo-width": 1.6 } });

  const tip = $("tip");
  for (const layer of ["points", "proposed"]) {
    map.on("mousemove", layer, (e) => {
      const f = e.features[0];
      tip.textContent = f.properties.name || f.properties.label;
      tip.style.left = e.point.x + "px"; tip.style.top = e.point.y + "px"; tip.style.display = "block";
    });
    map.on("mouseleave", layer, () => { tip.style.display = "none"; });
  }

  state.layersReady = true;
  prepareSimulation();
  wireTabs();
  const hash = location.hash.replace("#", "");
  setMode(["flow", "gauge", "physics", "climate", "feasibility"].includes(hash) ? hash : "flow");
}

function northForkLine() {
  // No committed flowline for the North Fork: a straight line Grant -> confluence, drawn faint and dashed.
  const a = state.data.byId["gage-06702500"], b = state.data.byId["gage-06707000"];
  if (!a || !b) return { type: "FeatureCollection", features: [] };
  return { type: "FeatureCollection", features: [{ type: "Feature", properties: { approx: true },
    geometry: { type: "LineString", coordinates: [[a.lon, a.lat], [b.lon, b.lat]] } }] };
}

/* -------------------------------------------------------------- tabs */

function wireTabs() {
  document.querySelectorAll("#tabs button").forEach((b) => {
    b.onclick = () => setMode(b.dataset.mode);
  });
  window.addEventListener("hashchange", () => {
    const h = location.hash.replace("#", "");
    if (h !== state.mode && ["flow", "gauge", "physics", "climate", "feasibility"].includes(h)) setMode(h);
  });
}

function setMode(mode) {
  state.mode = mode;
  location.hash = mode;
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.mode === mode));
  showProposed(mode === "feasibility");
  if (state.layersReady && map.getLayer("particles")) map.setLayoutProperty("particles", "visibility", mode === "flow" ? "visible" : "none");
  $("counter").style.display = mode === "flow" ? "block" : "none";
  if (mode !== "flow" && state.playing) pause();
  ({ flow: renderFlow, gauge: renderGauge, physics: renderPhysics, climate: renderClimate, feasibility: renderFeasibility })[mode]();
}

function missingNote(what, script) {
  return `<div class="missing">${esc(what)} is not available: run <code>${esc(script)}</code> (see README.md). The rest of the page still works.</div>`;
}

/* --------------------------------------------------------- simulation */

const SIM_CFG = {
  ticksPerDay: 30,          // at ~30 fps one simulated day per second at speed 1
  maxParticles: 3200,
  seedPerInchPerTick: 22,   // particles seeded per basin per tick per inch of daily liquid input
  overlandCellsPerTick: 1,  // illustrative hillslope speed (real overland flow is far slower and mostly subsurface)
  channelAccumThreshold: 60,// D8 cells (~7.5 km2) above which a cell is drawn as a small channel and moves faster
  reservoirDwellDays: 1.0,  // display dwell gage -> dam. DW: ~4 h to intake; the plant tracks the gage 1-2 days later
};

function prepareSimulation() {
  const d = state.data;
  if (!d.d8 || !d.channels || !d.forcings) return;
  const g = d.d8;
  // Rasterise mainstem reach vertices to grid cells so overland particles can join the channel.
  const cells = new Map();
  d.channels.mainstem.reaches.forEach((r, ri) => {
    r.coords.forEach((xy, vi) => {
      const c = Math.floor((xy[0] - g.bbox[0]) / g.dlon), row = Math.floor((g.bbox[3] - xy[1]) / g.dlat);
      for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
        const key = (row + dr) * g.ncols + (c + dc);
        if (!cells.has(key)) cells.set(key, [ri, vi]);
      }
    });
  });
  state.channelCells = cells;
  // Per-basin cell lists for seeding.
  state.basinCells = {};
  g.basin.forEach((b, i) => { if (b !== 255) (state.basinCells[b] = state.basinCells[b] || []).push(i); });
  // Flow tier per day (nearest tier by that day's flow) for Manning velocities.
  const tiers = Object.entries(d.channels.mainstem.travel).filter(([k]) => k !== "aug_2026_peak");
  state.tierByDay = d.forcings.series.flow.map((q) => {
    if (q === null) return tiers.find(([k]) => k === "median")[0];
    let best = tiers[0];
    for (const t of tiers) if (Math.abs(t[1].flow_cfs - q) < Math.abs(best[1].flow_cfs - q)) best = t;
    return best[0];
  });
  // Confluence join point: nearest mainstem vertex to gage 06707000
  const conf = d.byId["gage-06707000"];
  state.joinAt = conf ? nearestReachVertex(conf.lon, conf.lat) : [0, 0];
  state.day = d.forcings.dates.indexOf("2023-05-10") > 0 ? d.forcings.dates.indexOf("2023-05-10") : 0;
}

function nearestReachVertex(lon, lat) {
  let best = [0, 0], bd = Infinity;
  state.data.channels.mainstem.reaches.forEach((r, ri) => r.coords.forEach((xy, vi) => {
    const dd = segLenM(xy, [lon, lat]);
    if (dd < bd) { bd = dd; best = [ri, vi]; }
  }));
  return best;
}

function cellLonLat(i) {
  const g = state.data.d8, r = Math.floor(i / g.ncols), c = i % g.ncols;
  return [g.bbox[0] + (c + 0.5) * g.dlon, g.bbox[3] - (r + 0.5) * g.dlat];
}

const D8 = [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]];

function seedParticles() {
  const d = state.data, g = d.d8, f = d.forcings;
  if (state.particles.length > SIM_CFG.maxParticles) return;
  for (const [id, b] of Object.entries(f.basins)) {
    const bi = g.basins.indexOf(id);
    if (bi < 0) continue;
    const input = b.input_in[state.day];
    if (!input) continue;
    let n = input * SIM_CFG.seedPerInchPerTick;
    // Trumbull is nested inside Strontia's polygon; its cells are painted separately, so seed both from their own inputs
    const cells = state.basinCells[bi];
    if (!cells) continue;
    const whole = Math.floor(n); n = whole + (Math.random() < n - whole ? 1 : 0);
    for (let k = 0; k < n; k++) {
      const cell = cells[Math.floor(Math.random() * cells.length)];
      const [lon, lat] = cellLonLat(cell);
      state.particles.push({ m: "overland", cell, lon, lat, basin: id, age: 0 });
    }
  }
}

function stepParticle(p) {
  const d = state.data, g = d.d8, ch = d.channels;
  p.age++;
  if (p.m === "overland") {
    const steps = g.accum[p.cell] > SIM_CFG.channelAccumThreshold ? 3 : SIM_CFG.overlandCellsPerTick;
    for (let s = 0; s < steps; s++) {
      if (state.channelCells.has(p.cell)) {
        const [ri, vi] = state.channelCells.get(p.cell);
        p.m = "channel"; p.ri = ri; p.vi = vi; p.frac = 0; return true;
      }
      const dir = g.dir[p.cell];
      if (dir === 255) return outletHandoff(p);
      const [dr, dc] = D8[dir];
      p.cell += dr * g.ncols + dc;
      if (p.cell < 0 || p.cell >= g.dir.length) return false;
    }
    [p.lon, p.lat] = cellLonLat(p.cell);
    return true;
  }
  if (p.m === "channel") {
    const tier = ch.mainstem.travel[state.tierByDay[state.day]];
    let dist = tier.per_reach[p.ri].v_m_s * 86400 / SIM_CFG.ticksPerDay;
    while (dist > 0) {
      const r = ch.mainstem.reaches[p.ri];
      if (p.vi >= r.coords.length - 1) {
        if (p.ri === 0) { p.m = "reservoir"; p.s = 0; return true; }
        p.ri -= 1; p.vi = 0; p.frac = 0; continue;
      }
      const a = r.coords[p.vi], b = r.coords[p.vi + 1], L = segLenM(a, b);
      const remain = L * (1 - p.frac);
      if (dist >= remain) { dist -= remain; p.vi += 1; p.frac = 0; }
      else { p.frac += dist / L; dist = 0; }
    }
    const r = ch.mainstem.reaches[p.ri], vi = Math.min(p.vi, r.coords.length - 2);
    const a = r.coords[vi], b = r.coords[vi + 1];
    p.lon = a[0] + (b[0] - a[0]) * p.frac; p.lat = a[1] + (b[1] - a[1]) * p.frac;
    return true;
  }
  if (p.m === "reservoir") {
    const total = ch.gage_to_dam_m || 5000;
    p.s += total / (SIM_CFG.reservoirDwellDays * SIM_CFG.ticksPerDay);
    if (p.s >= total) { p.m = "conduit"; p.s = 0; return true; }
    [p.lon, p.lat] = pointAlong(ch.downstream_coords, p.s);
    return true;
  }
  if (p.m === "conduit") {
    const line = d.byId["conduit-26"];
    if (!line) { arrive(); return false; }
    const total = lineLen(line.coords);
    p.s += total / 4;  // Darcy bound says < 30 min; drawn over 4 ticks so it is visible
    if (p.s >= total) { arrive(); return false; }
    [p.lon, p.lat] = pointAlong(line.coords, p.s);
    return true;
  }
  if (p.m === "tunnel" || p.m === "northfork") {
    const total = lineLen(p.path);
    p.s += total / p.ticks;
    if (p.s >= total) {
      if (p.next === "northfork") {
        const a = d.byId["gage-06702500"], b = d.byId["gage-06707000"];
        p.m = "northfork"; p.path = [[a.lon, a.lat], [b.lon, b.lat]]; p.s = 0; p.ticks = 20; p.next = "join";
        return true;
      }
      if (p.next === "join") { p.m = "channel"; [p.ri, p.vi] = state.joinAt; p.frac = 0; return true; }
      return false;
    }
    [p.lon, p.lat] = pointAlong(p.path, p.s);
    return true;
  }
  return false;
}

function outletHandoff(p) {
  const d = state.data;
  if (p.basin === "blue-river-below-dillon") {
    const t = d.byId["roberts"]; if (!t) return false;
    const hrs = (d.channels.conduits.find((c) => c.id === "roberts") || {}).hours_at_darcy_v || 3.3;
    p.m = "tunnel"; p.path = t.coords; p.s = 0; p.ticks = Math.max(3, Math.round(hrs / 24 * SIM_CFG.ticksPerDay * 3)); p.next = "northfork";
    return true;
  }
  if (p.basin === "fraser-below-moffat") {
    const t = d.byId["moffat"]; if (!t) return false;
    p.m = "tunnel"; p.path = t.coords; p.s = 0; p.ticks = 6; p.next = "end"; state.counts.moffat++;
    return true;
  }
  if (p.basin === "williams-fork-parshall") { state.counts.williams++; return false; }
  return false;  // local pit or basin-edge sink inside the South Platte polygons
}

function pointAlong(coords, s) {
  for (let i = 0; i < coords.length - 1; i++) {
    const L = segLenM(coords[i], coords[i + 1]);
    if (s <= L) { const f = L ? s / L : 0; return [coords[i][0] + (coords[i + 1][0] - coords[i][0]) * f, coords[i][1] + (coords[i + 1][1] - coords[i][1]) * f]; }
    s -= L;
  }
  return coords[coords.length - 1];
}
function lineLen(coords) { let L = 0; for (let i = 0; i < coords.length - 1; i++) L += segLenM(coords[i], coords[i + 1]); return L; }

function arrive() { state.arrivedToday++; }

function simTick() {
  if (!state.playing) return;
  const f = state.data.forcings;
  for (let k = 0; k < state.speed; k++) {
    state.tick++;
    seedParticles();
    state.particles = state.particles.filter(stepParticle);
    if (state.tick % SIM_CFG.ticksPerDay === 0) {
      state.arrivals[state.day] = state.arrivedToday; state.arrivedToday = 0;
      if (state.day >= f.dates.length - 1) { pause(); break; }
      setDay(state.day + 1, false);
    }
  }
  drawParticles();
  updateCounter();
  // fixed cadence: ticksPerDay ticks per second at speed x1, independent of the map's frame rate
  state.timer = setTimeout(simTick, 1000 / SIM_CFG.ticksPerDay);
}

function drawParticles() {
  const src = map.getSource("particles");
  if (!src) return;
  src.setData({ type: "FeatureCollection", features: state.particles.map((p) => ({
    type: "Feature", properties: { c: MODE_COLORS[p.m] || "#fff" }, geometry: { type: "Point", coordinates: [p.lon, p.lat] } })) });
}

function play() {
  if (!state.data.d8) return;
  state.playing = true; $("play-btn") && ($("play-btn").textContent = "Pause");
  clearTimeout(state.timer);
  simTick();
}
function pause() { state.playing = false; clearTimeout(state.timer); $("play-btn") && ($("play-btn").textContent = "Play"); }

function setDay(i, redraw = true) {
  state.day = Math.max(0, Math.min(state.data.forcings.dates.length - 1, i));
  const r = $("day-range"); if (r) r.value = state.day;
  const t = $("day-label"); if (t) t.textContent = state.data.forcings.dates[state.day];
  updateDayReadout();
  shadeBasins();
  if (redraw) drawParticles();
}

function shadeBasins() {
  const d = state.data, f = d.forcings;
  if (!f || !d.basinIndex) return;
  for (const [id, b] of Object.entries(f.basins)) {
    const n = d.basinIndex[id]; if (n === undefined) continue;
    const v = b.input_in[state.day];
    const t = v === null ? 0 : Math.min(1, v / 0.8);
    map.setFeatureState({ source: "basins", id: n }, { fill: v === null ? "#666" : `rgb(${Math.round(120 + 100 * (1 - t))},${Math.round(160 + 60 * t)},${Math.round(200 + 55 * t)})`, op: 0.08 + 0.35 * t });
  }
}

function updateCounter() {
  const f = state.data.forcings, i = state.day;
  const by = {};
  for (const p of state.particles) by[p.m] = (by[p.m] || 0) + 1;
  const lastArr = state.arrivals[i - 1];
  $("counter").innerHTML = `<div><span class="k">Day</span> <b>${f.dates[i]}</b> &middot; flow tier <b>${state.tierByDay[i]}</b></div>
    <div><span class="k">Particles</span> ${state.particles.length} &middot; overland ${by.overland || 0} &middot; river ${(by.channel || 0) + (by.northfork || 0)} &middot; tunnel ${by.tunnel || 0} &middot; reservoir ${by.reservoir || 0}</div>
    <div><span class="k">Arrived at Foothills yesterday</span> ${lastArr === undefined ? "—" : lastArr} &middot; <span class="k">to Moffat</span> ${state.counts.moffat} &middot; <span class="k">Williams Fork (path not drawn)</span> ${state.counts.williams}</div>
    <div class="k" style="margin-top:4px">Particle counts are a picture of routing, not a volume.</div>`;
}
state.counts = { moffat: 0, williams: 0 };

/* ------------------------------------------------------------ flow panel */

function renderFlow() {
  const d = state.data;
  if (!d.d8 || !d.channels || !d.forcings) {
    content.innerHTML = `<h2>Flow simulation</h2>` +
      (!d.d8 ? missingNote("The D8 terrain grid", "python3 fetch_dem_d8.py") : "") +
      (!d.channels ? missingNote("The channel network", "python3 build_channels.py") : "") +
      (!d.forcings ? missingNote("The daily forcings", "python3 build_sim_data.py") : "") +
      `<p class="note">Basins, rivers and tunnels are still drawn from the committed geodata.</p>`;
    return;
  }
  const f = d.forcings;
  content.innerHTML = `
    <h2>Flow simulation</h2>
    <p>Each day, snowmelt (SWE lost at the basin's snow pillows) plus rain (the one NOAA gauge) seeds particles across each
      basin. They follow the terrain's steepest descent cell by cell (${badge("computed", "D8 on real terrain")}), join the
      South Platte at the mainstem, travel at the Manning velocity for that day's flow tier (${badge("computed")} given an
      ${badge("assumed", "assumed n, width")}), cross Strontia Springs and drop through Conduit 26 to Foothills.</p>
    <div id="timebar">
      <div class="time"><span id="day-label">${f.dates[state.day]}</span><span id="day-readout" class="note"></span></div>
      <input id="day-range" type="range" min="0" max="${f.dates.length - 1}" value="${state.day}">
      <div class="row">
        <button class="action" id="play-btn">${state.playing ? "Pause" : "Play"}</button>
        <button class="action quiet" id="speed-btn">Speed ×${state.speed}</button>
        <button class="action quiet" id="clear-btn">Clear</button>
      </div>
    </div>
    <div class="chips" id="storm-chips"></div>
    <div class="legend-row"><span class="dot" style="background:${MODE_COLORS.overland}"></span> overland (hillslope, illustrative speed)</div>
    <div class="legend-row"><span class="dot" style="background:${MODE_COLORS.channel}"></span> river, Manning velocity for the day's flow tier</div>
    <div class="legend-row"><span class="dot" style="background:${MODE_COLORS.tunnel}"></span> tunnel under the Divide (Roberts, Moffat)</div>
    <div class="legend-row"><span class="dot" style="background:${MODE_COLORS.reservoir}"></span> Strontia Springs Reservoir (display dwell ${SIM_CFG.reservoirDwellDays} day)</div>
    <div class="legend-row"><span class="dot" style="background:${MODE_COLORS.conduit}"></span> Conduit 26 to Foothills</div>
    <canvas id="input-chart" class="chart"></canvas>
    <div class="chart-label">Daily liquid input per basin (melt + rain, inches) around the current day; grey bars: NOAA precipitation at Strontia Springs Dam.</div>
    <details class="caveat"><summary>What is real here, and what is a cartoon</summary>
      <p>${badge("computed")} Terrain routing (D8 on Mapzen elevation), basin membership, daily melt from the pillows, rain from the
        gauge, reach slopes and lengths, Manning velocities per flow tier, tunnel head. ${badge("assumed")} Manning n = ${d.channels.assumed.manning_n},
        channel width ${d.channels.assumed.channel_width_m} m, an upstream flow-attenuation proxy, hillslope speed (one cell per tick; real
        hillslope water moves mostly through soil at centimetres per hour), the 1-day reservoir dwell. ${badge("general")} The D8 method itself,
        Manning's equation, the fact that most snowmelt reaches the river as groundwater rather than surface flow.</p>
      <p>${badge("quote", "Denver Water")} Water at the sentinel gage reaches the intake in about four hours, yet the plant record tracks the gage a day or two later,
        because the reservoir mixes what arrives (system.json). The dwell drawn here sits between those.</p>
      <p>Williams Fork water leaves its basin through tunnels whose geometry is not committed here, so its particles are counted and dropped at the
        outlet. Fraser water reaches the Moffat plant via Gross Reservoir, also not drawn. The North Fork below the Roberts Tunnel is a straight
        dashed line: no flowline committed.</p>
    </details>`;
  $("play-btn").onclick = () => (state.playing ? pause() : play());
  $("speed-btn").onclick = () => { state.speed = state.speed === 1 ? 3 : state.speed === 3 ? 8 : 1; $("speed-btn").textContent = `Speed ×${state.speed}`; };
  $("clear-btn").onclick = () => { state.particles = []; drawParticles(); };
  $("day-range").oninput = (e) => setDay(Number(e.target.value));
  const chips = $("storm-chips");
  (d.storms ? d.storms.events.concat([{ date: d.storms.reference_storm.date, prcp_in: "Aug 14–15 replay" }]) : []).forEach((ev) => {
    const c = document.createElement("span"); c.className = "chip";
    c.innerHTML = `<span class="dot" style="background:#d95926"></span>${ev.date}${typeof ev.prcp_in === "number" ? ` · ${ev.prcp_in} in` : ""}`;
    c.title = "Jump two days before this storm";
    c.onclick = () => { const i = f.dates.indexOf(ev.date); if (i >= 0) { setDay(Math.max(0, i - 2)); if (!state.playing) play(); } };
    chips.appendChild(c);
  });
  setDay(state.day);
  updateCounter();
  drawInputChart();
}

function updateDayReadout() {
  const f = state.data.forcings, i = state.day, el = $("day-readout");
  if (!el) return;
  const s = f.series;
  el.textContent = `flow ${fmt(s.flow[i], 0)} cfs · rain ${fmt(s.prcp[i], 2)} in · TOC ${fmt(s.toc[i], 2)} · Alk ${fmt(s.alk[i], 0)}`;
  drawInputChart();
}

function drawInputChart() {
  const cv = $("input-chart"); if (!cv) return;
  const f = state.data.forcings, i0 = Math.max(0, state.day - 45), i1 = Math.min(f.dates.length - 1, state.day + 15);
  const series = Object.entries(f.basins).map(([id, b]) => ({ id, values: b.input_in.slice(i0, i1 + 1) }));
  const bars = f.series.prcp.slice(i0, i1 + 1);
  drawChart(cv, { xs: f.dates.slice(i0, i1 + 1), lines: series.map((s, k) => ({ values: s.values, color: ["#4fb3ff", "#8fd3ff", "#ffd166", "#4cd7a5", "#ff9d3d"][k] })),
    bars: { values: bars, color: "rgba(255,255,255,0.25)" }, cursor: state.day - i0, ymax: 1.2, ylabel: "in/day" });
}

/* ----------------------------------------------------------- gauge panel */

function renderGauge() {
  const d = state.data, st = d.station, f = d.forcings;
  const lonlat = st ? [st.lon, st.lat] : [-105.12116, 39.43426];
  map.flyTo({ center: lonlat, zoom: 12.6, pitch: 62, bearing: -30, speed: 0.8 });
  if (!f) { content.innerHTML = `<h2>The rain gauge</h2>${missingNote("Daily forcings", "python3 build_sim_data.py")}`; return; }
  const i = state.day, p = f.series.prcp[i];
  content.innerHTML = `
    <h2>How the rain gauge records</h2>
    <p>${st ? `<b>${esc(st.name)}</b> (${st.id}) is a ${badge("general", "COOP")} station: a standard 8-inch can read once a day by an observer,
      which is why the record has a TOBS column. It has recorded since ${st.record_start}.` : "NOAA station USC00058022."}
      ${badge("computed", "sited")} It stands at <b>Strontia Springs Dam</b>, the outlet of the basin the simulation routes,
      not in the mountains where the snow and the storms are.</p>
    <div class="gauge-wrap">
      <svg id="gauge-svg" viewBox="0 0 90 150" width="90" height="150"></svg>
      <div class="gauge-read"><span class="note">Daily total, ${f.dates[i]}</span><b id="gauge-val">${fmt(p, 2)} in</b>
        <span class="note">flow ${fmt(f.series.flow[i], 0)} cfs · turbidity ${fmt(f.series.turb[i], 1)} FNU</span>
        <div class="row"><button class="action quiet" id="g-prev">◂ day</button><button class="action quiet" id="g-next">day ▸</button></div>
      </div>
    </div>
    <canvas id="gauge-chart" class="chart tall"></canvas>
    <div class="chart-label">Gauge (grey bars, inches) against river flow at Waterton (blue) and sentinel turbidity (orange), ±30 days.</div>
    <h3>What the 15-minute record shows that the gauge cannot</h3>
    <div class="chips" id="window-chips"></div>
    <canvas id="storm-chart" class="chart tall"></canvas>
    <div id="storm-caption" class="chart-label"></div>
    <h3>Why one gauge is the weakest link</h3>
    <ul>
      <li>${badge("computed")} The gauge's day boundary hides timing: a storm at 6 pm and one at 6 am are the same number.</li>
      <li>${badge("computed")} Of the ${d.storms ? d.storms.events.length : "top"} biggest gauge days, the river's daily-mean peak arrived
        ${d.storms ? `${Math.min(...d.storms.events.map((e) => e.flow_peak_offset_days))} to ${Math.max(...d.storms.events.map((e) => e.flow_peak_offset_days))}` : "several"} days later
        and ${d.storms ? d.storms.events.filter((e) => e.turb_peak_offset_days === null).length : "several"} left no turbidity response at all
        (winter snow, or rain that fell only at the dam).</li>
      <li>${badge("general")} Point gauges under-catch in wind and snow, and a summer convective cell is a few kilometres wide: the basin above is
        roughly 2,600 square miles, so most storms miss this can entirely, and the can misses most storms.</li>
      <li>${badge("general")} NOAA publishes this record about two days late; nothing built on it can be a 12-hour warning.</li>
      <li>${badge("general")} SNOTEL pillows also carry precipitation gauges (element PREC), and NEXRAD quantitative precipitation estimates cover the
        whole basin at ~1 km every few minutes. Neither is in the shipped data; both are the obvious upgrade (see 12-hour feasibility).</li>
    </ul>`;
  drawGaugeSvg(p);
  $("g-prev").onclick = () => { setDay(state.day - 1); renderGauge(); };
  $("g-next").onclick = () => { setDay(state.day + 1); renderGauge(); };
  const i0 = Math.max(0, i - 30), i1 = Math.min(f.dates.length - 1, i + 30);
  drawChart($("gauge-chart"), { xs: f.dates.slice(i0, i1 + 1), bars: { values: f.series.prcp.slice(i0, i1 + 1), color: "rgba(255,255,255,0.35)" }, ymax: 2,
    lines: [{ values: f.series.flow.slice(i0, i1 + 1), color: "#4fb3ff", axis: "right" }, { values: f.series.turb.slice(i0, i1 + 1), color: "#ff9d3d", axis: "right2" }],
    cursor: i - i0, ylabel: "in" });
  renderStormWindows();
}

function drawGaugeSvg(p) {
  const svg = $("gauge-svg"); if (!svg) return;
  const h = Math.min(1, (p || 0) / 3) * 100;  // 3 inches fills the tube
  svg.innerHTML = `
    <defs><linearGradient id="w" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#4fb3ff"/><stop offset="1" stop-color="#1c5cab"/></linearGradient></defs>
    <path d="M10 8 L80 8 L52 34 L52 40 L38 40 L38 34 Z" fill="none" stroke="#b8c0c8" stroke-width="2"/>
    <rect x="30" y="40" width="30" height="100" rx="3" fill="rgba(255,255,255,0.06)" stroke="#b8c0c8" stroke-width="2"/>
    <rect x="31" y="${140 - h}" width="28" height="${h}" fill="url(#w)"/>
    ${[0, 1, 2, 3].map((k) => `<line x1="60" x2="66" y1="${140 - k * 100 / 3}" y2="${140 - k * 100 / 3}" stroke="#b8c0c8"/><text x="68" y="${143 - k * 100 / 3}" font-size="7" fill="#b8c0c8">${k}"</text>`).join("")}
    <text x="45" y="148" font-size="6" text-anchor="middle" fill="#b8c0c8">8-inch can</text>`;
}

function renderStormWindows() {
  const sw = state.data.stormWindows, chips = $("window-chips");
  if (!sw) { chips.insertAdjacentHTML("afterend", missingNote("15-minute storm windows", "python3 fetch_storm_windows.py")); $("storm-chart").style.display = "none"; return; }
  sw.windows.forEach((w, k) => {
    const c = document.createElement("span"); c.className = "chip" + (state.selectedStorm === k ? "" : " off");
    c.innerHTML = `<span class="dot" style="background:#d95926"></span>${w.date}`;
    c.onclick = () => { state.selectedStorm = k; renderGauge(); };
    chips.appendChild(c);
  });
  if (state.selectedStorm === null) state.selectedStorm = sw.windows.length - 1;
  const w = sw.windows[state.selectedStorm];
  const q = w.series["06701900:discharge_cfs"] || [], turb = w.series["06707525:turbidity_fnu"] || [];
  const times = q.map((x) => x[0]);
  const turbMap = new Map(turb);
  const f = state.data.forcings;
  const bars = times.map((t) => { const day = t.slice(0, 10); const i = f.dates.indexOf(day); return t.endsWith("T12:00") && i >= 0 ? f.series.prcp[i] : null; });
  drawChart($("storm-chart"), { xs: times, lines: [{ values: q.map((x) => x[1]), color: "#4fb3ff" }, { values: times.map((t) => turbMap.get(t) ?? null), color: "#ff9d3d", axis: "right" }],
    bars: { values: bars, color: "rgba(255,255,255,0.35)", wide: true }, ylabel: "cfs / FNU", xIsTime: true });
  const tt = state.data.travel && state.data.travel.windows.find((x) => x.date === w.date);
  $("storm-caption").innerHTML = `${esc(w.why)}. Blue: Trumbull discharge (15-min). Orange: sentinel turbidity (15-min). Grey: the gauge's single daily number, drawn at noon.` +
    (tt ? ` ${badge("computed")} Trumbull rose ${fmt(tt.q_rise_cfs, 0)} cfs; turbidity cross-correlation lag ${tt.xcorr_lag_turb_h ?? "—"} h (r = ${fmt(tt.xcorr_r_turb, 2)}) against a Manning estimate of ${fmt(tt.manning_hours_at_peak, 1)} h for that flow.` : "");
}

/* --------------------------------------------------------- physics panel */

function renderPhysics() {
  const d = state.data, ch = d.channels, f = d.forcings, tt = d.travel;
  map.flyTo({ ...HOME, center: [-105.35, 39.35], zoom: 9.0, speed: 0.7 });
  if (!ch) { content.innerHTML = `<h2>Physics</h2>${missingNote("Channel network", "python3 build_channels.py")}`; return; }
  const tiers = Object.entries(ch.mainstem.travel);
  const tr = ch.mainstem.landmarks["gage-06701900"], ch_ = ch.mainstem.landmarks["res-cheesman"];
  content.innerHTML = `
    <h2>The physics, applied where it applies</h2>
    <div class="card"><h3>1. Continuity and Manning: the river</h3>
      <div class="formula">Q = v·A &nbsp;&nbsp; v = (1/n)·R^(2/3)·S^(1/2)</div>
      <p class="note">${badge("general")} Bernoulli's energy balance for open-channel flow, with friction, reduces to Manning's equation: velocity set by slope S,
        roughness n and hydraulic radius R. ${badge("computed")} Slopes from sampled terrain over ${ch.mainstem.n_reaches} reaches, ${(ch.mainstem.total_length_m / 1000).toFixed(0)} km;
        Q from the DWR record. ${badge("assumed")} n = ${ch.assumed.manning_n}, width ${ch.assumed.channel_width_m} m.</p>
      <table><tr><th>Flow tier</th><th class="num">cfs</th><th class="num">Trumbull → sentinel<br>${(tr.dist_to_gage_m / 1000).toFixed(1)} km</th><th class="num">Cheesman → sentinel<br>${(ch_.dist_to_gage_m / 1000).toFixed(1)} km</th><th class="num">headwater → sentinel</th></tr>
        ${tiers.map(([k, t]) => `<tr><td>${k.replace(/_/g, " ")}</td><td class="num">${t.flow_cfs}</td><td class="num">${fmt(t.hours_landmark_to_gage["gage-06701900"], 1)} h</td><td class="num">${fmt(t.hours_landmark_to_gage["res-cheesman"], 1)} h</td><td class="num">${fmt(t.hours_headwater_to_gage / 24, 1)} d</td></tr>`).join("")}
      </table>
      ${tt ? `<p class="note">${badge("computed", "empirical check")} Cross-correlating 15-minute Trumbull discharge with sentinel turbidity in ${tt.n_windows} storm windows gives
        ${tt.n_usable} usable lags (${(tt.xcorr_lag_turb_h.values || []).join(", ")} h; median ${fmt(tt.xcorr_lag_turb_h.median, 0)} h) against Manning's ${fmt(tt.manning_hours_at_same_flows.median, 1)} h
        at the same flows. Same order of magnitude; the remaining windows had lags near zero or negative, meaning the sediment came from side canyons between the gauges, not from Trumbull.</p>` : ""}
      <p class="note">${badge("quote", "Denver Water")} ${esc(ch.mainstem.denver_water_quote)}</p>
    </div>
    <div class="card"><h3>2. Bernoulli: the tunnels and the conduit</h3>
      <div class="formula">z₁ + p₁/ρg + v₁²/2g = z₂ + p₂/ρg + v₂²/2g + h_f &nbsp;&nbsp; h_f = f·(L/D)·v²/2g</div>
      <p class="note">${badge("general")} Between two free surfaces the pressure terms cancel and the head difference goes into velocity and friction, so
        v = √(2g·Δz / (1 + f·L/D)). ${badge("computed")} Δz from sampled portal elevations, L from the drawn lines. ${badge("assumed")} f = ${ch.assumed.darcy_f}, diameters below.
        These are hydraulic upper bounds on speed: valves and demand throttle the real flow far below them.</p>
      <table><tr><th>Conduit</th><th class="num">L km</th><th class="num">Δz m</th><th class="num">D m</th><th class="num">v bound</th><th class="num">≥ hours</th></tr>
        ${ch.conduits.map((c) => `<tr><td>${esc(c.name)}</td><td class="num">${(c.length_m / 1000).toFixed(1)}</td><td class="num">${fmt(c.head_m, 0)}</td><td class="num">${c.diameter_m_assumed}</td><td class="num">${fmt(c.v_darcy_m_s, 1)} m/s</td><td class="num">${fmt(c.hours_at_darcy_v, 1)}</td></tr>`).join("")}
      </table>
    </div>
    <div class="card"><h3>3. Continuity again: the reservoir</h3>
      ${ch.strontia ? `<div class="formula">τ = V / Q</div>
      <p class="note">${badge("computed")} Median Strontia storage ${ch.strontia.median_storage_af.toLocaleString()} acre-feet (DWR) over the flow tiers gives a nominal residence of
        ${Object.entries(ch.strontia.residence_days_at_tier).map(([k, v]) => `${k.replace(/_/g, " ")} ${v} d`).join(", ")}. ${badge("general")} Real reservoirs short-circuit: a dense,
        cold, turbid inflow can run along the old river channel to the intake in hours (Denver Water's four) while the bulk of the pool turns over in weeks.
        Which one happens depends on stratification, the Strontia sonde's question, visible on the system map's depth column.</p>` : `<p class="note">Storage history not available.</p>`}
    </div>
    <div class="card"><h3>4. Energy balance: degree-day snowmelt</h3>
      <div class="formula">melt = k · PDD</div>
      ${f && f.melt_fit && f.melt_fit.k_mm_per_degree_day_c !== undefined ? `<p class="note">${badge("general")} The degree-day method stands in for the full surface energy balance
        (shortwave, longwave, sensible, latent). ${badge("computed")} Fitted to Hoosier Pass SWE loss against positive degree-days from the NOAA station: k = ${f.melt_fit.k_mm_per_degree_day_c} mm per °C·day
        (n = ${f.melt_fit.n_days} melt days, r = ${f.melt_fit.pearson_r}). ${badge("general")} Textbook k is 3–6 mm/°C·day; ours is low because the air temperature comes from a station
        1,000 m below the pillow, so k absorbs the lapse rate, and because the fit is weak: sun, wind and rain-on-snow are not in a daily min/max.</p>
        <canvas id="melt-chart" class="chart"></canvas><div class="chart-label">Hoosier Pass daily SWE loss (in) against degree-days (°C) with the fitted line.</div>` : `<p class="note">Melt fit not available.</p>`}
    </div>
    <div class="card"><h3>5. What the physics says about warning time</h3>
      <p class="note">${badge("computed")} Add Denver Water's four hours from the sentinel to the intake and an event that starts at Trumbull physically reaches the plant in about
        ${fmt(tiers.find(([k]) => k === "high")[1].hours_landmark_to_gage["gage-06701900"] + 4, 0)}–${fmt(tiers.find(([k]) => k === "low")[1].hours_landmark_to_gage["gage-06701900"] + 4, 0)} hours, one from Cheesman in
        ${fmt(tiers.find(([k]) => k === "high")[1].hours_landmark_to_gage["res-cheesman"] + 4, 0)}–${fmt(tiers.find(([k]) => k === "low")[1].hours_landmark_to_gage["res-cheesman"] + 4, 0)} hours, and a side-canyon storm between Trumbull and the sentinel in under four.
        A 12-hour warning is therefore physically available only for changes that originate at or above Trumbull, and only if something upstream measures the thing being warned about.</p>
    </div>`;
  if (f && f.melt_fit && f.melt_fit.k_in_per_degree_day_c !== undefined) drawMeltScatter();
}

function drawMeltScatter() {
  const cv = $("melt-chart"); if (!cv) return;
  const f = state.data.forcings, swe = f.series.swe_hoosier, dd = f.derived.degree_days_c;
  const pts = [];
  for (let i = 1; i < swe.length; i++) {
    if (swe[i - 1] === null || swe[i] === null || dd[i] === null || swe[i] <= 0) continue;
    const m = swe[i - 1] - swe[i];
    if (m > 0 && dd[i] > 0) pts.push([dd[i], m]);
  }
  drawScatter(cv, pts, { xmax: 25, ymax: 2.5, line: (x) => f.melt_fit.k_in_per_degree_day_c * x, xlabel: "°C·day", ylabel: "in" });
}

/* --------------------------------------------------------- climate panel */

function renderClimate() {
  const e = state.data.enso, t = state.data.trends;
  map.flyTo({ ...HOME, speed: 0.7 });
  let html = `<h2>Climate context: El Niño and warming</h2>
    <p>Neither is a 12-hour predictor. Both shape the season the forecast lives in, so both were tested against the long records already in the repository rather than asserted.</p>`;
  if (!e) html += missingNote("ENSO context", "python3 fetch_oni.py");
  else {
    const h = e.headline || {};
    html += `<div class="card"><h3>El Niño / La Niña (ONI) against the snowpack</h3>
      <p class="note">${badge("general")} ${esc(e.general_knowledge[0])}</p>
      <canvas id="enso-chart" class="chart tall"></canvas>
      <div class="chart-label">Each dot is a water year since 1981: Nov–Jan ONI (x) against mean peak SWE over the mapped pillows (y), coloured by phase.</div>
      <p class="note">${badge("computed")} Spearman ρ = ${h.spearman_rho} (permutation p = ${h.p_perm}, n = ${h.n}). Mean peak SWE by phase:
        ${Object.entries(h.peak_swe_by_phase || {}).map(([k, v]) => `${k} ${v.mean_in} in (n=${v.n})`).join(", ")}.
        Per-station and reservoir tests are in <a href="results/enso_correlations.csv">results/enso_correlations.csv</a>; none reached p &lt; 0.05.
        The honest reading: on this mountain range, in these 46 years, knowing the ENSO phase tells you almost nothing about how much snow will fall.</p>
      <p class="note">${badge("general")} ${esc(e.general_knowledge[1])}</p>
    </div>`;
  }
  if (!t) html += missingNote("Long-record trends", "python3 analysis/03_trends.py");
  else {
    const pooled = t.pooled.filter((r) => r.series.includes("anomaly"));
    html += `<div class="card"><h3>Warming: is the snowpack changing?</h3>
      <p class="note">${badge("general")} ${esc(t.general_knowledge[0])}</p>
      <table><tr><th>Metric (mean anomaly, 11 pillows)</th><th class="num">per decade</th><th class="num">MK p</th><th class="num">n</th></tr>
        ${pooled.map((r) => `<tr><td>${esc(r.metric.replace(/_/g, " "))}</td><td class="num">${r.theil_sen_per_decade}</td><td class="num">${r.mk_p}</td><td class="num">${r.n_years}</td></tr>`).join("")}
      </table>
      <p class="note">${badge("computed")} Theil–Sen slope and Mann–Kendall test on per-station anomalies (so pillows added in 1999 do not fake a trend; a naive pooled mean
        showed a spurious −1.3 in/decade for exactly that reason). Over 1981–2026 these eleven pillows show no significant change in peak SWE or melt-out date.
        Four of ${t.rows.length} individual station tests pass p &lt; 0.05, about what chance gives. Full table: <a href="results/trends.csv">results/trends.csv</a>.</p>
      <p class="note">${badge("computed")} The NOAA temperature record here covers ${t.noaa_temperature.years_covered}: ${esc(t.noaa_temperature.verdict)}</p>
      <p class="note">${badge("general")} ${esc(t.general_knowledge[1])}</p>
      <canvas id="trend-chart" class="chart"></canvas>
      <div class="chart-label">Mean peak-SWE anomaly (in) by water year across the mapped pillows.</div>
    </div>`;
  }
  html += `<div class="card"><h3>Why this matters for a 12-hour forecast</h3>
    <p class="note">${badge("general")} Seasonal signals set the prior (how much snow, when it will melt, how much TOC-rich soil water will flush); they cannot move a forecast
      issued at breakfast for dinner. Weather physics at that horizon is convective initiation over 4,000 sq mi of mountains, which a single daily gauge cannot see
      and which even NWS high-resolution models place only in probability. The 12-hour problem is therefore an observation problem before it is a modelling one.</p></div>`;
  content.innerHTML = html;
  if (e) drawEnsoScatter();
  if (t) drawTrendChart();
}

function drawEnsoScatter() {
  const e = state.data.enso, cv = $("enso-chart"); if (!cv) return;
  const pts = [];
  for (const [wy, v] of Object.entries(e.water_years)) {
    const pk = v.peak_swe_in ? Object.values(v.peak_swe_in) : [];
    if (pk.length >= 3 && v.oni_ndj !== null && v.oni_ndj !== undefined) pts.push([v.oni_ndj, pk.reduce((a, b) => a + b, 0) / pk.length, v.phase, wy]);
  }
  drawScatter(cv, pts, { xmin: -2.5, xmax: 2.8, ymax: 30, color: (p) => p[2] === "El Nino" ? "#ff9d3d" : p[2] === "La Nina" ? "#4fb3ff" : "#b8c0c8", xlabel: "ONI (Nov–Jan)", ylabel: "peak SWE, in", zeroLine: true });
}

function drawTrendChart() {
  const t = state.data.trends, cv = $("trend-chart"); if (!cv) return;
  const years = Object.keys(t.pooled_series).sort();
  drawChart(cv, { xs: years, lines: [{ values: years.map((y) => t.pooled_series[y].peak_swe_in), color: "#dfeaff" }], ymin: -8, ymax: 8, ylabel: "in", zeroLine: true });
}

/* ----------------------------------------------------- feasibility panel */

function proposedSites() {
  const d = state.data, ch = d.channels, by = d.byId;
  const tr = ch && ch.mainstem.travel, lm = (tier, id) => (tr && tr[tier] ? tr[tier].hours_landmark_to_gage[id] : null);
  const range = (id) => tr ? `${fmt(lm("high", id) + 4, 0)}–${fmt(lm("low", id) + 4, 0)} h` : "—";
  const mid = (a, b, f) => [a.lon + (b.lon - a.lon) * f, a.lat + (b.lat - a.lat) * f];
  const trumbull = by["gage-06701900"], sentinel = by["gage-06707525"];
  return [
    { n: 1, label: "Online TOC/alkalinity at Foothills", lonlat: [by["plant-foothills"].lon, by["plant-foothills"].lat], lead: "0 h (target)", tier: "Essential",
      what: "A UV254/fDOM TOC surrogate and an alkalinity or conductivity analyser on the influent, logging every 15 minutes, or at least 2–3 grab samples a day.",
      why: "Today the target is one grab sample per day, so a 12-hour forecast cannot even be scored. Nothing else on this list can be evaluated until this exists (the PoC's first recommendation)." },
    { n: 2, label: "fDOM/UV254 at the sentinel gage", lonlat: [sentinel.lon, sentinel.lat], lead: "≈ 4 h to intake; 1–2 d in the plant record", tier: "High value, low cost",
      what: "Add an fDOM (fluorescent dissolved organic matter) or UV254 probe to the existing USGS sonde, which now reads turbidity, conductance, pH, temperature and DO.",
      why: "Turbidity is a sediment signal; TOC is dissolved. Nothing upstream currently measures the quantity being forecast. fDOM is its standard field proxy." },
    { n: 3, label: "Water quality at Trumbull", lonlat: [trumbull.lon, trumbull.lat], lead: range("gage-06701900"), tier: "High",
      what: "Turbidity, conductance and fDOM at the existing flow gauge 27.6 km upstream.",
      why: `Manning routing puts Trumbull water at the plant ${range("gage-06701900")} after it passes; the empirical storm lags agree in order of magnitude. This is the closest point that physically clears 12 hours for events that come down the mainstem.` },
    { n: 4, label: "Cheesman outlet water quality", lonlat: [by["res-cheesman"].lon, by["res-cheesman"].lat], lead: range("res-cheesman"), tier: "Medium",
      what: "A sonde below Cheesman Dam, where Denver Water controls releases.",
      why: "Summer mainstem flow is largely Cheesman release (general knowledge); release changes are known in advance, so pairing the schedule with outlet quality gives the longest deterministic lead on the mainstem." },
    { n: 5, label: "Side-canyon rain gauges (tipping bucket)", lonlat: mid(trumbull, sentinel, 0.4), lead: "1–4 h", tier: "High for storms",
      what: "Three or four telemetered tipping-bucket gauges on the tributaries between Trumbull and the sentinel, the stretch no gauge watches.",
      why: "In the Aug 2026 storm the sentinel jumped to 329 FNU while Trumbull barely moved: the sediment came from here. Only rain measured here, minutes after it falls, warns of that class of event." },
    { n: 6, label: "Second side-canyon gauge", lonlat: mid(trumbull, sentinel, 0.75), lead: "1–3 h", tier: "High for storms", what: "As above, lower reach.", why: "Convective cells are a few kilometres wide; two gauges 10 km apart see different storms." },
    { n: 7, label: "Radar QPE ingestion (NEXRAD KFTG)", lonlat: [-105.6, 39.15], lead: "0–6 h (nowcast)", tier: "Software only",
      what: "Pull MRMS/NEXRAD quantitative precipitation estimates for the basin polygons every 10 minutes; no hardware.",
      why: "Basin-wide rain at 1 km resolution replaces the single can at the dam. The system map already streams the archived radar for the Aug 2026 storm; this makes it an input rather than a picture." },
    { n: 8, label: "North Fork at Grant", lonlat: [by["gage-06702500"].lon, by["gage-06702500"].lat], lead: "unknown (no flowline committed)", tier: "Medium",
      what: "Water quality where Roberts Tunnel water enters the North Fork.",
      why: "Dillon water is a large, controllable share of the mainstem; its quality is Blue River quality, set on the other side of the Divide. The North Fork flowline is not in the repository, so its travel time is unmeasured here." },
    { n: 9, label: "SNOTEL precipitation (PREC element)", lonlat: [by["sntl-531"].lon, by["sntl-531"].lat], lead: "2–4 d via the river", tier: "Software only",
      what: "Fetch the precipitation and air-temperature elements the pillows already record; only SWE is in the shipped data.",
      why: "Headwater rain and melt-day temperatures, at elevation, where the water actually originates; the melt fit here is weak precisely because it had to use a valley thermometer." },
  ];
}

function showProposed(on) {
  if (!state.layersReady) return;
  const src = map.getSource("proposed");
  if (on && src) src.setData({ type: "FeatureCollection", features: proposedSites().map((s) => ({ type: "Feature",
    properties: { label: `${s.n}`, name: s.label }, geometry: { type: "Point", coordinates: s.lonlat } })) });
  for (const id of ["proposed", "proposed-labels"]) if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", on ? "visible" : "none");
}

function renderFeasibility() {
  const d = state.data, ch = d.channels;
  map.flyTo({ center: [-105.25, 39.35], zoom: 9.6, pitch: 55, bearing: -20, speed: 0.7 });
  const sites = ch ? proposedSites() : [];
  content.innerHTML = `
    <h2>Can Foothills get a 12-hour warning?</h2>
    <div class="warn">Not with today's measurements. The pre-registered study in <code>teams/poc-24h-toc-alk</code> found that at 24 hours "tomorrow looks like today"
      beats every model built from this data, and that 12 hours cannot even be tested because the target is one lab sample a day.</div>
    <h3>Why not, in order of severity</h3>
    <ul>
      <li>${badge("computed")} <b>The target is measured once a day.</b> A 12-hour forecast has nothing to be scored against; the recorded time is the sample date, not the hour.</li>
      <li>${badge("computed")} <b>Nothing upstream measures TOC.</b> The sentinel reads turbidity, conductance, pH, temperature, DO: sediment and salt, not dissolved carbon.</li>
      <li>${badge("computed")} <b>Rain is measured at the outlet.</b> The one gauge stands at Strontia Springs Dam, publishes two days late, and integrates a day into one number.</li>
      <li>${badge("computed")} <b>Side-canyon storms give under four hours.</b> Physics: the stretch between Trumbull and the sentinel has no gauge and drains straight into the reservoir.</li>
      <li>${badge("computed")} <b>Persistence is very hard to beat.</b> Day-to-day R² ≈ 0.96 for both targets; median 24 h change 0.03 mg/L TOC, 1.1 mg/L alkalinity (PoC).</li>
      <li>${badge("computed")} <b>Provisional data.</b> USGS, NRCS, NOAA and DWR revise after publication; a forecast inherits every later correction.</li>
      <li>${badge("computed")} <b>Five partial seasons, no winter.</b> No influent data January–March; TOC events cluster in 2023–24; the sonde has one partial season.</li>
    </ul>
    <h3>What the physics allows</h3>
    <p class="note">${ch ? `Manning routing over the ${(ch.mainstem.total_length_m / 1000).toFixed(0)} km mainstem plus Denver Water's four hours to the intake: an event passing Trumbull reaches the plant in
      ${fmt(ch.mainstem.travel.high.hours_landmark_to_gage["gage-06701900"] + 4, 0)}–${fmt(ch.mainstem.travel.low.hours_landmark_to_gage["gage-06701900"] + 4, 0)} h depending on flow, from Cheesman in
      ${fmt(ch.mainstem.travel.high.hours_landmark_to_gage["res-cheesman"] + 4, 0)}–${fmt(ch.mainstem.travel.low.hours_landmark_to_gage["res-cheesman"] + 4, 0)} h, from the headwaters in 2–4 days. So a 12-hour warning is physically available
      for anything that comes down the mainstem past Trumbull, and physically unavailable for anything generated in the last 28 km. The map shows where sensors would have to be to use that window.` : missingNote("Channel network", "python3 build_channels.py")}</p>
    <h3>What would make it possible</h3>
    <table><tr><th>#</th><th>Proposal</th><th>Lead it buys</th><th>Tier</th></tr>
      ${sites.map((s) => `<tr><td>${s.n}</td><td><b>${esc(s.label)}</b><br><span class="note">${esc(s.what)}</span><br><span class="note"><i>${esc(s.why)}</i></span></td><td class="num">${s.lead}</td><td>${s.tier}</td></tr>`).join("")}
    </table>
    <h3>And the modelling that follows</h3>
    <ul>
      <li>${badge("general")} <b>Event-conditional models.</b> Persistence wins on quiet days; train and score only on the storm and melt-onset days that matter, using 15-minute upstream data.</li>
      <li>${badge("general")} <b>Reframe to multi-day.</b> The PoC's lead sweep shows upstream data closing on persistence with lead; 3–7 days is where snow and flow information plausibly wins.</li>
      <li>${badge("general")} <b>Operator cost matrix.</b> The alkalinity onset flag already catches 87% of drops below 60 mg/L at 33% precision; whether that is useful depends on the cost of a false alarm, which only operators know.</li>
      <li>${badge("general")} <b>Travel-time-aware features.</b> Shift each upstream series by its routed travel time (this page's table) instead of a fixed daily lag.</li>
    </ul>
    <p class="note">The full argument, with every number's source, is in <a href="FEASIBILITY.md">FEASIBILITY.md</a>. Pink markers on the map are the proposed sites; numbers match the table.</p>`;
}

/* ----------------------------------------------------------- charting */

function setupCanvas(cv) {
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth || 340, h = cv.clientHeight || 120;
  cv.width = w * dpr; cv.height = h * dpr;
  const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
  return [g, w, h];
}

function drawChart(cv, o) {
  const [g, w, h] = setupCanvas(cv);
  const L = 34, R = o.lines && o.lines.some((l) => l.axis) ? 40 : 8, T = 6, B = 18;
  const n = o.xs.length, x = (i) => L + (i / Math.max(1, n - 1)) * (w - L - R);
  const finite = (arr) => arr.filter((v) => v !== null && v !== undefined && Number.isFinite(v));
  const leftVals = finite([...(o.bars ? o.bars.values : []), ...(o.lines || []).filter((l) => !l.axis).flatMap((l) => l.values)]);
  const ymin = o.ymin !== undefined ? o.ymin : 0;
  const ymax = o.ymax !== undefined ? Math.max(o.ymax, ...leftVals) : Math.max(0.01, ...leftVals) * 1.1;
  const y = (v) => T + (1 - (v - ymin) / (ymax - ymin)) * (h - T - B);
  g.strokeStyle = "rgba(255,255,255,0.12)"; g.lineWidth = 1;
  for (let k = 0; k <= 3; k++) { const yy = T + k * (h - T - B) / 3; g.beginPath(); g.moveTo(L, yy); g.lineTo(w - R, yy); g.stroke(); }
  g.fillStyle = "#b8c0c8"; g.font = "10px sans-serif"; g.textAlign = "right";
  g.fillText(fmt(ymax, ymax > 10 ? 0 : 1), L - 3, T + 9); g.fillText(fmt(ymin, 0), L - 3, h - B);
  if (o.ylabel) { g.textAlign = "left"; g.fillText(o.ylabel, 2, h - 4); }
  if (o.zeroLine && ymin < 0) { g.strokeStyle = "rgba(255,255,255,0.4)"; g.beginPath(); g.moveTo(L, y(0)); g.lineTo(w - R, y(0)); g.stroke(); }
  if (o.bars) {
    g.fillStyle = o.bars.color;
    const bw = o.bars.wide ? Math.max(3, (w - L - R) / n * 12) : Math.max(1, (w - L - R) / n - 1);
    o.bars.values.forEach((v, i) => { if (v !== null && v !== undefined) g.fillRect(x(i) - bw / 2, y(v), bw, y(ymin) - y(v)); });
  }
  const rightAxes = {};
  for (const ln of o.lines || []) {
    let yy = y;
    if (ln.axis) {
      const vals = finite(ln.values); const mx = Math.max(0.01, ...vals) * 1.1;
      rightAxes[ln.axis] = { mx, color: ln.color };
      yy = (v) => T + (1 - v / mx) * (h - T - B);
    }
    g.strokeStyle = ln.color; g.lineWidth = 1.6; g.beginPath(); let pen = false;
    ln.values.forEach((v, i) => { if (v === null || v === undefined || !Number.isFinite(v)) { pen = false; return; } if (!pen) { g.moveTo(x(i), yy(v)); pen = true; } else g.lineTo(x(i), yy(v)); });
    g.stroke();
  }
  let k = 0;
  for (const [, ax] of Object.entries(rightAxes)) { g.fillStyle = ax.color; g.textAlign = "left"; g.fillText(fmt(ax.mx, ax.mx > 10 ? 0 : 1), w - R + 3, T + 9 + 11 * k++); }
  g.fillStyle = "#b8c0c8"; g.textAlign = "left"; g.fillText(o.xIsTime ? o.xs[0].replace("T", " ") : o.xs[0], L, h - 5);
  g.textAlign = "right"; g.fillText(o.xIsTime ? o.xs[n - 1].replace("T", " ") : o.xs[n - 1], w - R, h - 5);
  if (o.cursor !== undefined && o.cursor >= 0 && o.cursor < n) { g.strokeStyle = "#fff"; g.lineWidth = 1; g.beginPath(); g.moveTo(x(o.cursor), T); g.lineTo(x(o.cursor), h - B); g.stroke(); }
}

function drawScatter(cv, pts, o) {
  const [g, w, h] = setupCanvas(cv);
  const L = 34, R = 8, T = 6, B = 18;
  const xmin = o.xmin !== undefined ? o.xmin : 0, xmax = o.xmax, ymin = o.ymin || 0, ymax = o.ymax;
  const x = (v) => L + (v - xmin) / (xmax - xmin) * (w - L - R), y = (v) => T + (1 - (v - ymin) / (ymax - ymin)) * (h - T - B);
  g.strokeStyle = "rgba(255,255,255,0.12)"; g.beginPath(); g.moveTo(L, T); g.lineTo(L, h - B); g.lineTo(w - R, h - B); g.stroke();
  if (o.zeroLine) { g.strokeStyle = "rgba(255,255,255,0.35)"; g.beginPath(); g.moveTo(x(0), T); g.lineTo(x(0), h - B); g.stroke(); }
  g.fillStyle = "#b8c0c8"; g.font = "10px sans-serif"; g.textAlign = "right"; g.fillText(fmt(ymax, 0), L - 3, T + 9); g.fillText(fmt(ymin, 0), L - 3, h - B);
  g.textAlign = "left"; g.fillText(o.xlabel || "", L, h - 5); g.textAlign = "right"; g.fillText(fmt(xmax, 0), w - R, h - 5); g.textAlign = "left"; g.fillText(o.ylabel || "", 2, T + 9);
  for (const p of pts) {
    if (p[0] < xmin || p[0] > xmax || p[1] > ymax) continue;
    g.fillStyle = o.color ? o.color(p) : "rgba(169,215,255,0.55)"; g.beginPath(); g.arc(x(p[0]), y(p[1]), 2.6, 0, Math.PI * 2); g.fill();
  }
  if (o.line) { g.strokeStyle = "#ffd166"; g.lineWidth = 1.5; g.beginPath(); g.moveTo(x(xmin), y(o.line(xmin))); g.lineTo(x(xmax), y(Math.min(ymax, o.line(xmax)))); g.stroke(); }
}

/* ------------------------------------------------------------------ go */

loadEverything().catch((e) => {
  content.innerHTML = `<div class="warn">Could not load the system data (${esc(e.message)}). Serve the repository with <code>python3 serve.py</code> from its root and open
    <code>/teams/terrain-flow-sim/</code>; opening the file directly cannot fetch JSON.</div>`;
});

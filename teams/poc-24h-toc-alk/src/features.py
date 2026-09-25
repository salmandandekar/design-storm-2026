"""Leakage-safe daily feature matrix and forecast labels.

Contract: in the wide daily frame returned by build_daily_features(), the
value in any column at calendar date T is computable from data timestamped
<= T as available at forecast issue time (end of day T). Concretely:

- Lab, USGS, DWR and SNOTEL series are near-real-time: usable at lag 0
  relative to T (values through day T).
- NOAA GHCN publishes ~2 days behind (per Jake), so its columns are shifted
  by NOAA_LATENCY_DAYS: the value stored at T is the reading from T-2.
- Strontia sonde casts are telemetered same-day (2026 only).

Labels for lead h are then y(T+h) joined against features at T; no feature
ever sees data newer than T. Rolling windows run on the calendar-reindexed
series with min_periods so small sensor gaps do not silently shift windows.

Feature sets (ablation ladder; cumulative):
  A: lab history only (scenario "lab known" only)
  B: + USGS gage
  C: + DWR flow / gage height (incl. Jake's turb x flow loading term)
  D: + SWE (Hoosier Pass), NOAA weather (latency-shifted), season encodings
  E: + Strontia sonde daily depth-band summaries (2026 overlap only)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D

NOAA_LATENCY_DAYS = 2

# Sonde depth bands, verified in 01_audit: low Vertical Position = warm
# surface water (Aug: 18 C at VP<5 vs 15 C at VP>43), so VP is depth in m.
SONDE_SURFACE_MAX_M = 5.0
SONDE_BOTTOM_MARGIN_M = 5.0  # bottom band = within 5 m of the day's max depth


def _roll(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=max(2, window - 3)).mean()


def build_daily_features() -> pd.DataFrame:
    """Wide daily frame on the full calendar; every column available at its date."""
    fth = D.load_foothills()
    usgs = D.load_usgs()
    dwr = D.load_dwr_telemetry()
    swe = D.load_swe_hoosier()
    noaa = D.load_noaa()

    start = min(x.index.min() for x in (fth, usgs, dwr, swe, noaa))
    end = max(x.index.max() for x in (fth, usgs, dwr, swe, noaa))
    cal = pd.date_range(start, end, freq="D", name="DATE")
    out = pd.DataFrame(index=cal)

    # --- Set A: lab history (only valid in the "lab known" scenario) ---
    for col, tag in [("TOC_mg_L", "toc"), ("Alk_mg_L", "alk")]:
        s = fth[col].reindex(cal)
        out[f"lab_{tag}"] = s
        out[f"lab_{tag}_diff1"] = s.diff()  # NaN across sampling gaps
        out[f"lab_{tag}_roll3"] = _roll(s, 3)
        out[f"lab_{tag}_roll7"] = _roll(s, 7)

    # --- Set B: USGS gage above Strontia ---
    u = usgs.reindex(cal)
    for col, tag in [
        ("Turbidity_Median", "turb"),
        ("Specific_Cond_Mean", "cond"),
        ("pH_Median", "ph"),
        ("Temp_C_Mean", "wtemp"),
        ("Dissolved_Oxygen_Mean", "do"),
    ]:
        s = u[col]
        out[f"usgs_{tag}"] = s
        out[f"usgs_{tag}_diff1"] = s.diff()
        out[f"usgs_{tag}_roll3"] = _roll(s, 3)
        out[f"usgs_{tag}_roll7"] = _roll(s, 7)
    out["usgs_turb_max"] = u["Turbidity_Max"]

    # --- Set C: DWR flow / gage height, plus the loading cross-term ---
    w = dwr.reindex(cal)
    out["dwr_flow"] = w["Flow_CFS"]
    out["dwr_flow_delta"] = w["Flow_CFS"].diff()
    out["dwr_flow_roll7"] = _roll(w["Flow_CFS"], 7)
    out["dwr_gageht"] = w["GageHeight_ft"]
    out["dwr_gageht_roll3"] = _roll(w["GageHeight_ft"], 3)
    out["dwr_gageht_roll7"] = _roll(w["GageHeight_ft"], 7)
    # Jake's strongest TOC predictor: 3-day turbidity x 7-day flow.
    out["turb_flow"] = out["usgs_turb_roll3"] * out["dwr_flow_roll7"]
    out["turb_cond"] = out["usgs_cond"] * out["usgs_turb_roll3"]

    # --- Set D: snowpack, NOAA weather (latency-shifted), season ---
    s = swe["SWE"].reindex(cal)
    out["swe"] = s
    out["swe_diff1"] = s.diff()  # negative = melt
    out["swe_roll7"] = _roll(s, 7)
    n = noaa.reindex(cal).shift(NOAA_LATENCY_DAYS)  # value at T is reading from T-2
    out["noaa_prcp_lag2"] = n["PRCP"]
    out["noaa_prcp_roll3"] = _roll(n["PRCP"], 3)
    out["noaa_prcp_roll7"] = _roll(n["PRCP"], 7)
    out["noaa_tmax_lag2"] = n["TMAX"]
    out["noaa_snow_lag2"] = n["SNOW"]
    doy = out.index.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    month = out.index.month
    out["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    return out


def build_sonde_daily() -> pd.DataFrame:
    """Set E: collapse Strontia casts to one row per day of depth-band summaries."""
    raw = D.load_strontia()
    ts = raw["Time stamp"]
    day = ts.dt.normalize()
    g = raw.assign(day=day)

    rows = []
    for d, sub in g.groupby("day"):
        vp = sub["Vertical Position"]
        surface = sub[vp <= SONDE_SURFACE_MAX_M]
        bottom = sub[vp >= vp.max() - SONDE_BOTTOM_MARGIN_M]
        row = {"DATE": d}
        for band, bsub in [("surf", surface), ("bot", bottom), ("all", sub)]:
            if len(bsub) == 0:
                continue
            row[f"sonde_temp_{band}"] = bsub["Temp C"].mean()
            row[f"sonde_cond_{band}"] = bsub["Conductivity"].mean()
            row[f"sonde_turb_{band}"] = bsub["Turbidity NTU"].mean()
        row["sonde_turb_max"] = sub["Turbidity NTU"].max()
        row["sonde_chl_surf"] = surface["Chl ug/L"].mean() if len(surface) else np.nan
        row["sonde_do_all"] = sub["ODO mg/L"].mean()
        row["sonde_ph_all"] = sub["pH"].mean()
        if len(surface) and len(bottom):
            row["sonde_strat"] = surface["Temp C"].mean() - bottom["Temp C"].mean()
        rows.append(row)
    out = pd.DataFrame(rows).set_index("DATE").sort_index()
    out.index.name = "DATE"
    return out


# Ablation ladder column groups (cumulative sets are unions of these).
GROUP_A = [
    "lab_toc", "lab_toc_diff1", "lab_toc_roll3", "lab_toc_roll7",
    "lab_alk", "lab_alk_diff1", "lab_alk_roll3", "lab_alk_roll7",
]
GROUP_B = [
    "usgs_turb", "usgs_turb_diff1", "usgs_turb_roll3", "usgs_turb_roll7", "usgs_turb_max",
    "usgs_cond", "usgs_cond_diff1", "usgs_cond_roll3", "usgs_cond_roll7",
    "usgs_ph", "usgs_ph_diff1", "usgs_ph_roll3", "usgs_ph_roll7",
    "usgs_wtemp", "usgs_wtemp_diff1", "usgs_wtemp_roll3", "usgs_wtemp_roll7",
    "usgs_do", "usgs_do_diff1", "usgs_do_roll3", "usgs_do_roll7",
]
GROUP_C = [
    "dwr_flow", "dwr_flow_delta", "dwr_flow_roll7",
    "dwr_gageht", "dwr_gageht_roll3", "dwr_gageht_roll7",
    "turb_flow", "turb_cond",
]
GROUP_D = [
    "swe", "swe_diff1", "swe_roll7",
    "noaa_prcp_lag2", "noaa_prcp_roll3", "noaa_prcp_roll7",
    "noaa_tmax_lag2", "noaa_snow_lag2",
    "doy_sin", "doy_cos", "month_sin", "month_cos",
]

FEATURE_SETS_LAB_KNOWN = {
    "A": GROUP_A,
    "AB": GROUP_A + GROUP_B,
    "ABC": GROUP_A + GROUP_B + GROUP_C,
    "ABCD": GROUP_A + GROUP_B + GROUP_C + GROUP_D,
}
FEATURE_SETS_LAB_BLIND = {
    "B": GROUP_B,
    "BC": GROUP_B + GROUP_C,
    "BCD": GROUP_B + GROUP_C + GROUP_D,
}

# Jake-equivalent reference sets (his notebook feature lists mapped onto this
# pipeline's column names), evaluated at lead 1 under the same harness.
JAKE_TOC = [
    "usgs_cond", "month_cos", "month_sin", "turb_flow", "usgs_turb_roll3",
    "noaa_prcp_roll7", "swe_roll7", "turb_cond", "usgs_turb", "usgs_turb_max",
]
JAKE_ALK = [
    "usgs_cond", "usgs_ph", "month_sin", "month_cos",
    "dwr_flow_roll7", "usgs_turb_roll3", "turb_flow", "usgs_do",
]


def make_labels(features: pd.DataFrame, target: str, lead: int) -> pd.DataFrame:
    """Labels for issue day T and lead h: level y(T+h), delta y(T+h)-y(T).

    Rows require an actual lab value at both T and T+h (no interpolation),
    so persistence and delta are always well defined.
    """
    fth = D.load_foothills()
    col = {"toc": "TOC_mg_L", "alk": "Alk_mg_L"}[target]
    y = fth[col].reindex(features.index)
    lab = pd.DataFrame(index=features.index)
    lab["y_now"] = y                      # lab value at T (known in scenario a)
    lab["y_future"] = y.shift(-lead)      # lab value at T+h (the target)
    lab["y_delta"] = lab["y_future"] - lab["y_now"]
    lab = lab.dropna(subset=["y_now", "y_future"])
    return lab

"""Canonical loaders for the Denver Water Design Storm datasets.

Each loader reads one original file from data/ (read-only), fixes only
documented artifacts, records every correction it makes in CLEANING_LOG,
and returns a DataFrame indexed by naive daily Timestamps.

Documented artifacts handled here:
- USGS_South_Platte.csv: -1e6 sentinel in Dissolved_Oxygen_Min -> NaN.
- MichiganCreek.csv: known-bad NRCS SWE of 9.0 on 2026-05-12..15 between
  zero readings (README). Loader flags it; modeling excludes this file.
- SouthPlatteTelemetry.csv Precip: cumulative counter with resets; converted
  to daily diffs for diagnostics only, never used as a predictor (Jake's
  guidance: "don't use this as a predictor. Data is really dirty.").
- Strontia xlsx: trailing spaces in column names stripped.

Readings are provisional (USGS revises after publication); nothing here
should be treated as settled truth.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"

DO_MIN_SENTINEL = -999_999.0  # exact sentinel present in the file; any DO < 0 is impossible
MICHIGAN_BAD_RANGE = ("2026-05-12", "2026-05-15")

CLEANING_LOG: list[str] = []


def _log(msg: str) -> None:
    if msg not in CLEANING_LOG:
        CLEANING_LOG.append(msg)


def _daily_index(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="raise")
    df = df.set_index(col).sort_index()
    df.index.name = "DATE"
    if df.index.duplicated().any():
        raise ValueError(f"duplicate dates in {col}")
    return df


def load_foothills() -> pd.DataFrame:
    """Target: daily grab-sample TOC_mg_L and Alk_mg_L at the Foothills influent."""
    df = pd.read_csv(DATA_DIR / "FoothillsInfluent.csv")
    df = _daily_index(df, "DATE")
    assert list(df.columns) == ["TOC_mg_L", "Alk_mg_L"]
    return df


def load_usgs() -> pd.DataFrame:
    """USGS gage 06707525 above Strontia: daily stats of the 15-min sonde."""
    df = pd.read_csv(DATA_DIR / "USGS_South_Platte.csv")
    df = _daily_index(df, "Date")
    bad = df["Dissolved_Oxygen_Min"] < 0  # catches the -999999 sentinel
    n_sentinel = int(bad.sum())
    if n_sentinel:
        df.loc[bad, "Dissolved_Oxygen_Min"] = np.nan
        _log(f"USGS Dissolved_Oxygen_Min: replaced {n_sentinel} negative sentinel value(s) (e.g. {DO_MIN_SENTINEL:g}) with NaN")
    return df.drop(columns=["site_no"])


def load_dwr_telemetry() -> pd.DataFrame:
    """Colorado DWR PLASPLCO telemetry: Flow_CFS, GageHeight_ft, cumulative Precip."""
    df = pd.read_csv(DATA_DIR / "SouthPlatteTelemetry.csv")
    df = _daily_index(df, "Date")
    # Diagnostics-only daily precip from the cumulative, reset-prone counter.
    diff = df["Precip"].diff()
    resets = int((diff < -0.5).sum())
    df["Precip_daily_diag"] = diff.clip(lower=0)
    _log(
        f"DWR Precip: cumulative counter with {resets} negative reset(s); "
        "daily diff kept for diagnostics only, excluded as a predictor"
    )
    return df


def load_dwr_flow() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "SouthPlatteFlow.csv")
    return _daily_index(df, "measDate")


def load_swe_hoosier() -> pd.DataFrame:
    """Hoosier Pass SNOTEL SWE - the sanctioned snow input (Sep 4 update)."""
    df = pd.read_csv(DATA_DIR / "HoosierPass.csv")
    return _daily_index(df, "DATE")


def load_swe_michigan() -> pd.DataFrame:
    """Michigan Creek SNOTEL SWE. Excluded from modeling: known-bad patch."""
    df = pd.read_csv(DATA_DIR / "MichiganCreek.csv")
    df = _daily_index(df, "DATE")
    bad = df.loc[MICHIGAN_BAD_RANGE[0] : MICHIGAN_BAD_RANGE[1], "SWE"]
    df["SWE_known_bad"] = False
    df.loc[bad.index, "SWE_known_bad"] = True
    _log(
        f"MichiganCreek SWE: flagged {len(bad)} known-bad day(s) "
        f"{MICHIGAN_BAD_RANGE[0]}..{MICHIGAN_BAD_RANGE[1]} (NRCS feed error, per README); file excluded from models"
    )
    return df


def load_noaa() -> pd.DataFrame:
    """NOAA GHCN USC00058022: PRCP (in), SNOW (in), TMAX/TMIN (deg F).

    Publishes ~2 days behind real time (Jake), so at issue day T only values
    through T-2 are treated as available.
    """
    df = pd.read_csv(DATA_DIR / "USC00058022.csv")
    df = _daily_index(df.drop(columns=["STATION"]), "DATE")
    return df


def load_strontia() -> pd.DataFrame:
    """Strontia Springs profiling sonde, 2026-04-07..08-19, one row per reading."""
    df = pd.read_excel(DATA_DIR / "Strontia 0407_0819.xlsx")
    stripped = {c: c.strip() for c in df.columns if c != c.strip()}
    if stripped:
        df = df.rename(columns=stripped)
        _log(f"Strontia xlsx: stripped trailing spaces from {len(stripped)} column name(s)")
    df["Time stamp"] = pd.to_datetime(df["Time stamp"])
    return df.sort_values("Time stamp").reset_index(drop=True)


ALL_DAILY_LOADERS = {
    "FoothillsInfluent": load_foothills,
    "USGS_South_Platte": load_usgs,
    "SouthPlatteTelemetry": load_dwr_telemetry,
    "SouthPlatteFlow": load_dwr_flow,
    "HoosierPass": load_swe_hoosier,
    "MichiganCreek": load_swe_michigan,
    "USC00058022": load_noaa,
}

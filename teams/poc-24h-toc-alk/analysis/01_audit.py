"""01: Audit every original dataset; write results/data_audit.csv + figures.

Verifies the documented schemas/artifacts and records: date ranges, calendar
gaps, per-column missingness, units, duplicates, sentinel handling, the
Michigan Creek bad patch, and the sonde's depth-orientation check.
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import data as D  # noqa: E402

TEAM = pathlib.Path(__file__).resolve().parents[1]
RESULTS = TEAM / "results"
FIGURES = TEAM / "figures"

UNITS = {
    "FoothillsInfluent": "TOC mg/L; Alk mg/L (grab samples)",
    "USGS_South_Platte": "cond uS/cm; temp C; turbidity FNU; DO mg/L",
    "SouthPlatteTelemetry": "flow cfs; gage ft; Precip cumulative (dirty)",
    "SouthPlatteFlow": "flow cfs",
    "HoosierPass": "SWE inches",
    "MichiganCreek": "SWE inches (known-bad 2026-05-12..15)",
    "USC00058022": "PRCP/SNOW inches; TMAX/TMIN deg F; ~2-day latency",
}


def audit_daily() -> pd.DataFrame:
    rows = []
    for name, loader in D.ALL_DAILY_LOADERS.items():
        df = loader()
        span = pd.date_range(df.index.min(), df.index.max(), freq="D")
        rows.append({
            "dataset": name,
            "rows": len(df),
            "start": df.index.min().date(),
            "end": df.index.max().date(),
            "calendar_days": len(span),
            "missing_days": len(span) - df.index.nunique(),
            "columns": ";".join(df.columns),
            "total_nans": int(df.isna().sum().sum()),
            "units": UNITS[name],
        })
    return pd.DataFrame(rows)


def audit_sonde() -> dict:
    raw = D.load_strontia()
    ts = raw["Time stamp"]
    day = ts.dt.normalize()
    aug = raw[ts.dt.month == 8]
    surf = aug[aug["Vertical Position"] <= 5]["Temp C"].mean()
    deep = aug[aug["Vertical Position"] >= 40]["Temp C"].mean()
    return {
        "dataset": "Strontia_sonde",
        "rows": len(raw),
        "start": ts.min().date(),
        "end": ts.max().date(),
        "calendar_days": (ts.max().normalize() - ts.min().normalize()).days + 1,
        "missing_days": int((ts.max().normalize() - ts.min().normalize()).days + 1 - day.nunique()),
        "columns": ";".join(c for c in raw.columns if c != "Time stamp"),
        "total_nans": int(raw.isna().sum().sum()),
        "units": (
            f"VP is depth in m (Aug temp {surf:.1f}C at VP<=5 vs {deep:.1f}C at VP>=40, "
            "warm=surface); temp C; cond uS/cm; turb NTU"
        ),
    }


def coverage_figure() -> None:
    fth = D.load_foothills()
    usgs = D.load_usgs()
    sonde = D.load_strontia()
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(fth.index, fth["TOC_mg_L"], ".", ms=2, color="tab:brown")
    axes[0].axhline(3.0, color="r", ls="--", lw=1, label="TOC 3 mg/L threshold")
    axes[0].set_ylabel("TOC mg/L")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Foothills influent record: winter gaps are structural (no Jan-Mar)")
    axes[1].plot(fth.index, fth["Alk_mg_L"], ".", ms=2, color="tab:blue")
    axes[1].axhline(60.0, color="r", ls="--", lw=1, label="Alk 60 mg/L threshold")
    axes[1].set_ylabel("Alk mg/L")
    axes[1].legend(loc="upper right")
    axes[2].plot(usgs.index, usgs["Turbidity_Median"], ".", ms=2, color="tab:gray",
                 label="USGS turbidity (median)")
    days = pd.DatetimeIndex(sonde["Time stamp"].dt.normalize().unique())
    axes[2].plot(days, [1] * len(days), "|", color="tab:green", ms=12,
                 label="Strontia sonde days (2026)")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("FNU (log)")
    axes[2].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "audit_coverage.png", dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    table = pd.concat([audit_daily(), pd.DataFrame([audit_sonde()])], ignore_index=True)
    table.to_csv(RESULTS / "data_audit.csv", index=False)
    with open(RESULTS / "cleaning_log.txt", "w") as fh:
        fh.write("Corrections applied by loaders (documented artifacts only):\n")
        for line in D.CLEANING_LOG:
            fh.write(f"- {line}\n")
        fh.write("\nAll readings are provisional (USGS revises after publication).\n")
    coverage_figure()
    print(table.to_string(index=False))
    print("\nCleaning log:")
    for line in D.CLEANING_LOG:
        print(" -", line)


if __name__ == "__main__":
    main()

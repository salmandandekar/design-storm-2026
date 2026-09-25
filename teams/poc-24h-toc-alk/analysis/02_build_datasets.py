"""02: Build the derived feature/label datasets and spot-check leakage safety.

Writes:
- derived/features_daily.csv  (wide daily frame; column at date T uses data <= T)
- derived/sonde_daily.csv     (Strontia casts collapsed to daily depth bands)

Denver Water's terms travel with these files (derived/TERMS.md).

Spot checks (fail loudly rather than silently mis-join):
- lab_toc at a sampled date equals FoothillsInfluent.csv on that date;
- noaa_prcp_lag2 at T equals the NOAA reading at T-2;
- usgs_turb_roll3 uses only days <= T;
- label y_future at (T, lead) equals the lab value at T+lead.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import data as D  # noqa: E402
from src import features as F  # noqa: E402

TEAM = pathlib.Path(__file__).resolve().parents[1]
DERIVED = TEAM / "derived"

TERMS_HEADER = (
    "# Derived from Denver Water Design Storm data; Denver Water's data terms "
    "apply (see TERMS.md in this folder). Readings are provisional.\n"
)


def spot_checks(feat: pd.DataFrame) -> None:
    fth = D.load_foothills()
    t = pd.Timestamp("2024-06-15")
    assert np.isclose(feat.loc[t, "lab_toc"], fth.loc[t, "TOC_mg_L"]), "lab join broken"

    noaa = D.load_noaa()
    t2 = pd.Timestamp("2023-07-10")
    assert np.isclose(feat.loc[t2, "noaa_prcp_lag2"], noaa.loc[t2 - pd.Timedelta(days=2), "PRCP"]), \
        "NOAA latency shift broken"

    usgs = D.load_usgs()
    t3 = pd.Timestamp("2023-06-20")
    window = usgs["Turbidity_Median"].reindex(pd.date_range(t3 - pd.Timedelta(days=2), t3))
    assert np.isclose(feat.loc[t3, "usgs_turb_roll3"], window.mean()), "roll3 window broken"

    lab = F.make_labels(feat, "toc", lead=1)
    t4 = lab.index[100]
    assert np.isclose(lab.loc[t4, "y_future"], fth.loc[t4 + pd.Timedelta(days=1), "TOC_mg_L"]), \
        "label lead broken"
    assert np.isclose(lab.loc[t4, "y_delta"], lab.loc[t4, "y_future"] - lab.loc[t4, "y_now"])
    print("spot checks passed: joins, latency shift, rolling window, labels")


def main() -> None:
    DERIVED.mkdir(exist_ok=True)
    feat = F.build_daily_features()
    spot_checks(feat)
    sonde = F.build_sonde_daily()

    for name, df in [("features_daily.csv", feat), ("sonde_daily.csv", sonde)]:
        path = DERIVED / name
        with open(path, "w") as fh:
            fh.write(TERMS_HEADER)
            df.to_csv(fh)
        print(f"wrote {path.name}: {df.shape[0]} rows x {df.shape[1]} cols")

    # Anchored-row counts for the ablation (same days across all feature sets).
    for target in ("toc", "alk"):
        lab = F.make_labels(feat, target, lead=1)
        anchored = feat.loc[lab.index, F.FEATURE_SETS_LAB_KNOWN["ABCD"]].dropna()
        print(f"{target}: {len(lab)} lead-1 label rows; {len(anchored)} anchored (full set D present)")


if __name__ == "__main__":
    main()

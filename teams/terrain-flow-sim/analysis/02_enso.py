#!/usr/bin/env python3
"""Flatten the ENSO join in sim-data/enso-context.json into a results table. Offline.

    python3 analysis/02_enso.py        # from teams/terrain-flow-sim/

Writes results/enso_correlations.csv (one row per station/reservoir/headline
test: Spearman rho of Nov-Jan ONI against peak SWE, melt-out day, or peak
storage; permutation p; n; phase means) so the memo and the page cite the
same numbers. The correlations themselves are computed by fetch_oni.py when
it joins the ONI to the committed records.
"""

import csv
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM = os.path.join(HERE, "sim-data")
RES = os.path.join(HERE, "results")


def main():
    e = json.load(open(os.path.join(SIM, "enso-context.json")))
    rows = []
    hl = e.get("headline", {})
    if hl:
        p = hl.get("peak_swe_by_phase", {})
        rows.append({"series": "all mapped SNOTEL (mean peak SWE)", "target": "peak_swe_in",
                     "spearman_rho": hl["spearman_rho"], "p_perm": hl["p_perm"], "n": hl["n"],
                     "el_nino_mean": p.get("El Nino", {}).get("mean_in"), "neutral_mean": p.get("Neutral", {}).get("mean_in"),
                     "la_nina_mean": p.get("La Nina", {}).get("mean_in")})
    for trip, st in e["stations"].items():
        p = st.get("peak_swe_by_phase", {})
        for key, target in (("peak_swe_vs_oni", "peak_swe_in"), ("melt_out_vs_oni", "melt_out_doy")):
            if key in st:
                rows.append({"series": st["name"], "target": target, "spearman_rho": st[key]["spearman_rho"],
                             "p_perm": st[key]["p_perm"], "n": st[key]["n"],
                             "el_nino_mean": p.get("El Nino", {}).get("mean_in") if target == "peak_swe_in" else "",
                             "neutral_mean": p.get("Neutral", {}).get("mean_in") if target == "peak_swe_in" else "",
                             "la_nina_mean": p.get("La Nina", {}).get("mean_in") if target == "peak_swe_in" else ""})
    for code, r in e["reservoirs"].items():
        if "peak_storage_vs_oni" in r:
            rows.append({"series": code, "target": "peak_storage_af", "spearman_rho": r["peak_storage_vs_oni"]["spearman_rho"],
                         "p_perm": r["peak_storage_vs_oni"]["p_perm"], "n": r["peak_storage_vs_oni"]["n"],
                         "el_nino_mean": "", "neutral_mean": "", "la_nina_mean": ""})
    os.makedirs(RES, exist_ok=True)
    cols = ["series", "target", "spearman_rho", "p_perm", "n", "el_nino_mean", "neutral_mean", "la_nina_mean"]
    with open(os.path.join(RES, "enso_correlations.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    sig = [r for r in rows if r["p_perm"] < 0.05]
    print(f"{len(rows)} tests written; {len(sig)} with p < 0.05 (expect ~{round(0.05 * len(rows), 1)} by chance)")


if __name__ == "__main__":
    main()

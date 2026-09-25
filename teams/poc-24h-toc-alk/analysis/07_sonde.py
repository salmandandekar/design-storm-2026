"""07: Strontia sonde secondary study on the 2026 overlap (anchored).

Question: on the identical 2026 days, does adding daily sonde depth-band
features (set E) to set D improve the 24 h forecast? Train on 2022-2025 with
sonde columns absent is impossible, so this study trains WITHIN 2026 using
an expanding-origin split (first 60% train, final 40% test, no shuffling)
for three matched configurations on the same anchored days:
  - ABCD (lab-known), ABCD+E (lab-known), BCD+E (lab-blind)

Hard limits stated in the report: one partial season, zero TOC>3 days in
2026, so this is correlation/lead-time evidence only, not event-warning
evidence. Also writes sonde-vs-influent correlations at lags 0..5 days.

Writes results/sonde_study.csv, results/sonde_correlations.csv.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import baselines as B  # noqa: E402
from src import evaluate as E  # noqa: E402
from src import features as F  # noqa: E402
from src import models as M  # noqa: E402
from src import runner as R  # noqa: E402

TEAM = pathlib.Path(__file__).resolve().parents[1]
LEAD = 1
TRAIN_FRAC = 0.6
SONDE_COLS_MIN_PRESENT = 0.9  # drop sonde columns present on <90% of overlap days


def main() -> None:
    feat = F.build_daily_features()
    sonde = F.build_sonde_daily()
    keep = [c for c in sonde.columns if sonde[c].notna().mean() >= SONDE_COLS_MIN_PRESENT]
    sonde = sonde[keep]
    feat_e = feat.join(sonde, how="left")

    rows = []
    for target in ("toc", "alk"):
        X_full, lab = R.anchored_frame(feat, target, LEAD)
        overlap = lab.index.intersection(sonde.dropna().index)
        X_e = feat_e.loc[overlap, keep]
        anchored = overlap[X_e.notna().all(axis=1)]
        lab_o = lab.loc[anchored]
        n = len(lab_o)
        n_train = int(n * TRAIN_FRAC)
        tr_idx, te_idx = anchored[:n_train], anchored[n_train:]
        configs = {
            "ABCD_lab_known": ("lab_known", F.FEATURE_SETS_LAB_KNOWN["ABCD"]),
            "ABCDE_lab_known": ("lab_known", F.FEATURE_SETS_LAB_KNOWN["ABCD"] + keep),
            "BCDE_lab_blind": ("lab_blind", F.FEATURE_SETS_LAB_BLIND["BCD"] + keep),
        }
        for cfg_name, (scenario, cols) in configs.items():
            X = feat_e.loc[anchored, cols]
            for model_name in M.REGRESSORS:
                pred = M.fit_predict_level(model_name, X.loc[tr_idx], X.loc[te_idx],
                                           lab_o.loc[tr_idx], lab_o.loc[te_idx], scenario)
                e = E.summarize_level_preds(lab_o.loc[te_idx, "y_future"], pred,
                                            B.persistence(lab_o.loc[te_idx]), LEAD)
                rows.append({"target": target, "config": cfg_name, "model": model_name,
                             "n_train": len(tr_idx), **e})
                print(f"{target} {cfg_name:16s} {model_name:5s} n_test={e['n']} "
                      f"MAE={e['mae']:.4f} skill={e['skill_vs_persistence']:+.3f}")
    pd.DataFrame(rows).to_csv(TEAM / "results" / "sonde_study.csv", index=False)

    # Lagged Spearman correlations: sonde feature at T vs influent at T+lag.
    from src import data as D
    fth = D.load_foothills()
    crows = []
    for lag in range(0, 6):
        shifted = sonde.copy()
        shifted.index = shifted.index + pd.Timedelta(days=lag)
        j = shifted.join(fth, how="inner")
        for col in keep:
            for tcol in ("TOC_mg_L", "Alk_mg_L"):
                sub = j[[col, tcol]].dropna()
                if len(sub) >= 20:
                    rho = sub[col].corr(sub[tcol], method="spearman")
                    crows.append({"sonde_feature": col, "target": tcol, "lag_days": lag,
                                  "spearman_rho": rho, "n": len(sub)})
    pd.DataFrame(crows).to_csv(TEAM / "results" / "sonde_correlations.csv", index=False)
    print("wrote sonde_study.csv, sonde_correlations.csv")


if __name__ == "__main__":
    main()

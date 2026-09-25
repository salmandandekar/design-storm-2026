"""03: Score the naive baselines on the rolling-origin folds (lead 1).

Run BEFORE any model: these are the bars. Also validates the harness (the
persistence row must show skill exactly 0 against itself, DM p = NaN).
Writes results/baselines_lead1.csv.
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import baselines as B  # noqa: E402
from src import evaluate as E  # noqa: E402
from src import features as F  # noqa: E402
from src import runner as R  # noqa: E402

TEAM = pathlib.Path(__file__).resolve().parents[1]
LEAD = 1


def main() -> None:
    feat = F.build_daily_features()
    rows = []
    for target in ("toc", "alk"):
        X_full, lab = R.anchored_frame(feat, target, LEAD)
        preds = {name: [] for name in ("persistence", "climatology", "persistence_drift")}
        trues, persists = [], []
        for fold_name, train, test in R.rolling_fold_slices(lab.index):
            lab_te = lab[test]
            train_end = lab.index[train].max()
            fold_preds = {
                "persistence": B.persistence(lab_te),
                "climatology": B.climatology(lab_te, target, train_end),
                "persistence_drift": B.persistence_drift(lab_te, target, LEAD),
            }
            for name, p in fold_preds.items():
                e = E.summarize_level_preds(lab_te["y_future"], p, B.persistence(lab_te), LEAD)
                rows.append({"target": target, "baseline": name, "fold": fold_name, **e})
                preds[name].append(p)
            trues.append(lab_te["y_future"])
            persists.append(B.persistence(lab_te))
        y_all = pd.concat(trues)
        p_persist = pd.concat(persists)
        for name, plist in preds.items():
            e = E.summarize_level_preds(y_all, pd.concat(plist), p_persist, LEAD)
            rows.append({"target": target, "baseline": name, "fold": "POOLED", **e})

    out = pd.DataFrame(rows)
    out.to_csv(TEAM / "results" / "baselines_lead1.csv", index=False)
    pooled = out[out["fold"] == "POOLED"]
    print(pooled[["target", "baseline", "n", "mae", "rmse", "skill_vs_persistence",
                  "skill_lo", "skill_hi", "dm_p"]].to_string(index=False))


if __name__ == "__main__":
    main()

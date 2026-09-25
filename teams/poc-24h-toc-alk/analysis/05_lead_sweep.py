"""05: Lead-time sweep (1, 2, 3, 4, 7 days) for the headline combos.

Reuses the ablation pipeline at each lead for the best lab-known and
lab-blind combos (fixed as ABCD/ridge-vs-tree winner per 04; to stay
pre-registered we sweep ALL three models on ABCD and BCD and report each,
rather than cherry-picking after seeing lead-1 results).

12-hour prediction is not testable from daily lab data; this sweep brackets
the 24-hour claim from above instead.

Writes results/lead_sweep.csv and figures/lead_sweep.png.
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import baselines as B  # noqa: E402
from src import evaluate as E  # noqa: E402
from src import features as F  # noqa: E402
from src import models as M  # noqa: E402
from src import runner as R  # noqa: E402

TEAM = pathlib.Path(__file__).resolve().parents[1]
LEADS = [1, 2, 3, 4, 7]
COMBOS = [("lab_known", "ABCD", F.FEATURE_SETS_LAB_KNOWN["ABCD"]),
          ("lab_blind", "BCD", F.FEATURE_SETS_LAB_BLIND["BCD"])]


def main() -> None:
    feat = F.build_daily_features()
    rows = []
    for target in ("toc", "alk"):
        for lead in LEADS:
            X_full, lab = R.anchored_frame(feat, target, lead)
            for scenario, set_name, cols in COMBOS:
                for model_name in M.REGRESSORS:
                    pooled_pred, pooled_true, pooled_persist = [], [], []
                    for fold_name, train, test in R.rolling_fold_slices(lab.index):
                        pred = M.fit_predict_level(model_name, X_full[cols][train],
                                                   X_full[cols][test], lab[train],
                                                   lab[test], scenario)
                        pooled_pred.append(pred)
                        pooled_true.append(lab[test]["y_future"])
                        pooled_persist.append(B.persistence(lab[test]))
                    e = E.summarize_level_preds(pd.concat(pooled_true), pd.concat(pooled_pred),
                                                pd.concat(pooled_persist), lead)
                    rows.append({"target": target, "lead": lead, "scenario": scenario,
                                 "set": set_name, "model": model_name, **e})
                    print(f"{target} lead={lead} {scenario:9s} {model_name:5s} "
                          f"MAE={e['mae']:.4f} persistMAE={e['mae_persistence']:.4f} "
                          f"skill={e['skill_vs_persistence']:+.3f} DM p={e['dm_p']:.3g}")
    out = pd.DataFrame(rows)
    out.to_csv(TEAM / "results" / "lead_sweep.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, target in zip(axes, ("toc", "alk")):
        sub = out[out["target"] == target]
        for scenario, style in [("lab_known", "-o"), ("lab_blind", "--s")]:
            best = (sub[sub["scenario"] == scenario]
                    .sort_values("mae").groupby("lead", as_index=False).first()
                    .sort_values("lead"))
            ax.plot(best["lead"], best["skill_vs_persistence"], style, label=scenario)
            ax.fill_between(best["lead"], best["skill_lo"], best["skill_hi"], alpha=0.15)
        ax.axhline(0, color="k", lw=1)
        ax.set_xlabel("lead (days)")
        ax.set_title(f"{target.upper()}: best-model skill vs persistence by lead")
        ax.legend()
    axes[0].set_ylabel("skill vs persistence (95% CI)")
    fig.tight_layout()
    fig.savefig(TEAM / "figures" / "lead_sweep.png", dpi=150)
    plt.close(fig)
    print("wrote lead_sweep.csv / lead_sweep.png")


if __name__ == "__main__":
    main()

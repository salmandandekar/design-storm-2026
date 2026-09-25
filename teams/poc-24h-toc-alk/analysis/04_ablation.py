"""04: Ablation ladder at lead 1 on rolling-origin folds, both scenarios.

Lab-known (primary): sets A, AB, ABC, ABCD; models trained on delta, scored
in level space. Lab-blind: sets B, BC, BCD; models trained on level.
Plus a Jake-equivalent reference (his notebook feature lists, RF, level
target) run under this same evaluation for continuity with his work.

Writes:
- results/ablation_lead1.csv          per-fold and pooled metrics
- results/predictions_lead1.csv       per-day pooled test predictions (reused by 06)
- results/regime_lead1.csv            wet->dry / dry->wet robustness for best combos
- figures/ablation_lead1.png
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

CONFIGS: list[tuple[str, str, list[str], str]] = []  # (scenario, set_name, cols, model)
for set_name, cols in F.FEATURE_SETS_LAB_KNOWN.items():
    for m in M.REGRESSORS:
        CONFIGS.append(("lab_known", set_name, cols, m))
for set_name, cols in F.FEATURE_SETS_LAB_BLIND.items():
    for m in M.REGRESSORS:
        CONFIGS.append(("lab_blind", set_name, cols, m))


def jake_config(target: str) -> tuple[str, str, list[str], str]:
    cols = F.JAKE_TOC if target == "toc" else F.JAKE_ALK
    return ("lab_blind", "jake_ref", cols, "rf")


def run_target(target: str, feat: pd.DataFrame):
    X_full, lab = R.anchored_frame(feat, target, LEAD)
    metric_rows, pred_rows = [], []
    configs = CONFIGS + [jake_config(target)]

    for scenario, set_name, cols, model_name in configs:
        X = X_full[cols]
        pooled_pred, pooled_true, pooled_persist = [], [], []
        for fold_name, train, test in R.rolling_fold_slices(lab.index):
            X_tr, X_te = X[train], X[test]
            lab_tr, lab_te = lab[train], lab[test]
            pred = M.fit_predict_level(model_name, X_tr, X_te, lab_tr, lab_te, scenario)
            persist = B.persistence(lab_te)
            e = E.summarize_level_preds(lab_te["y_future"], pred, persist, LEAD)
            metric_rows.append({"target": target, "scenario": scenario, "set": set_name,
                                "model": model_name, "fold": fold_name, **e})
            novelty = E.novelty_count(X_tr, X_te)
            pooled_pred.append(pred)
            pooled_true.append(lab_te["y_future"])
            pooled_persist.append(persist)
            for t in pred.index:
                pred_rows.append({
                    "target": target, "scenario": scenario, "set": set_name,
                    "model": model_name, "fold": fold_name, "date": t,
                    "y_now": lab_te.loc[t, "y_now"], "y_true": lab_te.loc[t, "y_future"],
                    "pred": pred.loc[t], "novelty": int(novelty.loc[t]),
                })
        e = E.summarize_level_preds(pd.concat(pooled_true), pd.concat(pooled_pred),
                                    pd.concat(pooled_persist), LEAD)
        metric_rows.append({"target": target, "scenario": scenario, "set": set_name,
                            "model": model_name, "fold": "POOLED", **e})
        print(f"{target} {scenario:9s} {set_name:8s} {model_name:5s} "
              f"pooled MAE={e['mae']:.4f} skill={e['skill_vs_persistence']:+.3f} "
              f"[{e['skill_lo']:+.3f},{e['skill_hi']:+.3f}] DM p={e['dm_p']:.3g}")
    return metric_rows, pred_rows, X_full, lab


def run_regimes(target: str, feat: pd.DataFrame, best: pd.DataFrame):
    """Wet->dry / dry->wet robustness for each scenario's best pooled combo."""
    X_full, lab = R.anchored_frame(feat, target, LEAD)
    rows = []
    for _, brow in best.iterrows():
        cols = (F.FEATURE_SETS_LAB_KNOWN | F.FEATURE_SETS_LAB_BLIND).get(brow["set"])
        if cols is None:
            cols = F.JAKE_TOC if target == "toc" else F.JAKE_ALK
        X = X_full[cols]
        for fold_name, train, test in R.regime_fold_slices(lab.index):
            pred = M.fit_predict_level(brow["model"], X[train], X[test],
                                       lab[train], lab[test], brow["scenario"])
            e = E.summarize_level_preds(lab[test]["y_future"], pred,
                                        B.persistence(lab[test]), LEAD)
            rows.append({"target": target, "scenario": brow["scenario"], "set": brow["set"],
                         "model": brow["model"], "fold": fold_name, **e})
    return rows


def figure(metrics: pd.DataFrame) -> None:
    pooled = metrics[(metrics["fold"] == "POOLED") & (metrics["set"] != "jake_ref")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, target in zip(axes, ("toc", "alk")):
        sub = pooled[pooled["target"] == target]
        order = ["A", "AB", "ABC", "ABCD", "B", "BC", "BCD"]
        xticks, xlabels = [], []
        for i, set_name in enumerate(order):
            for j, model in enumerate(M.REGRESSORS):
                row = sub[(sub["set"] == set_name) & (sub["model"] == model)]
                if row.empty:
                    continue
                r = row.iloc[0]
                x = i + (j - 1) * 0.22
                color = "tab:blue" if r["scenario"] == "lab_known" else "tab:orange"
                ax.errorbar(x, r["skill_vs_persistence"],
                            yerr=[[r["skill_vs_persistence"] - r["skill_lo"]],
                                  [r["skill_hi"] - r["skill_vs_persistence"]]],
                            fmt="o", ms=4, color=color, capsize=2)
            xticks.append(i)
            xlabels.append(set_name)
        ax.axhline(0, color="k", lw=1)
        ax.set_xticks(xticks, xlabels)
        ax.set_title(f"{target.upper()}: 24 h skill vs persistence (pooled, 95% CI)")
        ax.set_xlabel("feature set (blue = lab-known, orange = lab-blind)")
    axes[0].set_ylabel("skill = 1 - MAE_model / MAE_persistence")
    fig.tight_layout()
    fig.savefig(TEAM / "figures" / "ablation_lead1.png", dpi=150)
    plt.close(fig)


def main() -> None:
    feat = F.build_daily_features()
    all_metrics, all_preds, all_regimes = [], [], []
    for target in ("toc", "alk"):
        metric_rows, pred_rows, _, _ = run_target(target, feat)
        all_metrics += metric_rows
        all_preds += pred_rows
        mdf = pd.DataFrame(metric_rows)
        pooled = mdf[mdf["fold"] == "POOLED"]
        best = (pooled.sort_values("mae").groupby("scenario", as_index=False).first())
        all_regimes += run_regimes(target, feat, best)

    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(TEAM / "results" / "ablation_lead1.csv", index=False)
    pd.DataFrame(all_preds).to_csv(TEAM / "results" / "predictions_lead1.csv", index=False)
    pd.DataFrame(all_regimes).to_csv(TEAM / "results" / "regime_lead1.csv", index=False)
    figure(metrics)
    print("\nwrote ablation_lead1.csv, predictions_lead1.csv, regime_lead1.csv")


if __name__ == "__main__":
    main()

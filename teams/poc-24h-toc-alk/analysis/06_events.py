"""06: Event-onset scorecards, episode detection, novelty guardrail (lead 1).

Two complementary views, both pooled over rolling-origin folds:

1. Onset classification: among days NOT in state at T, predict entry into
   TOC > 3 / Alk < 60 at T+1.
   - Classifiers (logit / balanced RF) trained per fold on onset labels.
   - The best 04 regressors' level forecasts thresholded as implicit
     classifiers (pred crosses the threshold).
   Precision/recall with Wilson 95% CIs vs the onset base rate; the
   pre-registered insufficiency rule (<20 pooled onsets) is applied.

2. Episode scorecard: maximal in-state runs; was a warning raised on the
   onset day or within 2 days? False alarms per season.

Novelty guardrail: pooled |error| binned by count of features outside the
train 5th-95th percentile band (from 04 predictions), plus Spearman rho.

Writes results/events_onset.csv, results/episodes_{target}.csv,
results/novelty.csv; figures/event_timeline_{target}.png.
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import data as D  # noqa: E402
from src import evaluate as E  # noqa: E402
from src import events as EV  # noqa: E402
from src import features as F  # noqa: E402
from src import models as M  # noqa: E402
from src import runner as R  # noqa: E402

TEAM = pathlib.Path(__file__).resolve().parents[1]
LEAD = 1
MIN_ONSETS = 20  # pre-registered insufficiency rule


def onset_from_classifiers(target: str, feat: pd.DataFrame) -> list[dict]:
    X_full, lab = R.anchored_frame(feat, target, LEAD)
    cols = F.FEATURE_SETS_LAB_KNOWN["ABCD"]
    rows = []
    for model_name in M.CLASSIFIERS:
        pooled = []
        for fold_name, train, test in R.rolling_fold_slices(lab.index):
            elig_tr = EV.onset_labels(lab[train], target)
            elig_te = EV.onset_labels(lab[test], target)
            if elig_tr["onset"].nunique() < 2 or len(elig_te) == 0:
                continue
            gs = M.make_classifier(model_name)
            gs.fit(X_full[cols].loc[elig_tr.index], elig_tr["onset"])
            pred = pd.Series(gs.predict(X_full[cols].loc[elig_te.index]),
                             index=elig_te.index)
            pooled.append(pd.DataFrame({"onset": elig_te["onset"], "flag": pred}))
        if not pooled:
            continue
        dfp = pd.concat(pooled)
        rows.append(score_onsets(target, f"clf_{model_name}", dfp))
    return rows


def onset_from_regressor_preds(target: str, preds: pd.DataFrame) -> list[dict]:
    """Threshold the level forecasts of every 04 combo; keep the two scenarios' best-MAE combos."""
    rows = []
    sub = preds[preds["target"] == target].copy()
    sub["abs_err"] = (sub["pred"] - sub["y_true"]).abs()
    best = (sub.groupby(["scenario", "set", "model"])["abs_err"].mean()
            .reset_index().sort_values("abs_err").groupby("scenario").first().reset_index())
    for _, b in best.iterrows():
        p = sub[(sub["scenario"] == b["scenario"]) & (sub["set"] == b["set"])
                & (sub["model"] == b["model"])].set_index("date")
        now_state = EV.in_state(p["y_now"], target)
        elig = p[~now_state]
        dfp = pd.DataFrame({
            "onset": EV.in_state(elig["y_true"], target).astype(int),
            "flag": EV.in_state(elig["pred"], target).astype(int),
        })
        rows.append(score_onsets(target, f"reg_{b['scenario']}_{b['set']}_{b['model']}", dfp))
    return rows


def score_onsets(target: str, method: str, dfp: pd.DataFrame) -> dict:
    n = len(dfp)
    onsets = int(dfp["onset"].sum())
    flags = int(dfp["flag"].sum())
    tp = int(((dfp["onset"] == 1) & (dfp["flag"] == 1)).sum())
    prec, prec_lo, prec_hi = E.wilson(tp, flags)
    rec, rec_lo, rec_hi = E.wilson(tp, onsets)
    return {
        "target": target, "method": method, "eligible_days": n,
        "onsets": onsets, "flags": flags, "true_positives": tp,
        "base_rate": onsets / n if n else np.nan,
        "precision": prec, "precision_lo": prec_lo, "precision_hi": prec_hi,
        "recall": rec, "recall_lo": rec_lo, "recall_hi": rec_hi,
        "sufficient_by_rule": onsets >= MIN_ONSETS,
    }


def episode_scorecard(target: str, preds: pd.DataFrame) -> pd.DataFrame:
    col = {"toc": "TOC_mg_L", "alk": "Alk_mg_L"}[target]
    y = D.load_foothills()[col]
    sub = preds[(preds["target"] == target)].copy()
    sub["abs_err"] = (sub["pred"] - sub["y_true"]).abs()
    b = (sub.groupby(["scenario", "set", "model"])["abs_err"].mean()
         .reset_index().sort_values("abs_err").iloc[0])
    p = sub[(sub["scenario"] == b["scenario"]) & (sub["set"] == b["set"])
            & (sub["model"] == b["model"])].set_index("date")
    p.index = pd.to_datetime(p.index)
    test_years = sorted(p.index.year.unique())
    y_test = y[np.isin(y.index.year, test_years)]
    episodes = [ep for ep in EV.find_episodes(y_test, target)]
    flagged = p.index[EV.in_state(p["pred"], target)]
    card = EV.score_episodes(episodes, flagged)
    card.insert(0, "combo", f"{b['scenario']}_{b['set']}_{b['model']}")
    # false alarms: flagged days where truth never entered state that day
    truth_state = EV.in_state(p["y_true"], target)
    card.attrs["false_alarm_days"] = int((EV.in_state(p["pred"], target) & ~truth_state).sum())
    card.attrs["flagged_days"] = int(len(flagged))
    return card, p


def novelty_table(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in ("toc", "alk"):
        sub = preds[(preds["target"] == target) & (preds["scenario"] == "lab_known")
                    & (preds["set"] == "ABCD")].copy()
        sub["abs_err"] = (sub["pred"] - sub["y_true"]).abs()
        best_model = sub.groupby("model")["abs_err"].mean().idxmin()
        sub = sub[sub["model"] == best_model]
        bins = [-0.5, 0.5, 2.5, 5.5, np.inf]
        labels = ["0", "1-2", "3-5", ">5"]
        sub["nov_bin"] = pd.cut(sub["novelty"], bins=bins, labels=labels)
        g = sub.groupby("nov_bin", observed=True)["abs_err"].agg(["count", "mean"])
        rho, pval = stats.spearmanr(sub["novelty"], sub["abs_err"])
        for bin_name, r in g.iterrows():
            rows.append({"target": target, "novelty_bin": bin_name, "days": int(r["count"]),
                         "mae": r["mean"], "spearman_rho": rho, "spearman_p": pval})
    return pd.DataFrame(rows)


def timeline_figure(target: str, p: pd.DataFrame, card: pd.DataFrame) -> None:
    thr = EV.TOC_THRESHOLD if target == "toc" else EV.ALK_THRESHOLD
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(p.index, p["y_true"], ".", ms=3, color="tab:blue", label="measured (T+1)")
    ax.plot(p.index, p["pred"], ".", ms=2, color="tab:orange", alpha=0.7, label="forecast")
    ax.axhline(thr, color="r", ls="--", lw=1, label=f"threshold {thr:g}")
    for _, ep in card.iterrows():
        ax.axvspan(ep["start"], ep["end"], color="red", alpha=0.12)
        if ep["detected"]:
            ax.axvline(ep["first_warning"], color="green", lw=1.2)
    ax.set_title(f"{target.upper()} 24 h forecasts on test seasons; shaded = episodes, "
                 "green = first warning within 2 days of onset")
    ax.set_ylabel("mg/L")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(TEAM / "figures" / f"event_timeline_{target}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    feat = F.build_daily_features()
    preds = pd.read_csv(TEAM / "results" / "predictions_lead1.csv", parse_dates=["date"])
    onset_rows, episode_meta = [], []
    for target in ("toc", "alk"):
        onset_rows += onset_from_classifiers(target, feat)
        onset_rows += onset_from_regressor_preds(target, preds)
        card, p = episode_scorecard(target, preds)
        card.to_csv(TEAM / "results" / f"episodes_{target}.csv", index=False)
        episode_meta.append({"target": target, "episodes": len(card),
                             "detected_within_2d": int(card["detected"].sum()),
                             "false_alarm_days": card.attrs["false_alarm_days"],
                             "flagged_days": card.attrs["flagged_days"],
                             "combo": card["combo"].iloc[0] if len(card) else ""})
        timeline_figure(target, p, card)

    pd.DataFrame(onset_rows).to_csv(TEAM / "results" / "events_onset.csv", index=False)
    pd.DataFrame(episode_meta).to_csv(TEAM / "results" / "episodes_summary.csv", index=False)
    nov = novelty_table(preds)
    nov.to_csv(TEAM / "results" / "novelty.csv", index=False)
    print(pd.DataFrame(onset_rows).to_string(index=False))
    print(pd.DataFrame(episode_meta).to_string(index=False))
    print(nov.to_string(index=False))


if __name__ == "__main__":
    main()

"""Actionable-event definitions and episode-level scoring.

Events (thresholds from Jake's sample weights / classifier; the README flags
them as round numbers to confirm with operators):
- TOC event:        TOC > 3.0 mg/L
- Alkalinity event: Alk < 60.0 mg/L

Onset at lead h: not in state on the last observed lab day at/before T, in
state at T+h. Onsets are the operator-relevant warning; being in-state on a
day deep inside a long excursion is already known to the plant.

Episodes: maximal runs of consecutive observed in-state lab days, where the
previous observed day (within MAX_EPISODE_GAP_DAYS) was out of state.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TOC_THRESHOLD = 3.0     # mg/L, exceedance is the event
ALK_THRESHOLD = 60.0    # mg/L, deficit is the event
MAX_EPISODE_GAP_DAYS = 7
DETECTION_WINDOW_DAYS = 2  # flag on onset day or the next 2 days counts as caught


def in_state(y: pd.Series, target: str) -> pd.Series:
    if target == "toc":
        return y > TOC_THRESHOLD
    if target == "alk":
        return y < ALK_THRESHOLD
    raise KeyError(target)


def onset_labels(lab: pd.DataFrame, target: str) -> pd.DataFrame:
    """Rows eligible for onset prediction (not in state at T) with binary label."""
    now_state = in_state(lab["y_now"], target)
    fut_state = in_state(lab["y_future"], target)
    eligible = lab[~now_state].copy()
    eligible["onset"] = fut_state[~now_state].astype(int)
    return eligible


def find_episodes(y: pd.Series, target: str) -> list[dict]:
    """Maximal in-state runs over observed lab days."""
    state = in_state(y, target)
    episodes = []
    current = None
    prev_day = None
    for day, s in state.items():
        gap = (day - prev_day).days if prev_day is not None else None
        if s and current is None:
            current = {"start": day, "end": day}
        elif s and current is not None:
            if gap is not None and gap > MAX_EPISODE_GAP_DAYS:
                episodes.append(current)
                current = {"start": day, "end": day}
            else:
                current["end"] = day
        elif not s and current is not None:
            episodes.append(current)
            current = None
        prev_day = day
    if current is not None:
        episodes.append(current)
    for ep in episodes:
        ep["days"] = int((ep["end"] - ep["start"]).days) + 1
    return episodes


def score_episodes(episodes: list[dict], flagged_days: pd.DatetimeIndex,
                   window: int = DETECTION_WINDOW_DAYS) -> pd.DataFrame:
    """Per episode: was a warning raised on the onset day or within `window` days?"""
    rows = []
    for ep in episodes:
        hits = [d for d in flagged_days
                if ep["start"] <= d <= min(ep["start"] + pd.Timedelta(days=window), ep["end"])]
        first_hit = min(hits) if hits else pd.NaT
        rows.append({
            "start": ep["start"], "end": ep["end"], "days": ep["days"],
            "detected": bool(hits),
            "first_warning": first_hit,
            "lead_days": (first_hit - ep["start"]).days if hits else np.nan,
        })
    return pd.DataFrame(rows)

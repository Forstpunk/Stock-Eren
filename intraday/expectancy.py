"""Expectancy with bootstrap CIs, plus Bulkowski's breakeven failure rate.

expectancy_R = win_rate x avg_win_R - loss_rate x |avg_loss_R|   (net of costs)

No expectancy below ``config.min_sample`` trades: ``InsufficientSampleError`` is raised,
never a caveated number. Segment helpers return what could be computed and list what
could not, with the count that fell short.
"""
from __future__ import annotations

import math
from datetime import time

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from intraday.config import Config

BREAKEVEN_R = 0.5  # a trade that never reaches +0.5R is a "failure" in Bulkowski's sense


class InsufficientSampleError(Exception):
    def __init__(self, segment: str, n: int, minimum: int) -> None:
        self.segment = segment
        self.n = n
        self.minimum = minimum
        super().__init__(f"{segment}: {n} trades, below the minimum of {minimum}; no expectancy claimed")


class Expectancy(BaseModel):
    model_config = ConfigDict(frozen=True)

    segment: str
    n: int
    win_rate: float
    avg_win_r: float
    avg_loss_r: float  # negative or zero
    expectancy_r: float
    ci_low: float
    ci_high: float
    indistinguishable_from_zero: bool
    breakeven_failure_rate: float  # share of trades whose MFE never reached BREAKEVEN_R
    statement: str


def expectancy(r_net: pd.Series, mfe_r: pd.Series, segment: str, config: Config, seed: int) -> Expectancy:
    r = r_net.to_numpy(dtype="float64")
    m = mfe_r.to_numpy(dtype="float64")
    if len(r) != len(m):
        raise ValueError("r_net and mfe_r must be aligned")
    if np.isnan(r).any() or np.isnan(m).any():
        raise ValueError(f"{segment}: NaN in trade results")
    n = len(r)
    if n < config.min_sample:
        raise InsufficientSampleError(segment, n, config.min_sample)
    wins, losses = r[r > 0], r[r <= 0]
    win_rate = len(wins) / n
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    exp = win_rate * avg_win - (1 - win_rate) * abs(avg_loss)
    rng = np.random.default_rng(seed)
    boots = np.array([r[rng.integers(0, n, n)].mean() for _ in range(config.bootstrap_n)])
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    flat = lo <= 0 <= hi
    failure = float((m < BREAKEVEN_R).mean())
    statement = (
        f"{segment}: n={n}, win rate {win_rate:.1%}, avg win {avg_win:+.2f}R, avg loss {avg_loss:+.2f}R, "
        f"expectancy {exp:+.3f}R (95% CI {lo:+.3f} to {hi:+.3f}); breakeven failure rate {failure:.1%}. "
        + ("Indistinguishable from zero." if flat else "Distinguishable from zero.")
    )
    return Expectancy(
        segment=segment, n=n, win_rate=win_rate, avg_win_r=avg_win, avg_loss_r=avg_loss,
        expectancy_r=exp, ci_low=lo, ci_high=hi, indistinguishable_from_zero=flat,
        breakeven_failure_rate=failure, statement=statement,
    )


# ---- segment keys ---------------------------------------------------------------------


def rvol_bucket(value: float, threshold: float) -> str:
    if math.isnan(value):
        return "rvol n/a"
    if value < 1.0:
        return "rvol <1.0"
    if value < threshold:
        return f"rvol 1.0-{threshold:g}"
    return f"rvol >={threshold:g}"


def or_width_quartiles(values: pd.Series) -> pd.Series:
    """Quartile label per row, computed over the given trade set (NaN -> 'or_width n/a')."""
    labels = pd.Series("or_width n/a", index=values.index, dtype="object")
    ok = values.notna()
    if ok.sum() >= 4:
        codes = pd.qcut(values[ok], 4, labels=False, duplicates="drop")  # fewer bins if edges tie
        labels[ok] = ["or_width n/a" if math.isnan(c) else f"or_width Q{int(c) + 1}" for c in codes]
    return labels


def entry_half_hour(t: time) -> str:
    minutes = t.hour * 60 + t.minute
    start = minutes // 30 * 30
    return f"{start // 60:02d}:{start % 60:02d}"


def segment_keys(results: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Add the three segment columns to a results frame (from ``trades.results_frame``)."""
    out = results.copy()
    out["seg_rvol"] = [rvol_bucket(v, config.rvol_threshold) for v in out["rvol_breakout_bar"]]  # from features
    out["seg_or_width"] = or_width_quartiles(out["or_width_atr"])
    out["seg_half_hour"] = [entry_half_hour(pd.Timestamp(t).time()) for t in out["entry_time"]]
    return out


def segmented_expectancy(
    results: pd.DataFrame, by: str, config: Config, seed: int
) -> tuple[dict[str, Expectancy], dict[str, int]]:
    """(computed segments, skipped segments with their counts). One slippage level at a time."""
    if results["slippage_bps"].nunique() != 1:
        raise ValueError("segment one slippage level at a time")
    computed: dict[str, Expectancy] = {}
    skipped: dict[str, int] = {}
    for key, g in results.groupby(by, sort=True):
        try:
            computed[str(key)] = expectancy(g["r_net"], g["mfe_r"], str(key), config, seed)
        except InsufficientSampleError as exc:
            skipped[str(key)] = exc.n
    return computed, skipped

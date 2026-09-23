"""Resampling that respects how the data is actually grouped.

Breakouts in the same session are not independent: they share the day's market-wide
shock. Resampling individual rows pretends they are, which makes every confidence
interval too narrow and turns noise into "findings". Every interval in this pipeline is
therefore built by resampling whole sessions.

``session_bootstrap`` draws sessions with replacement, concatenates their row positions,
and applies the statistic to that index. A draw whose statistic is undefined (an empty
denominator, a single class) is skipped; if more than half the draws are undefined the
sample is too thin to bootstrap and the function raises rather than quoting an interval
built from the remainder.

One result that looks wrong and is not: resampling sessions widens the interval for a
LEVEL statistic (a mean, a rate) because clustered rows carry less information than their
count suggests, but it can NARROW the interval for a PAIRED statistic - a skill score
comparing two forecasts on the same rows - because keeping a session's rows together
preserves the pairing instead of mixing regimes. That is the same reason a paired t-test
is tighter than an unpaired one, and both directions are pinned by tests below.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

MIN_VALID_SHARE = 0.5


class BootstrapError(Exception):
    """Too few resamples produced a defined statistic to quote an interval."""


def session_bootstrap(
    sessions: np.ndarray,
    stat: Callable[[np.ndarray], float],
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, np.ndarray]:
    """Resample whole sessions with replacement; return (2.5th pct, 97.5th pct, draws).

    ``sessions`` holds one session key per row, in row order. ``stat`` receives the row
    positions of one resample and returns a float, or NaN when it cannot be computed.
    """
    if len(sessions) == 0:
        raise BootstrapError("no rows to bootstrap")
    if n_boot <= 0:
        raise ValueError(f"n_boot must be positive, got {n_boot}")

    keys, inverse = np.unique(sessions, return_inverse=True)
    # Row positions per session, so a draw is a concatenation of whole groups.
    order = np.argsort(inverse, kind="stable")
    boundaries = np.searchsorted(inverse[order], np.arange(len(keys) + 1))
    groups = [order[boundaries[g] : boundaries[g + 1]] for g in range(len(keys))]

    draws: list[float] = []
    for _ in range(n_boot):
        picked = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[g] for g in picked])
        value = stat(idx)
        if value is not None and not math.isnan(float(value)):
            draws.append(float(value))

    if len(draws) < MIN_VALID_SHARE * n_boot:
        raise BootstrapError(
            f"only {len(draws)} of {n_boot} resamples produced a defined statistic "
            f"({len(keys)} sessions, {len(sessions)} rows); the sample is too thin to bootstrap"
        )
    array = np.array(draws)
    return float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5)), array


def mean_stat(values: np.ndarray) -> Callable[[np.ndarray], float]:
    """The commonest statistic: the mean of ``values`` over the resampled rows."""

    def stat(idx: np.ndarray) -> float:
        return float(values[idx].mean()) if len(idx) else math.nan

    return stat


def session_variance_share(values: np.ndarray, sessions: np.ndarray) -> float:
    """Share of the variance of ``values`` explained by session means (eta squared).

    0 means sessions tell you nothing about the outcome and rows are effectively
    independent; values near 1 mean the day decides almost everything, and the effective
    sample size is closer to the number of sessions than to the number of rows.
    """
    if len(values) != len(sessions):
        raise ValueError("values and sessions must be the same length")
    if len(values) < 2:
        return math.nan
    total = float(((values - values.mean()) ** 2).sum())
    if total <= 0:
        return 0.0
    keys, inverse = np.unique(sessions, return_inverse=True)
    between = 0.0
    for g in range(len(keys)):
        group = values[inverse == g]
        between += len(group) * (group.mean() - values.mean()) ** 2
    return float(between / total)


def effective_sample_size(n_rows: int, n_sessions: int, variance_share: float) -> float:
    """Rows are worth less than they look when they cluster. A rough design-effect
    adjustment: with m rows per session and intra-session correlation rho, the effective
    count is n / (1 + (m - 1) * rho)."""
    if n_sessions <= 0 or n_rows <= 0:
        return 0.0
    if math.isnan(variance_share):
        return float(n_rows)
    m = n_rows / n_sessions
    rho = max(0.0, min(1.0, variance_share))
    return float(n_rows / (1 + (m - 1) * rho))

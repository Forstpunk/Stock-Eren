"""The session bootstrap: it must not be fooled by data that clusters by day.

The point of this module is calibration. If trades on one session share a shock, an iid
bootstrap quotes an interval far too narrow and declares edges that are not there. These
tests hold the session bootstrap to a nominal false-positive rate on data built with a
known-zero true effect and a strong shared shock.
"""
from __future__ import annotations

import numpy as np
import pytest

from intraday.stats import (
    BootstrapError,
    effective_sample_size,
    mean_stat,
    session_bootstrap,
    session_variance_share,
)

N_SESSIONS = 41
PER_SESSION = 20
SESSION_SHOCK_SD = 0.6
IDIOSYNCRATIC_SD = 1.0
SEEDS = range(40)  # enough to see a false-positive rate; keeps the test near a second


def clustered(seed: int, shock_sd: float = SESSION_SHOCK_SD, per_session: int = PER_SESSION):  # type: ignore[no-untyped-def]
    """Zero true mean, but every row in a session shares that session's shock."""
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0.0, shock_sd, N_SESSIONS)
    values = np.concatenate([
        shocks[s] + rng.normal(0.0, IDIOSYNCRATIC_SD, per_session) for s in range(N_SESSIONS)
    ])
    sessions = np.repeat(np.arange(N_SESSIONS), per_session)
    return values, sessions


def iid_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    """The old behaviour: resample rows, ignoring which session they came from."""
    n = len(values)
    draws = np.array([values[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def test_session_bootstrap_holds_its_nominal_false_positive_rate() -> None:
    """With no true effect, a 95% interval should miss zero about 5% of the time."""
    session_misses = iid_misses = 0
    for seed in SEEDS:
        values, sessions = clustered(seed)
        lo, hi, _ = session_bootstrap(sessions, mean_stat(values), 400, np.random.default_rng(seed))
        if lo > 0 or hi < 0:
            session_misses += 1
        ilo, ihi = iid_ci(values, 400, np.random.default_rng(seed))
        if ilo > 0 or ihi < 0:
            iid_misses += 1
    session_rate = session_misses / len(SEEDS)
    iid_rate = iid_misses / len(SEEDS)
    assert session_rate < 0.15, f"session bootstrap false-positive rate {session_rate:.0%}"
    assert iid_rate > session_rate, (
        f"the iid bootstrap should be the over-confident one: iid {iid_rate:.0%} vs session {session_rate:.0%}"
    )


def test_session_intervals_are_wider_when_data_clusters() -> None:
    widths_session, widths_iid = [], []
    for seed in SEEDS:
        values, sessions = clustered(seed)
        lo, hi, _ = session_bootstrap(sessions, mean_stat(values), 400, np.random.default_rng(seed))
        ilo, ihi = iid_ci(values, 400, np.random.default_rng(seed))
        widths_session.append(hi - lo)
        widths_iid.append(ihi - ilo)
    assert np.mean(widths_session) > 1.3 * np.mean(widths_iid)


def test_one_trade_per_session_matches_the_iid_bootstrap() -> None:
    """With no clustering left to find, the two methods must agree."""
    widths_session, widths_iid = [], []
    for seed in SEEDS:
        values, sessions = clustered(seed, per_session=1)
        lo, hi, _ = session_bootstrap(sessions, mean_stat(values), 400, np.random.default_rng(seed))
        ilo, ihi = iid_ci(values, 400, np.random.default_rng(seed))
        widths_session.append(hi - lo)
        widths_iid.append(ihi - ilo)
    ratio = np.mean(widths_session) / np.mean(widths_iid)
    assert 0.85 < ratio < 1.15, f"widths should be comparable, ratio was {ratio:.2f}"


def test_deterministic_under_a_fixed_seed() -> None:
    values, sessions = clustered(1)
    a = session_bootstrap(sessions, mean_stat(values), 200, np.random.default_rng(7))
    b = session_bootstrap(sessions, mean_stat(values), 200, np.random.default_rng(7))
    assert a[0] == b[0] and a[1] == b[1]
    assert np.array_equal(a[2], b[2])
    c = session_bootstrap(sessions, mean_stat(values), 200, np.random.default_rng(8))
    assert (a[0], a[1]) != (c[0], c[1])


def test_resamples_whole_sessions_not_rows() -> None:
    """Every draw must be a union of whole sessions, so row counts come in session-sized blocks."""
    sessions = np.repeat(np.arange(5), 4)
    seen: list[int] = []

    def stat(idx: np.ndarray) -> float:
        seen.append(len(idx))
        return float(len(idx))

    session_bootstrap(sessions, stat, 50, np.random.default_rng(0))
    assert seen and all(n == 20 for n in seen), "each draw is 5 sessions x 4 rows"


def test_raises_when_most_draws_are_undefined() -> None:
    sessions = np.repeat(np.arange(6), 3)
    with pytest.raises(BootstrapError, match="too thin to bootstrap"):
        session_bootstrap(sessions, lambda idx: float("nan"), 100, np.random.default_rng(0))
    with pytest.raises(BootstrapError, match="no rows"):
        session_bootstrap(np.array([]), mean_stat(np.array([])), 10, np.random.default_rng(0))


def test_variance_share_detects_clustering() -> None:
    clustered_values, sessions = clustered(3, shock_sd=3.0)
    flat_values, _ = clustered(3, shock_sd=0.0)
    assert session_variance_share(clustered_values, sessions) > 0.5
    assert session_variance_share(flat_values, sessions) < 0.2
    identical = np.repeat(np.arange(5.0), 4)
    assert session_variance_share(identical, np.repeat(np.arange(5), 4)) == pytest.approx(1.0)


def test_effective_sample_size_shrinks_with_clustering() -> None:
    assert effective_sample_size(800, 40, 0.0) == pytest.approx(800)
    assert effective_sample_size(800, 40, 1.0) == pytest.approx(40)
    middling = effective_sample_size(800, 40, 0.25)
    assert 40 < middling < 800


def test_paired_statistics_may_narrow_and_that_is_correct() -> None:
    """Session resampling widens a level statistic but can tighten a paired comparison.

    Two forecasts are scored on the same rows. The day sets the overall level (so the mean
    outcome is clustered), while the improvement one forecast makes over the other is
    steady across days. Mixing rows from different days, as an iid bootstrap does, breaks
    that pairing and adds noise the comparison does not actually have.
    """
    rng = np.random.default_rng(3)
    shocks = rng.normal(0.0, 1.5, N_SESSIONS)  # the day decides the level
    outcome = np.concatenate([shocks[s] + rng.normal(0, 0.3, PER_SESSION) for s in range(N_SESSIONS)])
    sessions = np.repeat(np.arange(N_SESSIONS), PER_SESSION)
    good = outcome - 0.1  # a steady improvement, identical on every day
    naive = outcome - 0.0

    level = mean_stat(outcome)
    paired = mean_stat(np.abs(naive - outcome) - np.abs(good - outcome))

    lo_s, hi_s, _ = session_bootstrap(sessions, level, 600, np.random.default_rng(0))
    lo_p, hi_p, _ = session_bootstrap(sessions, paired, 600, np.random.default_rng(0))
    r = np.random.default_rng(0)
    n = len(outcome)
    iid_level = np.array([outcome[r.integers(0, n, n)].mean() for _ in range(600)])
    diff = np.abs(naive - outcome) - np.abs(good - outcome)
    r = np.random.default_rng(0)
    iid_paired = np.array([diff[r.integers(0, n, n)].mean() for _ in range(600)])

    iid_level_width = np.percentile(iid_level, 97.5) - np.percentile(iid_level, 2.5)
    iid_paired_width = np.percentile(iid_paired, 97.5) - np.percentile(iid_paired, 2.5)
    assert (hi_s - lo_s) > iid_level_width, "a clustered level statistic must get a wider interval"
    assert (hi_p - lo_p) == pytest.approx(iid_paired_width, abs=0.02), (
        "an unclustered paired statistic should not be inflated by session resampling"
    )

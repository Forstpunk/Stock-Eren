"""The thirds table: counts are hand-checkable, a planted gap is found, noise is not,
the two-halves rule only passes when the gap holds in both, and the sample floor bites."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from intraday.analysis import MIN_GAP_PCT, bust_rate_by_third, run_study, split_date_at
from intraday.config import Config
from intraday.features import FEATURE_NAMES, REPORT_ONLY_NAMES, TESTED_NAMES


def table(values: list[float], labels: list[str], dates: list[date] | None = None) -> pd.DataFrame:
    n = len(values)
    days = dates or [date(2026, 6, 1) + timedelta(days=k // 6) for k in range(n)]
    df = pd.DataFrame({f: np.linspace(0, 1, n) for f in TESTED_NAMES})
    df["rvol_open_15m"] = values
    df["label"] = labels
    df["session_date"] = days
    return df


def test_thirds_are_countable_by_hand() -> None:
    values = list(range(9))  # 0..8 -> thirds are 0-2, 3-5, 6-8
    labels = ["BUSTED", "BUSTED", "NEITHER", "BUSTED", "NEITHER", "NEITHER", "NEITHER", "SUSTAINED", "NEITHER"]
    t = bust_rate_by_third(table(values, labels), "rvol_open_15m", "all")
    assert t is not None
    assert [third.n for third in t.thirds] == [3, 3, 3]
    assert [third.busts for third in t.thirds] == [2, 1, 0]
    assert [round(third.bust_rate_pct, 1) for third in t.thirds] == [66.7, 33.3, 0.0]
    assert t.n == 9 and t.busts == 3 and round(t.base_rate_pct, 1) == 33.3
    assert round(t.gap_pct, 1) == 66.7
    assert (t.thirds[0].lower, t.thirds[0].upper) == (0.0, pytest.approx(2.67, abs=0.01))


def test_nan_rows_are_dropped_not_imputed() -> None:
    values = [0, 1, 2, np.nan, 4, 5, 6, 7, 8]
    labels = ["BUSTED"] * 9
    t = bust_rate_by_third(table(values, labels), "rvol_open_15m", "all")
    assert t is not None and t.n == 8


def test_no_table_when_thirds_are_not_defined() -> None:
    assert bust_rate_by_third(table([1.0] * 9, ["BUSTED"] * 9), "rvol_open_15m", "all") is None
    assert bust_rate_by_third(table([1.0, 2.0], ["BUSTED", "NEITHER"]), "rvol_open_15m", "all") is None


def test_split_date_is_on_a_date_boundary() -> None:
    dates = pd.Series([date(2026, 1, 1)] * 5 + [date(2026, 1, 2)] * 5 + [date(2026, 1, 3)] * 5)
    assert split_date_at(dates, 0.7) == date(2026, 1, 3)
    assert split_date_at(dates, 0.5) == date(2026, 1, 2)
    assert split_date_at(pd.Series([date(2026, 1, 1)] * 3)) is None


def planted(n: int, gap: float, seed: int, bust_rate: float = 0.25) -> pd.DataFrame:
    """Busts get systematically lower rvol_open_15m; ``gap`` sets how much lower."""
    rng = np.random.default_rng(seed)
    is_bust = rng.random(n) < bust_rate
    df = pd.DataFrame({f: rng.normal(size=n) for f in TESTED_NAMES})
    df["rvol_open_15m"] = rng.normal(size=n) - gap * is_bust
    df["label"] = np.where(is_bust, "BUSTED", np.where(rng.random(n) < 0.4, "SUSTAINED", "NEITHER"))
    df["session_date"] = [date(2026, 6, 1) + timedelta(days=k // 8) for k in range(n)]
    return df


def test_planted_gap_is_found(config: Config) -> None:
    study = run_study(planted(1200, gap=1.5, seed=1), config)
    assert study.verdict == "signal"
    assert all(f.testable for f in study.findings)
    found = {f.feature for f in study.findings if f.holds}
    assert found == {"rvol_open_15m"}
    finding = next(f for f in study.findings if f.feature == "rvol_open_15m")
    assert finding.overall.gap_pct > MIN_GAP_PCT
    assert finding.first_half is not None and finding.second_half is not None
    assert "holds in both halves" in finding.statement
    assert "rvol_open_15m" in study.statement


def test_noise_is_not_a_signal(config: Config) -> None:
    study = run_study(planted(1200, gap=0.0, seed=2), config)
    assert study.verdict == "no_signal"
    assert not any(f.holds for f in study.findings)
    assert all(f.testable for f in study.findings)
    assert "NO SIGNAL" in study.statement


def test_gap_in_one_half_only_does_not_hold(config: Config) -> None:
    df = planted(1200, gap=0.0, seed=3)
    split = split_date_at(pd.to_datetime(df["session_date"]).dt.date)
    first = pd.to_datetime(df["session_date"]).dt.date < split
    busts = df["label"] == "BUSTED"
    df.loc[first & busts, "rvol_open_15m"] -= 4.0  # huge gap, but only in the first half
    study = run_study(df, config)
    finding = next(f for f in study.findings if f.feature == "rvol_open_15m")
    assert finding.first_half is not None and finding.first_half.gap_pct > MIN_GAP_PCT
    assert not finding.holds
    assert study.verdict == "no_signal"
    assert "does not hold in both halves" in finding.statement


def test_small_second_half_is_insufficient(config: Config) -> None:
    study = run_study(planted(120, gap=2.0, seed=4), config)
    assert study.verdict == "insufficient_sample"
    assert not any(f.testable for f in study.findings)
    assert "INSUFFICIENT SAMPLE" in study.statement


def test_feature_thin_in_one_half_is_not_testable(config: Config) -> None:
    """A feature present on only a few rows of a half cannot be tested, however big its gap.

    This is the RVOL case on a short history: it needs 20 prior sessions, so it barely
    exists early on, and a gap measured on a handful of failures is not a finding.
    """
    df = planted(1200, gap=3.0, seed=7)
    split = split_date_at(pd.to_datetime(df["session_date"]).dt.date)
    early = pd.to_datetime(df["session_date"]).dt.date < split
    keep = early & (np.arange(len(df)) % 40 == 0)  # only a thin slice of the first half survives
    df.loc[early & ~keep, "rvol_open_15m"] = np.nan

    study = run_study(df, config)
    finding = next(f for f in study.findings if f.feature == "rvol_open_15m")
    assert finding.first_half is not None and finding.first_half.busts < config.min_sample
    assert not finding.testable and not finding.holds
    assert "NOT TESTABLE" in finding.statement
    assert finding.overall.gap_pct > MIN_GAP_PCT, "the raw gap is still large - that is the trap"


def test_study_is_deterministic(config: Config) -> None:
    df = planted(600, gap=1.0, seed=5)
    assert run_study(df, config) == run_study(df, config)


def test_rejects_unknown_labels_and_missing_columns(config: Config) -> None:
    df = planted(300, gap=1.0, seed=6)
    bad = df.copy()
    bad.loc[bad.index[0], "label"] = "MAYBE"
    with pytest.raises(ValueError, match="unknown labels"):
        run_study(bad, config)
    with pytest.raises(ValueError, match="lacks"):
        run_study(df.drop(columns=["bar_body_ratio"]), config)
    with pytest.raises(ValueError, match="needs"):
        run_study(df.drop(columns=["session_date"]), config)

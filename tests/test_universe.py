"""Stocks-in-play selection: eligibility judged before the session, ranking blind to the future.

The literature's claim is that the ORB edge lives in choosing which names to trade each
morning, not in the breakout trigger. That only means anything if the choice is made from
information available at the time, so these tests are mostly about what the selector is NOT
allowed to see.
"""
from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from intraday.config import Config
from intraday.lookahead import assert_blind_to_poison, assert_no_lookahead_sweep
from intraday.universe import (
    eligible_pool,
    opening_rvol,
    rank_candidates,
    select_for_session,
)
from tests.synthetic import TEST_CALENDAR, random_daily, random_history

END = date(2026, 9, 25)
TARGET = date(2026, 9, 25)


@pytest.fixture(scope="module")
def config() -> Config:
    return Config()


def daily_with(close: float, volume: int, n: int = 40, seed: int = 1) -> pd.DataFrame:
    """Daily bars ending the day before TARGET, scaled to a given close and volume."""
    df = random_daily(n, seed=seed, end_day=TARGET - timedelta(days=1))
    scale = close / float(df["close"].iloc[-1])
    for col in ("open", "high", "low", "close"):
        df[col] = df[col] * scale
    df["volume"] = volume
    return df


# ---- eligibility is decided on yesterday's numbers ------------------------------------


def test_a_liquid_volatile_name_is_eligible(config: Config) -> None:
    daily = daily_with(close=1000.0, volume=2_000_000)  # Rs 200 cr turnover
    ok, close, atr_value, turnover, reasons = eligible_pool("AAA", TARGET, daily, TEST_CALENDAR, config)
    assert ok and reasons == ()
    assert close == pytest.approx(1000.0)
    assert turnover == pytest.approx(1000.0 * 2_000_000)
    assert atr_value > 0


def test_each_filter_excludes_and_says_why(config: Config) -> None:
    cheap = daily_with(close=10.0, volume=50_000_000)
    ok, *_, reasons = eligible_pool("AAA", TARGET, cheap, TEST_CALENDAR, config)
    assert not ok and any("price" in r for r in reasons)

    illiquid = daily_with(close=1000.0, volume=1_000)  # Rs 10 lakh turnover
    ok, *_, reasons = eligible_pool("AAA", TARGET, illiquid, TEST_CALENDAR, config)
    assert not ok and any("turnover" in r for r in reasons)

    # a name whose ATR is a tiny fraction of price: flat daily bars
    flat = daily_with(close=1000.0, volume=2_000_000)
    for col in ("open", "high", "low", "close"):
        flat[col] = 1000.0
    ok, *_, reasons = eligible_pool("AAA", TARGET, flat, TEST_CALENDAR, config)
    assert not ok and any("ATR" in r for r in reasons)


def test_eligibility_never_reads_the_session_being_traded(config: Config) -> None:
    """Yesterday decides today. A bar from the target session must change nothing."""
    daily = daily_with(close=1000.0, volume=2_000_000)
    before = eligible_pool("AAA", TARGET, daily, TEST_CALENDAR, config)

    with_today = daily.copy()
    today = with_today.iloc[[-1]].copy()
    today.index = pd.DatetimeIndex([pd.Timestamp(TARGET, tz=daily.index.tz)], name="ts")
    for col in ("open", "high", "low", "close"):
        today[col] = 5.0  # a catastrophic day that would fail every filter
    after = eligible_pool("AAA", TARGET, pd.concat([with_today, today]), TEST_CALENDAR, config)
    assert before == after


def test_no_prior_session_is_not_eligible(config: Config) -> None:
    daily = daily_with(close=1000.0, volume=2_000_000)
    early = daily[daily.index.date < daily.index.date.min() + timedelta(days=0)]
    ok, *_, reasons = eligible_pool("AAA", date(2000, 1, 3), early, TEST_CALENDAR, config)
    assert not ok and reasons == ("no prior session",)


# ---- the ranking is a point-in-time decision ---------------------------------------------


def test_opening_rvol_has_no_lookahead(config: Config) -> None:
    bars = random_history(25, seed=5, end_day=END, n_bars=73)

    def fn(b: pd.DataFrame, i: int) -> float:
        return opening_rvol(b, b.index[i].date(), config)

    start = int(np.argmax(bars.index.date == bars.index[-1].date()))
    assert_no_lookahead_sweep(fn, bars, range(start + 3, len(bars)))
    assert_blind_to_poison(fn, bars)


def test_opening_rvol_is_nan_before_the_range_completes(config: Config) -> None:
    bars = random_history(25, seed=6, end_day=END, n_bars=73)
    last = bars.index[-1].date()
    partial = bars[(bars.index.date < last) | (bars.index.time <= bars.index[1].time())]
    assert math.isnan(opening_rvol(partial, last, config))


def test_ranking_takes_the_heaviest_and_caps_the_count(config: Config) -> None:
    from intraday.universe import Candidate

    def candidate(name: str, rvol: float, eligible: bool = True) -> Candidate:
        return Candidate(
            symbol=name, session_date=TARGET, prior_close=1000.0, prior_atr=20.0, atr_pct=2.0,
            prior_turnover=2e8, opening_rvol=rvol, eligible=eligible, reasons=(),
        )

    pool = [candidate(f"S{k}", rvol=float(k)) for k in range(30)]
    pool.append(candidate("INELIGIBLE", rvol=99.0, eligible=False))
    pool.append(candidate("NO_RVOL", rvol=math.nan))

    chosen = rank_candidates(pool, config.model_copy(update={"stocks_in_play": 5}))
    assert [c.symbol for c in chosen] == ["S29", "S28", "S27", "S26", "S25"]
    assert "INELIGIBLE" not in {c.symbol for c in chosen}, "filters come before the ranking"
    assert "NO_RVOL" not in {c.symbol for c in chosen}, "an unknown RVOL cannot be ranked"


def test_selection_reports_what_it_rejected(config: Config) -> None:
    bars = {name: random_history(25, seed=k, end_day=END, n_bars=73) for k, name in enumerate(("AAA", "BBB", "CCC"))}
    daily = {
        "AAA": daily_with(close=1000.0, volume=2_000_000, seed=2),
        "BBB": daily_with(close=10.0, volume=2_000_000, seed=3),  # too cheap
        "CCC": daily_with(close=1000.0, volume=100, seed=4),  # too illiquid
    }
    chosen, report = select_for_session(END, bars, daily, TEST_CALENDAR, config)

    assert report.n_considered == 3
    assert report.n_eligible == 1
    assert [c.symbol for c in chosen] == ["AAA"]
    assert set(report.exclusion_counts) == {"price", "turnover"}
    assert "3 considered" in report.describe() and "1 traded" in report.describe()


def test_selection_is_empty_rather_than_wrong_when_nothing_qualifies(config: Config) -> None:
    bars = {"AAA": random_history(25, seed=9, end_day=END, n_bars=73)}
    daily = {"AAA": daily_with(close=5.0, volume=10, seed=8)}
    chosen, report = select_for_session(END, bars, daily, TEST_CALENDAR, config)
    assert chosen == [] and report.selected == ()
    assert report.n_eligible == 0, "a day with no qualifying name trades nothing, it does not relax"

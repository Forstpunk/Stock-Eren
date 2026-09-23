"""Features: lookahead harness, no dependence on future daily rows, poisoned fixture,
and hand-computed values."""
from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from intraday.config import Config
from intraday.features import ALL_COLUMNS, FEATURE_NAMES, FeatureContext, compute_features
from intraday.indicators import atr, rvol_at_time, rvol_bar
from intraday.lookahead import assert_blind_to_poison, assert_no_lookahead_sweep
from tests.synthetic import TEST_CALENDAR, random_daily, random_history

END = date(2026, 9, 25)
N_SESSIONS = 25  # > the 20-session rvol lookback


@pytest.fixture(scope="module")
def config() -> Config:
    return Config()


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    return random_history(N_SESSIONS, seed=21, end_day=END, n_bars=73)


@pytest.fixture(scope="module")
def daily() -> pd.DataFrame:
    # ends AFTER the last session: rows on/after the session date must be ignored
    return random_daily(40, seed=22, end_day=END + timedelta(days=7))


@pytest.fixture(scope="module")
def ctx(daily: pd.DataFrame, config: Config) -> FeatureContext:
    return FeatureContext(daily=daily, calendar=TEST_CALENDAR, config=config)


def last_session_start(bars: pd.DataFrame) -> int:
    return int(np.argmax(bars.index.date == bars.index[-1].date()))


# ---- lookahead ---------------------------------------------------------------------


@pytest.mark.parametrize("direction", ["long", "short"])
def test_features_no_lookahead_under_truncation(bars: pd.DataFrame, ctx: FeatureContext, direction: str) -> None:
    start = last_session_start(bars)

    def fn(b: pd.DataFrame, i: int) -> dict[str, float]:
        return compute_features(b, i, direction, ctx)

    assert_no_lookahead_sweep(fn, bars, range(start + 3, len(bars)))


def test_features_blind_to_poisoned_future(bars: pd.DataFrame, ctx: FeatureContext) -> None:
    start = last_session_start(bars)

    def fn(b: pd.DataFrame, i: int) -> dict[str, float]:
        return compute_features(b, i, "long", ctx) if i >= start + 3 else {}

    assert_blind_to_poison(fn, bars)


def test_features_ignore_daily_rows_after_the_decision(bars: pd.DataFrame, daily: pd.DataFrame, config: Config) -> None:
    start = last_session_start(bars)
    i = start + 10
    session_date = bars.index[i].date()
    full = FeatureContext(daily=daily, calendar=TEST_CALENDAR, config=config)
    truncated = FeatureContext(daily=daily[daily.index.date < session_date], calendar=TEST_CALENDAR, config=config)
    poisoned_daily = daily.copy()
    poisoned_daily.loc[poisoned_daily.index.date >= session_date, ["open", "high", "low", "close"]] *= 3.0
    poisoned = FeatureContext(daily=poisoned_daily, calendar=TEST_CALENDAR, config=config)
    a = compute_features(bars, i, "long", full)
    assert a == compute_features(bars, i, "long", truncated) == compute_features(bars, i, "long", poisoned)
    assert set(a) == set(ALL_COLUMNS)
    assert set(FEATURE_NAMES) < set(a)


# ---- values --------------------------------------------------------------------------


def test_hand_computed_values(bars: pd.DataFrame, daily: pd.DataFrame, ctx: FeatureContext, config: Config) -> None:
    start = last_session_start(bars)
    i = start + 10
    row = bars.iloc[i]
    f = compute_features(bars, i, "long", ctx)

    assert f["bar_body_ratio"] == pytest.approx(abs(row["close"] - row["open"]) / (row["high"] - row["low"]))
    assert f["minutes_since_open"] == 50.0
    assert f["rvol_open_15m"] == pytest.approx(rvol_at_time(bars, start + 2, config.rvol_lookback_sessions))
    assert f["rvol_breakout_bar"] == pytest.approx(rvol_bar(bars, i, config.rvol_lookback_sessions))

    session_date = bars.index[i].date()
    prev_day = max(d for d in set(daily.index.date) if d < session_date)
    prev_pos = int(np.flatnonzero(daily.index.date == prev_day)[0])
    atr_v = atr(daily, prev_pos, config.atr_period)
    or_bars = bars.iloc[start : start + 3]
    assert f["or_width_atr"] == pytest.approx((or_bars["high"].max() - or_bars["low"].min()) / atr_v)


def test_features_do_not_depend_on_direction(bars: pd.DataFrame, ctx: FeatureContext) -> None:
    """None of the three mechanisms is directional, so long and short agree."""
    i = last_session_start(bars) + 10
    assert compute_features(bars, i, "long", ctx) == compute_features(bars, i, "short", ctx)


def test_nan_when_context_is_missing(bars: pd.DataFrame, daily: pd.DataFrame, config: Config) -> None:
    start = last_session_start(bars)
    i = start + 10
    prev_day = max(d for d in set(daily.index.date) if d < bars.index[i].date())
    no_prev = FeatureContext(daily=daily[daily.index.date != prev_day], calendar=TEST_CALENDAR, config=config)
    f = compute_features(bars, i, "long", no_prev)
    assert math.isnan(f["or_width_atr"])
    for name in ("bar_body_ratio", "minutes_since_open", "rvol_open_15m", "rvol_breakout_bar"):
        assert not math.isnan(f[name]), name


def test_rvol_nan_without_enough_sessions(daily: pd.DataFrame, config: Config) -> None:
    short_hist = random_history(5, seed=21, end_day=END, n_bars=73)
    ctx = FeatureContext(daily=daily, calendar=TEST_CALENDAR, config=config)
    f = compute_features(short_hist, last_session_start(short_hist) + 10, "long", ctx)
    assert math.isnan(f["rvol_open_15m"]) and math.isnan(f["rvol_breakout_bar"])
    assert not math.isnan(f["or_width_atr"])

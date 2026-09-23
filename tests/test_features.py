"""Features: lookahead harness, no dependence on future daily/index rows, poisoned
fixture, hand-computed values, and the direction-signing rule.

Every feature here is pre-registered in RESEARCH.md with an expected direction. These
tests check the arithmetic and the blindness, never the direction - that is what the
study measures.
"""
from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from intraday.config import Config
from intraday.expiry_calendar import ExpiryCalendar
from intraday.features import (
    ALL_COLUMNS,
    CONTEXT_NAMES,
    FEATURE_NAMES,
    REPORT_ONLY_NAMES,
    TESTED_NAMES,
    FeatureContext,
    compute_features,
)
from intraday.indicators import atr, index_breakout_state, rvol_at_time, rvol_bar
from intraday.lookahead import assert_blind_to_poison, assert_no_lookahead_sweep, poison_future
from tests.synthetic import TEST_CALENDAR, random_daily, random_history

END = date(2026, 9, 25)
N_SESSIONS = 25  # > the 14-session rvol lookback


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
def index_bars() -> pd.DataFrame:
    """The index on the same grid; full 75 bars, like the real NIFTY50 frame."""
    return random_history(N_SESSIONS, seed=23, end_day=END)


@pytest.fixture(scope="module")
def expiries() -> ExpiryCalendar:
    return ExpiryCalendar({date(2026, 9, 24)})


@pytest.fixture(scope="module")
def ctx(daily: pd.DataFrame, index_bars: pd.DataFrame, expiries: ExpiryCalendar, config: Config) -> FeatureContext:
    return FeatureContext(
        daily=daily, index_bars=index_bars, calendar=TEST_CALENDAR, expiries=expiries, config=config
    )


def last_session_start(bars: pd.DataFrame) -> int:
    return int(np.argmax(bars.index.date == bars.index[-1].date()))


# ---- the name groups are the contract the study relies on ----------------------------


def test_name_groups_are_disjoint_and_complete() -> None:
    assert set(FEATURE_NAMES) & set(REPORT_ONLY_NAMES) == set()
    assert set(FEATURE_NAMES) & set(CONTEXT_NAMES) == set()
    assert set(REPORT_ONLY_NAMES) & set(CONTEXT_NAMES) == set()
    assert TESTED_NAMES == FEATURE_NAMES + REPORT_ONLY_NAMES
    assert set(ALL_COLUMNS) == set(TESTED_NAMES) | set(CONTEXT_NAMES)


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


def test_index_breakout_state_has_no_lookahead(index_bars: pd.DataFrame, config: Config) -> None:
    """The index features must be as blind as the stock ones: later index bars cannot
    change what the index had done by the decision stamp."""
    start = last_session_start(index_bars)

    def fn(b: pd.DataFrame, i: int) -> float:
        return float(index_breakout_state(b, b.index[i], config))

    assert_no_lookahead_sweep(fn, index_bars, range(start + 3, len(index_bars)))
    assert_blind_to_poison(fn, index_bars)


def test_features_ignore_index_and_daily_rows_after_the_decision(
    bars: pd.DataFrame, daily: pd.DataFrame, index_bars: pd.DataFrame,
    expiries: ExpiryCalendar, config: Config,
) -> None:
    start = last_session_start(bars)
    i = start + 10
    ts = bars.index[i]
    session_date = ts.date()

    def ctx_of(d: pd.DataFrame, ix: pd.DataFrame) -> FeatureContext:
        return FeatureContext(
            daily=d, index_bars=ix, calendar=TEST_CALENDAR, expiries=expiries, config=config
        )

    full = ctx_of(daily, index_bars)
    truncated = ctx_of(daily[daily.index.date < session_date], index_bars[index_bars.index <= ts])
    poisoned_daily = daily.copy()
    poisoned_daily.loc[poisoned_daily.index.date >= session_date, ["open", "high", "low", "close"]] *= 3.0
    poisoned = ctx_of(poisoned_daily, poison_future(index_bars))

    a = compute_features(bars, i, "long", full)
    assert a == compute_features(bars, i, "long", truncated)
    assert a == compute_features(bars, i, "long", poisoned)
    assert set(a) == set(ALL_COLUMNS)


# ---- values --------------------------------------------------------------------------


def test_hand_computed_values(
    bars: pd.DataFrame, daily: pd.DataFrame, index_bars: pd.DataFrame, ctx: FeatureContext, config: Config
) -> None:
    start = last_session_start(bars)
    i = start + 10
    row = bars.iloc[i]
    ts = bars.index[i]
    f = compute_features(bars, i, "long", ctx)

    assert f["bar_body_ratio"] == pytest.approx(abs(row["close"] - row["open"]) / (row["high"] - row["low"]))
    assert f["minutes_since_open"] == 50.0
    assert f["day_of_week"] == float(ts.date().weekday())
    assert f["rvol_open_15m"] == pytest.approx(rvol_at_time(bars, start + 2, config.rvol_lookback_sessions))
    assert f["rvol_breakout_bar"] == pytest.approx(rvol_bar(bars, i, config.rvol_lookback_sessions))

    prev_day = max(d for d in set(daily.index.date) if d < ts.date())
    prev_pos = int(np.flatnonzero(daily.index.date == prev_day)[0])
    prev = daily.iloc[prev_pos]
    atr_v = atr(daily, prev_pos, config.atr_period)

    or_bars = bars.iloc[start : start + 3]
    or_high, or_low = or_bars["high"].max(), or_bars["low"].min()
    assert f["or_width_atr"] == pytest.approx((or_high - or_low) / atr_v)
    assert f["breakout_depth_atr"] == pytest.approx(abs(row["close"] - or_high) / atr_v)
    assert f["prior_day_return_atr_signed"] == pytest.approx((prev["close"] - prev["open"]) / atr_v)

    # relative strength: both legs measured from their own session opens
    stock_move = row["close"] / bars["open"].iloc[start] - 1
    ix = index_bars.loc[ts]
    index_open = index_bars[index_bars.index.date == ts.date()]["open"].iloc[0]
    index_move = ix["close"] / index_open - 1
    assert f["rel_strength_vs_index"] == pytest.approx((stock_move - index_move) * 100)

    # gap in ATR units, derived from the intraday previous close only
    prev_close_intraday = bars["close"].iloc[start - 1]
    assert f["gap_atr_signed"] == pytest.approx((bars["open"].iloc[start] - prev_close_intraday) / atr_v)

    assert f["index_or_agrees"] in (0.0, 1.0)


def test_index_or_agrees_matches_the_index_state(bars: pd.DataFrame, ctx: FeatureContext, config: Config) -> None:
    i = last_session_start(bars) + 10
    ts = bars.index[i]
    state = index_breakout_state(ctx.index_bars, ts, config)
    long = compute_features(bars, i, "long", ctx)["index_or_agrees"]
    short = compute_features(bars, i, "short", ctx)["index_or_agrees"]
    assert long == (1.0 if state == 1 else 0.0)
    assert short == (1.0 if state == -1 else 0.0)


def test_signed_features_flip_with_direction(bars: pd.DataFrame, ctx: FeatureContext) -> None:
    i = last_session_start(bars) + 10
    long = compute_features(bars, i, "long", ctx)
    short = compute_features(bars, i, "short", ctx)
    for name in ("rel_strength_vs_index", "gap_atr_signed", "prior_day_return_atr_signed"):
        assert short[name] == pytest.approx(-long[name]), name
    for name in ("rvol_open_15m", "rvol_breakout_bar", "bar_body_ratio", "or_width_atr",
                 "minutes_since_open", "day_of_week", "is_expiry_day"):
        same = short[name] == long[name] or (math.isnan(short[name]) and math.isnan(long[name]))
        assert same, name
    # depth is measured from whichever boundary was broken, so it is direction-specific
    assert short["breakout_depth_atr"] != long["breakout_depth_atr"]


def test_expiry_flag_is_nan_outside_the_covered_range(
    bars: pd.DataFrame, daily: pd.DataFrame, index_bars: pd.DataFrame, config: Config
) -> None:
    i = last_session_start(bars) + 10
    session_date = bars.index[i].date()

    covering = FeatureContext(
        daily=daily, index_bars=index_bars, calendar=TEST_CALENDAR,
        expiries=ExpiryCalendar({session_date, session_date - timedelta(days=30)}), config=config,
    )
    assert compute_features(bars, i, "long", covering)["is_expiry_day"] == 1.0

    elsewhere = FeatureContext(
        daily=daily, index_bars=index_bars, calendar=TEST_CALENDAR,
        expiries=ExpiryCalendar({date(2020, 1, 30), date(2020, 2, 27)}), config=config,
    )
    assert math.isnan(compute_features(bars, i, "long", elsewhere)["is_expiry_day"])

    empty = FeatureContext(
        daily=daily, index_bars=index_bars, calendar=TEST_CALENDAR,
        expiries=ExpiryCalendar(set()), config=config,
    )
    assert math.isnan(compute_features(bars, i, "long", empty)["is_expiry_day"])


def test_nan_when_context_is_missing(
    bars: pd.DataFrame, daily: pd.DataFrame, index_bars: pd.DataFrame,
    expiries: ExpiryCalendar, config: Config,
) -> None:
    start = last_session_start(bars)
    i = start + 10
    ts = bars.index[i]
    prev_day = max(d for d in set(daily.index.date) if d < ts.date())

    no_daily = FeatureContext(
        daily=daily[daily.index.date != prev_day], index_bars=index_bars,
        calendar=TEST_CALENDAR, expiries=expiries, config=config,
    )
    f = compute_features(bars, i, "long", no_daily)
    for name in ("or_width_atr", "breakout_depth_atr", "gap_atr_signed", "prior_day_return_atr_signed"):
        assert math.isnan(f[name]), name
    assert not math.isnan(f["rel_strength_vs_index"]), "the index leg does not need the daily frame"

    no_index = FeatureContext(
        daily=daily, index_bars=index_bars.iloc[0:0], calendar=TEST_CALENDAR,
        expiries=expiries, config=config,
    )
    g = compute_features(bars, i, "long", no_index)
    assert math.isnan(g["rel_strength_vs_index"]) and math.isnan(g["index_or_agrees"])
    assert not math.isnan(g["bar_body_ratio"])


def test_index_lookup_is_exact_never_nearest(
    bars: pd.DataFrame, daily: pd.DataFrame, index_bars: pd.DataFrame,
    expiries: ExpiryCalendar, config: Config,
) -> None:
    """A missing index bar must give NaN, not the neighbouring minute's bar."""
    i = last_session_start(bars) + 10
    ts = bars.index[i]
    holed = index_bars[index_bars.index != ts]
    ctx = FeatureContext(
        daily=daily, index_bars=holed, calendar=TEST_CALENDAR, expiries=expiries, config=config
    )
    assert math.isnan(compute_features(bars, i, "long", ctx)["rel_strength_vs_index"])


def test_rvol_nan_without_enough_sessions(
    daily: pd.DataFrame, index_bars: pd.DataFrame, expiries: ExpiryCalendar, config: Config
) -> None:
    short_hist = random_history(5, seed=21, end_day=END, n_bars=73)
    ctx = FeatureContext(
        daily=daily, index_bars=index_bars, calendar=TEST_CALENDAR, expiries=expiries, config=config
    )
    f = compute_features(short_hist, last_session_start(short_hist) + 10, "long", ctx)
    assert math.isnan(f["rvol_open_15m"]) and math.isnan(f["rvol_breakout_bar"])
    assert not math.isnan(f["or_width_atr"])

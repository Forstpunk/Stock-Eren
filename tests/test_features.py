"""Features: lookahead harness (intraday), no dependence on future daily/index rows,
poisoned fixture, and hand-computed values."""
from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from intraday.config import Config
from intraday.features import FEATURE_NAMES, FeatureContext, compute_features
from intraday.indicators import atr, session_sigma, session_vwap
from intraday.lookahead import assert_blind_to_poison, assert_no_lookahead_sweep, poison_future
from tests.synthetic import TEST_CALENDAR, random_daily, random_history

END = date(2026, 9, 25)
N_SESSIONS = 25  # > rvol lookback of 20


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    return random_history(N_SESSIONS, seed=21, end_day=END, n_bars=73)


@pytest.fixture(scope="module")
def daily() -> pd.DataFrame:
    # 40 days ending AFTER the last session: rows on/after the session date must be ignored
    return random_daily(40, seed=22, end_day=END + timedelta(days=7))


@pytest.fixture(scope="module")
def index_bars() -> pd.DataFrame:
    return random_history(N_SESSIONS, seed=23, end_day=END)  # full 75 bars


@pytest.fixture(scope="module")
def ctx(daily: pd.DataFrame, index_bars: pd.DataFrame, config: Config) -> FeatureContext:
    return FeatureContext(daily=daily, index_bars=index_bars, calendar=TEST_CALENDAR, config=config)


@pytest.fixture(scope="module")
def config() -> Config:
    return Config()


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


def test_features_ignore_daily_and_index_rows_after_decision(
    bars: pd.DataFrame, daily: pd.DataFrame, index_bars: pd.DataFrame, config: Config
) -> None:
    start = last_session_start(bars)
    i = start + 10
    ts = bars.index[i]
    session_date = ts.date()
    full = FeatureContext(daily=daily, index_bars=index_bars, calendar=TEST_CALENDAR, config=config)
    truncated = FeatureContext(
        daily=daily[daily.index.date < session_date],
        index_bars=index_bars[index_bars.index <= ts],
        calendar=TEST_CALENDAR,
        config=config,
    )
    poisoned_daily = daily.copy()
    future = poisoned_daily.index.date >= session_date
    poisoned_daily.loc[future, ["open", "high", "low", "close"]] *= 3.0
    poisoned = FeatureContext(
        daily=poisoned_daily, index_bars=poison_future(index_bars), calendar=TEST_CALENDAR, config=config
    )
    a = compute_features(bars, i, "long", full)
    b = compute_features(bars, i, "long", truncated)
    c = compute_features(bars, i, "long", poisoned)
    assert a == b == c
    assert set(a) == set(FEATURE_NAMES)


# ---- values --------------------------------------------------------------------------


def test_hand_computed_values(bars: pd.DataFrame, daily: pd.DataFrame, index_bars: pd.DataFrame, ctx: FeatureContext, config: Config) -> None:
    start = last_session_start(bars)
    i = start + 10
    row = bars.iloc[i]
    f = compute_features(bars, i, "long", ctx)

    or_bars = bars.iloc[start : start + 3]
    or_high, or_low = or_bars["high"].max(), or_bars["low"].min()
    assert f["prior_touches"] == float((bars["high"].iloc[start + 3 : i] >= or_high).sum())
    assert f["bar_body_ratio"] == pytest.approx(abs(row["close"] - row["open"]) / (row["high"] - row["low"]))
    assert f["minutes_since_open"] == 50.0

    session_date = bars.index[i].date()
    prev_day = max(d for d in set(daily.index.date) if d < session_date)
    assert TEST_CALENDAR.is_trading_day(prev_day)
    prev_pos = int(np.flatnonzero(daily.index.date == prev_day)[0])
    prev = daily.iloc[prev_pos]
    atr_v = atr(daily, prev_pos, config.atr_period)
    assert f["atr_pct"] == pytest.approx(atr_v / prev["close"] * 100)
    assert f["or_width_atr"] == pytest.approx((or_high - or_low) / atr_v)
    today_open = bars["open"].iloc[start]
    assert f["open_position_in_prior_range"] == pytest.approx((today_open - prev["low"]) / (prev["high"] - prev["low"]))
    prev_close_intraday = bars["close"].iloc[start - 1]
    assert f["gap_atr"] == pytest.approx((today_open - prev_close_intraday) / atr_v)

    ib = index_bars.loc[bars.index[i]]
    assert f["index_agreement"] == pytest.approx((ib["close"] / ib["open"] - 1) * 1e4)

    vwap, sigma = session_vwap(bars, i), session_sigma(bars, i)
    assert f["vwap_distance_sigma"] == pytest.approx(((row["close"] - vwap) / vwap) / sigma)
    assert not math.isnan(f["rvol_breakout_bar"]) and not math.isnan(f["rvol_open_15m"])


def test_signed_features_flip_with_direction(bars: pd.DataFrame, ctx: FeatureContext) -> None:
    i = last_session_start(bars) + 10
    long = compute_features(bars, i, "long", ctx)
    short = compute_features(bars, i, "short", ctx)
    for name in ("vwap_distance_sigma", "index_agreement", "gap_atr"):
        assert short[name] == pytest.approx(-long[name])
    assert short["open_position_in_prior_range"] == pytest.approx(1.0 - long["open_position_in_prior_range"])
    for name in ("or_width_atr", "bar_body_ratio", "minutes_since_open", "atr_pct", "rvol_breakout_bar", "rvol_open_15m"):
        assert short[name] == long[name]


def test_nan_when_context_is_missing(bars: pd.DataFrame, daily: pd.DataFrame, index_bars: pd.DataFrame, config: Config) -> None:
    start = last_session_start(bars)
    i = start + 10
    session_date = bars.index[i].date()
    prev_day = max(d for d in set(daily.index.date) if d < session_date)
    no_prev = FeatureContext(
        daily=daily[daily.index.date != prev_day],
        index_bars=index_bars[index_bars.index != bars.index[i]],
        calendar=TEST_CALENDAR,
        config=config,
    )
    f = compute_features(bars, i, "long", no_prev)
    for name in ("atr_pct", "or_width_atr", "gap_atr", "open_position_in_prior_range", "index_agreement"):
        assert math.isnan(f[name]), name
    for name in ("prior_touches", "bar_body_ratio", "minutes_since_open", "vwap_distance_sigma", "rvol_breakout_bar"):
        assert not math.isnan(f[name]), name


def test_rvol_nan_without_enough_sessions(daily: pd.DataFrame, index_bars: pd.DataFrame, config: Config) -> None:
    short_hist = random_history(5, seed=21, end_day=END, n_bars=73)
    ctx = FeatureContext(daily=daily, index_bars=index_bars, calendar=TEST_CALENDAR, config=config)
    f = compute_features(short_hist, last_session_start(short_hist) + 10, "long", ctx)
    assert math.isnan(f["rvol_breakout_bar"]) and math.isnan(f["rvol_open_15m"])
    assert not math.isnan(f["atr_pct"])

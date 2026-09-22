"""Every indicator: (1) no lookahead under truncation, (2) blind to the poisoned future,
(3) value pinned against an independent hand computation."""
from __future__ import annotations

import math
from datetime import date, time

import numpy as np
import pandas as pd
import pytest

from intraday.config import Config
from intraday.indicators import (
    atr,
    gap_pct,
    opening_range,
    rvol_at_time,
    session_sigma,
    session_vwap,
)
from intraday.lookahead import assert_blind_to_poison, assert_no_lookahead_sweep
from tests.synthetic import TEST_CALENDAR, random_daily, random_history

LOOKBACK = 5
ATR_PERIOD = 14


@pytest.fixture(scope="module")
def history() -> pd.DataFrame:
    """8 sessions x 73 bars (the TAIL_COLLAPSED shape), chained."""
    return random_history(8, seed=3, n_bars=73)


@pytest.fixture(scope="module")
def daily() -> pd.DataFrame:
    return random_daily(40, seed=5)


def _or(bars: pd.DataFrame, i: int) -> dict[str, float]:
    try:
        r = opening_range(bars, i, Config())
    except ValueError:
        return {"high": math.nan, "low": math.nan, "width": math.nan, "end_index": math.nan}
    return r.model_dump()


def _rvol(bars: pd.DataFrame, i: int) -> float:
    return rvol_at_time(bars, i, LOOKBACK)


def _gap(bars: pd.DataFrame, i: int) -> float:
    return gap_pct(bars, i, TEST_CALENDAR)


def _atr(daily: pd.DataFrame, i: int) -> float:
    return atr(daily, i, ATR_PERIOD)


INTRADAY = [session_vwap, session_sigma, _or, _rvol, _gap]


# ---- lookahead ------------------------------------------------------------------------


@pytest.mark.parametrize("fn", INTRADAY, ids=lambda f: f.__name__)
def test_no_lookahead_under_truncation(fn, history: pd.DataFrame) -> None:  # type: ignore[no-untyped-def]
    last_session_start = int(np.argmax(history.index.date == history.index[-1].date()))
    assert_no_lookahead_sweep(fn, history, range(last_session_start - 5, len(history)))


@pytest.mark.parametrize("fn", INTRADAY, ids=lambda f: f.__name__)
def test_blind_to_poisoned_future(fn, history: pd.DataFrame) -> None:  # type: ignore[no-untyped-def]
    assert_blind_to_poison(fn, history)


def test_atr_no_lookahead(daily: pd.DataFrame) -> None:
    assert_no_lookahead_sweep(_atr, daily)


# ---- hand-computed values --------------------------------------------------------------


def test_vwap_matches_loop(history: pd.DataFrame) -> None:
    start = int(np.argmax(history.index.date == history.index[-1].date()))
    i = start + 10
    num = 0.0
    den = 0.0
    for k in range(start, i + 1):
        row = history.iloc[k]
        tp = (row["high"] + row["low"] + row["close"]) / 3
        num += tp * row["volume"]
        den += row["volume"]
    assert session_vwap(history, i) == pytest.approx(num / den, rel=1e-12)


def test_vwap_resets_daily_and_is_nan_without_volume(history: pd.DataFrame) -> None:
    start = int(np.argmax(history.index.date == history.index[-1].date()))
    first = history.iloc[start]
    assert session_vwap(history, start) == pytest.approx((first["high"] + first["low"] + first["close"]) / 3)
    zeroed = history.copy()
    zeroed.iloc[start, zeroed.columns.get_loc("volume")] = 0
    assert math.isnan(session_vwap(zeroed, start))
    assert not math.isnan(session_vwap(zeroed, start + 1))


def test_sigma_matches_numpy(history: pd.DataFrame) -> None:
    start = int(np.argmax(history.index.date == history.index[-1].date()))
    assert math.isnan(session_sigma(history, start))
    assert math.isnan(session_sigma(history, start + 1))
    i = start + 12
    close = history["close"].iloc[start : i + 1].to_numpy()
    expected = np.std(close[1:] / close[:-1] - 1, ddof=1)
    assert session_sigma(history, i) == pytest.approx(expected)


def test_opening_range_values_and_guards(history: pd.DataFrame, config: Config) -> None:
    start = int(np.argmax(history.index.date == history.index[-1].date()))
    with pytest.raises(ValueError, match="incomplete"):
        opening_range(history, start + 1, config)
    r = opening_range(history, start + 2, config)
    first3 = history.iloc[start : start + 3]
    assert r.high == first3["high"].max() and r.low == first3["low"].min()
    assert r.width == pytest.approx(r.high - r.low)
    assert r.end_index == start + 2
    assert opening_range(history, start + 40, config) == r
    with pytest.raises(ValueError, match="opening range undefined"):
        opening_range(history.iloc[start + 1 :], 5, config)


def test_rvol_time_matched(history: pd.DataFrame) -> None:
    dates = sorted(set(history.index.date))
    last = dates[-1]
    start = int(np.argmax(history.index.date == last))
    i = start + 6  # 09:45 bar
    t = history.index[i].time()
    assert t == time(9, 45)
    today = history["volume"].iloc[start : i + 1].sum()
    prior = dates[-LOOKBACK - 1 : -1]
    cum = []
    for d in prior:
        s = history[(history.index.date == d) & np.array([ts.time() <= t for ts in history.index])]
        cum.append(s["volume"].sum())
    assert rvol_at_time(history, i, LOOKBACK) == pytest.approx(today / np.mean(cum))


def test_rvol_nan_without_enough_history(history: pd.DataFrame) -> None:
    dates = sorted(set(history.index.date))
    start_day3 = int(np.argmax(history.index.date == dates[2]))
    assert math.isnan(rvol_at_time(history, start_day3 + 5, LOOKBACK))  # only 2 prior sessions
    assert math.isnan(rvol_at_time(history, 5, LOOKBACK))


def test_gap_pct_uses_previous_trading_day_only(history: pd.DataFrame) -> None:
    dates = sorted(set(history.index.date))
    last = dates[-1]
    start = int(np.argmax(history.index.date == last))
    prev = history[history.index.date == dates[-2]]
    expected = (history["open"].iloc[start] - prev["close"].iloc[-1]) / prev["close"].iloc[-1] * 100
    assert gap_pct(history, start, TEST_CALENDAR) == pytest.approx(expected)
    assert gap_pct(history, start + 30, TEST_CALENDAR) == pytest.approx(expected)
    # remove the previous session: the gap becomes undefined, not "vs two days ago"
    holed = history[history.index.date != dates[-2]]
    hs = int(np.argmax(holed.index.date == last))
    assert math.isnan(gap_pct(holed, hs, TEST_CALENDAR))
    assert math.isnan(gap_pct(history, 3, TEST_CALENDAR))


def test_gap_pct_skips_weekend_and_holiday() -> None:
    hist = random_history(3, seed=9, end_day=date(2026, 9, 15))  # 10th, 11th, 15th (14th holiday, weekend)
    dates = sorted(set(hist.index.date))
    assert dates == [date(2026, 9, 10), date(2026, 9, 11), date(2026, 9, 15)]
    start = int(np.argmax(hist.index.date == dates[-1]))
    assert not math.isnan(gap_pct(hist, start, TEST_CALENDAR))


def test_atr_matches_wilder_by_hand(daily: pd.DataFrame) -> None:
    assert math.isnan(atr(daily, ATR_PERIOD - 1, ATR_PERIOD))
    h, l, c = (daily[k].to_numpy() for k in ("high", "low", "close"))
    trs = [max(h[k] - l[k], abs(h[k] - c[k - 1]), abs(l[k] - c[k - 1])) for k in range(1, len(daily))]
    value = sum(trs[:ATR_PERIOD]) / ATR_PERIOD
    assert atr(daily, ATR_PERIOD, ATR_PERIOD) == pytest.approx(value)
    for k in range(ATR_PERIOD, len(trs)):
        value = (value * (ATR_PERIOD - 1) + trs[k]) / ATR_PERIOD
    assert atr(daily, len(daily) - 1, ATR_PERIOD) == pytest.approx(value)


def test_indicators_reject_bad_frames(history: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="sorted"):
        session_vwap(history.iloc[::-1], 0)
    with pytest.raises(IndexError):
        session_vwap(history, len(history))
    naive = history.copy()
    naive.index = naive.index.tz_localize(None)
    with pytest.raises(ValueError, match="tz-aware"):
        session_sigma(naive, 3)

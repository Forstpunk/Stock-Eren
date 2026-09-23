"""Crafted sessions for each label, the one-per-direction rule, the 15:15 cutoff, and the
lookahead harness on breakout detection."""
from __future__ import annotations

from datetime import date, datetime, time

import pandas as pd
import pytest

from intraday.config import IST, Config
from intraday.labelling import Label, detect_breakout_at, label_breakouts, label_session
from intraday.lookahead import assert_blind_to_poison, assert_no_lookahead_sweep
from tests.synthetic import TEST_CALENDAR, random_daily, random_history, random_session
from tests.test_validate import make_session

DAY = date(2026, 9, 7)
ATR = 4.0  # with OR width 2: sustain at 0.5 ATR = 2.0 (1.0x width), bust below 0.25 ATR = 1.0 (0.5x width)


def flat_session(n: int = 73) -> pd.DataFrame:
    """OR = [99, 101] (width 2) over the first three bars, then flat inside the range."""
    df = make_session(day=DAY, n=n)
    df["open"] = 100.0
    df["close"] = 100.0
    df["high"] = 100.2
    df["low"] = 99.8
    df.iloc[0, df.columns.get_loc("high")] = 101.0
    df.iloc[1, df.columns.get_loc("low")] = 99.0
    return df


def set_bar(df: pd.DataFrame, k: int, o: float, h: float, l: float, c: float) -> None:
    df.iloc[k, [df.columns.get_loc(x) for x in ("open", "high", "low", "close")]] = [o, h, l, c]


def test_no_breakout_when_closes_stay_inside(config: Config) -> None:
    assert label_session(flat_session(), "X", ATR, config) == []


def test_sustained_long(config: Config) -> None:
    df = flat_session()
    set_bar(df, 10, 100.5, 101.6, 100.4, 101.5)  # close 101.5 > OR high 101 -> long breakout
    set_bar(df, 14, 101.5, 103.2, 101.4, 103.0)  # high >= 101 + 2.0 -> sustained
    (e,) = label_session(df, "X", ATR, config)
    assert e.direction == "long" and e.label is Label.SUSTAINED
    assert e.breakout_index == 10 and e.resolved_index == 14
    assert e.breakout_time.time() == time(10, 5) and e.minutes_since_open == 50
    assert e.or_high == 101.0 and e.or_low == 99.0 and e.or_width == 2.0 and e.atr == ATR
    assert e.max_extension_atr == pytest.approx((103.2 - 101.0) / ATR)


def test_busted_long(config: Config) -> None:
    df = flat_session()
    set_bar(df, 10, 100.5, 101.7, 100.4, 101.5)  # long breakout; extension 0.7 < 0.5x2 = 1.0
    set_bar(df, 12, 101.0, 101.2, 100.0, 100.3)  # back inside
    set_bar(df, 15, 99.5, 99.6, 98.5, 98.7)  # close through OR low before 15:15 -> busted
    e, opposite = label_session(df, "X", ATR, config)  # the bust bar is also the first short breakout
    assert e.label is Label.BUSTED and e.resolved_index == 15
    assert opposite.direction == "short" and opposite.breakout_index == 15


def test_busted_short(config: Config) -> None:
    df = flat_session()
    set_bar(df, 8, 99.4, 99.5, 98.5, 98.6)  # close < 99 -> short breakout, extension 0.5 < 1.0
    set_bar(df, 20, 100.5, 101.6, 100.4, 101.3)  # close through OR high -> busted
    e, _ = label_session(df, "X", ATR, config)
    assert e.direction == "short" and e.label is Label.BUSTED and e.resolved_index == 20


def test_touch_is_not_a_breakout(config: Config) -> None:
    df = flat_session()
    set_bar(df, 10, 100.5, 102.5, 100.4, 100.9)  # high beyond, close inside
    assert label_session(df, "X", ATR, config) == []


def test_neither_when_extended_past_half_then_reversed(config: Config) -> None:
    df = flat_session()
    set_bar(df, 10, 100.5, 102.2, 100.4, 101.5)  # long breakout, extension 1.2 >= 1.0 (0.5x width)
    set_bar(df, 15, 99.5, 99.6, 98.5, 98.7)  # reversal through OR low, but too extended to be a bust
    e, _ = label_session(df, "X", ATR, config)
    assert e.label is Label.NEITHER and e.resolved_index is None
    assert e.max_extension_atr == pytest.approx(1.2 / ATR)


def test_neither_when_bust_would_be_after_cutoff(config: Config) -> None:
    df = flat_session(75)
    set_bar(df, 60, 100.5, 101.6, 100.4, 101.5)  # breakout at 14:15
    set_bar(df, 72, 99.5, 99.6, 98.5, 98.7)  # 15:15 bar: not before cutoff
    e, _ = label_session(df, "X", ATR, config)
    assert e.label is Label.NEITHER
    df2 = df.copy()
    set_bar(df2, 71, 99.5, 99.6, 98.5, 98.7)  # 15:10 bar: before cutoff
    e2, _ = label_session(df2, "X", ATR, config)
    assert e2.label is Label.BUSTED


def test_sustain_beats_bust_within_the_same_bar(config: Config) -> None:
    df = flat_session()
    set_bar(df, 10, 100.5, 101.6, 100.4, 101.5)
    set_bar(df, 12, 101.5, 103.5, 98.0, 98.5)  # hits +1.0R intrabar and closes through OR low
    e, _ = label_session(df, "X", ATR, config)
    assert e.label is Label.SUSTAINED


def test_one_breakout_per_direction_first_wins(config: Config) -> None:
    df = flat_session()
    set_bar(df, 10, 100.5, 101.6, 100.4, 101.5)  # first long
    set_bar(df, 12, 100.0, 100.2, 99.9, 100.0)  # back inside
    set_bar(df, 20, 100.5, 101.9, 100.4, 101.8)  # second long close: ignored
    set_bar(df, 30, 99.5, 99.6, 98.5, 98.7)  # first short (also busts the long)
    events = label_session(df, "X", ATR, config)
    assert [(e.direction, e.breakout_index) for e in events] == [("long", 10), ("short", 30)]
    assert events[0].label is Label.BUSTED and events[0].resolved_index == 30


def test_breakout_on_bar_right_after_range(config: Config) -> None:
    df = flat_session()
    set_bar(df, 3, 100.5, 101.6, 100.4, 101.5)
    (e,) = label_session(df, "X", ATR, config)
    assert e.breakout_index == 3 and e.minutes_since_open == 15


def test_zero_width_range_raises(config: Config) -> None:
    df = flat_session()
    df.iloc[0, df.columns.get_loc("high")] = 100.2
    df.iloc[1, df.columns.get_loc("low")] = 99.8
    df.iloc[:3, df.columns.get_loc("high")] = 100.0
    df.iloc[:3, df.columns.get_loc("low")] = 100.0
    set_bar(df, 10, 100.5, 101.6, 100.4, 101.5)
    with pytest.raises(ValueError, match="width"):
        label_session(df, "X", ATR, config)


def test_missing_atr_raises_and_is_skipped_at_frame_level(config: Config) -> None:
    df = flat_session()
    set_bar(df, 10, 100.5, 101.6, 100.4, 101.5)
    with pytest.raises(ValueError, match="no prior-day ATR"):
        label_session(df, "X", float("nan"), config)
    hist = random_history(3, seed=11, n_bars=73)
    daily = random_daily(30, seed=12, end_day=date(2026, 9, 25))
    first_day = sorted(set(hist.index.date))[0]
    prev = max(d for d in set(daily.index.date) if d < first_day)
    holed = daily[daily.index.date != prev]  # first session loses its prior-day ATR
    events, skipped = label_breakouts(hist, "X", holed, TEST_CALENDAR, config)
    assert skipped == [first_day]
    assert all(e.session_date != first_day for e in events)


def test_multi_session_offsets(config: Config) -> None:
    hist = random_history(4, seed=11, n_bars=73)
    daily = random_daily(30, seed=12, end_day=date(2026, 9, 25))
    events, skipped = label_breakouts(hist, "X", daily, TEST_CALENDAR, config)
    assert events and skipped == [], "random walks should produce some breakouts"
    for e in events:
        assert hist.index[e.breakout_index].date() == e.session_date
        assert hist.index[e.breakout_index].to_pydatetime() == e.breakout_time
        assert float(hist["close"].iloc[e.breakout_index]) == e.breakout_close
    per_session_dir = {(e.session_date, e.direction) for e in events}
    assert len(per_session_dir) == len(events)


def test_partial_start_session_raises(config: Config) -> None:
    df = random_session(seed=1, day=DAY).iloc[2:]
    with pytest.raises(ValueError, match="opening range undefined"):
        label_session(df, "X", ATR, config)


# ---- lookahead on detection --------------------------------------------------------


def _detect(bars: pd.DataFrame, i: int) -> str:
    return detect_breakout_at(bars, i, Config()) or "none"


def test_detection_has_no_lookahead(session: pd.DataFrame) -> None:
    assert_no_lookahead_sweep(_detect, session)
    assert_blind_to_poison(_detect, session)


def test_detection_no_lookahead_on_history() -> None:
    hist = random_history(3, seed=5, n_bars=73)
    assert_no_lookahead_sweep(_detect, hist)
    assert_blind_to_poison(_detect, hist)


def test_detection_agrees_with_label_session(config: Config) -> None:
    hist = random_history(3, seed=8, n_bars=73)
    daily = random_daily(30, seed=9, end_day=date(2026, 9, 25))
    events, _ = label_breakouts(hist, "X", daily, TEST_CALENDAR, config)
    detected = [(i, d) for i in range(len(hist)) if (d := detect_breakout_at(hist, i, config))]
    assert [(e.breakout_index, e.direction) for e in events] == detected


def test_base_rates_table() -> None:
    from intraday.labelling import base_rates

    df = pd.DataFrame({
        "label": ["SUSTAINED", "BUSTED", "NEITHER", "NEITHER"],
        "direction": ["long", "long", "short", "short"],
    })
    overall = base_rates(df)
    assert overall.loc["all", "n"] == 4 and overall.loc["all", "NEITHER_pct"] == 50.0
    by_dir = base_rates(df, "direction")
    assert by_dir.loc["long", "BUSTED"] == 1 and by_dir.loc["short", "SUSTAINED"] == 0


def test_breakout_depth_cannot_coexist_with_a_bust(config: Config) -> None:
    """Why breakout_depth_atr is context, not a tested feature.

    resolve() measures the excursion from the breakout bar onward, so the breakout bar's
    own close counts. A close already beyond the bust threshold can never be labelled
    BUSTED, which makes the feature a partial restatement of the label.
    """
    from intraday.features import CONTEXT_NAMES, FEATURE_NAMES

    assert "breakout_depth_atr" in CONTEXT_NAMES
    assert "breakout_depth_atr" not in FEATURE_NAMES

    # OR is [99, 101], width 2, ATR 4 -> bust below 1.0, sustain at 2.0 beyond the boundary.
    for depth_atr, close in ((0.30, 102.2), (0.60, 103.4)):
        df = flat_session()
        set_bar(df, 10, 100.5, close + 0.1, 100.4, close)  # deep close through the OR high
        set_bar(df, 20, 99.5, 99.6, 98.0, 98.2)  # then straight back through the other side
        events = label_session(df, "X", ATR, config)
        long = next(e for e in events if e.direction == "long")
        depth = abs(long.breakout_close - long.or_high) / ATR
        assert depth == pytest.approx(depth_atr, abs=0.02)
        assert long.label is not Label.BUSTED, (
            f"a close {depth:.2f} ATR beyond the boundary was labelled BUSTED, which resolve() cannot do"
        )

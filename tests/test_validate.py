"""Synthetic sessions triggering each verdict, plus the same-day tail rule."""
from __future__ import annotations

from datetime import date, datetime, time

import numpy as np
import pandas as pd
import pytest

from intraday.config import IST, Config
from intraday.trading_calendar import CalendarRangeError, TradingCalendar
from intraday.validate import (
    Verdict,
    drop_settling_tail,
    expected_slots,
    split_sessions,
    validate_session,
)

MONDAY = date(2026, 9, 7)
FETCHED_LATER = datetime(2026, 9, 8, 10, 0, tzinfo=IST)


@pytest.fixture
def calendar() -> TradingCalendar:
    return TradingCalendar({date(2026, 9, 14): "Ganesh Chaturthi"})


def make_session(day: date = MONDAY, n: int | None = None, config: Config = Config()) -> pd.DataFrame:
    slots = expected_slots(config)[:n]
    idx = pd.DatetimeIndex([datetime.combine(day, t, tzinfo=IST) for t in slots], name="ts")
    close = 100.0 + np.arange(len(idx)) * 0.1
    return pd.DataFrame(
        {
            "open": close - 0.05,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": np.full(len(idx), 1000, dtype="int64"),
        },
        index=idx,
    )


def test_expected_slots(config: Config) -> None:
    slots = expected_slots(config)
    assert len(slots) == 75
    assert slots[0] == time(9, 15)
    assert slots[-1] == time(15, 25)


def test_clean_session(config: Config, calendar: TradingCalendar) -> None:
    v = validate_session(make_session(), "X", MONDAY, config, calendar, FETCHED_LATER)
    assert v.verdict is Verdict.CLEAN
    assert v.bar_count == 75 and v.missing_slots == () and v.reasons == ()


def test_missing_final_two_slots_is_tail_collapsed(config: Config, calendar: TradingCalendar) -> None:
    v = validate_session(make_session(n=73), "X", MONDAY, config, calendar, FETCHED_LATER)
    assert v.verdict is Verdict.TAIL_COLLAPSED
    assert v.missing_slots == ("15:20", "15:25")
    assert "73 of 75" in v.reasons[0]
    one = validate_session(make_session(n=74), "X", MONDAY, config, calendar, FETCHED_LATER)
    assert one.verdict is Verdict.TAIL_COLLAPSED


def test_missing_three_tail_slots_is_partial(config: Config, calendar: TradingCalendar) -> None:
    v = validate_session(make_session(n=72), "X", MONDAY, config, calendar, FETCHED_LATER)
    assert v.verdict is Verdict.PARTIAL
    assert v.missing_slots == ("15:15", "15:20", "15:25")


def test_missing_interior_slots_is_partial(config: Config, calendar: TradingCalendar) -> None:
    full = make_session()
    df = full.drop(full.index[[30, 31]])
    v = validate_session(df, "X", MONDAY, config, calendar, FETCHED_LATER)
    assert v.verdict is Verdict.PARTIAL
    assert v.missing_slots == ("11:45", "11:50")


def test_same_day_dropped_tail_is_partial_not_collapsed(config: Config, calendar: TradingCalendar) -> None:
    same_day = datetime.combine(MONDAY, time(16, 0), tzinfo=IST)
    df, dropped = drop_settling_tail(make_session(), MONDAY, same_day)
    v = validate_session(df, "X", MONDAY, config, calendar, same_day, dropped)
    assert v.verdict is Verdict.PARTIAL
    assert v.dropped_tail_bars == 2


def test_window_violation_is_corrupt(config: Config, calendar: TradingCalendar) -> None:
    df = make_session()
    idx = df.index.to_list()
    idx[-1] = datetime.combine(MONDAY, time(15, 30), tzinfo=IST)
    df.index = pd.DatetimeIndex(idx, name="ts")
    v = validate_session(df, "X", MONDAY, config, calendar, FETCHED_LATER)
    assert v.verdict is Verdict.CORRUPT
    assert any("outside the session grid" in r for r in v.reasons)


def test_off_grid_timestamp_is_corrupt(config: Config, calendar: TradingCalendar) -> None:
    df = make_session()
    idx = df.index.to_list()
    idx[10] = datetime.combine(MONDAY, time(10, 7), tzinfo=IST)
    df.index = pd.DatetimeIndex(idx, name="ts")
    v = validate_session(df, "X", MONDAY, config, calendar, FETCHED_LATER)
    assert v.verdict is Verdict.CORRUPT


def test_ohlc_sanity_is_corrupt(config: Config, calendar: TradingCalendar) -> None:
    df = make_session()
    df.iloc[5, df.columns.get_loc("low")] = df.iloc[5]["close"] + 1.0
    v = validate_session(df, "X", MONDAY, config, calendar, FETCHED_LATER)
    assert v.verdict is Verdict.CORRUPT
    assert any("low above open/close at 09:40" in r for r in v.reasons)

    df = make_session()
    df.iloc[3, df.columns.get_loc("open")] = 0.0
    v = validate_session(df, "X", MONDAY, config, calendar, FETCHED_LATER)
    assert v.verdict is Verdict.CORRUPT
    assert any("non-positive" in r for r in v.reasons)


def test_duplicate_index_is_corrupt(config: Config, calendar: TradingCalendar) -> None:
    df = make_session(n=74)
    df = pd.concat([df, df.iloc[[-1]]])
    v = validate_session(df, "X", MONDAY, config, calendar, FETCHED_LATER)
    assert v.verdict is Verdict.CORRUPT
    assert any("duplicate timestamp" in r for r in v.reasons)


def test_unsorted_index_is_corrupt(config: Config, calendar: TradingCalendar) -> None:
    df = make_session().iloc[::-1]
    v = validate_session(df, "X", MONDAY, config, calendar, FETCHED_LATER)
    assert v.verdict is Verdict.CORRUPT
    assert any("not ascending" in r for r in v.reasons)


def test_zero_volume_majority_is_suspect(config: Config, calendar: TradingCalendar) -> None:
    df = make_session()
    df.iloc[:40, df.columns.get_loc("volume")] = 0
    v = validate_session(df, "X", MONDAY, config, calendar, FETCHED_LATER)
    assert v.verdict is Verdict.SUSPECT
    assert v.zero_volume_bars == 40

    df = make_session()
    df.iloc[:37, df.columns.get_loc("volume")] = 0
    v = validate_session(df, "X", MONDAY, config, calendar, FETCHED_LATER)
    assert v.verdict is Verdict.CLEAN
    assert v.zero_volume_bars == 37


def test_holiday_with_data_is_corrupt(config: Config, calendar: TradingCalendar) -> None:
    holiday = date(2026, 9, 14)
    v = validate_session(make_session(day=holiday), "X", holiday, config, calendar, FETCHED_LATER)
    assert v.verdict is Verdict.CORRUPT
    assert any("Ganesh Chaturthi" in r for r in v.reasons)


def test_weekend_with_data_is_corrupt(config: Config, calendar: TradingCalendar) -> None:
    saturday = date(2026, 9, 12)
    v = validate_session(make_session(day=saturday), "X", saturday, config, calendar, FETCHED_LATER)
    assert v.verdict is Verdict.CORRUPT
    assert any("weekend" in r for r in v.reasons)


def test_worst_verdict_wins_and_all_reasons_kept(config: Config, calendar: TradingCalendar) -> None:
    df = make_session(n=70)
    df.iloc[:60, df.columns.get_loc("volume")] = 0
    v = validate_session(df, "X", MONDAY, config, calendar, FETCHED_LATER)
    assert v.verdict is Verdict.SUSPECT
    assert len(v.reasons) == 2


def test_date_outside_calendar_raises(config: Config, calendar: TradingCalendar) -> None:
    day = date(2027, 1, 4)
    with pytest.raises(CalendarRangeError):
        validate_session(make_session(day=day), "X", day, config, calendar, FETCHED_LATER)


def test_drop_settling_tail_only_same_day() -> None:
    df = make_session(n=60)
    same_day = datetime.combine(MONDAY, time(14, 20), tzinfo=IST)
    trimmed, dropped = drop_settling_tail(df, MONDAY, same_day)
    assert dropped == 2 and len(trimmed) == 58
    assert trimmed.index[-1].time() == time(14, 0)  # 60 bars end 14:10; drop 14:05, 14:10

    untouched, dropped = drop_settling_tail(df, MONDAY, FETCHED_LATER)
    assert dropped == 0 and len(untouched) == 60


def test_split_sessions_groups_by_ist_date() -> None:
    a = make_session(day=date(2026, 9, 7))
    b = make_session(day=date(2026, 9, 8))
    parts = split_sessions(pd.concat([a, b]))
    assert list(parts) == [date(2026, 9, 7), date(2026, 9, 8)]
    assert len(parts[date(2026, 9, 8)]) == 75


def test_calendar_csv_loads_and_covers_2024_2026() -> None:
    from pathlib import Path

    cal = TradingCalendar.from_csv(Path("data/nse_holidays.csv"))
    assert cal.first_year == 2024 and cal.last_year == 2026
    assert not cal.is_trading_day(date(2026, 9, 14))
    assert cal.is_trading_day(date(2026, 9, 15))
    assert not cal.is_trading_day(date(2026, 9, 13))
    with pytest.raises(CalendarRangeError):
        cal.is_trading_day(date(2023, 12, 29))


def test_index_is_exempt_from_zero_volume_rule(config: Config, calendar: TradingCalendar) -> None:
    df = make_session()
    df["volume"] = 0
    v = validate_session(df, "NIFTY50", MONDAY, config, calendar, FETCHED_LATER, expect_volume=False)
    assert v.verdict is Verdict.CLEAN
    assert v.zero_volume_bars == 75


def test_daily_row_verdicts(calendar: TradingCalendar) -> None:
    from intraday.validate import validate_daily_row

    def row(day: date, **over: float) -> pd.DataFrame:
        base = {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 5_000_000}
        base.update(over)
        idx = pd.DatetimeIndex([datetime.combine(day, time(0, 0), tzinfo=IST)], name="ts")
        return pd.DataFrame({k: [v] for k, v in base.items()}, index=idx).astype({"volume": "int64"})

    ok = validate_daily_row(row(MONDAY), "X", MONDAY, calendar, FETCHED_LATER)
    assert ok.verdict is Verdict.CLEAN and ok.expected_bars == 1
    holiday = date(2026, 9, 14)
    assert validate_daily_row(row(holiday), "X", holiday, calendar, FETCHED_LATER).verdict is Verdict.CORRUPT
    assert validate_daily_row(row(MONDAY, low=100.8), "X", MONDAY, calendar, FETCHED_LATER).verdict is Verdict.CORRUPT
    with pytest.raises(ValueError, match="one daily row"):
        validate_daily_row(pd.concat([row(MONDAY), row(MONDAY)]), "X", MONDAY, calendar, FETCHED_LATER)


def test_daily_row_flags_possible_unadjusted_corporate_action(config: Config, calendar: TradingCalendar) -> None:
    """A 1:2 split shows up as the open halving overnight, with no real move behind it."""
    from intraday.validate import validate_daily_row

    def row(open_: float) -> pd.DataFrame:
        idx = pd.DatetimeIndex([datetime.combine(MONDAY, time(0, 0), tzinfo=IST)], name="ts")
        return pd.DataFrame(
            {"open": [open_], "high": [open_ * 1.01], "low": [open_ * 0.99],
             "close": [open_], "volume": [5_000_000]},
            index=idx,
        ).astype({"volume": "int64"})

    split = validate_daily_row(row(500.0), "X", MONDAY, calendar, FETCHED_LATER, previous_close=1000.0, config=config)
    assert split.verdict is Verdict.SUSPECT
    assert "possible unadjusted corporate action" in split.reasons[0]
    assert "0.50x" in split.reasons[0]

    ordinary = validate_daily_row(row(1020.0), "X", MONDAY, calendar, FETCHED_LATER, previous_close=1000.0, config=config)
    assert ordinary.verdict is Verdict.CLEAN

    # Without a previous close there is nothing to compare against, so no claim is made.
    unknown = validate_daily_row(row(500.0), "X", MONDAY, calendar, FETCHED_LATER, previous_close=None, config=config)
    assert unknown.verdict is Verdict.CLEAN


def test_corporate_action_check_never_masks_a_corrupt_row(config: Config, calendar: TradingCalendar) -> None:
    """CORRUPT outranks SUSPECT: a broken row is not downgraded to 'probably a split'."""
    from intraday.validate import validate_daily_row

    idx = pd.DatetimeIndex([datetime.combine(MONDAY, time(0, 0), tzinfo=IST)], name="ts")
    broken = pd.DataFrame(
        {"open": [500.0], "high": [400.0], "low": [499.0], "close": [500.0], "volume": [1]}, index=idx
    ).astype({"volume": "int64"})
    v = validate_daily_row(broken, "X", MONDAY, calendar, FETCHED_LATER, previous_close=1000.0, config=config)
    assert v.verdict is Verdict.CORRUPT


def test_split_suspect_ratio_bounds_are_validated() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="straddle 1.0"):
        Config(split_suspect_ratio=(1.2, 1.6))
    with pytest.raises(ValidationError, match="straddle 1.0"):
        Config(split_suspect_ratio=(0.6, 0.9))

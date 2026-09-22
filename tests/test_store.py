"""Store: session replacement preserves other sessions; tiers move with verdicts; logs written."""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from intraday.config import IST, Config
from intraday.store import BarStore, read_jsonl
from intraday.trading_calendar import TradingCalendar
from intraday.validate import Verdict, validate_session
from tests.test_validate import FETCHED_LATER, make_session

D1, D2, D3 = date(2026, 9, 7), date(2026, 9, 8), date(2026, 9, 9)


@pytest.fixture
def store(tmp_path: Path) -> BarStore:
    return BarStore(tmp_path, "5m")


@pytest.fixture
def calendar() -> TradingCalendar:
    return TradingCalendar({date(2026, 9, 14): "Ganesh Chaturthi"})


def _validated(day: date, n: int | None, config: Config, calendar: TradingCalendar):  # type: ignore[no-untyped-def]
    df = make_session(day=day, n=n)
    return validate_session(df, "X", day, config, calendar, FETCHED_LATER), df


def test_roundtrip_preserves_tz_and_dtypes(store: BarStore, config: Config, calendar: TradingCalendar) -> None:
    store.put_sessions("X", [_validated(D1, None, config, calendar)])
    out = store.read_research("X")
    assert str(out.index.tz) == "Asia/Kolkata"
    assert out.index[0] == datetime(2026, 9, 7, 9, 15, tzinfo=IST)
    assert out["volume"].dtype == "int64" and out["close"].dtype == "float64"
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]


def test_replacement_touches_only_that_session(store: BarStore, config: Config, calendar: TradingCalendar) -> None:
    store.put_sessions("X", [_validated(d, None, config, calendar) for d in (D1, D2, D3)])
    before = store.read_research("X")
    assert len(before) == 225

    v2, df2 = _validated(D2, None, config, calendar)
    df2 = df2.copy()
    df2["close"] = 999.0
    events = store.put_sessions("X", [(v2, df2)])

    after = store.read_research("X")
    assert len(after) == 225
    assert (after.loc[after.index.date == D2, "close"] == 999.0).all()
    pd.testing.assert_frame_equal(after[after.index.date == D1], before[before.index.date == D1])
    pd.testing.assert_frame_equal(after[after.index.date == D3], before[before.index.date == D3])
    assert events[0]["event"] == "replaced" and events[0]["previous_rows"] == 75


def test_verdict_change_moves_session_between_tiers(store: BarStore, config: Config, calendar: TradingCalendar) -> None:
    store.put_sessions("X", [_validated(D1, 70, config, calendar), _validated(D2, None, config, calendar)])
    assert store.research_sessions("X") == [D2]
    assert set(store.read_quarantine("X").index.date) == {D1}

    store.put_sessions("X", [_validated(D1, None, config, calendar)])
    assert store.research_sessions("X") == [D1, D2]
    assert store.read_quarantine("X").empty
    log = read_jsonl(store.fetch_log_path)
    moved = [e for e in log if e["session_date"] == D1.isoformat() and e["event"] == "replaced"]
    assert moved[0]["previous_tier"] == "quarantine" and moved[0]["tier"] == "bars"


def test_verdict_log_last_wins(store: BarStore, config: Config, calendar: TradingCalendar) -> None:
    store.put_sessions("X", [_validated(D1, 70, config, calendar)])
    store.put_sessions("X", [_validated(D1, None, config, calendar)])
    latest = store.load_verdicts()
    assert latest[("X", D1)].verdict is Verdict.CLEAN
    assert len(read_jsonl(store.verdicts_path)) == 2


def test_rejects_frame_with_foreign_dates(store: BarStore, config: Config, calendar: TradingCalendar) -> None:
    v, _ = _validated(D1, None, config, calendar)
    with pytest.raises(ValueError, match="other dates"):
        store.put_sessions("X", [(v, make_session(day=D2))])


def test_tail_collapsed_lands_in_research_tier(store: BarStore, config: Config, calendar: TradingCalendar) -> None:
    store.put_sessions("X", [_validated(D1, 73, config, calendar)])
    assert store.research_sessions("X") == [D1]
    assert store.read_quarantine("X").empty


def test_intervals_do_not_cross_contaminate_the_verdict_log(tmp_path: Path, config: Config, calendar: TradingCalendar) -> None:
    from intraday.validate import validate_daily_row

    intraday_store, daily_store = BarStore(tmp_path, "5m"), BarStore(tmp_path, "1d")
    intraday_store.put_sessions("X", [_validated(D1, 70, config, calendar)])  # PARTIAL
    row = make_session(day=D1, n=1)
    daily_store.put_sessions("X", [(validate_daily_row(row, "X", D1, calendar, FETCHED_LATER), row)])  # CLEAN
    assert intraday_store.load_verdicts()[("X", D1)].verdict is Verdict.PARTIAL
    assert daily_store.load_verdicts()[("X", D1)].verdict is Verdict.CLEAN
    assert intraday_store.verdicts_path == daily_store.verdicts_path


def test_old_format_verdict_log_is_refused(tmp_path: Path) -> None:
    store = BarStore(tmp_path, "5m")
    store.verdicts_path.write_text('{"symbol": "X"}' + chr(10), encoding="utf-8")
    with pytest.raises(ValueError, match="no interval field"):
        store.load_verdicts()

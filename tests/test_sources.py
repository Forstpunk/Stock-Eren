"""Source-boundary contract tests. No network: the raw frame is synthetic."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from intraday.config import IST, Config
from intraday.sources import BAR_COLUMNS, DataUnavailableError, assert_bar_contract, make_source
from intraday.sources.yfinance_source import YFinanceSource, normalise_raw, to_yahoo_symbol

START = datetime(2026, 9, 1, tzinfo=IST)
END = datetime(2026, 9, 2, tzinfo=IST)


def _raw_frame(n: int = 3, tz: str | None = "Asia/Kolkata") -> pd.DataFrame:
    idx = pd.date_range("2026-09-01 09:15", periods=n, freq="5min", tz=tz, name="Datetime")
    return pd.DataFrame(
        {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Volume": 1000},
        index=idx,
    )


def test_normalise_raw_matches_contract() -> None:
    df = normalise_raw(_raw_frame(), "RELIANCE", "5m", START, END)
    assert tuple(df.columns) == BAR_COLUMNS
    assert str(df.index.tz) == "Asia/Kolkata"
    assert df["volume"].dtype == "int64"
    assert df["close"].dtype == "float64"


def test_normalise_converts_utc_to_ist() -> None:
    raw = _raw_frame(tz="UTC")
    df = normalise_raw(raw, "RELIANCE", "5m", START, END)
    assert df.index[0] == pd.Timestamp("2026-09-01 09:15", tz="UTC").tz_convert(IST)


def test_empty_frame_raises() -> None:
    with pytest.raises(DataUnavailableError, match="empty"):
        normalise_raw(_raw_frame(0), "RELIANCE", "5m", START, END)


def test_naive_index_raises() -> None:
    with pytest.raises(DataUnavailableError, match="tz-naive"):
        normalise_raw(_raw_frame(tz=None), "RELIANCE", "5m", START, END)


def test_duplicate_timestamps_raise() -> None:
    raw = pd.concat([_raw_frame(2), _raw_frame(2)]).sort_index()
    with pytest.raises(DataUnavailableError, match="duplicate"):
        normalise_raw(raw, "RELIANCE", "5m", START, END)


def test_contract_rejects_wrong_columns() -> None:
    df = normalise_raw(_raw_frame(), "X", "5m", START, END).rename(columns={"close": "Close"})
    with pytest.raises(DataUnavailableError, match="columns"):
        assert_bar_contract(df, "X", "5m", START, END)


def test_yahoo_symbol_mapping() -> None:
    assert to_yahoo_symbol("RELIANCE") == "RELIANCE.NS"
    assert to_yahoo_symbol("NIFTY50") == "^NSEI"
    with pytest.raises(ValueError):
        to_yahoo_symbol("RELIANCE.NS")


def test_fetch_rejects_range_beyond_history_cap() -> None:
    src = YFinanceSource()
    with pytest.raises(DataUnavailableError, match="at most 59 days"):
        src.fetch("RELIANCE", "5m", datetime(2020, 1, 1, tzinfo=IST), datetime(2020, 2, 1, tzinfo=IST))


def test_fetch_rejects_naive_datetimes() -> None:
    with pytest.raises(DataUnavailableError, match="tz-aware"):
        YFinanceSource().fetch("RELIANCE", "5m", datetime(2026, 9, 1), datetime(2026, 9, 2))


def test_make_source_yfinance() -> None:
    assert make_source(Config(source="yfinance")).name == "yfinance"


def test_make_source_kite_is_not_a_fallback() -> None:
    with pytest.raises(NotImplementedError):
        make_source(Config(source="kite"))

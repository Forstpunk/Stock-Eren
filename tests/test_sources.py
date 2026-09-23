"""Source-boundary contract tests. No network: the raw frame is synthetic."""
from __future__ import annotations

from datetime import datetime, timedelta

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


# ---- Kite adapter: fake clients only, never the network ------------------------------


class FakeKite:
    """Stands in for kiteconnect.KiteConnect. Records calls so chunking can be asserted."""

    def __init__(self, instruments: list[dict] | None = None, rows: list[dict] | None = None,
                 raises: Exception | None = None) -> None:
        self._instruments = instruments if instruments is not None else [
            {"tradingsymbol": "RELIANCE", "instrument_token": 738561, "instrument_type": "EQ"},
            {"tradingsymbol": "TCS", "instrument_token": 2953217, "instrument_type": "EQ"},
        ]
        self._rows = rows or []
        self._raises = raises
        self.calls: list[dict] = []

    def instruments(self, exchange: str) -> list[dict]:
        return self._instruments

    def historical_data(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(kwargs)
        if self._raises is not None:
            raise self._raises
        return self._rows


def make_kite_source(tmp_path, fake: FakeKite):  # type: ignore[no-untyped-def]
    import os

    from intraday.sources.kite import KiteSource

    os.environ["KITE_API_KEY"] = "test-key"
    os.environ["KITE_ACCESS_TOKEN"] = "test-token"
    src = KiteSource(cache_dir=tmp_path)
    src._kite = fake
    src._min_seconds_between_requests = 0.0
    return src


def test_kite_history_cap_is_not_the_per_request_cap() -> None:
    """Conflating the two silently truncates a study, so they must differ for minute bars."""
    from intraday.sources.kite import _MAX_DAYS_PER_REQUEST, KiteSource

    assert _MAX_DAYS_PER_REQUEST["minute"] == 60
    assert KiteSource.max_history_days["1m"] > 60, "1m total history is years, not one request"
    assert KiteSource.max_history_days["5m"] > _MAX_DAYS_PER_REQUEST["5minute"]


def test_kite_unresolvable_symbol_names_survivorship(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from intraday.sources import SymbolNotResolvable

    src = make_kite_source(tmp_path, FakeKite())
    with pytest.raises(SymbolNotResolvable) as exc:
        src.instrument_token("DELISTEDCO")
    message = str(exc.value)
    assert "delisted" in message and "renamed" in message
    assert "never" in message and "substituted" in message
    assert isinstance(exc.value, DataUnavailableError), "callers catch the base class"


def test_kite_resolves_a_listed_symbol_and_caches_the_dump(tmp_path) -> None:  # type: ignore[no-untyped-def]
    fake = FakeKite()
    src = make_kite_source(tmp_path, fake)
    assert src.instrument_token("RELIANCE") == 738561
    assert list(tmp_path.glob("instruments_nse_*.parquet")), "the dump should be cached on disk"


def test_kite_expired_token_is_named_not_retried(tmp_path) -> None:  # type: ignore[no-untyped-def]
    class TokenException(Exception):
        pass

    src = make_kite_source(tmp_path, FakeKite(raises=TokenException("token expired")))
    start = datetime.now(tz=IST) - timedelta(days=5)
    end = datetime.now(tz=IST)
    with pytest.raises(DataUnavailableError, match="06:00 IST"):
        src.fetch("RELIANCE", "5m", start, end)


def test_kite_chunks_by_the_per_request_cap(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from intraday.sources.kite import _MAX_DAYS_PER_REQUEST

    fake = FakeKite()
    src = make_kite_source(tmp_path, fake)
    start = datetime.now(tz=IST) - timedelta(days=250)
    end = datetime.now(tz=IST)
    with pytest.raises(DataUnavailableError, match="no bars"):
        src.fetch("RELIANCE", "5m", start, end)  # the fake returns nothing
    assert len(fake.calls) == 3, f"250 days at {_MAX_DAYS_PER_REQUEST['5minute']}/request needs 3 calls"


def test_kite_normalises_rows_to_the_bar_contract() -> None:
    from intraday.sources.kite import normalise_kite

    rows = pd.DataFrame({
        "date": pd.date_range("2026-09-07 09:15", periods=3, freq="5min", tz="Asia/Kolkata"),
        "open": [100.0, 101.0, 102.0], "high": [101.0, 102.0, 103.0],
        "low": [99.0, 100.0, 101.0], "close": [100.5, 101.5, 102.5], "volume": [10, 20, 30],
    })
    df = normalise_kite(rows, "RELIANCE", "5m", START, END)
    assert tuple(df.columns) == BAR_COLUMNS
    assert str(df.index.tz) == "Asia/Kolkata" and df["volume"].dtype == "int64"

    naive = rows.copy()
    naive["date"] = pd.date_range("2026-09-07 09:15", periods=3, freq="5min")  # Kite sometimes tz-naive
    df2 = normalise_kite(naive, "RELIANCE", "5m", START, END)
    assert str(df2.index.tz) == "Asia/Kolkata"
    assert df2.index[0].hour == 9 and df2.index[0].minute == 15, "naive stamps are IST, not UTC"

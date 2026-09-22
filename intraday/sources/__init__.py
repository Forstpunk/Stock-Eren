"""Bar-source boundary.

Exactly one source is active per run, selected by ``Config.source``. This is a
port/adapter boundary, not a fallback chain: if the configured source cannot
deliver, the run fails.

Contract every implementation honours:
- returns a DataFrame indexed by tz-aware Asia/Kolkata timestamps, ascending, unique
- columns exactly ``["open", "high", "low", "close", "volume"]``; float except volume (int64)
- raises ``DataUnavailableError`` instead of returning an empty frame
- handles its own chunking and rate limiting
"""
from __future__ import annotations

from datetime import datetime
from typing import Protocol

import pandas as pd

from intraday.config import IST, Config

BAR_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


class DataUnavailableError(Exception):
    """The configured source could not deliver the requested bars."""

    def __init__(self, symbol: str, interval: str, start: datetime, end: datetime, reason: str) -> None:
        self.symbol = symbol
        self.interval = interval
        self.start = start
        self.end = end
        self.reason = reason
        super().__init__(f"{symbol} {interval} {start:%Y-%m-%d} -> {end:%Y-%m-%d}: {reason}")


class BarSource(Protocol):
    name: str
    max_history_days: dict[str, int]  # interval -> days available

    def fetch(self, symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame: ...


def assert_bar_contract(df: pd.DataFrame, symbol: str, interval: str, start: datetime, end: datetime) -> None:
    """Raise DataUnavailableError if ``df`` violates the source contract."""

    def fail(reason: str) -> None:
        raise DataUnavailableError(symbol, interval, start, end, reason)

    if df.empty:
        fail("source returned no bars")
    if tuple(df.columns) != BAR_COLUMNS:
        fail(f"columns {list(df.columns)} != {list(BAR_COLUMNS)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        fail(f"index is {type(df.index).__name__}, not DatetimeIndex")
    if df.index.tz is None or str(df.index.tz) != str(IST):
        fail(f"index tz is {df.index.tz}, expected {IST}")
    if not df.index.is_monotonic_increasing:
        fail("index is not ascending")
    if not df.index.is_unique:
        dup = df.index[df.index.duplicated()][0]
        fail(f"duplicate timestamp {dup}")
    for col in ("open", "high", "low", "close"):
        if not pd.api.types.is_float_dtype(df[col]):
            fail(f"column {col} is {df[col].dtype}, expected float")
    if not pd.api.types.is_integer_dtype(df["volume"]):
        fail(f"column volume is {df['volume'].dtype}, expected int")


def make_source(config: Config) -> BarSource:
    """Build the one source named by config. Unknown or unimplemented sources raise."""
    if config.source == "yfinance":
        from intraday.sources.yfinance_source import YFinanceSource

        return YFinanceSource()
    if config.source == "kite":
        raise NotImplementedError("Kite Connect source is not implemented yet; see intraday/sources/kite.py")
    raise ValueError(f"unknown source {config.source!r}")

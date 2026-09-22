"""yfinance adapter. Free; 60 days of 5-minute history; NSE symbols carry a '.NS' suffix."""
from __future__ import annotations

import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from intraday.config import IST
from intraday.sources import BAR_COLUMNS, DataUnavailableError, assert_bar_contract

_RAW_COLUMNS = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}


def to_yahoo_symbol(symbol: str) -> str:
    """NSE equity ticker -> Yahoo ticker. Index symbols are mapped explicitly."""
    if symbol == "NIFTY50":
        return "^NSEI"
    if symbol.startswith("^") or "." in symbol:
        raise ValueError(f"symbol {symbol!r} must be a bare NSE ticker such as 'RELIANCE'")
    return f"{symbol}.NS"


def normalise_raw(raw: pd.DataFrame, symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Reshape a yfinance history frame into the bar contract. Raises rather than fixes."""
    if raw.empty:
        raise DataUnavailableError(symbol, interval, start, end, "yfinance returned an empty frame")
    missing = [c for c in _RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise DataUnavailableError(symbol, interval, start, end, f"yfinance frame lacks columns {missing}")
    if not isinstance(raw.index, pd.DatetimeIndex):
        raise DataUnavailableError(symbol, interval, start, end, f"index is {type(raw.index).__name__}")
    if raw.index.tz is None:
        raise DataUnavailableError(symbol, interval, start, end, "yfinance index is tz-naive")

    df = raw[list(_RAW_COLUMNS)].rename(columns=_RAW_COLUMNS)
    df.index = df.index.tz_convert(IST)
    df.index.name = "ts"
    if df["volume"].isna().any():
        first = df.index[df["volume"].isna()][0]
        raise DataUnavailableError(symbol, interval, start, end, f"NaN volume at {first}")
    df = df.astype({"open": "float64", "high": "float64", "low": "float64", "close": "float64", "volume": "int64"})
    df = df[list(BAR_COLUMNS)]
    assert_bar_contract(df, symbol, interval, start, end)
    return df


class YFinanceSource:
    name = "yfinance"
    # Total lookback Yahoo serves per interval. Yahoo's "within the last 60 days" rule is
    # strict: a 60-day span is rejected (verified 2026-09-21), 59 is the largest that works.
    max_history_days: dict[str, int] = {"1m": 29, "2m": 59, "5m": 59, "15m": 59, "30m": 59, "1h": 729, "1d": 3650}
    # Span Yahoo accepts in a single request per interval.
    _max_days_per_request: dict[str, int] = {"1m": 7, "2m": 59, "5m": 59, "15m": 59, "30m": 59, "1h": 729, "1d": 3650}
    _min_seconds_between_requests: float = 0.5

    def __init__(self) -> None:
        self._last_request_at: float = 0.0

    def fetch(self, symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
        if interval not in self.max_history_days:
            raise DataUnavailableError(symbol, interval, start, end, f"interval {interval!r} not served by yfinance")
        if start.tzinfo is None or end.tzinfo is None:
            raise DataUnavailableError(symbol, interval, start, end, "start/end must be tz-aware")
        if start >= end:
            raise DataUnavailableError(symbol, interval, start, end, "start must precede end")
        now = datetime.now(tz=IST)
        oldest_allowed = now - timedelta(days=self.max_history_days[interval])
        if start.date() < oldest_allowed.date():
            raise DataUnavailableError(
                symbol, interval, start, end,
                f"yfinance serves at most {self.max_history_days[interval]} days of {interval}; "
                f"oldest allowed start is {oldest_allowed:%Y-%m-%d %H:%M}",
            )

        ticker = yf.Ticker(to_yahoo_symbol(symbol))
        chunks: list[pd.DataFrame] = []
        span = timedelta(days=self._max_days_per_request[interval])
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + span, end)
            self._throttle()
            try:
                raw = ticker.history(
                    start=chunk_start, end=chunk_end, interval=interval,
                    auto_adjust=False, actions=False, prepost=False, raise_errors=True,
                )
            except Exception as exc:  # yfinance raises a mix of types; the boundary re-labels, never swallows
                raise DataUnavailableError(
                    symbol, interval, chunk_start, chunk_end, f"{type(exc).__name__}: {exc}"
                ) from exc
            chunks.append(raw)
            chunk_start = chunk_end

        raw_all = pd.concat(chunks) if len(chunks) > 1 else chunks[0]
        return normalise_raw(raw_all, symbol, interval, start, end)

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        wait = self._min_seconds_between_requests - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.monotonic()

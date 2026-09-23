"""Kite Connect bar source. ~10 years of 5-minute history, subscription required.

Satisfies ``intraday.sources.BarSource``. Handles, internally:

- instrument_token lookup from the daily NSE instruments dump, cached per day
- per-request day caps: 100 days for ``5minute``, 60 for ``minute``, 2000 for ``day``
- the 3 requests/second limit on the historical endpoint
- the daily access-token expiry at 06:00 IST, reported as a DataUnavailableError that
  names the expiry rather than silently re-authenticating
- Kite's tz-naive IST timestamps, localised explicitly

Credentials come from the environment, never from the repo:

    KITE_API_KEY, KITE_ACCESS_TOKEN

The access token is issued by the daily login flow and expires at 06:00 IST the next
morning. ``python -m intraday login`` prints the steps.
"""
from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from intraday.config import IST
from intraday.sources import BAR_COLUMNS, DataUnavailableError, SymbolNotResolvable, assert_bar_contract

# pipeline interval -> Kite interval
_INTERVALS = {"1m": "minute", "3m": "3minute", "5m": "5minute", "15m": "15minute", "30m": "30minute", "1d": "day"}
_MAX_DAYS_PER_REQUEST = {"minute": 60, "3minute": 100, "5minute": 100, "15minute": 200, "30minute": 200, "day": 2000}
_TOKEN_EXPIRY_HOUR = 6  # IST


class KiteCredentialsError(Exception):
    """API key or access token missing from the environment."""


def _credentials() -> tuple[str, str]:
    api_key = os.environ.get("KITE_API_KEY", "").strip()
    access_token = os.environ.get("KITE_ACCESS_TOKEN", "").strip()
    if not api_key or not access_token:
        raise KiteCredentialsError(
            "set KITE_API_KEY and KITE_ACCESS_TOKEN in the environment. The access token is "
            "issued by the daily login flow and expires at 06:00 IST; run "
            "'python -m intraday login' for the steps."
        )
    return api_key, access_token


def token_is_stale(issued_at: datetime, now: datetime) -> bool:
    """A Kite access token dies at the first 06:00 IST after it was issued."""
    expiry = issued_at.replace(hour=_TOKEN_EXPIRY_HOUR, minute=0, second=0, microsecond=0)
    if issued_at >= expiry:
        expiry += timedelta(days=1)
    return now >= expiry


class KiteSource:
    name = "kite"
    # Total history available, NOT the per-request cap - those are different numbers and
    # conflating them silently truncates a study. The per-request caps live in
    # _MAX_DAYS_PER_REQUEST below (60 days for minute candles, 100 for 5minute, and so on).
    # VERIFY: Kite's own docs publish the per-request caps but not a single authoritative
    # retention figure. The developer forum states roughly 3 years for 1-minute candles and
    # longer for coarser intervals, and retention is subscription dependent. 1m is set to
    # 1095 days on that basis; the others assume a full 10-year plan. If a fetch fails with
    # an empty response for an old range, lower these rather than assuming the data is gone.
    max_history_days: dict[str, int] = {"1m": 1095, "3m": 3650, "5m": 3650, "15m": 3650, "30m": 3650, "1d": 3650}
    _min_seconds_between_requests: float = 1 / 3  # 3 req/sec on the historical endpoint

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._api_key, self._access_token = _credentials()
        self._cache_dir = cache_dir or Path("data") / "kite"
        self._last_request_at = 0.0
        self._instruments: dict[str, int] | None = None
        self._kite = None

    # ---- connection ------------------------------------------------------------------

    def _client(self):  # type: ignore[no-untyped-def]
        if self._kite is None:
            try:
                from kiteconnect import KiteConnect
            except ImportError as exc:
                raise DataUnavailableError(
                    "-", "-", datetime.now(tz=IST), datetime.now(tz=IST),
                    "kiteconnect is not installed; pip install kiteconnect",
                ) from exc
            self._kite = KiteConnect(api_key=self._api_key)
            self._kite.set_access_token(self._access_token)
        return self._kite

    # ---- instruments -----------------------------------------------------------------

    def instrument_token(self, symbol: str) -> int:
        """NSE trading symbol -> instrument_token. Raises if the symbol is not listed."""
        if self._instruments is None:
            self._instruments = self._load_instruments()
        token = self._instruments.get(symbol)
        if token is None:
            raise SymbolNotResolvable(
                symbol, "-", datetime.now(tz=IST), datetime.now(tz=IST),
                f"{symbol} is not in today's NSE instrument list. It may be delisted, renamed, "
                "merged, or simply misspelt. The instrument dump only describes instruments that "
                "exist today, so a study built from it silently excludes names that have since "
                "disappeared - survivorship bias. This symbol is excluded and reported, never "
                "substituted.",
            )
        return token

    def _load_instruments(self) -> dict[str, int]:
        """Today's NSE instruments, cached on disk: the dump is large and changes daily."""
        today = datetime.now(tz=IST).date()
        path = self._cache_dir / f"instruments_nse_{today:%Y-%m-%d}.parquet"
        if path.exists():
            frame = pd.read_parquet(path)
        else:
            self._throttle()
            try:
                rows = self._client().instruments("NSE")
            except Exception as exc:
                raise self._translate(exc, "-", "-", datetime.now(tz=IST), datetime.now(tz=IST)) from exc
            frame = pd.DataFrame(rows)
            if frame.empty:
                raise DataUnavailableError("-", "-", datetime.now(tz=IST), datetime.now(tz=IST), "empty instrument dump")
            frame = frame.loc[frame["instrument_type"] == "EQ", ["tradingsymbol", "instrument_token"]]
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
        return dict(zip(frame["tradingsymbol"], frame["instrument_token"].astype(int)))

    # ---- fetch -----------------------------------------------------------------------

    def fetch(self, symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
        if interval not in _INTERVALS:
            raise DataUnavailableError(symbol, interval, start, end, f"interval {interval!r} not served by Kite")
        if start.tzinfo is None or end.tzinfo is None:
            raise DataUnavailableError(symbol, interval, start, end, "start/end must be tz-aware")
        if start >= end:
            raise DataUnavailableError(symbol, interval, start, end, "start must precede end")
        oldest = datetime.now(tz=IST) - timedelta(days=self.max_history_days[interval])
        if start.date() < oldest.date():
            raise DataUnavailableError(
                symbol, interval, start, end,
                f"Kite serves at most {self.max_history_days[interval]} days of {interval}; "
                f"oldest allowed start is {oldest:%Y-%m-%d}",
            )

        kite_interval = _INTERVALS[interval]
        token = self.instrument_token(symbol)
        span = timedelta(days=_MAX_DAYS_PER_REQUEST[kite_interval])
        chunks: list[pd.DataFrame] = []
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + span, end)
            self._throttle()
            try:
                rows = self._client().historical_data(
                    instrument_token=token,
                    from_date=chunk_start.astimezone(IST).replace(tzinfo=None),
                    to_date=chunk_end.astimezone(IST).replace(tzinfo=None),
                    interval=kite_interval,
                    continuous=False,
                    oi=False,
                )
            except Exception as exc:
                raise self._translate(exc, symbol, interval, chunk_start, chunk_end) from exc
            if rows:
                chunks.append(pd.DataFrame(rows))
            chunk_start = chunk_end

        if not chunks:
            raise DataUnavailableError(symbol, interval, start, end, "Kite returned no bars")
        return normalise_kite(pd.concat(chunks, ignore_index=True), symbol, interval, start, end)

    def _translate(self, exc: Exception, symbol: str, interval: str, start: datetime, end: datetime) -> DataUnavailableError:
        """Re-label a kiteconnect exception. An expired token is named as such, never retried."""
        name = type(exc).__name__
        if name in ("TokenException", "PermissionException"):
            return DataUnavailableError(
                symbol, interval, start, end,
                f"{name}: the Kite access token is invalid or expired (it dies at 06:00 IST). "
                f"Re-run the login flow and update KITE_ACCESS_TOKEN. Original: {exc}",
            )
        return DataUnavailableError(symbol, interval, start, end, f"{name}: {exc}")

    def _throttle(self) -> None:
        wait = self._min_seconds_between_requests - (time.monotonic() - self._last_request_at)
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.monotonic()


def normalise_kite(raw: pd.DataFrame, symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Kite rows -> the bar contract. Raises rather than repairing."""
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(raw.columns)
    if missing:
        raise DataUnavailableError(symbol, interval, start, end, f"Kite rows lack columns {sorted(missing)}")
    df = raw.copy()
    ts = pd.to_datetime(df["date"])
    # Kite returns tz-aware IST on most endpoints and naive IST on some; handle both explicitly.
    df.index = ts.dt.tz_localize(IST) if ts.dt.tz is None else ts.dt.tz_convert(IST)
    df.index.name = "ts"
    df = df.drop(columns=["date"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if df[["open", "high", "low", "close"]].isna().any().any() or df["volume"].isna().any():
        first = df.index[df.isna().any(axis=1)][0]
        raise DataUnavailableError(symbol, interval, start, end, f"NaN values at {first}")
    df = df.astype({"open": "float64", "high": "float64", "low": "float64", "close": "float64", "volume": "int64"})
    df = df[list(BAR_COLUMNS)]
    assert_bar_contract(df, symbol, interval, start, end)
    return df


LOGIN_INSTRUCTIONS = """Kite Connect daily login

The access token expires at 06:00 IST every morning; this has to be done once a day.

1. Create an app at https://developers.kite.trade/apps (Rs 2,000 one-off) and subscribe
   to historical data (Rs 2,000/month at the time of writing). Note the api_key and
   api_secret.

2. Open this in a browser, log in, and copy the request_token from the redirect URL:

       https://kite.trade/connect/login?api_key=<API_KEY>&v=3

3. Exchange it for an access token:

       python -c "from kiteconnect import KiteConnect; \\
         k=KiteConnect(api_key='<API_KEY>'); \\
         print(k.generate_session('<REQUEST_TOKEN>', api_secret='<API_SECRET>')['access_token'])"

4. Put both in the environment before running the pipeline:

       $env:KITE_API_KEY='<API_KEY>'
       $env:KITE_ACCESS_TOKEN='<ACCESS_TOKEN>'

5. Select the source:

       python -m intraday update --source kite --days 2000

Nothing in this repo stores a key or a token. If the token has expired mid-run the fetch
fails with a message naming the expiry - it never re-authenticates on its own."""

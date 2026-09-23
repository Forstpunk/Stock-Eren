"""Pure indicator functions. No state, no caching.

Every function takes a bar frame and a positional index ``i`` and uses only rows at
positions ``<= i``. Undefined values are returned as NaN, never as a default.

Two frame shapes are used:

- intraday frame: one symbol's research sessions concatenated in time order, 5m grid,
  tz-aware Asia/Kolkata index. Session boundaries are calendar dates. Functions that
  need history (RVOL, gap) look back across dates; functions that reset daily (VWAP,
  sigma, opening range) only look at rows sharing the date of row ``i``.
- daily frame: one row per trading day, used by ``atr``.
"""
from __future__ import annotations

import math
from datetime import date, time, timedelta

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from intraday.config import Config
from intraday.trading_calendar import TradingCalendar
from intraday.validate import expected_slots


class OpeningRange(BaseModel):
    model_config = ConfigDict(frozen=True)

    high: float
    low: float
    width: float
    end_index: int  # position of the last opening-range bar in the frame


def _check_frame(bars: pd.DataFrame, i: int) -> None:
    if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
        raise ValueError("bars must have a tz-aware DatetimeIndex")
    if not bars.index.is_monotonic_increasing:
        raise ValueError("bars must be sorted ascending")
    if not 0 <= i < len(bars):
        raise IndexError(f"index {i} outside 0..{len(bars) - 1}")


def session_start_pos(bars: pd.DataFrame, i: int) -> int:
    """Position of the first row sharing the date of row ``i``."""
    day = bars.index[i].date()
    dates = bars.index.date
    return int(np.searchsorted(dates, day, side="left"))


def _today(bars: pd.DataFrame, i: int) -> pd.DataFrame:
    """Rows of the session containing ``i``, up to and including ``i``."""
    return bars.iloc[session_start_pos(bars, i) : i + 1]


# ---- daily-reset indicators ---------------------------------------------------------


def session_vwap(bars: pd.DataFrame, i: int) -> float:
    """Cumulative typical-price x volume / cumulative volume for the session containing
    ``i``, through bar ``i``. NaN while cumulative volume is zero (yfinance's 09:15 bar)."""
    _check_frame(bars, i)
    today = _today(bars, i)
    volume = today["volume"].to_numpy(dtype="float64")
    cum_vol = volume.sum()
    if cum_vol <= 0:
        return math.nan
    typical = (today["high"] + today["low"] + today["close"]).to_numpy(dtype="float64") / 3.0
    return float((typical * volume).sum() / cum_vol)


def session_sigma(bars: pd.DataFrame, i: int) -> float:
    """Sample stdev of close-to-close bar returns within the session, through bar ``i``.
    NaN with fewer than two returns (three bars)."""
    _check_frame(bars, i)
    close = _today(bars, i)["close"].to_numpy(dtype="float64")
    if len(close) < 3:
        return math.nan
    returns = close[1:] / close[:-1] - 1.0
    return float(returns.std(ddof=1))


def opening_range(bars: pd.DataFrame, i: int, config: Config) -> OpeningRange:
    """High/low/width of the first ``config.opening_range_minutes`` of the session
    containing ``i``. Raises if the range is not yet complete at ``i`` or if the session
    does not start on the session-start slot (a partial-start session has no range)."""
    _check_frame(bars, i)
    start = session_start_pos(bars, i)
    n = config.opening_range_bars
    if bars.index[start].time() != config.session_start:
        raise ValueError(
            f"session {bars.index[i].date()} starts at {bars.index[start].time()}, not "
            f"{config.session_start}; opening range undefined"
        )
    end = start + n - 1
    if i < end:
        raise ValueError(f"opening range incomplete at index {i}: needs bars through index {end}")
    window = bars.iloc[start : end + 1]
    if window.index[-1].time() != expected_slots(config)[n - 1]:
        raise ValueError(f"opening range bars are not contiguous on {bars.index[i].date()}")
    high = float(window["high"].max())
    low = float(window["low"].min())
    return OpeningRange(high=high, low=low, width=high - low, end_index=end)


# ---- cross-session indicators -------------------------------------------------------


def rvol_at_time(bars: pd.DataFrame, i: int, lookback: int) -> float:
    """Cumulative volume today through bar ``i`` divided by the mean, over the trailing
    ``lookback`` sessions, of cumulative volume through the same time of day.
    Time-of-day matched. NaN if fewer than ``lookback`` prior sessions exist or the
    trailing mean is zero."""
    _check_frame(bars, i)
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    start = session_start_pos(bars, i)
    cutoff: time = bars.index[i].time()
    today_vol = float(bars["volume"].iloc[start : i + 1].sum())

    history = bars.iloc[:start]
    if history.empty:
        return math.nan
    prior_dates = pd.unique(history.index.date)
    if len(prior_dates) < lookback:
        return math.nan
    window_dates = set(prior_dates[-lookback:])
    hist_dates = history.index.date
    hist_times = np.array([ts.time() for ts in history.index])
    mask = np.array([d in window_dates for d in hist_dates]) & (hist_times <= cutoff)
    matched = history.loc[mask, "volume"]
    per_session = matched.groupby(hist_dates[mask]).sum()
    denom = float(per_session.sum()) / lookback  # sessions with no bar <= cutoff count as zero
    if denom <= 0:
        return math.nan
    return today_vol / denom


def rvol_bar(bars: pd.DataFrame, i: int, lookback: int) -> float:
    """Volume of bar ``i`` divided by the mean volume of the bar at the same time of day
    over the trailing ``lookback`` sessions. NaN without ``lookback`` prior sessions or
    if the trailing mean is zero."""
    _check_frame(bars, i)
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    start = session_start_pos(bars, i)
    slot: time = bars.index[i].time()
    history = bars.iloc[:start]
    if history.empty:
        return math.nan
    prior_dates = pd.unique(history.index.date)
    if len(prior_dates) < lookback:
        return math.nan
    window_dates = set(prior_dates[-lookback:])
    mask = np.array([ts.date() in window_dates and ts.time() == slot for ts in history.index])
    denom = float(history.loc[mask, "volume"].sum()) / lookback  # sessions lacking the slot count as zero
    if denom <= 0:
        return math.nan
    return float(bars["volume"].iloc[i]) / denom


def gap_pct(bars: pd.DataFrame, i: int, calendar: TradingCalendar) -> float:
    """(today's first open - previous trading day's last close) / previous close x 100.
    NaN if the previous trading day is absent from ``bars`` (never substitutes an older close)."""
    _check_frame(bars, i)
    start = session_start_pos(bars, i)
    today: date = bars.index[i].date()
    prev_day = previous_trading_day(today, calendar)
    history = bars.iloc[:start]
    if history.empty or history.index[-1].date() != prev_day:
        return math.nan
    prev_close = float(history["close"].iloc[-1])
    today_open = float(bars["open"].iloc[start])
    return (today_open - prev_close) / prev_close * 100.0


def previous_trading_day(day: date, calendar: TradingCalendar) -> date:
    d = day - timedelta(days=1)
    while not calendar.is_trading_day(d):
        d -= timedelta(days=1)
    return d


# ---- daily indicators ---------------------------------------------------------------


def atr_prior_day(daily: pd.DataFrame, session_date: date, calendar: TradingCalendar, period: int) -> float:
    """ATR through the previous trading day's daily row. NaN if that exact day is absent
    from ``daily`` (an older row is never substituted)."""
    prev = previous_trading_day(session_date, calendar)
    hits = np.flatnonzero(daily.index.date == prev)
    if len(hits) == 0:
        return math.nan
    return atr(daily, int(hits[0]), period)


def atr(daily: pd.DataFrame, i: int, period: int) -> float:
    """Wilder ATR through daily row ``i``. Needs ``period + 1`` rows (the first true range
    uses the prior close); NaN before that."""
    _check_frame(daily, i)
    if period <= 0:
        raise ValueError("period must be positive")
    if i < period:
        return math.nan
    window = daily.iloc[: i + 1]
    high = window["high"].to_numpy(dtype="float64")
    low = window["low"].to_numpy(dtype="float64")
    close = window["close"].to_numpy(dtype="float64")
    prev_close = close[:-1]
    tr = np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - prev_close),
        np.abs(low[1:] - prev_close),
    ])
    value = float(tr[:period].mean())
    for x in tr[period:]:
        value = (value * (period - 1) + float(x)) / period
    return value


# ---- index alignment ------------------------------------------------------------------


def index_bar_at(index_bars: pd.DataFrame, ts: pd.Timestamp) -> pd.Series | None:
    """The index bar stamped exactly ``ts``, or None.

    Exact match only. A nearest-neighbour lookup would quietly compare a stock bar against
    an index bar from a different minute, which is a small lie that compounds.
    """
    if index_bars.empty or ts not in index_bars.index:
        return None
    row = index_bars.loc[ts]
    return row.iloc[0] if isinstance(row, pd.DataFrame) else row


def index_session_open(index_bars: pd.DataFrame, session_date: date) -> float:
    """Open of the index's first bar of ``session_date``. NaN if the session is absent."""
    same_day = index_bars[index_bars.index.date == session_date]
    return math.nan if same_day.empty else float(same_day["open"].iloc[0])


def index_breakout_state(
    index_bars: pd.DataFrame, ts: pd.Timestamp, direction: str, config: Config
) -> float:
    """Has the index closed beyond its own opening range in ``direction``, at or before ``ts``?

    Returns 1.0 if any bar up to and including ``ts`` closed beyond the boundary on that
    side, 0.0 if none did, and NaN when the index has no bars for that session or no
    complete opening range yet.

    ANY such close counts, not just the session's first breakout. An index that broke up
    at 09:35 and is back inside by 11:00 has still shown buyers willing to pay through the
    level, and a stock breaking up at 11:05 is not fighting the index. Reading only the
    first breakout would also make the answer for one direction depend on whether the
    OTHER direction happened to break earlier, which is not what was pre-registered.

    Uses only index bars at or before ``ts``, so it is as blind to the future as the stock
    features are.
    """
    if direction not in ("long", "short"):
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")
    same_day = index_bars[index_bars.index.date == ts.date()]
    same_day = same_day[same_day.index <= ts]
    if same_day.empty:
        return math.nan
    try:
        rng = opening_range(same_day, len(same_day) - 1, config)
    except (ValueError, IndexError):
        return math.nan  # range not complete yet, or the session does not start on the grid
    after = same_day["close"].iloc[rng.end_index + 1 :]
    if after.empty:
        return 0.0
    broke = (after > rng.high) if direction == "long" else (after < rng.low)
    return float(bool(broke.any()))

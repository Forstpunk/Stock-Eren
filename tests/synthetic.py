"""Synthetic sessions for tests: a seeded random walk on the real NSE 5m grid."""
from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from intraday.config import IST, Config
from intraday.trading_calendar import TradingCalendar
from intraday.validate import expected_slots

TEST_CALENDAR = TradingCalendar({date(2026, 9, 14): "Ganesh Chaturthi", date(2026, 10, 2): "Gandhi Jayanti"})


def random_session(
    seed: int = 0,
    day: date = date(2026, 9, 7),
    start_price: float = 1000.0,
    bar_sigma: float = 0.0015,
    config: Config = Config(),
) -> pd.DataFrame:
    """One full 75-bar session. OHLC is internally consistent; volume is lognormal."""
    rng = np.random.default_rng(seed)
    slots = expected_slots(config)
    n = len(slots)
    idx = pd.DatetimeIndex([datetime.combine(day, t, tzinfo=IST) for t in slots], name="ts")
    returns = rng.normal(0.0, bar_sigma, n)
    close = start_price * np.exp(np.cumsum(returns))
    open_ = np.concatenate([[start_price], close[:-1]]) * (1 + rng.normal(0, bar_sigma / 4, n))
    wick = np.abs(rng.normal(0, bar_sigma, n)) * close
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    volume = rng.lognormal(mean=11.0, sigma=0.5, size=n).astype("int64")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
    )


def trading_days_ending(end_day: date, n: int) -> list[date]:
    days: list[date] = []
    d = end_day
    while len(days) < n:
        if TEST_CALENDAR.is_trading_day(d):
            days.append(d)
        d -= timedelta(days=1)
    days.reverse()
    return days


def random_history(
    n_sessions: int, seed: int = 0, end_day: date = date(2026, 9, 25), n_bars: int | None = None
) -> pd.DataFrame:
    """``n_sessions`` consecutive trading days ending on ``end_day``, chained so each
    session opens near the previous close. ``n_bars`` truncates each session (e.g. 73)."""
    frames = []
    price = 1000.0
    for k, day in enumerate(trading_days_ending(end_day, n_sessions)):
        s = random_session(seed=seed * 1000 + k, day=day, start_price=price)
        if n_bars is not None:
            s = s.iloc[:n_bars]
        frames.append(s)
        price = float(s["close"].iloc[-1]) * (1 + np.random.default_rng(seed * 7 + k).normal(0, 0.004))
    return pd.concat(frames)


def random_daily(n_days: int, seed: int = 0, end_day: date = date(2026, 9, 25)) -> pd.DataFrame:
    """Daily bars on trading days ending ``end_day``; index at midnight IST like yfinance."""
    rng = np.random.default_rng(seed)
    days = trading_days_ending(end_day, n_days)
    close = 1000.0 * np.exp(np.cumsum(rng.normal(0, 0.012, n_days)))
    open_ = np.concatenate([[1000.0], close[:-1]]) * (1 + rng.normal(0, 0.004, n_days))
    wick = np.abs(rng.normal(0, 0.006, n_days)) * close
    idx = pd.DatetimeIndex([datetime.combine(day, datetime.min.time(), tzinfo=IST) for day in days], name="ts")
    return pd.DataFrame({
        "open": open_,
        "high": np.maximum(open_, close) + wick,
        "low": np.minimum(open_, close) - wick,
        "close": close,
        "volume": rng.lognormal(15, 0.4, n_days).astype("int64"),
    }, index=idx)

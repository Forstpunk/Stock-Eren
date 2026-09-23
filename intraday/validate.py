"""Session validation: every fetched session gets a named verdict. Nothing is silently fixed.

Verdict severity: CORRUPT > SUSPECT > PARTIAL > TAIL_COLLAPSED > CLEAN. Every failing
check appends a reason; the worst one becomes the verdict.

Research set = CLEAN + TAIL_COLLAPSED. TAIL_COLLAPSED is a session that passes every
check except that its final one or two slots are absent and it was not a same-day fetch:
yfinance folds 15:15-15:30 (closing auction included) into the 15:15 bar, whose close
equals the official daily close (verified 2026-09-21 on 20 NSE equities). The session is
complete in content; only the last 15 minutes are at coarser resolution.
"""
from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta
from enum import Enum

import pandas as pd
from pydantic import BaseModel, ConfigDict

from intraday.config import Config
from intraday.sources import BAR_COLUMNS
from intraday.trading_calendar import TradingCalendar

SETTLING_TAIL_BARS = 2  # final bars of a same-day fetch carry late-settling closes
ZERO_VOLUME_SUSPECT_SHARE = 0.5
MAX_COLLAPSED_TAIL_SLOTS = 2  # at most the last two slots may be folded into the prior bar


class Verdict(str, Enum):
    CLEAN = "CLEAN"
    TAIL_COLLAPSED = "TAIL_COLLAPSED"
    PARTIAL = "PARTIAL"
    SUSPECT = "SUSPECT"
    CORRUPT = "CORRUPT"


_SEVERITY = {
    Verdict.CLEAN: 0,
    Verdict.TAIL_COLLAPSED: 1,
    Verdict.PARTIAL: 2,
    Verdict.SUSPECT: 3,
    Verdict.CORRUPT: 4,
}
RESEARCH_VERDICTS = frozenset({Verdict.CLEAN, Verdict.TAIL_COLLAPSED})


class SessionVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    session_date: date
    verdict: Verdict
    bar_count: int
    expected_bars: int
    zero_volume_bars: int
    missing_slots: tuple[str, ...]
    dropped_tail_bars: int
    reasons: tuple[str, ...]
    fetched_at: datetime


def expected_slots(config: Config) -> list[time]:
    """Bar start times of a full session: 09:15, 09:20, ... 15:25 for 5m."""
    anchor = datetime(2000, 1, 1, config.session_start.hour, config.session_start.minute)
    step = timedelta(minutes=config.interval_minutes)
    return [(anchor + k * step).time() for k in range(config.bars_per_session)]


def split_sessions(bars: pd.DataFrame) -> dict[date, pd.DataFrame]:
    """Group a multi-day frame by IST calendar date, preserving order."""
    if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
        raise ValueError("bars must be indexed by tz-aware timestamps")
    return {d: g for d, g in bars.groupby(bars.index.date, sort=True)}


def drop_settling_tail(session: pd.DataFrame, session_date: date, fetched_at: datetime) -> tuple[pd.DataFrame, int]:
    """Drop the final bars of a session fetched on the same calendar day. Returns (frame, dropped)."""
    if session_date != fetched_at.date():
        return session, 0
    n = min(SETTLING_TAIL_BARS, len(session))
    return session.iloc[: len(session) - n], n


def validate_session(
    session: pd.DataFrame,
    symbol: str,
    session_date: date,
    config: Config,
    calendar: TradingCalendar,
    fetched_at: datetime,
    dropped_tail_bars: int = 0,
    expect_volume: bool = True,
) -> SessionVerdict:
    """Validate one session. ``expect_volume=False`` for an index, which has no traded volume."""
    if session.empty:
        raise ValueError(f"{symbol} {session_date}: validate_session called with no bars")
    if tuple(session.columns) != BAR_COLUMNS:
        raise ValueError(f"{symbol} {session_date}: columns {list(session.columns)} != {list(BAR_COLUMNS)}")

    reasons: list[str] = []
    worst = Verdict.CLEAN

    def flag(verdict: Verdict, reason: str) -> None:
        nonlocal worst
        reasons.append(f"{verdict.value}: {reason}")
        if _SEVERITY[verdict] > _SEVERITY[worst]:
            worst = verdict

    # 6. Non-trading day returning data. Raises if the calendar does not cover the date.
    if not calendar.is_trading_day(session_date):
        why = calendar.holiday_name(session_date) or "weekend"
        flag(Verdict.CORRUPT, f"{session_date} is not a trading day ({why}) yet has {len(session)} bars")

    # 4. Monotonic, unique index.
    if not session.index.is_monotonic_increasing:
        flag(Verdict.CORRUPT, "timestamps are not ascending")
    if not session.index.is_unique:
        dup = session.index[session.index.duplicated()][0]
        flag(Verdict.CORRUPT, f"duplicate timestamp {dup.time()}")

    # 2. Window and grid. Every bar starts inside [session_start, session_end) on the interval grid.
    slots = expected_slots(config)
    slot_set = set(slots)
    wrong_date = [ts for ts in session.index if ts.date() != session_date]
    if wrong_date:
        flag(Verdict.CORRUPT, f"{len(wrong_date)} bars dated outside {session_date}, first {wrong_date[0]}")
    off_grid = [ts for ts in session.index if ts.time() not in slot_set]
    if off_grid:
        flag(Verdict.CORRUPT, f"{len(off_grid)} bars outside the session grid, first {off_grid[0].time()}")

    # 3. OHLC sanity.
    ohlc = session[["open", "high", "low", "close"]]
    if ohlc.isna().any().any() or session["volume"].isna().any():
        first = session.index[session.isna().any(axis=1)][0]
        flag(Verdict.CORRUPT, f"NaN values at {first.time()}")
    else:
        nonpositive = (ohlc <= 0).any(axis=1)
        if nonpositive.any():
            flag(Verdict.CORRUPT, f"non-positive price at {session.index[nonpositive][0].time()}")
        bad_low = session["low"] > ohlc[["open", "close"]].min(axis=1)
        if bad_low.any():
            flag(Verdict.CORRUPT, f"low above open/close at {session.index[bad_low][0].time()}")
        bad_high = session["high"] < ohlc[["open", "close"]].max(axis=1)
        if bad_high.any():
            flag(Verdict.CORRUPT, f"high below open/close at {session.index[bad_high][0].time()}")
        if (session["volume"] < 0).any():
            flag(Verdict.CORRUPT, "negative volume")

    # 1. Bar count. A missing tail of at most MAX_COLLAPSED_TAIL_SLOTS on a completed
    #    (not same-day) session is the known close-collapse, not a data hole.
    present = set(ts.time() for ts in session.index)
    missing_times = [t for t in slots if t not in present]
    missing = tuple(t.strftime("%H:%M") for t in missing_times)
    if missing:
        tail_only = (
            dropped_tail_bars == 0
            and len(missing_times) <= MAX_COLLAPSED_TAIL_SLOTS
            and missing_times == slots[-len(missing_times):]
        )
        detail = ", ".join(missing) if len(missing) <= 6 else f"{len(missing)} slots"
        flag(
            Verdict.TAIL_COLLAPSED if tail_only else Verdict.PARTIAL,
            f"{len(session)} of {config.bars_per_session} bars; missing {detail}",
        )

    # 5. Zero-volume bars. Always counted; only judged where volume is meaningful.
    zero_volume = int((session["volume"] == 0).sum())
    if expect_volume and zero_volume > ZERO_VOLUME_SUSPECT_SHARE * len(session):
        flag(Verdict.SUSPECT, f"{zero_volume} of {len(session)} bars have zero volume")

    return SessionVerdict(
        symbol=symbol,
        session_date=session_date,
        verdict=worst,
        bar_count=len(session),
        expected_bars=config.bars_per_session,
        zero_volume_bars=zero_volume,
        missing_slots=missing,
        dropped_tail_bars=dropped_tail_bars,
        reasons=tuple(reasons),
        fetched_at=fetched_at,
    )


def validate_daily_row(
    row: pd.DataFrame,
    symbol: str,
    session_date: date,
    calendar: TradingCalendar,
    fetched_at: datetime,
    previous_close: float | None = None,
    config: Config | None = None,
) -> SessionVerdict:
    """One daily bar is one session. Checks: trading day, OHLC sanity, and - when
    ``previous_close`` is given - whether the open gapped so far from it that an
    unadjusted corporate action is the likelier explanation than a real move.

    Same-day rows are the caller's job to drop (an in-progress daily bar is not a bar)."""
    if len(row) != 1:
        raise ValueError(f"{symbol} {session_date}: expected one daily row, got {len(row)}")
    if tuple(row.columns) != BAR_COLUMNS:
        raise ValueError(f"{symbol} {session_date}: columns {list(row.columns)} != {list(BAR_COLUMNS)}")
    reasons: list[str] = []
    if not calendar.is_trading_day(session_date):
        why = calendar.holiday_name(session_date) or "weekend"
        reasons.append(f"CORRUPT: {session_date} is not a trading day ({why}) yet has a daily bar")
    r = row.iloc[0]
    if any(math.isnan(float(r[c])) for c in ("open", "high", "low", "close")) or math.isnan(float(r["volume"])):
        reasons.append("CORRUPT: NaN values")
    elif min(r["open"], r["high"], r["low"], r["close"]) <= 0:
        reasons.append("CORRUPT: non-positive price")
    elif r["low"] > min(r["open"], r["close"]) or r["high"] < max(r["open"], r["close"]):
        reasons.append("CORRUPT: OHLC inconsistent")
    elif r["volume"] < 0:
        reasons.append("CORRUPT: negative volume")

    verdict = Verdict.CORRUPT if reasons else Verdict.CLEAN
    if verdict is Verdict.CLEAN and previous_close is not None and previous_close > 0:
        low, high = (config or Config()).split_suspect_ratio
        ratio = float(r["open"]) / previous_close
        if not low <= ratio <= high:
            verdict = Verdict.SUSPECT
            reasons.append(
                f"SUSPECT: possible unadjusted corporate action - open {float(r['open']):.2f} is "
                f"{ratio:.2f}x the previous close {previous_close:.2f}, outside {low}-{high}"
            )

    return SessionVerdict(
        symbol=symbol,
        session_date=session_date,
        verdict=verdict,
        bar_count=1,
        expected_bars=1,
        zero_volume_bars=int(r["volume"] == 0),
        missing_slots=(),
        dropped_tail_bars=0,
        reasons=tuple(reasons),
        fetched_at=fetched_at,
    )

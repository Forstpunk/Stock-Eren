"""Opening-range breakout labelling: SUSTAINED / BUSTED / NEITHER.

Definitions (config-driven, defaults in brackets):

- Breakout: the first bar after the opening range whose CLOSE is beyond the range
  boundary. Close, not touch. One per direction per session; the first one.
- SUSTAINED: from the breakout bar onward, a bar's extreme extends >= 0.5 ATR
  (prior-day ATR) beyond the boundary before any bust condition.
- BUSTED: the move never extends >= 0.25 ATR beyond the boundary, and a bar CLOSES
  through the opposite boundary before the bust cutoff [15:15]. A close through the
  opposite boundary necessarily traversed the range, so no separate "closed inside
  first" condition is required.
- A session without a prior-day ATR has no threshold and is not labelled; it is
  counted and reported, never defaulted.
- NEITHER: everything else. Kept as a third class; forcing a binary label distorts
  base rates.

Within a bar, the sustain check (on the extreme) is evaluated before the bust check
(on the close), because the close is the last print of the bar.

``detect_breakout_at`` uses only bars <= i and is tested against the lookahead harness.
``resolve`` walks forward by construction: it is the outcome, not a feature.
"""
from __future__ import annotations

import math
from datetime import date, datetime, time
from enum import Enum
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

from intraday.config import Config
from intraday.indicators import OpeningRange, atr_prior_day, opening_range, session_start_pos
from intraday.store import BarStore
from intraday.trading_calendar import TradingCalendar

BREAKOUTS_FILE = "breakouts.parquet"

Direction = Literal["long", "short"]


class Label(str, Enum):
    SUSTAINED = "SUSTAINED"
    BUSTED = "BUSTED"
    NEITHER = "NEITHER"


class BreakoutEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    session_date: date
    direction: Direction
    breakout_index: int  # position in the frame passed to label_breakouts
    breakout_time: datetime
    minutes_since_open: int
    or_high: float
    or_low: float
    or_width: float
    breakout_close: float
    atr: float  # prior-day ATR that defined the thresholds
    label: Label
    resolved_index: int | None  # bar at which SUSTAINED/BUSTED was decided; None for NEITHER
    resolved_time: datetime | None
    max_extension_atr: float  # max favourable excursion / ATR through session end (outcome-side)


def detect_breakout_at(bars: pd.DataFrame, i: int, config: Config) -> Direction | None:
    """Direction if bar ``i`` is the session's FIRST close beyond the opening range in
    that direction, else None. Uses only bars <= i."""
    if i < session_start_pos(bars, i) + config.opening_range_bars:
        return None  # still inside the opening range: nothing can break out yet
    rng = opening_range(bars, i, config)  # raises on a session without a valid range
    close = float(bars["close"].iloc[i])
    if close > rng.high:
        direction: Direction = "long"
        earlier = bars["close"].iloc[rng.end_index + 1 : i] > rng.high
    elif close < rng.low:
        direction = "short"
        earlier = bars["close"].iloc[rng.end_index + 1 : i] < rng.low
    else:
        return None
    return None if bool(earlier.any()) else direction


def resolve(
    session: pd.DataFrame, breakout_pos: int, direction: Direction, rng: OpeningRange, atr_value: float, config: Config
) -> tuple[Label, int | None, float]:
    """Walk forward from the breakout bar. Returns (label, resolved position, max extension in ATR)."""
    if rng.width <= 0:
        raise ValueError(f"opening range width is {rng.width}; cannot label {session.index[0].date()}")
    if not atr_value > 0:
        raise ValueError(f"ATR {atr_value} is not positive; cannot label {session.index[0].date()}")
    sign = 1.0 if direction == "long" else -1.0
    boundary = rng.high if direction == "long" else rng.low
    opposite = rng.low if direction == "long" else rng.high
    sustain_at = config.sustain_extension_atr * atr_value
    bust_below = config.bust_extension_atr * atr_value

    highs = session["high"].to_numpy(dtype="float64")
    lows = session["low"].to_numpy(dtype="float64")
    closes = session["close"].to_numpy(dtype="float64")
    times = [ts.time() for ts in session.index]

    max_ext = 0.0
    outcome: tuple[Label, int | None] | None = None
    for k in range(breakout_pos, len(session)):
        extreme = highs[k] if direction == "long" else lows[k]
        max_ext = max(max_ext, sign * (extreme - boundary))
        if outcome is None:
            if max_ext >= sustain_at:
                outcome = (Label.SUSTAINED, k)
            elif (
                sign * (closes[k] - opposite) < 0
                and times[k] < config.bust_cutoff
                and max_ext < bust_below
            ):
                outcome = (Label.BUSTED, k)
    label, pos = outcome if outcome is not None else (Label.NEITHER, None)
    return label, pos, max_ext / atr_value


def label_session(
    session: pd.DataFrame, symbol: str, atr_value: float, config: Config, frame_offset: int = 0
) -> list[BreakoutEvent]:
    """All breakout events (at most one per direction) for one session frame.
    ``atr_value`` is the prior-day ATR; ``frame_offset`` is the session's first position in
    the enclosing multi-session frame."""
    if session.empty:
        return []
    if math.isnan(atr_value):
        raise ValueError(f"{symbol} {session.index[0].date()}: no prior-day ATR; session cannot be labelled")
    rng = opening_range(session, len(session) - 1, config)
    events: list[BreakoutEvent] = []
    seen: set[str] = set()
    session_open = datetime.combine(session.index[0].date(), config.session_start, tzinfo=session.index.tz)
    for k in range(rng.end_index + 1, len(session)):
        direction = detect_breakout_at(session, k, config)
        if direction is None or direction in seen:
            continue
        seen.add(direction)
        label, pos, max_ext_atr = resolve(session, k, direction, rng, atr_value, config)
        ts = session.index[k]
        events.append(
            BreakoutEvent(
                symbol=symbol,
                session_date=ts.date(),
                direction=direction,
                breakout_index=frame_offset + k,
                breakout_time=ts.to_pydatetime(),
                minutes_since_open=int((ts.to_pydatetime() - session_open).total_seconds() // 60),
                or_high=rng.high,
                or_low=rng.low,
                or_width=rng.width,
                breakout_close=float(session["close"].iloc[k]),
                atr=atr_value,
                label=label,
                resolved_index=None if pos is None else frame_offset + pos,
                resolved_time=None if pos is None else session.index[pos].to_pydatetime(),
                max_extension_atr=max_ext_atr,
            )
        )
        if len(seen) == 2:
            break
    return events


def label_breakouts(
    bars: pd.DataFrame, symbol: str, daily: pd.DataFrame, calendar: TradingCalendar, config: Config
) -> tuple[list[BreakoutEvent], list[date]]:
    """Label every session in a multi-session research frame. Returns (events, sessions
    skipped for lack of a prior-day ATR)."""
    if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
        raise ValueError("bars must have a tz-aware DatetimeIndex")
    events: list[BreakoutEvent] = []
    skipped: list[date] = []
    offset = 0
    for day, session in bars.groupby(bars.index.date, sort=True):
        atr_value = atr_prior_day(daily, day, calendar, config.atr_period)
        if math.isnan(atr_value):
            skipped.append(day)
        else:
            events.extend(label_session(session, symbol, atr_value, config, frame_offset=offset))
        offset += len(session)
    return events, skipped


def events_to_frame(events: list[BreakoutEvent]) -> pd.DataFrame:
    if not events:
        raise ValueError("no breakout events to tabulate")
    df = pd.DataFrame([e.model_dump() for e in events])
    df["label"] = df["label"].map(lambda v: v.value if isinstance(v, Label) else v)
    return df


def label_universe(
    store: BarStore, daily_store: BarStore, calendar: TradingCalendar, config: Config
) -> tuple[pd.DataFrame, dict[str, list[date]]]:
    """Label every research session of every equity in the store (the index is not traded).
    Returns (events frame, sessions skipped per symbol for lack of a prior-day ATR)."""
    symbols = [s for s in store.symbols() if s != config.index_symbol]
    if not symbols:
        raise ValueError(f"no symbols in {store.root}; run fetch first")
    events: list[BreakoutEvent] = []
    skipped: dict[str, list[date]] = {}
    for symbol in symbols:
        bars = store.read_research(symbol)
        daily = daily_store.read_research(symbol)
        if bars.empty or daily.empty:
            raise ValueError(f"{symbol}: missing intraday or daily research bars")
        ev, sk = label_breakouts(bars, symbol, daily, calendar, config)
        events.extend(ev)
        if sk:
            skipped[symbol] = sk
    return events_to_frame(events), skipped


def save_breakouts(df: pd.DataFrame, data_dir: Path) -> Path:
    path = data_dir / BREAKOUTS_FILE
    df.to_parquet(path, index=False)
    return path


def load_breakouts(data_dir: Path) -> pd.DataFrame:
    path = data_dir / BREAKOUTS_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run label first")
    return pd.read_parquet(path)


def base_rates(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Share of SUSTAINED / BUSTED / NEITHER, overall or per group. Counts included."""
    labels = [l.value for l in Label]
    if by is None:
        counts = df["label"].value_counts().reindex(labels, fill_value=0).to_frame().T
        counts.index = ["all"]
    else:
        counts = pd.crosstab(df[by], df["label"]).reindex(columns=labels, fill_value=0)
    out = counts.copy()
    out["n"] = counts.sum(axis=1)
    for l in labels:
        out[f"{l}_pct"] = (counts[l] / out["n"] * 100).round(1)
    return out

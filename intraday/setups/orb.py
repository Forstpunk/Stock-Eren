"""S1 - ORB + VWAP filter (baseline).

Signal: the session's first close beyond the opening range in a direction (the labelled
breakout event), AND the breakout close is on the breakout side of the session VWAP.
Entry: next bar open, no later than ``last_entry_time``. Stop: ``stop_atr_multiple`` x
prior-day ATR. Exit: stop or EOD (base), plus the 1R-partial variant recorded separately.

``signal_at`` decides from bars <= i only and is tested against the lookahead harness.
"""
from __future__ import annotations

import math

import pandas as pd

from intraday.config import Config
from intraday.indicators import session_vwap
from intraday.labelling import detect_breakout_at
from intraday.setups.common import VARIANTS, build_trade, entry_allowed
from intraday.trades import Direction, Trade

NAME = "orb"


def signal_at(bars: pd.DataFrame, i: int, config: Config) -> Direction | None:
    """Breakout direction at bar ``i`` if it also passes the VWAP filter, else None."""
    direction = detect_breakout_at(bars, i, config)
    if direction is None:
        return None
    vwap = session_vwap(bars, i)
    if math.isnan(vwap):
        return None  # VWAP undefined (no volume yet): the filter cannot pass
    close = float(bars["close"].iloc[i])
    if direction == "long" and close > vwap:
        return "long"
    if direction == "short" and close < vwap:
        return "short"
    return None


def trades_for_session(
    session: pd.DataFrame,
    symbol: str,
    atr_value: float,
    features_by_direction: dict[str, dict[str, float]],
    config: Config,
) -> list[Trade]:
    """All S1 trades (both variants) for one session. ``features_by_direction`` holds the
    Stage 5 vector for each labelled breakout direction in this session."""
    trades: list[Trade] = []
    seen: set[str] = set()
    for i in range(config.opening_range_bars, len(session)):
        direction = signal_at(session, i, config)
        if direction is None or direction in seen:
            continue
        seen.add(direction)
        entry_pos = i + 1
        if not entry_allowed(session, entry_pos, config):
            continue
        if direction not in features_by_direction:
            raise ValueError(f"{symbol} {session.index[i].date()}: no feature vector for the {direction} breakout")
        for variant in VARIANTS:
            trades.append(build_trade(
                NAME, variant, symbol, session, session.index[i].date(), direction, entry_pos,
                atr_value, features_by_direction[direction], config,
            ))
        if len(seen) == 2:
            break
    return trades

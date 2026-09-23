"""Shared setup mechanics: the one exit simulator and the trade builder.

Exit rules (both setups):
- stop: an ATR multiple below/above entry. Filled AT the stop price when a bar trades
  through it during the bar, but at the bar's OPEN when the bar opens already beyond the
  stop - a gap through the level fills where the market reopens, not where the order sat.
  Checked from the entry bar inclusive
- eod: close of the session's last bar (the 15:15 bar on TAIL_COLLAPSED sessions,
  whose close is the official close)
- partial_1r variant: half the position exits at +1R (entry + risk) the first time a
  bar's extreme reaches it; the stop then moves to entry; the remainder exits at that
  breakeven stop or EOD. The recorded exit price is the size-weighted blend, so R
  arithmetic and costs flow through the same ``evaluate`` path. Costs assume one exit
  order; at Rs 50,000 the brokerage cap does not bind so two orders cost the same.
- if a bar touches both the stop and the target, the stop is assumed to fill first
  (worst case).
"""
from __future__ import annotations

from datetime import date
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

from intraday.config import Config
from intraday.trades import Direction, ExitReason, Trade

Variant = Literal["base", "partial_1r"]
VARIANTS: tuple[Variant, ...] = ("base", "partial_1r")


class Exit(BaseModel):
    model_config = ConfigDict(frozen=True)

    exit_index: int
    exit_price: float
    reason: ExitReason
    mfe_price: float


def simulate_exit(session: pd.DataFrame, entry_pos: int, direction: Direction, entry_price: float, stop: float, variant: Variant) -> Exit:
    if not 0 <= entry_pos < len(session):
        raise IndexError(f"entry position {entry_pos} outside session of {len(session)} bars")
    sign = 1.0 if direction == "long" else -1.0
    risk = sign * (entry_price - stop)
    if risk <= 0:
        raise ValueError(f"stop {stop} is not on the losing side of entry {entry_price} for a {direction}")
    highs = session["high"].to_numpy(dtype="float64")
    lows = session["low"].to_numpy(dtype="float64")
    closes = session["close"].to_numpy(dtype="float64")
    opens = session["open"].to_numpy(dtype="float64")
    last = len(session) - 1
    target = entry_price + sign * risk
    mfe = entry_price
    partial_done = False
    active_stop = stop

    def favourable(k: int) -> float:
        return highs[k] if direction == "long" else lows[k]

    def adverse(k: int) -> float:
        return lows[k] if direction == "long" else highs[k]

    for k in range(entry_pos, last + 1):
        mfe = max(mfe, favourable(k)) if direction == "long" else min(mfe, favourable(k))
        stop_hit = sign * (adverse(k) - active_stop) <= 0
        if stop_hit:
            # A bar that opens beyond the stop gapped through it: the fill is the open,
            # which is worse than the stop. Taking the stop price here would credit the
            # trade with a price that was never available.
            gapped_through = sign * (opens[k] - active_stop) <= 0
            final = opens[k] if gapped_through else active_stop
            reason: ExitReason = "stop"
            return Exit(exit_index=k, exit_price=_blend(partial_done, target, final), reason=reason, mfe_price=mfe)
        if variant == "partial_1r" and not partial_done and sign * (favourable(k) - target) >= 0:
            partial_done = True
            active_stop = entry_price
    final = closes[last]
    reason = "target" if partial_done else "eod"
    return Exit(exit_index=last, exit_price=_blend(partial_done, target, final), reason=reason, mfe_price=mfe)


def _blend(partial_done: bool, target: float, final: float) -> float:
    return 0.5 * target + 0.5 * final if partial_done else final


def build_trade(
    setup: str,
    variant: Variant,
    symbol: str,
    session: pd.DataFrame,
    session_date: date,
    direction: Direction,
    entry_pos: int,
    atr_value: float,
    features: dict[str, float],
    config: Config,
) -> Trade:
    """Entry at the open of ``entry_pos``, stop at the ATR multiple, exit per ``variant``."""
    entry_price = float(session["open"].iloc[entry_pos])
    sign = 1.0 if direction == "long" else -1.0
    stop = entry_price - sign * config.stop_atr_multiple * atr_value
    exit_ = simulate_exit(session, entry_pos, direction, entry_price, stop, variant)
    return Trade(
        setup=setup,
        variant=variant,
        symbol=symbol,
        session_date=session_date,
        direction=direction,
        entry_index=entry_pos,
        entry_time=session.index[entry_pos].to_pydatetime(),
        entry_price=entry_price,
        exit_index=exit_.exit_index,
        exit_time=session.index[exit_.exit_index].to_pydatetime(),
        exit_price=exit_.exit_price,
        exit_reason=exit_.reason,
        stop_price=stop,
        atr=atr_value,
        stop_atr_multiple=config.stop_atr_multiple,
        mfe_price=exit_.mfe_price,
        features=features,
    )


def entry_allowed(session: pd.DataFrame, entry_pos: int, config: Config) -> bool:
    """The entry bar exists and is not later than the last-entry time."""
    return entry_pos < len(session) and session.index[entry_pos].time() <= config.last_entry_time

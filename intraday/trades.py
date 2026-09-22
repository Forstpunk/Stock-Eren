"""Trade records and the one evaluation path from prices to net R.

A ``Trade`` is what a setup (or the random benchmark) produces: reference entry and
exit prices, the stop that defines R, the ATR the stop was derived from, and the full
feature vector at decision time. ``evaluate`` turns it into a ``TradeResult`` at one
slippage level: gross P&L, itemised costs, net P&L, and both in R.

R = quantity x |entry - stop|. Quantity = floor(position_inr / entry). Slippage is a
cost item (bps x turnover), never a price adjustment, so every number ties back to the
same breakdown.
"""
from __future__ import annotations

import math
from datetime import date, datetime
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

from intraday.config import Config
from intraday.costs import CostBreakdown, round_trip_costs

Direction = Literal["long", "short"]
ExitReason = Literal["stop", "eod", "target", "time"]


class Trade(BaseModel):
    model_config = ConfigDict(frozen=True)

    setup: str
    variant: str
    symbol: str
    session_date: date
    direction: Direction
    entry_index: int
    entry_time: datetime
    entry_price: float
    exit_index: int
    exit_time: datetime
    exit_price: float
    exit_reason: ExitReason
    stop_price: float
    atr: float
    stop_atr_multiple: float
    mfe_price: float  # best price seen between entry and exit, in the trade's favour
    features: dict[str, float]


class TradeResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    trade: Trade
    slippage_bps: int
    quantity: int
    risk_inr: float
    buy_turnover: float
    sell_turnover: float
    gross_pnl: float
    costs: CostBreakdown
    net_pnl: float
    r_gross: float
    r_net: float
    mfe_r: float
    duration_bars: int


def quantity_for(entry_price: float, config: Config) -> int:
    if entry_price <= 0:
        raise ValueError(f"entry price must be positive, got {entry_price}")
    qty = math.floor(config.position_inr / entry_price)
    if qty < 1:
        raise ValueError(f"position {config.position_inr} buys no shares at {entry_price}")
    return qty


def evaluate(trade: Trade, slippage_bps: int, config: Config) -> TradeResult:
    if trade.exit_index < trade.entry_index:
        raise ValueError(f"{trade.symbol} {trade.session_date}: exit index precedes entry index")
    # exit_index == entry_index is a same-bar stop-out (entry at open, stop hit intrabar)
    sign = 1.0 if trade.direction == "long" else -1.0
    risk_per_share = sign * (trade.entry_price - trade.stop_price)
    if risk_per_share <= 0:
        raise ValueError(
            f"{trade.symbol} {trade.session_date}: stop {trade.stop_price} is not on the losing side of "
            f"entry {trade.entry_price} for a {trade.direction}"
        )
    qty = quantity_for(trade.entry_price, config)
    risk_inr = qty * risk_per_share
    if trade.direction == "long":
        buy_turnover, sell_turnover = qty * trade.entry_price, qty * trade.exit_price
    else:
        sell_turnover, buy_turnover = qty * trade.entry_price, qty * trade.exit_price
    gross = sign * (trade.exit_price - trade.entry_price) * qty
    costs = round_trip_costs(buy_turnover, sell_turnover, slippage_bps)
    net = gross - costs.total
    mfe = sign * (trade.mfe_price - trade.entry_price) * qty
    return TradeResult(
        trade=trade,
        slippage_bps=slippage_bps,
        quantity=qty,
        risk_inr=risk_inr,
        buy_turnover=buy_turnover,
        sell_turnover=sell_turnover,
        gross_pnl=gross,
        costs=costs,
        net_pnl=net,
        r_gross=gross / risk_inr,
        r_net=net / risk_inr,
        mfe_r=mfe / risk_inr,
        duration_bars=trade.exit_index - trade.entry_index,
    )


def results_frame(results: list[TradeResult]) -> pd.DataFrame:
    """Flat table: one row per result, trade fields + features + evaluation."""
    if not results:
        raise ValueError("no trade results to tabulate")
    rows = []
    for r in results:
        t = r.trade
        rows.append({
            "setup": t.setup, "variant": t.variant, "symbol": t.symbol, "session_date": t.session_date,
            "direction": t.direction, "entry_time": t.entry_time, "exit_time": t.exit_time,
            "entry_price": t.entry_price, "exit_price": t.exit_price, "stop_price": t.stop_price,
            "exit_reason": t.exit_reason, "atr": t.atr, "stop_atr_multiple": t.stop_atr_multiple,
            "slippage_bps": r.slippage_bps, "quantity": r.quantity, "risk_inr": r.risk_inr,
            "gross_pnl": r.gross_pnl, "cost_total": r.costs.total, "cost_slippage": r.costs.slippage,
            "net_pnl": r.net_pnl, "r_gross": r.r_gross, "r_net": r.r_net, "mfe_r": r.mfe_r,
            "duration_bars": r.duration_bars,
            **t.features,
        })
    return pd.DataFrame(rows)

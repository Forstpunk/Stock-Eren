"""Itemised NSE intraday cost model (retail discount broker). Never a single number.

Per order, both sides unless stated:
- brokerage:      min(20, 0.03% x turnover)
- STT:            0.025% of turnover, SELL side only
- exchange txn:   0.00297% of turnover
- stamp duty:     0.003% of turnover, BUY side only
- SEBI fee:       0.0001% of turnover
- GST:            18% of (brokerage + exchange txn + SEBI fee)
- slippage:       bps x turnover

A round trip has one buy leg and one sell leg regardless of direction; a short simply
sells first. STT and stamp attach to the side, not to the order of the legs.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

BROKERAGE_CAP_INR = 20.0
BROKERAGE_RATE = 0.0003
STT_RATE_SELL = 0.00025
EXCHANGE_RATE = 0.0000297
STAMP_RATE_BUY = 0.00003
SEBI_RATE = 0.000001
GST_RATE = 0.18


class CostBreakdown(BaseModel):
    model_config = ConfigDict(frozen=True)

    buy_turnover: float
    sell_turnover: float
    slippage_bps: int
    brokerage_buy: float
    brokerage_sell: float
    stt: float
    exchange_buy: float
    exchange_sell: float
    stamp: float
    sebi_buy: float
    sebi_sell: float
    gst: float
    slippage_buy: float
    slippage_sell: float

    @property
    def statutory(self) -> float:
        """Everything except slippage."""
        return (
            self.brokerage_buy + self.brokerage_sell + self.stt + self.exchange_buy + self.exchange_sell
            + self.stamp + self.sebi_buy + self.sebi_sell + self.gst
        )

    @property
    def slippage(self) -> float:
        return self.slippage_buy + self.slippage_sell

    @property
    def total(self) -> float:
        return self.statutory + self.slippage

    def items(self) -> dict[str, float]:
        return {
            "brokerage_buy": self.brokerage_buy,
            "brokerage_sell": self.brokerage_sell,
            "stt": self.stt,
            "exchange_buy": self.exchange_buy,
            "exchange_sell": self.exchange_sell,
            "stamp": self.stamp,
            "sebi_buy": self.sebi_buy,
            "sebi_sell": self.sebi_sell,
            "gst": self.gst,
            "slippage_buy": self.slippage_buy,
            "slippage_sell": self.slippage_sell,
            "total": self.total,
        }


def brokerage(turnover: float) -> float:
    return min(BROKERAGE_CAP_INR, BROKERAGE_RATE * turnover)


def round_trip_costs(buy_turnover: float, sell_turnover: float, slippage_bps: int) -> CostBreakdown:
    if buy_turnover <= 0 or sell_turnover <= 0:
        raise ValueError(f"turnover must be positive, got buy={buy_turnover} sell={sell_turnover}")
    if slippage_bps < 0:
        raise ValueError(f"slippage_bps must be non-negative, got {slippage_bps}")
    b_buy, b_sell = brokerage(buy_turnover), brokerage(sell_turnover)
    x_buy, x_sell = EXCHANGE_RATE * buy_turnover, EXCHANGE_RATE * sell_turnover
    return CostBreakdown(
        buy_turnover=buy_turnover,
        sell_turnover=sell_turnover,
        slippage_bps=slippage_bps,
        brokerage_buy=b_buy,
        brokerage_sell=b_sell,
        stt=STT_RATE_SELL * sell_turnover,
        exchange_buy=x_buy,
        exchange_sell=x_sell,
        stamp=STAMP_RATE_BUY * buy_turnover,
        sebi_buy=SEBI_RATE * buy_turnover,
        sebi_sell=SEBI_RATE * sell_turnover,
        gst=GST_RATE * (b_buy + b_sell + x_buy + x_sell + SEBI_RATE * (buy_turnover + sell_turnover)),
        slippage_buy=slippage_bps / 10_000 * buy_turnover,
        slippage_sell=slippage_bps / 10_000 * sell_turnover,
    )

"""Itemised NSE intraday cost model (retail discount broker). Never a single number.

Per order, both sides unless stated:
- brokerage:      min(20, 0.03% x turnover)
- STT:            0.025% of turnover, SELL side only
- exchange txn:   0.00297% of turnover
- stamp duty:     0.003% of turnover, BUY side only
- SEBI fee:       0.0001% of turnover
- GST:            18% of (brokerage + exchange txn + SEBI fee)
- slippage:       bps x turnover, optionally widened by how large the order is for the bar

A round trip has one buy leg and one sell leg regardless of direction; a short simply
sells first. STT and stamp attach to the side, not to the order of the legs.

On slippage. A flat bps figure says a Rs 50,000 order costs the same in a bar that traded
Rs 2 crore as in one that traded Rs 5 lakh, which is false and false in the direction that
flatters a backtest: the thin bars are exactly where a strategy's fills are worst.
``participation_slippage_bps`` widens the assumed slippage with the order's share of the
bar's traded value, using a square-root impact term - the standard shape, chosen because it
is standard and not fitted to anything here. It is off unless a bar value is supplied, so
the flat model remains the default and the comparison between them stays visible.
"""
from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict

BROKERAGE_CAP_INR = 20.0
BROKERAGE_RATE = 0.0003
STT_RATE_SELL = 0.00025
EXCHANGE_RATE = 0.0000297
STAMP_RATE_BUY = 0.00003
SEBI_RATE = 0.000001
GST_RATE = 0.18

# Impact coefficient for the square-root model: extra bps = k * sqrt(order / bar value).
# A round number, taken as a modelling assumption rather than a measurement, because
# nothing in this data set can calibrate it. It is deliberately not tuned.
IMPACT_COEFFICIENT_BPS = 10.0
MAX_PARTICIPATION = 1.0


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



def participation_slippage_bps(order_value: float, bar_value: float, base_bps: float) -> float:
    """Slippage in bps for an order of ``order_value`` into a bar that traded ``bar_value``.

    Returns ``base_bps`` plus a square-root impact term in the order's participation rate.
    An order at 1% of a bar's value adds about 1 bp; at 25%, about 5; at 100%, about 10.
    Participation above 1.0 is capped, because beyond that the model has nothing useful to
    say and pretending otherwise would invent precision.

    With no bar value known, the flat assumption is returned unchanged rather than guessed at.
    """
    if base_bps < 0:
        raise ValueError(f"base_bps must be non-negative, got {base_bps}")
    if order_value <= 0:
        raise ValueError(f"order value must be positive, got {order_value}")
    if bar_value is None or not bar_value > 0:
        return base_bps
    participation = min(order_value / bar_value, MAX_PARTICIPATION)
    return base_bps + IMPACT_COEFFICIENT_BPS * math.sqrt(participation)

"""Matched random benchmark. Every strategy trade gets a random twin:

- a random eligible symbol (one with a research session that day), same session
- a random direction
- a random entry bar inside the setup's entry window
- the same holding duration in bars (capped at the session's last bar)
- the same R definition: stop = same ATR multiple of the twin symbol's prior-day ATR
- the same costs at the same slippage

The twin holds for the duration without a stop and exits at the close of its exit bar;
the strategy's exit rule already shaped its duration, so matching the duration isolates
entry selection. Deterministic under ``seed``. Edge is the mean net R difference over
the pairs with a paired bootstrap CI.
"""
from __future__ import annotations

import math
from datetime import date, time

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from intraday.config import Config
from intraday.indicators import atr_prior_day
from intraday.stats import mean_stat, session_bootstrap
from intraday.store import BarStore
from intraday.trades import Trade, TradeResult, evaluate
from intraday.trading_calendar import TradingCalendar

BENCHMARK_SETUP = "random"


class SessionUniverse:
    """Bars and prior-day ATR per (symbol, date) for every equity in the store."""

    def __init__(self, store: BarStore, daily_store: BarStore, calendar: TradingCalendar, config: Config) -> None:
        self.config = config
        self.sessions: dict[tuple[str, date], pd.DataFrame] = {}
        self.atr: dict[tuple[str, date], float] = {}
        symbols = [s for s in store.symbols() if s != config.index_symbol]
        if not symbols:
            raise ValueError("store has no equity symbols")
        for symbol in symbols:
            bars = store.read_research(symbol)
            daily = daily_store.read_research(symbol)
            for day, frame in bars.groupby(bars.index.date):
                a = atr_prior_day(daily, day, calendar, config.atr_period)
                if math.isnan(a) or a <= 0:
                    continue  # no ATR -> no R -> not eligible
                self.sessions[(symbol, day)] = frame
                self.atr[(symbol, day)] = a

    def eligible(self, day: date) -> list[str]:
        return sorted(s for (s, d) in self.sessions if d == day)


def random_twin(
    strategy: TradeResult,
    universe: SessionUniverse,
    window: tuple[time, time],
    rng: np.random.Generator,
    config: Config,
) -> Trade:
    day = strategy.trade.session_date
    candidates = universe.eligible(day)
    if not candidates:
        raise ValueError(f"no eligible symbol on {day} for the random benchmark")
    symbol = candidates[int(rng.integers(0, len(candidates)))]
    bars = universe.sessions[(symbol, day)]
    times = np.array([ts.time() for ts in bars.index])
    in_window = np.flatnonzero((times >= window[0]) & (times <= window[1]))
    if len(in_window) == 0:
        raise ValueError(f"{symbol} {day}: no bars inside the entry window {window}")
    duration = strategy.duration_bars
    latest = len(bars) - 1 - duration
    feasible = in_window[in_window <= latest]
    entry = int(feasible[int(rng.integers(0, len(feasible)))]) if len(feasible) else int(in_window[0])
    exit_ = min(entry + duration, len(bars) - 1)  # duration 0 = enter at open, exit at that bar's close
    direction = "long" if rng.random() < 0.5 else "short"
    sign = 1.0 if direction == "long" else -1.0
    entry_price = float(bars["open"].iloc[entry])
    atr_value = universe.atr[(symbol, day)]
    stop = entry_price - sign * config.stop_atr_multiple * atr_value
    path = bars.iloc[entry : exit_ + 1]
    mfe = float(path["high"].max()) if direction == "long" else float(path["low"].min())
    return Trade(
        setup=BENCHMARK_SETUP,
        variant=strategy.trade.variant,
        symbol=symbol,
        session_date=day,
        direction=direction,
        entry_index=entry,
        entry_time=bars.index[entry].to_pydatetime(),
        entry_price=entry_price,
        exit_index=exit_,
        exit_time=bars.index[exit_].to_pydatetime(),
        exit_price=float(bars["close"].iloc[exit_]),
        exit_reason="time",
        stop_price=stop,
        atr=atr_value,
        stop_atr_multiple=config.stop_atr_multiple,
        mfe_price=mfe,
        features={},
    )


def matched_random(
    strategy: list[TradeResult],
    universe: SessionUniverse,
    window: tuple[time, time],
    config: Config,
    seed: int,
) -> list[TradeResult]:
    """One random twin per strategy result, evaluated at the same slippage."""
    if not strategy:
        raise ValueError("no strategy trades to benchmark")
    rng = np.random.default_rng(seed)
    return [evaluate(random_twin(r, universe, window, rng, config), r.slippage_bps, config) for r in strategy]


class EdgeVsRandom(BaseModel):
    model_config = ConfigDict(frozen=True)

    n: int
    slippage_bps: int
    mean_strategy_r: float
    mean_random_r: float
    edge_r: float
    ci_low: float
    ci_high: float
    indistinguishable_from_zero: bool
    statement: str


def edge_vs_random(strategy: list[TradeResult], random: list[TradeResult], config: Config, seed: int) -> EdgeVsRandom:
    if len(strategy) != len(random) or not strategy:
        raise ValueError("strategy and random results must be non-empty and paired")
    bps = {r.slippage_bps for r in strategy} | {r.slippage_bps for r in random}
    if len(bps) != 1:
        raise ValueError(f"mixed slippage levels {bps}; benchmark one level at a time")
    s = np.array([r.r_net for r in strategy])
    b = np.array([r.r_net for r in random])
    diff = s - b
    # A pair belongs to the strategy trade's session; same-day pairs share that day's shock.
    sessions = np.array([str(r.trade.session_date) for r in strategy])
    rng = np.random.default_rng(seed)
    n = len(diff)
    lo, hi, _ = session_bootstrap(sessions, mean_stat(diff), config.bootstrap_n, rng)
    edge = float(diff.mean())
    flat = lo <= 0 <= hi
    statement = (
        f"{n} trades at {bps.pop()} bps: strategy {s.mean():+.3f}R vs matched random {b.mean():+.3f}R, "
        f"edge {edge:+.3f}R (95% CI {lo:+.3f} to {hi:+.3f}). "
        + ("The interval spans zero: no edge over random entries is detected." if flat
           else "The interval excludes zero.")
    )
    return EdgeVsRandom(
        n=n, slippage_bps=strategy[0].slippage_bps,
        mean_strategy_r=float(s.mean()), mean_random_r=float(b.mean()),
        edge_r=edge, ci_low=lo, ci_high=hi, indistinguishable_from_zero=flat, statement=statement,
    )

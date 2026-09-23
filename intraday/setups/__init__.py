"""Setups. ``run_setup`` builds every trade for a named setup over the research store,
attaching each breakout's Stage 5 feature vector, and returns the trades plus a count of
signals skipped for entering after ``last_entry_time``.

- orb        S1, ORB + VWAP filter (baseline). Not gated.
- failed_orb S2, failed-ORB reversal. Gated: refuses unless Stage 6 says "signal".
"""
from __future__ import annotations

import math
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict

from intraday.config import Config
from intraday.features import ALL_COLUMNS
from intraday.indicators import atr_prior_day
from intraday.setups import failed_orb, orb
from intraday.store import BarStore
from intraday.trades import Trade
from intraday.trading_calendar import TradingCalendar

SessionRunner = Callable[[pd.DataFrame, str, float, dict[str, dict[str, float]], Config], list[Trade]]

SETUPS: dict[str, SessionRunner] = {
    orb.NAME: orb.trades_for_session,
    failed_orb.NAME: failed_orb.trades_for_session,
}


class SetupRun(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    setup: str
    trades: list[Trade]
    sessions_seen: int
    sessions_without_atr: int
    skipped_late_entries: int


def features_by_session(features: pd.DataFrame) -> dict[tuple[str, date], dict[str, dict[str, float]]]:
    out: dict[tuple[str, date], dict[str, dict[str, float]]] = {}
    for _, row in features.iterrows():
        key = (str(row["symbol"]), pd.Timestamp(row["session_date"]).date())
        out.setdefault(key, {})[str(row["direction"])] = {f: float(row[f]) for f in ALL_COLUMNS}
    return out


def run_setup(
    name: str,
    store: BarStore,
    daily_store: BarStore,
    features: pd.DataFrame,
    calendar: TradingCalendar,
    config: Config,
    data_dir: Path,
) -> SetupRun:
    if name not in SETUPS:
        raise ValueError(f"unknown setup {name!r}; known: {sorted(SETUPS)}")
    if name == failed_orb.NAME:
        failed_orb.assert_gate_open(data_dir, config)
    runner = SETUPS[name]
    feats = features_by_session(features)
    symbols = [s for s in store.symbols() if s != config.index_symbol]
    trades: list[Trade] = []
    seen = no_atr = late = 0
    for symbol in symbols:
        bars = store.read_research(symbol)
        daily = daily_store.read_research(symbol)
        for day, session in bars.groupby(bars.index.date, sort=True):
            seen += 1
            atr_value = atr_prior_day(daily, day, calendar, config.atr_period)
            if math.isnan(atr_value):
                no_atr += 1
                continue
            before = len(trades)
            trades.extend(runner(session, symbol, atr_value, feats.get((symbol, day), {}), config))
            late += _late_signals(name, session, atr_value, config) - (len(trades) - before) // 2
    return SetupRun(setup=name, trades=trades, sessions_seen=seen, sessions_without_atr=no_atr, skipped_late_entries=late)


def _late_signals(name: str, session: pd.DataFrame, atr_value: float, config: Config) -> int:
    """Number of distinct signal directions in the session (regardless of entry time)."""
    seen: set[str] = set()
    for i in range(config.opening_range_bars, len(session)):
        d = orb.signal_at(session, i, config) if name == orb.NAME else failed_orb.signal_at(session, i, atr_value, config)
        if d is not None:
            seen.add(d)
    return len(seen)

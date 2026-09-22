"""Feature extraction: one feature per suspected mechanism, computed at the decision
moment (the close of the breakout bar) from bars at positions <= the breakout index,
daily bars strictly before the session date, and index bars at or before the same
timestamp. Undefined values are NaN, never defaults.

Signed features are oriented so that a positive value means "in the breakout direction":
``vwap_distance_sigma``, ``index_agreement``, ``gap_atr``, ``open_position_in_prior_range``.
"""
from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from intraday.config import Config
from intraday.indicators import (
    atr_prior_day,
    gap_pct,
    opening_range,
    previous_trading_day,
    rvol_at_time,
    rvol_bar,
    session_sigma,
    session_start_pos,
    session_vwap,
)
from intraday.labelling import Direction
from intraday.store import BarStore
from intraday.trading_calendar import TradingCalendar

FEATURES_FILE = "features.parquet"

FEATURE_NAMES: tuple[str, ...] = (
    "rvol_breakout_bar",
    "rvol_open_15m",
    "vwap_distance_sigma",
    "or_width_atr",
    "prior_touches",
    "index_agreement",
    "gap_atr",
    "open_position_in_prior_range",
    "bar_body_ratio",
    "minutes_since_open",
    "atr_pct",
)

FEATURE_MECHANISM: dict[str, str] = {
    "rvol_breakout_bar": "no real participation behind the break",
    "rvol_open_15m": "was the session busy at all",
    "vwap_distance_sigma": "move already extended",
    "or_width_atr": "wide range = energy spent",
    "prior_touches": "stops clustered at a repeatedly-tested level",
    "index_agreement": "Nifty direction in the same bar",
    "gap_atr": "gap-fade pressure",
    "open_position_in_prior_range": "where today opened vs yesterday",
    "bar_body_ratio": "absorption: volume without price progress",
    "minutes_since_open": "liquidity by time of day",
    "atr_pct": "baseline volatility",
}


class FeatureContext(BaseModel):
    """Everything a feature may consult besides the intraday frame."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    daily: pd.DataFrame  # the symbol's daily research bars (any span; only rows < session date are used)
    index_bars: pd.DataFrame  # the index's intraday research bars
    calendar: TradingCalendar
    config: Config


def _prior_daily_row(daily: pd.DataFrame, session_date: date, calendar: TradingCalendar) -> int | None:
    """Position of the previous trading day's daily row, or None if that exact day is absent."""
    prev = previous_trading_day(session_date, calendar)
    dates = daily.index.date
    hits = np.flatnonzero(dates == prev)
    return int(hits[0]) if len(hits) else None


def compute_features(bars: pd.DataFrame, i: int, direction: Direction, ctx: FeatureContext) -> dict[str, float]:
    """All features for a breakout whose breakout bar is at position ``i`` of ``bars``."""
    cfg = ctx.config
    sign = 1.0 if direction == "long" else -1.0
    ts = bars.index[i]
    session_date = ts.date()
    start = session_start_pos(bars, i)
    rng = opening_range(bars, i, cfg)
    row = bars.iloc[i]
    close = float(row["close"])

    # --- daily context: ATR and prior range, strictly from before the session date
    prior_pos = _prior_daily_row(ctx.daily, session_date, ctx.calendar)
    if prior_pos is None:
        atr_value = math.nan
        prior_close = prior_high = prior_low = math.nan
    else:
        atr_value = atr_prior_day(ctx.daily, session_date, ctx.calendar, cfg.atr_period)
        prior = ctx.daily.iloc[prior_pos]
        prior_close, prior_high, prior_low = float(prior["close"]), float(prior["high"]), float(prior["low"])

    # --- vwap distance in session-sigma units
    vwap = session_vwap(bars, i)
    sigma = session_sigma(bars, i)
    if math.isnan(vwap) or math.isnan(sigma) or sigma <= 0:
        vwap_distance_sigma = math.nan
    else:
        vwap_distance_sigma = sign * ((close - vwap) / vwap) / sigma

    # --- prior touches of the boundary between the range and the breakout bar
    between = bars.iloc[rng.end_index + 1 : i]
    if direction == "long":
        prior_touches = int((between["high"] >= rng.high).sum())
    else:
        prior_touches = int((between["low"] <= rng.low).sum())

    # --- index agreement: the index bar at the same timestamp, signed by direction
    if ts in ctx.index_bars.index:
        ib = ctx.index_bars.loc[ts]
        index_agreement = sign * (float(ib["close"]) / float(ib["open"]) - 1.0) * 1e4  # bps
    else:
        index_agreement = math.nan

    # --- gap in ATR units, signed by direction. gap_pct is relative to the intraday prior
    #     close, so the price gap is derived from it alone (never from the daily close).
    today_open = float(bars["open"].iloc[start])
    gap = gap_pct(bars, i, ctx.calendar)
    if math.isnan(gap) or math.isnan(atr_value) or atr_value <= 0:
        gap_atr = math.nan
    else:
        gap_price = today_open - today_open / (1.0 + gap / 100.0)
        gap_atr = sign * gap_price / atr_value

    # --- where today opened inside yesterday's range, measured towards the breakout direction
    if math.isnan(prior_high) or prior_high <= prior_low:
        open_position = math.nan
    else:
        raw = (today_open - prior_low) / (prior_high - prior_low)
        open_position = raw if direction == "long" else 1.0 - raw

    # --- breakout bar body
    bar_range = float(row["high"]) - float(row["low"])
    bar_body_ratio = math.nan if bar_range <= 0 else abs(close - float(row["open"])) / bar_range

    session_open = pd.Timestamp.combine(session_date, cfg.session_start).tz_localize(bars.index.tz)
    minutes_since_open = float((ts - session_open).total_seconds() // 60)

    return {
        "rvol_breakout_bar": rvol_bar(bars, i, cfg.rvol_lookback_sessions),
        "rvol_open_15m": rvol_at_time(bars, rng.end_index, cfg.rvol_lookback_sessions),
        "vwap_distance_sigma": vwap_distance_sigma,
        "or_width_atr": math.nan if math.isnan(atr_value) or atr_value <= 0 else rng.width / atr_value,
        "prior_touches": float(prior_touches),
        "index_agreement": index_agreement,
        "gap_atr": gap_atr,
        "open_position_in_prior_range": open_position,
        "bar_body_ratio": bar_body_ratio,
        "minutes_since_open": minutes_since_open,
        "atr_pct": math.nan if math.isnan(atr_value) or math.isnan(prior_close) else atr_value / prior_close * 100.0,
    }


def build_feature_table(
    breakouts: pd.DataFrame,
    store: BarStore,
    daily_store: BarStore,
    calendar: TradingCalendar,
    config: Config,
) -> pd.DataFrame:
    """One row per breakout event: identifiers, label, outcome-side extension, and features."""
    if breakouts.empty:
        raise ValueError("no breakouts to featurise")
    index_bars = store.read_research(config.index_symbol)
    if index_bars.empty:
        raise ValueError(f"no research bars for index {config.index_symbol}")
    rows: list[dict[str, object]] = []
    for symbol, group in breakouts.groupby("symbol", sort=True):
        bars = store.read_research(symbol)
        daily = daily_store.read_research(symbol)
        if bars.empty or daily.empty:
            raise ValueError(f"{symbol}: missing intraday or daily research bars")
        ctx = FeatureContext(daily=daily, index_bars=index_bars, calendar=calendar, config=config)
        for _, e in group.iterrows():
            i = int(e["breakout_index"])
            if bars.index[i] != pd.Timestamp(e["breakout_time"]):
                raise ValueError(
                    f"{symbol} {e['session_date']}: breakout_index {i} no longer points at "
                    f"{e['breakout_time']}; re-run label after fetch"
                )
            feats = compute_features(bars, i, e["direction"], ctx)
            if feats["minutes_since_open"] != float(e["minutes_since_open"]):
                raise ValueError(f"{symbol} {e['session_date']}: minutes_since_open disagrees with label")
            rows.append({
                "symbol": symbol,
                "session_date": e["session_date"],
                "direction": e["direction"],
                "breakout_index": i,
                "breakout_time": e["breakout_time"],
                "label": e["label"],
                "max_extension_atr": e["max_extension_atr"],
                **feats,
            })
    return pd.DataFrame(rows)


def save_features(df: pd.DataFrame, data_dir: Path) -> Path:
    path = data_dir / FEATURES_FILE
    df.to_parquet(path, index=False)
    return path


def load_features(data_dir: Path) -> pd.DataFrame:
    path = data_dir / FEATURES_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run features first")
    return pd.read_parquet(path)

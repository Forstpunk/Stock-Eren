"""Feature extraction.

Three features, one per mechanism actually worth testing. Each is computed at the
decision moment (the close of the breakout bar) from bars at positions <= the breakout
index. Undefined values are NaN, never defaults.

    rvol_open_15m       was the session busy at all
    rvol_breakout_bar   was there real participation behind the break
    bar_body_ratio      absorption: volume without price progress

Two more columns ride along for segmentation only. They are NOT tested as mechanisms,
because both are partly definitional: a breakout's outcome depends on the range width
(the bust definition is a width-scaled distance) and on how much session is left.

    or_width_atr        opening range width in ATR units
    minutes_since_open  time of day
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict

from intraday.config import Config
from intraday.indicators import atr_prior_day, opening_range, rvol_at_time, rvol_bar
from intraday.labelling import Direction
from intraday.store import BarStore
from intraday.trading_calendar import TradingCalendar

FEATURES_FILE = "features.parquet"

FEATURE_NAMES: tuple[str, ...] = ("rvol_open_15m", "rvol_breakout_bar", "bar_body_ratio")

FEATURE_MECHANISM: dict[str, str] = {
    "rvol_open_15m": "was the session busy at all",
    "rvol_breakout_bar": "real participation behind the break",
    "bar_body_ratio": "absorption: volume without price progress",
}

# What each feature is expected to show, in plain words, for the report text.
FEATURE_EXPECTATION: dict[str, str] = {
    "rvol_open_15m": "quiet opens should fail more often",
    "rvol_breakout_bar": "breaks on thin volume should fail more often",
    "bar_body_ratio": "small bodies (price stalling) should fail more often",
}

CONTEXT_NAMES: tuple[str, ...] = ("or_width_atr", "minutes_since_open")
ALL_COLUMNS: tuple[str, ...] = FEATURE_NAMES + CONTEXT_NAMES


class FeatureContext(BaseModel):
    """Everything a feature may consult besides the intraday frame."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    daily: pd.DataFrame  # the symbol's daily research bars; only rows before the session date are used
    calendar: TradingCalendar
    config: Config


def compute_features(bars: pd.DataFrame, i: int, direction: Direction, ctx: FeatureContext) -> dict[str, float]:
    """Features and context columns for a breakout whose breakout bar is at position ``i``."""
    cfg = ctx.config
    ts = bars.index[i]
    rng = opening_range(bars, i, cfg)
    row = bars.iloc[i]

    atr_value = atr_prior_day(ctx.daily, ts.date(), ctx.calendar, cfg.atr_period)
    bar_range = float(row["high"]) - float(row["low"])
    session_open = pd.Timestamp.combine(ts.date(), cfg.session_start).tz_localize(bars.index.tz)

    return {
        "rvol_open_15m": rvol_at_time(bars, rng.end_index, cfg.rvol_lookback_sessions),
        "rvol_breakout_bar": rvol_bar(bars, i, cfg.rvol_lookback_sessions),
        "bar_body_ratio": math.nan if bar_range <= 0 else abs(float(row["close"]) - float(row["open"])) / bar_range,
        "or_width_atr": math.nan if math.isnan(atr_value) or atr_value <= 0 else rng.width / atr_value,
        "minutes_since_open": float((ts - session_open).total_seconds() // 60),
    }


def build_feature_table(
    breakouts: pd.DataFrame,
    store: BarStore,
    daily_store: BarStore,
    calendar: TradingCalendar,
    config: Config,
) -> pd.DataFrame:
    """One row per breakout event: identifiers, label, and the feature/context columns."""
    if breakouts.empty:
        raise ValueError("no breakouts to featurise")
    rows: list[dict[str, object]] = []
    for symbol, group in breakouts.groupby("symbol", sort=True):
        bars = store.read_research(symbol)
        daily = daily_store.read_research(symbol)
        if bars.empty or daily.empty:
            raise ValueError(f"{symbol}: missing intraday or daily research bars")
        ctx = FeatureContext(daily=daily, calendar=calendar, config=config)
        for _, e in group.iterrows():
            i = int(e["breakout_index"])
            if bars.index[i] != pd.Timestamp(e["breakout_time"]):
                raise ValueError(
                    f"{symbol} {e['session_date']}: breakout_index {i} no longer points at "
                    f"{e['breakout_time']}; re-run the study after a fetch"
                )
            feats = compute_features(bars, i, e["direction"], ctx)
            if feats["minutes_since_open"] != float(e["minutes_since_open"]):
                raise ValueError(f"{symbol} {e['session_date']}: minutes_since_open disagrees with the label")
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
        raise FileNotFoundError(f"{path} not found; run study first")
    return pd.read_parquet(path)

"""Feature extraction.

Each feature is computed at the decision moment - the close of the breakout bar - from
bars at positions <= the breakout index. Undefined values are NaN, never defaults.
Definitions and expected directions are pre-registered in RESEARCH.md before implementation.

Three groups, and the difference between them is what a verdict may rest on.

FEATURE_NAMES - directional, testable. A pre-registered expected direction exists, so a
gap that holds in both periods is evidence.

    rvol_open_15m            was the session busy at all
    rvol_breakout_bar        real participation behind the break
    bar_body_ratio           absorption: volume without price progress
    rel_strength_vs_index    moving on its own demand, not the market's tide
    index_or_agrees          the index broke its own range the same way
    gap_atr_signed           the move began overnight, in this direction

REPORT_ONLY_NAMES - two-sided. Both stories are tellable, so there is no honest direction
to predict and these can never on their own produce a signal verdict.

    prior_day_return_atr_signed  continuation or exhaustion
    is_expiry_day                expiry positioning distorts ranges

CONTEXT_NAMES - segmentation only, never tested as mechanisms, because each is partly
definitional: the outcome depends on the range width (a failure is a width-scaled
distance), on how much session is left to resolve in, and - for breakout_depth_atr - on
the labelling thresholds themselves.

    or_width_atr        opening range width in ATR units
    minutes_since_open  time of day
    day_of_week         0-4
    breakout_depth_atr  how far the close sits beyond the broken boundary, in ATR

breakout_depth_atr was pre-registered as directional and demoted in code review before any
result was interpreted. ``labelling.resolve`` measures the move from the breakout bar
onward, so the breakout bar's own excursion already counts: a close at depth >= the bust
threshold can never be labelled BUSTED, and one at depth >= the sustain threshold is
SUSTAINED on the spot. Testing it would measure the definition, not a mechanism.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict

from intraday.config import Config
from intraday.expiry_calendar import ExpiryCalendar
from intraday.indicators import (
    atr_prior_day,
    gap_pct,
    index_bar_at,
    index_breakout_state,
    index_session_open,
    opening_range,
    previous_trading_day,
    rvol_at_time,
    rvol_bar,
    session_start_pos,
)
from intraday.labelling import Direction
from intraday.store import BarStore
from intraday.trading_calendar import TradingCalendar

FEATURES_FILE = "features.parquet"

FEATURE_NAMES: tuple[str, ...] = (
    "rvol_open_15m",
    "rvol_breakout_bar",
    "bar_body_ratio",
    "rel_strength_vs_index",
    "index_or_agrees",
    "gap_atr_signed",
)

# Pre-registered as two-sided: reported, never able to produce a signal verdict alone.
REPORT_ONLY_NAMES: tuple[str, ...] = ("prior_day_return_atr_signed", "is_expiry_day")

CONTEXT_NAMES: tuple[str, ...] = (
    "or_width_atr", "minutes_since_open", "day_of_week", "breakout_depth_atr",
)

TESTED_NAMES: tuple[str, ...] = FEATURE_NAMES + REPORT_ONLY_NAMES
ALL_COLUMNS: tuple[str, ...] = FEATURE_NAMES + REPORT_ONLY_NAMES + CONTEXT_NAMES

FEATURE_MECHANISM: dict[str, str] = {
    "rvol_open_15m": "was the session busy at all",
    "rvol_breakout_bar": "real participation behind the break",
    "bar_body_ratio": "absorption: volume without price progress",
    "rel_strength_vs_index": "moving on its own demand, not the market's tide",
    "index_or_agrees": "the index broke its own range the same way",
    "gap_atr_signed": "the move began overnight, in this direction",
    "breakout_depth_atr": "how far beyond the boundary the close sat (context only)",
    "prior_day_return_atr_signed": "continuation or exhaustion (two-sided)",
    "is_expiry_day": "expiry positioning distorts ranges (two-sided)",
}

# What each feature is expected to show, in plain words, for the report text.
FEATURE_EXPECTATION: dict[str, str] = {
    "rvol_open_15m": "quiet opens should fail more often",
    "rvol_breakout_bar": "breaks on thin volume should fail more often",
    "bar_body_ratio": "small bodies (price stalling) should fail more often",
    "rel_strength_vs_index": "breakouts lagging the index should fail more often",
    "index_or_agrees": "breakouts fighting the index should fail more often",
    "gap_atr_signed": "breakouts against the overnight gap should fail more often",
    "breakout_depth_atr": "no direction tested; partly fixed by the labelling thresholds",
    "prior_day_return_atr_signed": "no direction predicted; reported only",
    "is_expiry_day": "no direction predicted; reported only",
}

# Plain-English names for the report, so it never has to print an identifier.
FEATURE_PLAIN: dict[str, str] = {
    "rvol_open_15m": "quiet first 15 minutes",
    "rvol_breakout_bar": "thin volume on the breakout bar",
    "bar_body_ratio": "small candle body (price stalling)",
    "rel_strength_vs_index": "lagging the index today",
    "index_or_agrees": "index breaking the same way",
    "gap_atr_signed": "overnight gap against the breakout",
    "breakout_depth_atr": "how decisively the level broke",
    "prior_day_return_atr_signed": "yesterday's move",
    "is_expiry_day": "expiry day",
}


class FeatureContext(BaseModel):
    """Everything a feature may consult besides the intraday frame."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    daily: pd.DataFrame  # the symbol's daily research bars; only rows before the session date are used
    index_bars: pd.DataFrame  # the index's intraday research bars; only bars <= the decision stamp are used
    calendar: TradingCalendar
    expiries: ExpiryCalendar
    config: Config


def compute_features(bars: pd.DataFrame, i: int, direction: Direction, ctx: FeatureContext) -> dict[str, float]:
    """Features and context columns for a breakout whose breakout bar is at position ``i``.

    Signed features are multiplied by -1 for short breakouts, so a positive value always
    means "in the direction of the breakout".
    """
    cfg = ctx.config
    ts = bars.index[i]
    session_date = ts.date()
    sign = 1.0 if direction == "long" else -1.0
    start = session_start_pos(bars, i)
    rng = opening_range(bars, i, cfg)
    row = bars.iloc[i]
    close = float(row["close"])

    atr_value = atr_prior_day(ctx.daily, session_date, ctx.calendar, cfg.atr_period)
    has_atr = not math.isnan(atr_value) and atr_value > 0
    bar_range = float(row["high"]) - float(row["low"])
    session_open = pd.Timestamp.combine(session_date, cfg.session_start).tz_localize(bars.index.tz)

    # --- relative strength against the index, both measured from their own session opens
    stock_open = float(bars["open"].iloc[start])
    index_row = index_bar_at(ctx.index_bars, ts)
    index_open = index_session_open(ctx.index_bars, session_date)
    if index_row is None or math.isnan(index_open) or index_open <= 0 or stock_open <= 0:
        rel_strength = math.nan
    else:
        stock_move = close / stock_open - 1.0
        index_move = float(index_row["close"]) / index_open - 1.0
        rel_strength = sign * (stock_move - index_move) * 100.0  # percentage points

    # --- has the index broken its own opening range the same way, by now?
    state = index_breakout_state(ctx.index_bars, ts, cfg)
    if isinstance(state, float) and math.isnan(state):
        index_agrees = math.nan
    else:
        index_agrees = 1.0 if state == (1 if direction == "long" else -1) else 0.0

    # --- overnight gap in ATR units, signed. Derived from gap_pct alone so there is one
    #     definition of "the previous close" rather than two that can disagree.
    gap = gap_pct(bars, i, ctx.calendar)
    if math.isnan(gap) or not has_atr:
        gap_signed = math.nan
    else:
        gap_price = stock_open - stock_open / (1.0 + gap / 100.0)
        gap_signed = sign * gap_price / atr_value

    # --- how far through the broken boundary this close actually is
    boundary = rng.high if direction == "long" else rng.low
    depth = math.nan if not has_atr else abs(close - boundary) / atr_value

    # --- yesterday's own move, signed (two-sided: reported, never a verdict on its own)
    prior_return = math.nan
    if has_atr:
        prev_day = previous_trading_day(session_date, ctx.calendar)
        prev_rows = ctx.daily[ctx.daily.index.date == prev_day]
        if not prev_rows.empty:
            prev = prev_rows.iloc[0]
            prior_return = sign * (float(prev["close"]) - float(prev["open"])) / atr_value

    expiry = ctx.expiries.is_expiry(session_date)

    return {
        "rvol_open_15m": rvol_at_time(bars, rng.end_index, cfg.rvol_lookback_sessions),
        "rvol_breakout_bar": rvol_bar(bars, i, cfg.rvol_lookback_sessions),
        "bar_body_ratio": math.nan if bar_range <= 0 else abs(close - float(row["open"])) / bar_range,
        "rel_strength_vs_index": rel_strength,
        "index_or_agrees": index_agrees,
        "gap_atr_signed": gap_signed,
        "breakout_depth_atr": depth,
        "prior_day_return_atr_signed": prior_return,
        "is_expiry_day": math.nan if expiry is None else float(expiry),
        "or_width_atr": math.nan if not has_atr else rng.width / atr_value,
        "minutes_since_open": float((ts - session_open).total_seconds() // 60),
        "day_of_week": float(session_date.weekday()),
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
    index_bars = store.read_research(config.index_symbol)
    if index_bars.empty:
        raise ValueError(
            f"no research bars for the index {config.index_symbol}; the index features cannot be "
            "computed and would be NaN for every row"
        )
    expiries = ExpiryCalendar.from_csv(config.data_dir / "nse_expiries.csv")
    rows: list[dict[str, object]] = []
    for symbol, group in breakouts.groupby("symbol", sort=True):
        bars = store.read_research(symbol)
        daily = daily_store.read_research(symbol)
        if bars.empty or daily.empty:
            raise ValueError(f"{symbol}: missing intraday or daily research bars")
        ctx = FeatureContext(
            daily=daily, index_bars=index_bars, calendar=calendar, expiries=expiries, config=config
        )
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

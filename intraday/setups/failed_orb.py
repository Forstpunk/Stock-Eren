"""S2 - Failed-ORB reversal (the primary hypothesis). GATED on the Stage 6 verdict.

Signal: bar ``i`` is the bust bar of the session's first breakout in a direction, i.e.
by the close of bar ``i`` the breakout has met the BUSTED definition (never extended
>= bust_extension_atr beyond its boundary, closed through the opposite boundary before
the cutoff). Trade the reversal: enter at the next bar's open in the OPPOSITE direction
to the original breakout. Stop: ATR multiple. Exit: stop or EOD; 1R-partial variant
recorded separately. Each trade carries the original breakout's Stage 5 feature vector.

``signal_at`` resolves the breakout on the frame truncated at ``i``, so it can only see
what a trader at that close could see; it is tested against the lookahead harness.

The gate: ``assert_gate_open`` reads the latest study and raises unless its verdict is
"signal" - i.e. unless some feature separates failed breakouts in both halves of the
period. There is no override flag.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from intraday.config import Config
from intraday.indicators import opening_range, session_start_pos
from intraday.labelling import Label, detect_breakout_at, resolve
from intraday.setups.common import VARIANTS, build_trade, entry_allowed
from intraday.trades import Direction, Trade

NAME = "failed_orb"


class GateClosedError(Exception):
    """Stage 6 has not produced an out-of-sample signal; S2 may not run."""


STUDY_FILE = "study.json"


def assert_gate_open(data_dir: Path, config: Config) -> str:
    path = data_dir / STUDY_FILE
    if not path.exists():
        raise GateClosedError(f"{path} not found: run the study before backtesting {NAME}")
    verdict = json.loads(path.read_text(encoding="utf-8"))["verdict"]
    if verdict != "signal":
        raise GateClosedError(
            f"the study verdict is {verdict!r} ({path.name}); {NAME} runs only on a 'signal' verdict. "
            "This is the research gate, not a bug."
        )
    return verdict


def signal_at(bars: pd.DataFrame, i: int, atr_value: float, config: Config) -> Direction | None:
    """Reversal direction if bar ``i`` completes a bust of the session's first breakout."""
    start = session_start_pos(bars, i)
    if i < start + config.opening_range_bars + 1:
        return None
    session = bars.iloc[start : i + 1]  # truncated at i: nothing later is visible
    rng = opening_range(session, len(session) - 1, config)
    for k in range(config.opening_range_bars, len(session) - 1):
        direction = detect_breakout_at(session, k, config)
        if direction is None:
            continue
        label, pos, _ = resolve(session, k, direction, rng, atr_value, config)
        if label is Label.BUSTED and pos == len(session) - 1:
            return "short" if direction == "long" else "long"
    return None


def trades_for_session(
    session: pd.DataFrame,
    symbol: str,
    atr_value: float,
    features_by_direction: dict[str, dict[str, float]],
    config: Config,
) -> list[Trade]:
    trades: list[Trade] = []
    seen: set[str] = set()
    for i in range(config.opening_range_bars + 1, len(session)):
        direction = signal_at(session, i, atr_value, config)
        if direction is None or direction in seen:
            continue
        seen.add(direction)
        original = "long" if direction == "short" else "short"
        entry_pos = i + 1
        if not entry_allowed(session, entry_pos, config):
            continue
        if original not in features_by_direction:
            raise ValueError(f"{symbol} {session.index[i].date()}: no feature vector for the {original} breakout")
        for variant in VARIANTS:
            trades.append(build_trade(
                NAME, variant, symbol, session, session.index[i].date(), direction, entry_pos,
                atr_value, features_by_direction[original], config,
            ))
    return trades

"""Lookahead detection. Built before any trading logic, because intraday lookahead
bugs are invisible and produce spectacular fake results.

Two mechanisms, both mechanical:

1. ``assert_no_lookahead(fn, bars, decision_index)`` calls ``fn`` twice, once on the
   full session and once on the session truncated at ``decision_index`` (inclusive:
   the decision bar itself is known once it has closed). Any difference in the value
   computed for that index is lookahead. So is raising on the truncated frame only.

2. ``poison_future(bars, from_time)`` returns a copy of the frame in which every bar of
   the LAST session at or after ``from_time`` is replaced by a huge favourable move
   (earlier sessions are untouched, so multi-session frames work). A function that
   peeks will produce different values on the poisoned frame for decisions made before
   the first poisoned bar; ``assert_blind_to_poison`` checks exactly that.

``fn`` has the signature ``fn(bars: pd.DataFrame, index: int) -> value`` where ``value``
is a scalar, a dict of scalars, a pandas Series/DataFrame, a numpy array, or a pydantic
model. Comparison is field by field; NaN equals NaN.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from datetime import time
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

Computation = Callable[[pd.DataFrame, int], Any]

POISON_FROM = time(11, 0)
POISON_MULTIPLIER = 1.5  # every poisoned bar trades at 1.5x the last honest close
POISON_VOLUME_MULTIPLIER = 50


class LookaheadError(AssertionError):
    """A computation for bar ``index`` depends on bars after ``index``."""

    def __init__(self, name: str, index: int, field: str, full: Any, truncated: Any) -> None:
        self.name = name
        self.index = index
        self.field = field
        super().__init__(
            f"{name} at index {index}: field {field!r} differs between the reference session and "
            f"the truncated/poisoned one: reference={full!r} other={truncated!r}"
        )


def _flatten(value: Any, prefix: str = "value") -> dict[str, Any]:
    """Reduce any supported return type to {field_name: scalar}."""
    if isinstance(value, BaseModel):
        return _flatten(value.model_dump(), prefix)
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            out.update(_flatten(v, f"{prefix}.{k}"))
        return out
    if isinstance(value, pd.DataFrame):
        out = {}
        for col in value.columns:
            out.update(_flatten(value[col], f"{prefix}.{col}"))
        return out
    if isinstance(value, pd.Series):
        return {f"{prefix}[{i}]": v for i, v in value.items()}
    if isinstance(value, np.ndarray):
        return {f"{prefix}[{i}]": v for i, v in enumerate(value.ravel().tolist())}
    if isinstance(value, (list, tuple)):
        out = {}
        for i, v in enumerate(value):
            out.update(_flatten(v, f"{prefix}[{i}]"))
        return out
    return {prefix: value}


def _equal(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b
    try:
        if pd.isna(a) and pd.isna(b):
            return True
    except (TypeError, ValueError):
        pass
    try:
        return bool(a == b)
    except Exception:  # objects that cannot be compared are, by definition, not equal
        return False


def _compare(name: str, index: int, full_value: Any, other_value: Any) -> None:
    full = _flatten(full_value)
    other = _flatten(other_value)
    if full.keys() != other.keys():
        missing = sorted(set(full) ^ set(other))
        raise LookaheadError(name, index, missing[0], full.get(missing[0]), other.get(missing[0]))
    for field, a in full.items():
        b = other[field]
        if not _equal(a, b):
            raise LookaheadError(name, index, field, a, b)


def assert_no_lookahead(fn: Computation, bars: pd.DataFrame, decision_index: int, name: str | None = None) -> None:
    """Raise LookaheadError if ``fn(bars, decision_index)`` uses any bar after ``decision_index``."""
    label = name or getattr(fn, "__name__", repr(fn))
    if not 0 <= decision_index < len(bars):
        raise IndexError(f"{label}: decision_index {decision_index} outside 0..{len(bars) - 1}")
    full_value = fn(bars, decision_index)
    truncated = bars.iloc[: decision_index + 1]
    try:
        truncated_value = fn(truncated, decision_index)
    except Exception as exc:
        raise LookaheadError(label, decision_index, "<call>", full_value, f"raised {type(exc).__name__}: {exc}") from exc
    _compare(label, decision_index, full_value, truncated_value)


def assert_no_lookahead_sweep(
    fn: Computation, bars: pd.DataFrame, indices: Iterable[int] | None = None, name: str | None = None
) -> None:
    """Run ``assert_no_lookahead`` at every index (default: every bar of the session)."""
    for i in (indices if indices is not None else range(len(bars))):
        assert_no_lookahead(fn, bars, i, name)


def poison_future(bars: pd.DataFrame, from_time: time = POISON_FROM) -> pd.DataFrame:
    """Copy of ``bars`` where every bar of the last session at or after ``from_time`` is a
    huge favourable move."""
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError("poison_future needs a DatetimeIndex")
    last_day = bars.index[-1].date()
    mask = np.array([ts.date() == last_day and ts.time() >= from_time for ts in bars.index])
    if not mask.any():
        raise ValueError(f"no bars at or after {from_time} on {last_day} to poison")
    if mask.all():
        raise ValueError(f"every bar is at or after {from_time}; nothing honest remains")
    honest_close = float(bars["close"].iloc[int(np.argmax(mask)) - 1])
    poisoned = bars.copy()
    n = int(mask.sum())
    ramp = honest_close * (1 + (POISON_MULTIPLIER - 1) * np.linspace(0.2, 1.0, n))
    poisoned.loc[mask, "open"] = ramp * 0.995
    poisoned.loc[mask, "close"] = ramp
    poisoned.loc[mask, "high"] = ramp * 1.01
    poisoned.loc[mask, "low"] = ramp * 0.99
    poisoned.loc[mask, "volume"] = (bars["volume"].iloc[~mask].median() * POISON_VOLUME_MULTIPLIER).astype("int64")
    return poisoned


def first_poisoned_index(bars: pd.DataFrame, poisoned: pd.DataFrame) -> int:
    """Position of the first row where the two frames differ."""
    if len(bars) != len(poisoned):
        raise ValueError("honest and poisoned frames must have the same length")
    differs = (bars.to_numpy() != poisoned.to_numpy()).any(axis=1)
    if not differs.any():
        raise ValueError("poisoned frame is identical to the honest one")
    return int(np.argmax(differs))


def assert_blind_to_poison(
    fn: Computation, bars: pd.DataFrame, from_time: time = POISON_FROM, name: str | None = None
) -> None:
    """For every decision index before the first poisoned bar, ``fn`` must give identical
    values on the honest and the poisoned frame. Any difference means ``fn`` looked past
    its decision bar."""
    label = name or getattr(fn, "__name__", repr(fn))
    poisoned = poison_future(bars, from_time)
    cutoff = first_poisoned_index(bars, poisoned)
    for i in range(cutoff):
        _compare(f"{label} (poisoned future)", i, fn(bars, i), fn(poisoned, i))

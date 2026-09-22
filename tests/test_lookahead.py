"""Harness self-tests. The cheating functions here exist only to prove the detector works."""
from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd
import pytest

from intraday.lookahead import (
    LookaheadError,
    assert_blind_to_poison,
    assert_no_lookahead,
    assert_no_lookahead_sweep,
    first_poisoned_index,
    poison_future,
)

# ---- honest computations -----------------------------------------------------------


def running_mean_close(bars: pd.DataFrame, i: int) -> float:
    return float(bars["close"].iloc[: i + 1].mean())


def running_stats(bars: pd.DataFrame, i: int) -> dict[str, float]:
    window = bars.iloc[: i + 1]
    return {"high": float(window["high"].max()), "low": float(window["low"].min()), "n": float(len(window))}


# ---- cheats (harness self-test only) ------------------------------------------------


def cheat_session_high(bars: pd.DataFrame, i: int) -> float:
    return float(bars["high"].max())  # whole session, not up to i


def cheat_next_close(bars: pd.DataFrame, i: int) -> float:
    return float(bars["close"].iloc[i + 1])  # raises on the truncated frame


def cheat_centred_mean(bars: pd.DataFrame, i: int) -> dict[str, float]:
    return {"mean": float(bars["close"].iloc[max(0, i - 2) : i + 3].mean()), "n": float(i)}


def cheat_rolling_centred(bars: pd.DataFrame, i: int) -> pd.Series:
    return bars["close"].rolling(5, center=True).mean().iloc[: i + 1]


# ---- tests -----------------------------------------------------------------------


def test_honest_scalar_passes(session: pd.DataFrame) -> None:
    assert_no_lookahead_sweep(running_mean_close, session)


def test_honest_dict_passes(session: pd.DataFrame) -> None:
    assert_no_lookahead_sweep(running_stats, session)


def test_session_high_cheat_is_caught(session: pd.DataFrame) -> None:
    with pytest.raises(LookaheadError, match="cheat_session_high at index 10"):
        assert_no_lookahead(cheat_session_high, session, 10)


def test_next_close_cheat_is_caught_via_raise(session: pd.DataFrame) -> None:
    with pytest.raises(LookaheadError, match="raised IndexError"):
        assert_no_lookahead(cheat_next_close, session, 10)


def test_dict_cheat_names_the_field(session: pd.DataFrame) -> None:
    with pytest.raises(LookaheadError) as exc:
        assert_no_lookahead(cheat_centred_mean, session, 20)
    assert exc.value.field == "value.mean"


def test_series_cheat_names_the_position(session: pd.DataFrame) -> None:
    with pytest.raises(LookaheadError) as exc:
        assert_no_lookahead(cheat_rolling_centred, session, 20)
    assert exc.value.field.startswith("value[")


def test_nan_equals_nan(session: pd.DataFrame) -> None:
    def rolling_20(bars: pd.DataFrame, i: int) -> float:
        return float(bars["close"].rolling(20).mean().iloc[i])

    assert_no_lookahead(rolling_20, session, 5)  # NaN on both sides


def test_bad_index_raises(session: pd.DataFrame) -> None:
    with pytest.raises(IndexError):
        assert_no_lookahead(running_mean_close, session, 75)


# ---- poisoned fixture ---------------------------------------------------------------


def test_poison_replaces_only_from_cutoff(session: pd.DataFrame, poisoned_session: pd.DataFrame) -> None:
    cutoff = first_poisoned_index(session, poisoned_session)
    assert session.index[cutoff].time() == time(11, 0)
    pd.testing.assert_frame_equal(session.iloc[:cutoff], poisoned_session.iloc[:cutoff])
    honest_close = session["close"].iloc[cutoff - 1]
    assert (poisoned_session["close"].iloc[cutoff:] > honest_close * 1.05).all()
    assert (poisoned_session["low"] <= poisoned_session[["open", "close"]].min(axis=1)).all()
    assert (poisoned_session["high"] >= poisoned_session[["open", "close"]].max(axis=1)).all()


def test_poison_rejects_degenerate_cutoffs(session: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        poison_future(session, time(9, 15))
    with pytest.raises(ValueError):
        poison_future(session, time(16, 0))


def test_honest_function_is_blind_to_poison(session: pd.DataFrame) -> None:
    assert_blind_to_poison(running_stats, session)


def test_peeking_function_scores_impossibly_well_on_poison(session: pd.DataFrame) -> None:
    def cheat_future_return(bars: pd.DataFrame, i: int) -> float:
        return float(bars["close"].iloc[-1] / bars["close"].iloc[i] - 1)

    cutoff = first_poisoned_index(session, poison_future(session))
    honest = np.array([cheat_future_return(session, i) for i in range(cutoff)])
    poisoned = np.array([cheat_future_return(poison_future(session), i) for i in range(cutoff)])
    assert poisoned.min() > 0.4 and abs(honest).max() < 0.1  # "impossibly well"
    with pytest.raises(LookaheadError, match="poisoned future"):
        assert_blind_to_poison(cheat_future_return, session)


def test_poison_touches_only_the_last_session() -> None:
    from tests.synthetic import random_history

    hist = random_history(3)
    poisoned = poison_future(hist)
    last_day = hist.index[-1].date()
    earlier = hist.index.date != last_day
    pd.testing.assert_frame_equal(hist[earlier], poisoned[earlier])
    is_last_after_11 = (hist.index.date == last_day) & np.array([t.time() >= time(11, 0) for t in hist.index])
    assert first_poisoned_index(hist, poisoned) == int(np.argmax(is_last_after_11))

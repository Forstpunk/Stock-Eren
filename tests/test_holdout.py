"""The sealed holdout.

Looking at a test period repeatedly is how it stops being out-of-sample: each decision made
after a peek fits the model to it a little more, and nothing in the numbers shows it. The
seal exists so that cannot happen by accident, which means it has to be enforced where the
bars are read, not politely honoured by each caller.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from intraday.config import Config
from intraday.store import BarStore, SealedHoldoutError
from intraday.trading_calendar import TradingCalendar
from intraday.validate import validate_session
from tests.synthetic import TEST_CALENDAR, random_history
from tests.test_validate import FETCHED_LATER

SEAL = date(2026, 9, 21)


@pytest.fixture
def stocked(tmp_path: Path) -> tuple[Path, list[date]]:
    """Six sessions, three either side of the seal."""
    config = Config()
    store = BarStore(tmp_path, "5m")
    hist = random_history(6, seed=77, end_day=date(2026, 9, 23), n_bars=73)
    sessions = [
        (validate_session(f, "AAA", d, config, TEST_CALENDAR, FETCHED_LATER), f)
        for d, f in hist.groupby(hist.index.date)
    ]
    store.put_sessions("AAA", sessions)
    return tmp_path, sorted({d for d, _ in [(v.session_date, f) for v, f in sessions]})


def test_sealed_store_hides_the_holdout(stocked) -> None:  # type: ignore[no-untyped-def]
    root, all_dates = stocked
    open_store = BarStore(root, "5m")
    sealed = BarStore(root, "5m", holdout_from=SEAL)

    assert max(open_store.research_sessions("AAA")) >= SEAL, "the fixture must straddle the seal"
    visible = sealed.research_sessions("AAA")
    assert visible, "the training period must still be readable"
    assert all(d < SEAL for d in visible)
    assert len(visible) < len(open_store.research_sessions("AAA"))


def test_unsealed_store_sees_everything(stocked) -> None:  # type: ignore[no-untyped-def]
    root, _ = stocked
    sealed = BarStore(root, "5m", holdout_from=SEAL)
    unsealed = BarStore(root, "5m", holdout_from=SEAL, unsealed=True)
    assert len(unsealed.read_research("AAA")) > len(sealed.read_research("AAA"))
    assert unsealed.read_research("AAA").equals(BarStore(root, "5m").read_research("AAA"))


def test_reading_the_holdout_directly_is_refused_while_sealed(stocked) -> None:  # type: ignore[no-untyped-def]
    root, _ = stocked
    sealed = BarStore(root, "5m", holdout_from=SEAL)
    with pytest.raises(SealedHoldoutError, match="sealed"):
        sealed.read_holdout("AAA")

    unsealed = BarStore(root, "5m", holdout_from=SEAL, unsealed=True)
    held = unsealed.read_holdout("AAA")
    assert not held.empty and all(d >= SEAL for d in set(held.index.date))


def test_no_holdout_configured_changes_nothing(stocked) -> None:  # type: ignore[no-untyped-def]
    root, _ = stocked
    plain = BarStore(root, "5m")
    assert not plain.sealed
    assert plain.read_holdout("AAA").empty, "with no seal there is no holdout to return"
    assert len(plain.read_research("AAA")) > 0


def test_for_config_builds_a_sealed_store(tmp_path: Path) -> None:
    sealed = BarStore.for_config(Config(data_dir=tmp_path, holdout_from=SEAL), "5m")
    assert sealed.sealed and sealed.holdout_from == SEAL

    unsealed = BarStore.for_config(
        Config(data_dir=tmp_path, holdout_from=SEAL, holdout_unsealed=True), "5m"
    )
    assert not unsealed.sealed

    none = BarStore.for_config(Config(data_dir=tmp_path), "5m")
    assert not none.sealed and none.holdout_from is None


def test_the_seal_survives_the_whole_pipeline(stocked) -> None:  # type: ignore[no-untyped-def]
    """Labelling reads bars through the store, so it must inherit the seal automatically.

    This is the point of enforcing it in one place: a caller cannot forget.
    """
    from intraday.labelling import label_breakouts
    from tests.synthetic import random_daily

    root, _ = stocked
    daily = random_daily(40, seed=78, end_day=date(2026, 9, 23))
    sealed = BarStore(root, "5m", holdout_from=SEAL)
    events, _ = label_breakouts(sealed.read_research("AAA"), "AAA", daily, TEST_CALENDAR, Config())
    assert all(e.session_date < SEAL for e in events), "a labelled breakout leaked from the holdout"


def test_config_accepts_a_holdout_date() -> None:
    assert Config().holdout_from is None, "sealing is opt-in; it costs history"
    assert Config(holdout_from=SEAL).holdout_from == SEAL
    assert Config(holdout_from=SEAL).holdout_unsealed is False

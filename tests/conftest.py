from __future__ import annotations

import pandas as pd
import pytest

from intraday.config import Config
from intraday.lookahead import poison_future
from tests.synthetic import random_session


@pytest.fixture
def config() -> Config:
    return Config()


@pytest.fixture
def session() -> pd.DataFrame:
    """An honest synthetic session: 75 bars, seeded random walk."""
    return random_session(seed=42)


@pytest.fixture
def poisoned_session(session: pd.DataFrame) -> pd.DataFrame:
    """The same session with every bar from 11:00 replaced by a huge favourable move.

    Every feature and setup from Stage 3 onward is tested against this: any computation
    for a bar before 11:00 must be identical on ``session`` and ``poisoned_session``.
    """
    return poison_future(session)

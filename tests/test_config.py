from __future__ import annotations

import pytest
from pydantic import ValidationError

from intraday.config import Config, interval_to_minutes


def test_bars_per_session_is_derived(config: Config) -> None:
    assert config.session_minutes == 375
    assert config.interval_minutes == 5
    assert config.bars_per_session == 75
    assert config.opening_range_bars == 3


def test_config_is_frozen(config: Config) -> None:
    with pytest.raises(ValidationError):
        config.min_sample = 10  # type: ignore[misc]


def test_interval_must_divide_session() -> None:
    with pytest.raises(ValidationError, match="does not divide"):
        Config(interval="7m")


def test_opening_range_must_align_to_interval() -> None:
    with pytest.raises(ValidationError, match="not a multiple"):
        Config(opening_range_minutes=17)


def test_bad_interval_string() -> None:
    with pytest.raises(ValueError, match="must look like"):
        interval_to_minutes("5min")

"""Frozen research configuration. One instance per run; nothing mutates it."""
from __future__ import annotations

import re
from datetime import time
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

IST = ZoneInfo("Asia/Kolkata")

_INTERVAL_RE = re.compile(r"^(\d+)m$")


def interval_to_minutes(interval: str) -> int:
    """Parse a minute interval string such as '5m'. Raises on anything else."""
    match = _INTERVAL_RE.match(interval)
    if match is None:
        raise ValueError(f"interval must look like '5m', got {interval!r}")
    minutes = int(match.group(1))
    if minutes <= 0:
        raise ValueError(f"interval must be positive, got {interval!r}")
    return minutes


class Config(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Data source - exactly one is active per run. Failure is failure, not a switch.
    source: Literal["yfinance", "kite"] = "yfinance"
    data_dir: Path = Path("data")
    index_symbol: str = "NIFTY50"

    # Session
    session_start: time = time(9, 15)
    session_end: time = time(15, 30)
    interval: str = "5m"
    opening_range_minutes: int = Field(default=15, gt=0)

    # Daily bars (for ATR and gap): fetched alongside intraday, validated per row.
    daily_interval: str = "1d"
    daily_history_days: int = Field(default=200, gt=0)

    # Indicators
    rvol_lookback_sessions: int = Field(default=20, gt=0)
    atr_period: int = Field(default=14, gt=0)

    # Trades, benchmark, expectancy
    position_inr: float = Field(default=50_000.0, gt=0)
    stop_atr_multiple: float = Field(default=1.0, gt=0)
    last_entry_time: time = time(14, 30)  # no new positions after this bar
    bootstrap_n: int = Field(default=1000, gt=0)
    benchmark_seed: int = 0

    # Breakout labelling. Thresholds are multiples of the prior-day ATR, not of the
    # opening-range width: width-unit thresholds made the resolution rate a function of
    # range width (wide ranges resolved as NEITHER 73% of the time on the yfinance pilot).
    # Fixed before any confirmatory run; ratio 2:1 preserved from the original spec.
    sustain_extension_atr: float = Field(default=0.5, gt=0)
    bust_extension_atr: float = Field(default=0.25, gt=0)
    bust_cutoff: time = time(15, 15)

    # Research thresholds
    slippage_bps: tuple[int, ...] = (5, 10, 20)
    min_sample: int = Field(default=30, gt=0)
    rvol_threshold: float = Field(default=2.0, gt=0)
    gap_threshold_pct: float = Field(default=1.0, gt=0)
    atr_pct_threshold: float = Field(default=1.5, gt=0)
    turnover_threshold_inr: float = Field(default=10 * 1e7, gt=0)  # Rs 10 crore

    @computed_field  # type: ignore[prop-decorator]
    @property
    def interval_minutes(self) -> int:
        return interval_to_minutes(self.interval)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def session_minutes(self) -> int:
        start = self.session_start.hour * 60 + self.session_start.minute
        end = self.session_end.hour * 60 + self.session_end.minute
        return end - start

    @computed_field  # type: ignore[prop-decorator]
    @property
    def bars_per_session(self) -> int:
        return self.session_minutes // self.interval_minutes

    @computed_field  # type: ignore[prop-decorator]
    @property
    def opening_range_bars(self) -> int:
        return self.opening_range_minutes // self.interval_minutes

    @computed_field  # type: ignore[prop-decorator]
    @property
    def first_entry_time(self) -> time:
        """Earliest possible entry bar: the bar after the first post-range bar."""
        minutes = (self.session_start.hour * 60 + self.session_start.minute
                   + self.opening_range_minutes + self.interval_minutes)
        return time(minutes // 60, minutes % 60)

    @model_validator(mode="after")
    def _check_consistency(self) -> Config:
        if self.session_minutes <= 0:
            raise ValueError(
                f"session_end {self.session_end} must be after session_start {self.session_start}"
            )
        if self.session_minutes % self.interval_minutes != 0:
            raise ValueError(
                f"interval {self.interval} does not divide the {self.session_minutes}-minute session"
            )
        if self.opening_range_minutes % self.interval_minutes != 0:
            raise ValueError(
                f"opening_range_minutes {self.opening_range_minutes} is not a multiple of {self.interval}"
            )
        if self.bust_extension_atr >= self.sustain_extension_atr:
            raise ValueError("bust_extension_atr must be below sustain_extension_atr")
        if not (self.session_start < self.bust_cutoff <= self.session_end):
            raise ValueError(f"bust_cutoff {self.bust_cutoff} must lie inside the session")
        if not (self.session_start < self.last_entry_time < self.session_end):
            raise ValueError(f"last_entry_time {self.last_entry_time} must lie inside the session")
        if not self.slippage_bps or any(b < 0 for b in self.slippage_bps):
            raise ValueError(f"slippage_bps must be non-empty and non-negative, got {self.slippage_bps}")
        return self


DEFAULT_CONFIG = Config()

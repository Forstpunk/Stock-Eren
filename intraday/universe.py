"""Which symbols to study, chosen per session rather than fixed in advance.

The published ORB result (Zarattini & Aziz, RESEARCH.md item 1) does not come from a fixed
list of large caps. It rebuilds the universe every morning: from a pool of liquid names,
take the ones trading unusually heavily in the first minutes, and trade those. Their claim
is that the edge lives in that selection step, not in the breakout trigger. We have never
tested it, because we ran a fixed list of twenty.

Two stages, both of which must be blind to the future:

- ``eligible_pool`` filters candidates on what was knowable before the session opened:
  the previous day's close, its ATR and its turnover. A name that was too cheap, too quiet
  or too illiquid yesterday is out today.
- ``stocks_in_play`` ranks the survivors by opening relative volume - the session's own
  first minutes against the same minutes on recent sessions - and takes the top N.

The second stage uses bars from the session being traded, which is legitimate: the
selection happens after the opening range closes and before any entry. It is still a
decision made at a point in time, so it is computed from bars at or before that point and
carries the same lookahead tests as every feature.
"""
from __future__ import annotations

import math
from datetime import date

import pandas as pd
from pydantic import BaseModel, ConfigDict

from intraday.config import Config
from intraday.indicators import atr_prior_day, rvol_at_time, session_start_pos
from intraday.trading_calendar import TradingCalendar


class Candidate(BaseModel):
    """One symbol's case for being traded on one session, with the numbers behind it."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    session_date: date
    prior_close: float
    prior_atr: float
    atr_pct: float  # prior ATR as a percentage of the prior close
    prior_turnover: float  # previous session's traded value, in rupees
    opening_rvol: float  # this session's opening volume against its recent norm
    eligible: bool
    reasons: tuple[str, ...]  # why it was excluded, empty when eligible


def eligible_pool(
    symbol: str,
    session_date: date,
    daily: pd.DataFrame,
    calendar: TradingCalendar,
    config: Config,
) -> tuple[bool, float, float, float, tuple[str, ...]]:
    """Was this name worth considering, judged only on data from before the session?

    Returns (eligible, prior close, prior ATR, prior turnover, reasons for exclusion).
    """
    reasons: list[str] = []
    prev_rows = daily[daily.index.date < session_date]
    if prev_rows.empty:
        return False, math.nan, math.nan, math.nan, ("no prior session",)

    prior = prev_rows.iloc[-1]
    prior_close = float(prior["close"])
    prior_turnover = prior_close * float(prior["volume"])
    prior_atr = atr_prior_day(daily, session_date, calendar, config.atr_period)

    if prior_close < config.min_price_inr:
        reasons.append(f"price {prior_close:.1f} below {config.min_price_inr}")
    if math.isnan(prior_atr) or prior_atr <= 0:
        reasons.append("no prior-day ATR")
    elif prior_atr / prior_close * 100 < config.atr_pct_threshold:
        reasons.append(f"ATR {prior_atr / prior_close * 100:.2f}% below {config.atr_pct_threshold}%")
    if prior_turnover < config.turnover_threshold_inr:
        reasons.append(f"turnover {prior_turnover / 1e7:.1f} cr below {config.turnover_threshold_inr / 1e7:.0f} cr")

    return not reasons, prior_close, prior_atr, prior_turnover, tuple(reasons)


def opening_rvol(bars: pd.DataFrame, session_date: date, config: Config) -> float:
    """Relative volume through the end of the opening range. NaN without enough history."""
    same_day = bars[bars.index.date == session_date]
    if same_day.empty:
        return math.nan
    position = bars.index.get_loc(same_day.index[0])
    end = int(position) + config.opening_range_bars - 1
    if end >= len(bars) or bars.index[end].date() != session_date:
        return math.nan  # the opening range has not completed on this session
    return rvol_at_time(bars, end, config.rvol_lookback_sessions)


def rank_candidates(
    candidates: list[Candidate], config: Config
) -> list[Candidate]:
    """The eligible ones with a usable RVOL, heaviest first, capped at the day's size."""
    usable = [c for c in candidates if c.eligible and not math.isnan(c.opening_rvol)]
    usable.sort(key=lambda c: (-c.opening_rvol, c.symbol))
    return usable[: config.stocks_in_play]


class SelectionReport(BaseModel):
    """What was chosen on one session and, just as importantly, what was not."""

    model_config = ConfigDict(frozen=True)

    session_date: date
    n_considered: int
    n_eligible: int
    n_with_rvol: int
    selected: tuple[str, ...]
    exclusion_counts: dict[str, int]  # first reason -> how many names it removed

    def describe(self) -> str:
        return (
            f"{self.session_date}: {self.n_considered} considered, {self.n_eligible} eligible, "
            f"{self.n_with_rvol} with a usable opening RVOL, {len(self.selected)} traded"
        )


def select_for_session(
    session_date: date,
    bars_by_symbol: dict[str, pd.DataFrame],
    daily_by_symbol: dict[str, pd.DataFrame],
    calendar: TradingCalendar,
    config: Config,
) -> tuple[list[Candidate], SelectionReport]:
    """The day's stocks in play, plus every candidate's numbers for auditing."""
    candidates: list[Candidate] = []
    exclusions: dict[str, int] = {}
    for symbol in sorted(bars_by_symbol):
        daily = daily_by_symbol.get(symbol)
        if daily is None or daily.empty:
            exclusions["no daily bars"] = exclusions.get("no daily bars", 0) + 1
            continue
        ok, close, atr_value, turnover, reasons = eligible_pool(
            symbol, session_date, daily, calendar, config
        )
        if reasons:
            head = reasons[0].split(" ")[0]
            exclusions[head] = exclusions.get(head, 0) + 1
        candidates.append(Candidate(
            symbol=symbol,
            session_date=session_date,
            prior_close=close,
            prior_atr=atr_value,
            atr_pct=math.nan if math.isnan(atr_value) or close <= 0 else atr_value / close * 100,
            prior_turnover=turnover,
            opening_rvol=opening_rvol(bars_by_symbol[symbol], session_date, config),
            eligible=ok,
            reasons=reasons,
        ))

    chosen = rank_candidates(candidates, config)
    report = SelectionReport(
        session_date=session_date,
        n_considered=len(bars_by_symbol),
        n_eligible=sum(1 for c in candidates if c.eligible),
        n_with_rvol=sum(1 for c in candidates if c.eligible and not math.isnan(c.opening_rvol)),
        selected=tuple(c.symbol for c in chosen),
        exclusion_counts=exclusions,
    )
    return chosen, report

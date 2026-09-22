"""Benchmark: deterministic under seed, matched on session/duration/window/R rule.
Expectancy: formula by hand, CI flag, InsufficientSampleError, segments."""
from __future__ import annotations

from datetime import date, time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from intraday.benchmark import SessionUniverse, edge_vs_random, matched_random, random_twin
from intraday.config import Config
from intraday.expectancy import (
    InsufficientSampleError,
    entry_half_hour,
    expectancy,
    or_width_quartiles,
    rvol_bucket,
    segment_keys,
    segmented_expectancy,
)
from intraday.indicators import atr_prior_day
from intraday.store import BarStore
from intraday.trades import Trade, TradeResult, evaluate, results_frame
from intraday.validate import validate_daily_row, validate_session
from tests.synthetic import TEST_CALENDAR, random_daily, random_history
from tests.test_validate import FETCHED_LATER

END = date(2026, 9, 25)
WINDOW = (time(9, 30), time(15, 0))


@pytest.fixture(scope="module")
def stores(tmp_path_factory: pytest.TempPathFactory) -> tuple[BarStore, BarStore]:
    root = tmp_path_factory.mktemp("data")
    config = Config()
    store, daily_store = BarStore(root, "5m"), BarStore(root, "1d")
    for k, symbol in enumerate(("AAA", "BBB", "CCC")):
        hist = random_history(6, seed=100 + k, end_day=END, n_bars=73)
        sessions = [
            (validate_session(f, symbol, d, config, TEST_CALENDAR, FETCHED_LATER), f)
            for d, f in hist.groupby(hist.index.date)
        ]
        store.put_sessions(symbol, sessions)
        daily = random_daily(30, seed=200 + k, end_day=END)
        rows = [(validate_daily_row(r, symbol, d, TEST_CALENDAR, FETCHED_LATER), r) for d, r in daily.groupby(daily.index.date)]
        daily_store.put_sessions(symbol, rows)
    return store, daily_store


@pytest.fixture(scope="module")
def universe(stores: tuple[BarStore, BarStore]) -> SessionUniverse:
    return SessionUniverse(stores[0], stores[1], TEST_CALENDAR, Config())


def strategy_results(stores: tuple[BarStore, BarStore], config: Config, bps: int = 10) -> list[TradeResult]:
    """Fake 'strategy': long at 10:05 open, hold 12 bars, on AAA every session."""
    store, daily_store = stores
    bars = store.read_research("AAA")
    daily = daily_store.read_research("AAA")
    out = []
    for day, s in bars.groupby(bars.index.date):
        a = atr_prior_day(daily, day, TEST_CALENDAR, config.atr_period)
        if np.isnan(a):
            continue
        e = 10
        x = e + 12
        t = Trade(
            setup="fake", variant="base", symbol="AAA", session_date=day, direction="long",
            entry_index=e, entry_time=s.index[e].to_pydatetime(), entry_price=float(s["open"].iloc[e]),
            exit_index=x, exit_time=s.index[x].to_pydatetime(), exit_price=float(s["close"].iloc[x]),
            exit_reason="time", stop_price=float(s["open"].iloc[e]) - a, atr=a, stop_atr_multiple=1.0,
            mfe_price=float(s["high"].iloc[e : x + 1].max()),
            features={"rvol_breakout_bar": 1.5, "or_width_atr": 0.3 + 0.05 * len(out)},
        )
        out.append(evaluate(t, bps, config))
    return out


def test_universe_requires_atr(universe: SessionUniverse) -> None:
    days = sorted({d for (_, d) in universe.sessions})
    assert len(days) == 6
    assert universe.eligible(days[-1]) == ["AAA", "BBB", "CCC"]
    assert all(v > 0 for v in universe.atr.values())


def test_benchmark_deterministic_under_seed(stores, universe, config: Config) -> None:  # type: ignore[no-untyped-def]
    strat = strategy_results(stores, config)
    a = matched_random(strat, universe, WINDOW, config, seed=7)
    b = matched_random(strat, universe, WINDOW, config, seed=7)
    c = matched_random(strat, universe, WINDOW, config, seed=8)
    assert [r.trade for r in a] == [r.trade for r in b]
    assert [r.trade for r in a] != [r.trade for r in c]
    assert edge_vs_random(strat, a, config, seed=1) == edge_vs_random(strat, b, config, seed=1)


def test_twin_matches_session_duration_window_and_r_rule(stores, universe, config: Config) -> None:  # type: ignore[no-untyped-def]
    strat = strategy_results(stores, config)
    twins = matched_random(strat, universe, WINDOW, config, seed=3)
    directions = set()
    for s, t in zip(strat, twins):
        assert t.trade.session_date == s.trade.session_date
        assert t.duration_bars == s.duration_bars == 12
        assert WINDOW[0] <= t.trade.entry_time.time() <= WINDOW[1]
        assert t.slippage_bps == s.slippage_bps
        assert t.trade.setup == "random" and t.trade.symbol in ("AAA", "BBB", "CCC")
        assert t.trade.atr == universe.atr[(t.trade.symbol, t.trade.session_date)]
        assert abs(t.trade.entry_price - t.trade.stop_price) == pytest.approx(config.stop_atr_multiple * t.trade.atr)
        directions.add(t.trade.direction)
    assert directions == {"long", "short"}


def test_edge_vs_random_paired_bootstrap(stores, universe, config: Config) -> None:  # type: ignore[no-untyped-def]
    strat = strategy_results(stores, config)
    twins = matched_random(strat, universe, WINDOW, config, seed=3)
    e = edge_vs_random(strat, twins, config, seed=0)
    assert e.n == len(strat) and e.slippage_bps == 10
    assert e.edge_r == pytest.approx(e.mean_strategy_r - e.mean_random_r)
    assert e.ci_low <= e.edge_r <= e.ci_high
    assert e.indistinguishable_from_zero == (e.ci_low <= 0 <= e.ci_high)
    with pytest.raises(ValueError, match="mixed slippage"):
        edge_vs_random(strat, strategy_results(stores, config, bps=5), config, seed=0)


def test_random_twin_rejects_empty_universe_day(universe, stores, config: Config) -> None:  # type: ignore[no-untyped-def]
    strat = strategy_results(stores, config)[0]
    bad = strat.model_copy(update={"trade": strat.trade.model_copy(update={"session_date": date(2026, 1, 5)})})
    with pytest.raises(ValueError, match="no eligible symbol"):
        random_twin(bad, universe, WINDOW, np.random.default_rng(0), config)


# ---- expectancy ------------------------------------------------------------------------


def test_expectancy_formula_by_hand(config: Config) -> None:
    r = pd.Series([2.0] * 20 + [-1.0] * 20)
    m = pd.Series([2.5] * 20 + [0.2] * 20)
    e = expectancy(r, m, "hand", config, seed=0)
    assert e.n == 40 and e.win_rate == 0.5
    assert e.avg_win_r == 2.0 and e.avg_loss_r == -1.0
    assert e.expectancy_r == pytest.approx(0.5 * 2.0 - 0.5 * 1.0)
    assert e.breakeven_failure_rate == 0.5
    assert e.ci_low <= 0.5 <= e.ci_high
    assert not e.indistinguishable_from_zero


def test_expectancy_flags_zero_spanning_ci(config: Config) -> None:
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0, 1.0, 200))
    e = expectancy(r, pd.Series(np.abs(r)), "noise", config, seed=0)
    assert e.indistinguishable_from_zero and "Indistinguishable" in e.statement


def test_expectancy_raises_below_min_sample(config: Config) -> None:
    with pytest.raises(InsufficientSampleError, match="29 trades, below the minimum of 30"):
        expectancy(pd.Series([1.0] * 29), pd.Series([1.0] * 29), "thin", config, seed=0)


def test_segment_helpers(config: Config) -> None:
    assert rvol_bucket(0.5, 2.0) == "rvol <1.0"
    assert rvol_bucket(1.5, 2.0) == "rvol 1.0-2"
    assert rvol_bucket(2.0, 2.0) == "rvol >=2"
    assert rvol_bucket(float("nan"), 2.0) == "rvol n/a"
    assert entry_half_hour(time(9, 35)) == "09:30" and entry_half_hour(time(10, 0)) == "10:00"
    q = or_width_quartiles(pd.Series([0.1, 0.2, 0.3, 0.4, np.nan]))
    assert list(q) == ["or_width Q1", "or_width Q2", "or_width Q3", "or_width Q4", "or_width n/a"]


def test_segmented_expectancy_lists_skipped(stores, config: Config) -> None:  # type: ignore[no-untyped-def]
    strat = strategy_results(stores, config)
    df = segment_keys(results_frame(strat * 10), config)  # 60 rows, all rvol 1.5 -> one bucket
    computed, skipped = segmented_expectancy(df, "seg_rvol", config, seed=0)
    assert list(computed) == ["rvol 1.0-2"] and skipped == {}
    by_width, skipped_w = segmented_expectancy(df, "seg_or_width", config, seed=0)
    assert set(by_width) | set(skipped_w) == {"or_width Q1", "or_width Q2", "or_width Q3", "or_width Q4"}
    computed2, skipped2 = segmented_expectancy(df.iloc[:20], "seg_rvol", config, seed=0)
    assert computed2 == {} and skipped2 == {"rvol 1.0-2": 20}

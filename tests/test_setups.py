"""Setups: exit simulator by hand, S1/S2 signals through the lookahead harness, the VWAP
filter, entry-time rule, the S2 gate, and an end-to-end run on a temporary store."""
from __future__ import annotations

import json
from datetime import date, time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from intraday.config import Config
from intraday.lookahead import assert_blind_to_poison, assert_no_lookahead_sweep
from intraday.setups import run_setup
from intraday.setups import failed_orb, orb
from intraday.setups.common import build_trade, simulate_exit
from intraday.setups.failed_orb import GateClosedError, assert_gate_open
from intraday.store import BarStore
from intraday.trades import evaluate
from intraday.validate import validate_daily_row, validate_session
from tests.synthetic import TEST_CALENDAR, random_daily, random_history
from tests.test_labelling import ATR, DAY, flat_session, set_bar
from tests.test_validate import FETCHED_LATER

# ---- exit simulator --------------------------------------------------------------------


def test_exit_eod_and_mfe() -> None:
    df = flat_session()
    set_bar(df, 30, 100.0, 103.0, 99.9, 100.5)
    e = simulate_exit(df, 10, "long", 100.0, 96.0, "base")
    assert e.reason == "eod" and e.exit_index == 72 and e.exit_price == 100.0
    assert e.mfe_price == 103.0


def test_exit_stop_filled_at_stop_price() -> None:
    df = flat_session()
    set_bar(df, 15, 99.0, 99.1, 95.0, 96.0)
    e = simulate_exit(df, 10, "long", 100.0, 96.5, "base")
    assert e.reason == "stop" and e.exit_index == 15 and e.exit_price == 96.5
    s = simulate_exit(df, 10, "short", 100.0, 100.3, "base")  # flat highs at 100.2 never touch 100.3 ...
    assert s.reason == "eod"
    set_bar(df, 20, 100.0, 100.4, 99.9, 100.1)
    s2 = simulate_exit(df, 10, "short", 100.0, 100.3, "base")
    assert s2.reason == "stop" and s2.exit_index == 20 and s2.exit_price == 100.3


def test_same_bar_stop_out() -> None:
    df = flat_session()
    set_bar(df, 10, 100.0, 100.1, 95.0, 96.0)
    e = simulate_exit(df, 10, "long", 100.0, 97.0, "base")
    assert e.exit_index == 10 and e.reason == "stop"


def test_partial_1r_blends_and_moves_stop_to_breakeven() -> None:
    df = flat_session()  # closes 100, highs 100.2, lows 99.8
    df.iloc[11:, df.columns.get_loc("low")] = 100.05  # lows above entry: breakeven stop not touched by noise
    set_bar(df, 20, 100.1, 104.5, 100.05, 104.0)  # reaches +1R (entry 100, risk 4 -> target 104)
    set_bar(df, 40, 100.0, 100.1, 99.5, 99.6)  # dips below entry -> breakeven stop
    base = simulate_exit(df, 10, "long", 100.0, 96.0, "base")
    part = simulate_exit(df, 10, "long", 100.0, 96.0, "partial_1r")
    assert base.reason == "eod" and base.exit_price == 100.0
    assert part.reason == "stop" and part.exit_index == 40
    assert part.exit_price == pytest.approx(0.5 * 104.0 + 0.5 * 100.0)  # half at target, half at breakeven
    df2 = flat_session()
    df2.iloc[11:, df2.columns.get_loc("low")] = 100.05
    set_bar(df2, 20, 100.1, 104.5, 100.05, 104.0)
    part2 = simulate_exit(df2, 10, "long", 100.0, 96.0, "partial_1r")
    assert part2.reason == "target" and part2.exit_index == 72
    assert part2.exit_price == pytest.approx(0.5 * 104.0 + 0.5 * 100.0)


def test_stop_and_target_same_bar_is_worst_case() -> None:
    df = flat_session()
    set_bar(df, 20, 100.0, 105.0, 95.0, 100.0)
    part = simulate_exit(df, 10, "long", 100.0, 96.0, "partial_1r")
    assert part.reason == "stop" and part.exit_price == 96.0


def test_build_trade_uses_next_bar_open_and_atr_stop(config: Config) -> None:
    df = flat_session()
    set_bar(df, 11, 101.7, 102.0, 101.5, 101.8)
    t = build_trade("orb", "base", "X", df, DAY, "long", 11, ATR, {"f": 1.0}, config)
    assert t.entry_price == 101.7 and t.stop_price == pytest.approx(101.7 - config.stop_atr_multiple * ATR)
    assert t.entry_time.time() == time(10, 10) and t.features == {"f": 1.0}
    r = evaluate(t, 10, config)
    assert r.risk_inr == pytest.approx(r.quantity * ATR)


# ---- S1 --------------------------------------------------------------------------------


def _s1(bars: pd.DataFrame, i: int) -> str:
    return orb.signal_at(bars, i, Config()) or "none"


def test_s1_signal_no_lookahead(session: pd.DataFrame) -> None:
    assert_no_lookahead_sweep(_s1, session, range(3, len(session)))
    assert_blind_to_poison(_s1, session)


def test_s1_vwap_filter_blocks_breakout_below_vwap(config: Config) -> None:
    df = flat_session()
    df["volume"] = 1000
    df.iloc[0, df.columns.get_loc("volume")] = 10_000_000  # VWAP pinned near bar 0's typical price
    df.iloc[0, [df.columns.get_loc(c) for c in ("open", "high", "low", "close")]] = [110.0, 110.5, 109.5, 110.0]
    df.iloc[1, df.columns.get_loc("high")] = 110.5  # OR high 110.5, VWAP ~110
    set_bar(df, 10, 110.0, 110.7, 109.9, 110.6)  # closes above OR high, but below? no: 110.6 > vwap 110
    assert orb.signal_at(df, 10, config) == "long"
    df.iloc[0, [df.columns.get_loc(c) for c in ("open", "high", "low", "close")]] = [112.0, 112.5, 111.5, 112.0]
    df.iloc[0, df.columns.get_loc("high")] = 110.5  # keep OR high at 110.5 but VWAP ~111.3
    df.iloc[0, df.columns.get_loc("low")] = 110.4
    assert orb.signal_at(df, 10, config) is None  # 110.6 < VWAP -> filtered out


def test_s1_trades_both_variants_and_skips_late_entries(config: Config) -> None:
    df = flat_session()
    set_bar(df, 10, 100.5, 101.6, 100.4, 101.5)  # long breakout, VWAP ~100 -> passes
    set_bar(df, 66, 99.5, 99.6, 98.5, 98.7)  # short at 14:45 -> entry 14:50 > 14:30: skipped
    feats = {"long": {"x": 1.0}, "short": {"x": 2.0}}
    trades = orb.trades_for_session(df, "X", ATR, feats, config)
    assert [(t.variant, t.direction, t.entry_index) for t in trades] == [("base", "long", 11), ("partial_1r", "long", 11)]
    assert trades[0].features == {"x": 1.0}


def test_s1_missing_feature_vector_raises(config: Config) -> None:
    df = flat_session()
    set_bar(df, 10, 100.5, 101.6, 100.4, 101.5)
    with pytest.raises(ValueError, match="no feature vector"):
        orb.trades_for_session(df, "X", ATR, {}, config)


# ---- S2 --------------------------------------------------------------------------------


def busted_long_session() -> pd.DataFrame:
    df = flat_session()
    set_bar(df, 10, 100.5, 101.7, 100.4, 101.5)  # long breakout, extension 0.7 < 0.25 ATR = 1.0
    set_bar(df, 15, 99.5, 99.6, 98.5, 98.7)  # closes through OR low at 10:30 -> bust
    return df


def test_s2_signal_fires_on_bust_bar_only(config: Config) -> None:
    df = busted_long_session()
    fired = [i for i in range(len(df)) if failed_orb.signal_at(df, i, ATR, config)]
    assert fired == [15]
    assert failed_orb.signal_at(df, 15, ATR, config) == "short"


def test_s2_no_signal_when_breakout_extended(config: Config) -> None:
    df = busted_long_session()
    set_bar(df, 12, 101.5, 103.0, 101.4, 102.0)  # extended 2.0 = 0.5 ATR -> sustained, not bustable
    assert all(failed_orb.signal_at(df, i, ATR, config) is None for i in range(len(df)))


def _s2(bars: pd.DataFrame, i: int) -> str:
    return failed_orb.signal_at(bars, i, 4.0, Config()) or "none"


def test_s2_signal_no_lookahead(session: pd.DataFrame) -> None:
    assert_no_lookahead_sweep(_s2, session, range(4, len(session)))
    assert_blind_to_poison(_s2, session)
    hist = random_history(3, seed=5, n_bars=73)
    assert_no_lookahead_sweep(_s2, hist, range(len(hist) - 73, len(hist)))


def test_s2_trade_is_the_reversal_with_original_features(config: Config) -> None:
    df = busted_long_session()
    trades = failed_orb.trades_for_session(df, "X", ATR, {"long": {"x": 1.0}}, config)
    assert [(t.variant, t.direction, t.entry_index) for t in trades] == [("base", "short", 16), ("partial_1r", "short", 16)]
    assert trades[0].features == {"x": 1.0}
    assert trades[0].stop_price == pytest.approx(trades[0].entry_price + ATR)


def test_s2_gate(tmp_path: Path, config: Config) -> None:
    from intraday.setups.failed_orb import STUDY_FILE

    with pytest.raises(GateClosedError, match="not found"):
        assert_gate_open(tmp_path, config)
    path = tmp_path / STUDY_FILE
    for closed in ("insufficient_sample", "no_signal"):
        path.write_text(json.dumps({"verdict": closed}), encoding="utf-8")
        with pytest.raises(GateClosedError, match="research gate"):
            assert_gate_open(tmp_path, config)
    path.write_text(json.dumps({"verdict": "signal"}), encoding="utf-8")
    assert assert_gate_open(tmp_path, config) == "signal"


def test_real_data_gate_is_closed(config: Config) -> None:
    """The project's own diagnostic must currently keep S2 shut."""
    with pytest.raises(GateClosedError):
        assert_gate_open(Path("data"), config)


# ---- end to end on a temporary store ----------------------------------------------------


def test_run_setup_end_to_end(tmp_path: Path, config: Config) -> None:
    store, daily_store = BarStore(tmp_path, "5m"), BarStore(tmp_path, "1d")
    hist = random_history(6, seed=31, end_day=date(2026, 9, 25), n_bars=73)
    store.put_sessions("AAA", [
        (validate_session(f, "AAA", d, config, TEST_CALENDAR, FETCHED_LATER), f) for d, f in hist.groupby(hist.index.date)
    ])
    daily = random_daily(30, seed=32, end_day=date(2026, 9, 25))
    daily_store.put_sessions("AAA", [
        (validate_daily_row(r, "AAA", d, TEST_CALENDAR, FETCHED_LATER), r) for d, r in daily.groupby(daily.index.date)
    ])
    from intraday.features import ALL_COLUMNS
    from intraday.labelling import label_breakouts

    events, _ = label_breakouts(hist, "AAA", daily, TEST_CALENDAR, config)
    features = pd.DataFrame([{
        "symbol": e.symbol, "session_date": e.session_date, "direction": e.direction,
        **{f: float(k) for k, f in enumerate(ALL_COLUMNS)},
    } for e in events])
    run = run_setup("orb", store, daily_store, features, TEST_CALENDAR, config, tmp_path)
    assert run.sessions_seen == 6 and run.sessions_without_atr == 0
    assert run.trades and len(run.trades) % 2 == 0
    for t in run.trades:
        assert t.entry_time.time() >= config.first_entry_time and t.entry_time.time() <= config.last_entry_time
        assert set(t.features) == set(ALL_COLUMNS)
    with pytest.raises(GateClosedError):
        run_setup("failed_orb", store, daily_store, features, TEST_CALENDAR, config, tmp_path)


def test_run_backtest_end_to_end(tmp_path: Path, config: Config) -> None:
    from intraday.backtest import load_backtest, run_backtest, save_backtest
    from intraday.features import ALL_COLUMNS
    from intraday.labelling import label_breakouts

    store, daily_store = BarStore(tmp_path, "5m"), BarStore(tmp_path, "1d")
    for k, sym in enumerate(("AAA", "BBB")):
        hist = random_history(8, seed=41 + k, end_day=date(2026, 9, 25), n_bars=73)
        store.put_sessions(sym, [
            (validate_session(f, sym, d, config, TEST_CALENDAR, FETCHED_LATER), f) for d, f in hist.groupby(hist.index.date)
        ])
        daily = random_daily(30, seed=51 + k, end_day=date(2026, 9, 25))
        daily_store.put_sessions(sym, [
            (validate_daily_row(r, sym, d, TEST_CALENDAR, FETCHED_LATER), r) for d, r in daily.groupby(daily.index.date)
        ])
    rows = []
    for sym in ("AAA", "BBB"):
        events, _ = label_breakouts(store.read_research(sym), sym, daily_store.read_research(sym), TEST_CALENDAR, config)
        rows += [{"symbol": e.symbol, "session_date": e.session_date, "direction": e.direction,
                  **{f: float(k) for k, f in enumerate(ALL_COLUMNS)}} for e in events]
    features = pd.DataFrame(rows)
    summary, table = run_backtest("orb", (5, 20), store, daily_store, features, TEST_CALENDAR, config, tmp_path)
    assert summary.slippage_bps == (5, 20) and len(summary.variants) == 4
    assert set(table["role"]) == {"strategy", "random"}
    assert (table[table.role == "strategy"].groupby("slippage_bps").size() == summary.n_signals * 2).all()
    for v in summary.variants:
        assert v.edge.n == v.n and v.overall is None or v.overall.n == v.n  # tiny sample -> insufficient
    save_backtest(summary, table, tmp_path)
    assert load_backtest("orb", tmp_path) == summary

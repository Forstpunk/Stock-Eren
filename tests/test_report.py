"""Report: verdict rule, integrity summary from synthetic verdicts, caveats verbatim,
and an integration build against the project's own artifacts when present."""
from __future__ import annotations

import io
from datetime import date, datetime
from pathlib import Path

import pytest
from rich.console import Console

from intraday.backtest import BacktestSummary, VariantSummary
from intraday.benchmark import EdgeVsRandom
from intraday.config import IST, Config
from intraday.expectancy import Expectancy
from intraday.report import CAVEATS, build_report, integrity_summary, render_report, setup_verdict
from intraday.validate import SessionVerdict, Verdict


def _edge(lo: float, hi: float) -> EdgeVsRandom:
    return EdgeVsRandom(n=100, slippage_bps=10, mean_strategy_r=0.1, mean_random_r=0.0, edge_r=(lo + hi) / 2,
                        ci_low=lo, ci_high=hi, indistinguishable_from_zero=lo <= 0 <= hi, statement="")


def _exp(lo: float, hi: float) -> Expectancy:
    return Expectancy(segment="s", n=100, win_rate=0.5, avg_win_r=1.0, avg_loss_r=-0.5, expectancy_r=(lo + hi) / 2,
                      ci_low=lo, ci_high=hi, indistinguishable_from_zero=lo <= 0 <= hi, breakeven_failure_rate=0.5, statement="")


def _summary(variants: list[VariantSummary]) -> BacktestSummary:
    return BacktestSummary(setup="orb", run_at=datetime(2026, 9, 22, tzinfo=IST), sessions_seen=10, sessions_without_atr=0,
                           skipped_late_entries=0, n_signals=100, slippage_bps=(5, 20), variants=tuple(variants))


def _variant(bps: int, edge: EdgeVsRandom, overall: Expectancy | None, variant: str = "base") -> VariantSummary:
    return VariantSummary(variant=variant, slippage_bps=bps, n=100, edge=edge, overall=overall, overall_n=100,
                          segments={}, skipped_segments={})


def test_setup_verdict_rule() -> None:
    edge_at_all = _summary([_variant(5, _edge(0.05, 0.3), _exp(0.05, 0.3)), _variant(20, _edge(0.02, 0.2), _exp(0.01, 0.2))])
    assert setup_verdict(edge_at_all) == "edge detected"
    dies_at_20 = _summary([_variant(5, _edge(0.05, 0.3), _exp(0.05, 0.3)), _variant(20, _edge(-0.1, 0.2), _exp(-0.1, 0.1))])
    assert setup_verdict(dies_at_20) == "no edge"
    beats_random_but_loses_money = _summary([_variant(5, _edge(0.05, 0.3), _exp(-0.3, -0.1))])
    assert setup_verdict(beats_random_but_loses_money) == "no edge"
    thin = _summary([_variant(5, _edge(0.05, 0.3), None)])
    assert setup_verdict(thin) == "insufficient sample"
    with pytest.raises(ValueError, match="no base variant"):
        setup_verdict(_summary([_variant(5, _edge(0.05, 0.3), _exp(0.05, 0.3), variant="partial_1r")]))


def test_integrity_summary_counts_equities_only() -> None:
    def v(sym: str, d: date, verdict: Verdict, missing: tuple[str, ...] = ()) -> SessionVerdict:
        return SessionVerdict(symbol=sym, session_date=d, verdict=verdict, bar_count=73, expected_bars=75, zero_volume_bars=1,
                              missing_slots=missing, dropped_tail_bars=0, reasons=(), fetched_at=datetime(2026, 9, 22, tzinfo=IST))

    d1, d2 = date(2026, 9, 7), date(2026, 9, 8)
    verdicts = {
        ("A", d1): v("A", d1, Verdict.CLEAN), ("A", d2): v("A", d2, Verdict.PARTIAL, ("15:15", "15:20")),
        ("B", d1): v("B", d1, Verdict.TAIL_COLLAPSED, ("15:20",)), ("B", d2): v("B", d2, Verdict.CORRUPT),
        ("NIFTY50", d1): v("NIFTY50", d1, Verdict.SUSPECT),
    }
    daily = {("A", d1): v("A", d1, Verdict.CLEAN), ("A", d2): v("A", d2, Verdict.CORRUPT)}
    log = [{"event": "added"}, {"event": "fetch_run", "ts": "t1"}, {"event": "fetch_run", "ts": "t2"}]
    s = integrity_summary(verdicts, daily, log, "NIFTY50")
    assert s.sessions_validated == 4 and s.symbols == 2 and s.research_sessions == 2
    assert s.verdict_counts == {"CLEAN": 1, "TAIL_COLLAPSED": 1, "PARTIAL": 1, "SUSPECT": 0, "CORRUPT": 1}
    assert s.per_symbol_non_research == {"A": 1, "B": 1}
    assert s.daily_rows == 2 and s.daily_corrupt == 1
    assert s.last_fetch == {"event": "fetch_run", "ts": "t2"}


def test_caveats_verbatim() -> None:
    assert CAVEATS == (
        "Research output. Not advice, not a prediction, not a recommendation.\n"
        "- Current data source: yfinance, 60-day limit, one market regime only.\n"
        "- Results from a single regime do not generalise. Treat as preliminary\n"
        "  until re-run on multi-year history.\n"
        "- Costs modelled; real slippage varies with size and liquidity.\n"
        '- "No edge detected" is a valid and useful result.'
    )


@pytest.mark.skipif(
    not (Path("data/backtest_orb.json").exists() and Path("data/study.json").exists()),
    reason="project artifacts not present",
)
def test_report_builds_from_project_artifacts(config: Config) -> None:
    report = build_report(config)
    console = Console(record=True, width=150, file=io.StringIO())
    render_report(report, console)
    text = console.export_text()
    for heading in (
        "THE SHORT VERSION", "In four sentences", "Verdicts", "1. Data integrity", "2. Base rates",
        "3. What separates a failed breakout", "Cost of trading it", "Caveats",
    ):
        assert heading in text
    if report.forecast is not None:
        assert "4. Forecast" in text and "Calibration" in text
        # a gated setup must never be reported from a stale artifact
        assert ("failed_orb" not in report.backtests) or report.study.verdict == "signal"
    assert text.rstrip().endswith('"No edge detected" is a valid and useful result.')
    assert "orb" in report.setup_verdicts
    # the four sentences are numbered and come before any table
    assert text.index("1. Data:") < text.index("1. Data integrity")


def test_backtest_older_than_the_features_is_refused(tmp_path, config: Config) -> None:  # type: ignore[no-untyped-def]
    """A run killed part-way leaves a backtest describing data that no longer exists.

    Blending two eras of results in one report is worse than reporting nothing, so an
    artifact older than features.parquet is named and dropped rather than loaded.
    """
    import json
    import os
    import shutil

    source = Path("data")
    if not (source / "features.parquet").exists() or not (source / "backtest_orb.json").exists():
        pytest.skip("project artifacts not present")

    for name in ("features.parquet", "breakouts.parquet", "study.json", "predictions.parquet",
                 "backtest_orb.json", "backtest_orb.parquet", "nse_holidays.csv", "nse_expiries.csv",
                 "session_verdicts.jsonl", "fetch_log.jsonl"):
        if (source / name).exists():
            shutil.copy2(source / name, tmp_path / name)
    for tier in ("bars", "quarantine"):
        if (source / tier).exists():
            shutil.copytree(source / tier, tmp_path / tier, dirs_exist_ok=True)

    cfg = config.model_copy(update={"data_dir": tmp_path})
    fresh = build_report(cfg)
    assert "orb" in fresh.backtests and fresh.stale_backtests == ()

    # age the backtest a minute behind the features
    features_mtime = (tmp_path / "features.parquet").stat().st_mtime
    for suffix in ("json", "parquet"):
        os.utime(tmp_path / f"backtest_orb.{suffix}", (features_mtime - 60, features_mtime - 60))

    stale = build_report(cfg)
    assert "orb" not in stale.backtests
    assert stale.stale_backtests == ("orb",)

    console = Console(record=True, width=150, file=io.StringIO())
    render_report(stale, console)
    text = console.export_text()
    assert "NOT REPORTED" in text and "Re-run study" in text

"""Report assembly and rendering, from persisted artifacts only.

Order: run header -> verdicts -> data integrity -> base rates -> diagnostic ->
edge vs benchmark -> expectancy -> segments (per setup) -> caveats (verbatim).

Nothing is recomputed here. Missing core artifacts raise with the command to run;
the gated setup (failed_orb) is reported as "gate closed" when absent.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict
from rich.console import Console
from rich.table import Table

from intraday.backtest import BacktestSummary, load_backtest, render_backtest
from intraday.config import IST, Config
from intraday.diagnostic import DiagnosticReport
from intraday.diagnostic_render import render as render_diagnostic
from intraday.labelling import base_rates, load_breakouts
from intraday.setups import SETUPS
from intraday.setups.failed_orb import NAME as GATED_SETUP
from intraday.store import BarStore, read_jsonl
from intraday.validate import RESEARCH_VERDICTS, SessionVerdict, Verdict

CAVEATS = """Research output. Not advice, not a prediction, not a recommendation.
- Current data source: yfinance, 60-day limit, one market regime only.
- Results from a single regime do not generalise. Treat as preliminary
  until re-run on multi-year history.
- Costs modelled; real slippage varies with size and liquidity.
- "No edge detected" is a valid and useful result."""

SetupVerdict = Literal["edge detected", "no edge", "insufficient sample"]


class IntegritySummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    sessions_validated: int
    verdict_counts: dict[str, int]
    research_sessions: int
    symbols: int
    per_symbol_non_research: dict[str, int]
    missing_slot_counts: dict[str, int]
    daily_rows: int
    daily_corrupt: int
    last_fetch: dict[str, object] | None


class Report(BaseModel):
    model_config = ConfigDict(frozen=True)

    generated_at: datetime
    config: dict[str, object]
    integrity: IntegritySummary
    n_breakouts: int
    base_overall: dict[str, float]
    base_by_month: dict[str, dict[str, float]]
    diagnostic: DiagnosticReport
    diagnostic_sensitivity: dict[int, DiagnosticReport]
    backtests: dict[str, BacktestSummary]
    setup_verdicts: dict[str, SetupVerdict]
    gate_closed: bool


# ---- pieces ---------------------------------------------------------------------------


def integrity_summary(
    verdicts: dict[tuple[str, object], SessionVerdict], daily_verdicts: dict[tuple[str, object], SessionVerdict],
    fetch_log: list[dict[str, object]], index_symbol: str,
) -> IntegritySummary:
    equities = {k: v for k, v in verdicts.items() if k[0] != index_symbol}
    counts = Counter(v.verdict.value for v in equities.values())
    non_research: Counter[str] = Counter()
    slots: Counter[str] = Counter()
    for (sym, _), v in equities.items():
        if v.verdict not in RESEARCH_VERDICTS:
            non_research[sym] += 1
        if v.verdict in (Verdict.PARTIAL, Verdict.TAIL_COLLAPSED):
            slots.update(v.missing_slots)
    runs = [r for r in fetch_log if r.get("event") == "fetch_run"]
    return IntegritySummary(
        sessions_validated=len(equities),
        verdict_counts={v.value: counts.get(v.value, 0) for v in Verdict},
        research_sessions=sum(1 for v in equities.values() if v.verdict in RESEARCH_VERDICTS),
        symbols=len({k[0] for k in equities}),
        per_symbol_non_research=dict(sorted(non_research.items())),
        missing_slot_counts=dict(slots.most_common(6)),
        daily_rows=len(daily_verdicts),
        daily_corrupt=sum(1 for v in daily_verdicts.values() if v.verdict is Verdict.CORRUPT),
        last_fetch=runs[-1] if runs else None,
    )


def setup_verdict(summary: BacktestSummary) -> SetupVerdict:
    """edge detected: base variant beats random with a CI above zero AND has positive
    expectancy with a CI above zero at EVERY slippage level. insufficient sample: any
    level lacks an expectancy. no edge: otherwise."""
    base = [v for v in summary.variants if v.variant == "base"]
    if not base:
        raise ValueError(f"{summary.setup}: no base variant in summary")
    if any(v.overall is None for v in base):
        return "insufficient sample"
    if all(v.edge.ci_low > 0 and v.overall is not None and v.overall.ci_low > 0 for v in base):
        return "edge detected"
    return "no edge"


def _rates(df: pd.DataFrame) -> dict[str, float]:
    r = base_rates(df).iloc[0]
    return {"n": float(r["n"]), "SUSTAINED_pct": float(r["SUSTAINED_pct"]), "BUSTED_pct": float(r["BUSTED_pct"]), "NEITHER_pct": float(r["NEITHER_pct"])}


# ---- build -------------------------------------------------------------------------------


def build_report(config: Config) -> Report:
    data_dir = config.data_dir
    store = BarStore(data_dir, config.interval)
    daily_store = BarStore(data_dir, config.daily_interval)
    verdicts = store.load_verdicts()
    if not verdicts:
        raise FileNotFoundError(f"{store.verdicts_path} is empty; run fetch first")
    integrity = integrity_summary(verdicts, daily_store.load_verdicts(), read_jsonl(store.fetch_log_path), config.index_symbol)

    breakouts = load_breakouts(data_dir)
    by_month = breakouts.assign(month=breakouts["session_date"].astype(str).str.slice(0, 7))
    base_by_month = {m: _rates(g) for m, g in by_month.groupby("month")}

    diag_path = data_dir / f"diagnostic_rvol{config.rvol_lookback_sessions}.json"
    if not diag_path.exists():
        raise FileNotFoundError(f"{diag_path} not found; run diagnose first")
    diagnostic = DiagnosticReport.model_validate_json(diag_path.read_text(encoding="utf-8"))
    sensitivity: dict[int, DiagnosticReport] = {}
    for p in sorted(data_dir.glob("diagnostic_rvol*.json")):
        lb = int(p.stem.removeprefix("diagnostic_rvol"))
        if lb != config.rvol_lookback_sessions:
            sensitivity[lb] = DiagnosticReport.model_validate_json(p.read_text(encoding="utf-8"))

    backtests: dict[str, BacktestSummary] = {}
    for name in SETUPS:
        p = data_dir / f"backtest_{name}.json"
        if p.exists():
            backtests[name] = load_backtest(name, data_dir)
        elif name != GATED_SETUP:
            raise FileNotFoundError(f"{p} not found; run backtest --setup {name} first")
    gate_closed = GATED_SETUP not in backtests and diagnostic.verdict != "signal"

    return Report(
        generated_at=datetime.now(tz=IST),
        config={k: v for k, v in json.loads(config.model_dump_json()).items()},
        integrity=integrity,
        n_breakouts=len(breakouts),
        base_overall=_rates(breakouts),
        base_by_month=base_by_month,
        diagnostic=diagnostic,
        diagnostic_sensitivity=sensitivity,
        backtests=backtests,
        setup_verdicts={name: setup_verdict(s) for name, s in backtests.items()},
        gate_closed=gate_closed,
    )


# ---- render ------------------------------------------------------------------------------


def render_report(report: Report, console: Console) -> None:
    c = report.config
    console.rule("[bold]Intraday research report - NSE opening-range breakouts")
    console.print(
        f"generated {report.generated_at:%Y-%m-%d %H:%M} IST | source {c['source']} | interval {c['interval']} | "
        f"opening range {c['opening_range_minutes']}m | thresholds sustain >= {c['sustain_extension_atr']} ATR, "
        f"bust < {c['bust_extension_atr']} ATR | slippage {c['slippage_bps']} bps | min sample {c['min_sample']} | "
        f"rvol lookback {c['rvol_lookback_sessions']} | stop {c['stop_atr_multiple']} ATR | position Rs {c['position_inr']:,.0f}"
    )

    console.print("\n[bold]Verdicts[/bold]")
    console.print(f"  diagnostic (does anything predict busts out of sample?): [bold]{report.diagnostic.verdict}[/bold]")
    for name, v in report.setup_verdicts.items():
        console.print(f"  setup {name}: [bold]{v}[/bold]")
    if report.gate_closed:
        console.print(f"  setup {GATED_SETUP}: [bold]gate closed[/bold] (requires a 'signal' diagnostic verdict)")

    i = report.integrity
    console.print("\n[bold]1. Data integrity[/bold]")
    console.print(
        f"  {i.sessions_validated} equity sessions validated across {i.symbols} symbols: "
        + "  ".join(f"{k} {v}" for k, v in i.verdict_counts.items())
        + f"  -> research set {i.research_sessions}"
    )
    if i.missing_slot_counts:
        console.print("  slots missing (PARTIAL/TAIL_COLLAPSED): " + ", ".join(f"{k} x{v}" for k, v in i.missing_slot_counts.items()))
    if i.per_symbol_non_research:
        console.print("  non-research sessions per symbol: " + ", ".join(f"{k} {v}" for k, v in i.per_symbol_non_research.items()))
    console.print(f"  daily bars: {i.daily_rows} rows, {i.daily_corrupt} CORRUPT (quarantined)")
    if i.last_fetch:
        lf = i.last_fetch
        console.print(f"  last fetch: {str(lf.get('ts'))[:16]} source={lf.get('source')} days={lf.get('days')} failed={lf.get('failed')}")

    console.print(f"\n[bold]2. Base rates[/bold]  ({report.n_breakouts} breakouts)")
    t = Table()
    for col in ("period", "n", "SUSTAINED %", "BUSTED %", "NEITHER %"):
        t.add_column(col, justify="left" if col == "period" else "right")
    o = report.base_overall
    t.add_row("all", f"{o['n']:.0f}", f"{o['SUSTAINED_pct']:.1f}", f"{o['BUSTED_pct']:.1f}", f"{o['NEITHER_pct']:.1f}")
    for m, r in report.base_by_month.items():
        t.add_row(m, f"{r['n']:.0f}", f"{r['SUSTAINED_pct']:.1f}", f"{r['BUSTED_pct']:.1f}", f"{r['NEITHER_pct']:.1f}")
    console.print(t)
    console.print(f"  breakeven failure rate (Bulkowski) for this definition: {o['BUSTED_pct']:.1f}%")

    console.print("\n[bold]3. Diagnostic[/bold]")
    render_diagnostic(report.diagnostic, console, f"rvol lookback {c['rvol_lookback_sessions']} (as specified)")
    for lb, d in report.diagnostic_sensitivity.items():
        console.print(
            f"\n  sensitivity run, rvol lookback {lb}: verdict {d.verdict}; train AUC {d.train_auc:.3f}, "
            f"test AUC {d.test_auc:.3f} (CI {d.test_auc_ci[0]:.3f}-{d.test_auc_ci[1]:.3f}); "
            f"{d.n_test} test rows, bust rate {d.base_rate_test:.1%}"
        )

    console.print("\n[bold]4-6. Edge vs benchmark, expectancy, segments[/bold]")
    for name, s in report.backtests.items():
        render_backtest(s, console)
    if report.gate_closed:
        console.print(f"\n  {GATED_SETUP}: not run - the Stage 6 gate is closed (verdict {report.diagnostic.verdict}).")

    console.print("\n[bold]7. Caveats[/bold]")
    console.print(CAVEATS, markup=False, highlight=False)


def save_report_text(console: Console, data_dir: Path) -> Path:
    path = data_dir / "report.txt"
    path.write_text(console.export_text(), encoding="utf-8")
    return path

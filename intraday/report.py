"""Report assembly and rendering, from persisted artifacts only.

The report opens with four plain sentences - data, base rates, what separates failures,
what trading it costs - and only then shows the tables behind them. Nothing is
recomputed here; every number comes from a file a previous step wrote.
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

from intraday.analysis import MIN_GAP_PCT, Study
from intraday.backtest import BacktestSummary, load_backtest, render_backtest
from intraday.config import IST, Config
from intraday.features import FEATURE_MECHANISM
from intraday.forecast import Score as ForecastScore
from intraday.forecast import load_predictions, score as score_forecast
from intraday.labelling import base_rates, load_breakouts
from intraday.setups import SETUPS
from intraday.setups.failed_orb import NAME as GATED_SETUP
from intraday.setups.failed_orb import STUDY_FILE
from intraday.store import BarStore, read_jsonl
from intraday.validate import RESEARCH_VERDICTS, SessionVerdict, Verdict

CAVEATS = """Research output. Not advice, not a prediction, not a recommendation.
- Current data source: yfinance, 60-day limit, one market regime only.
- Results from a single regime do not generalise. Treat as preliminary
  until re-run on multi-year history.
- Costs modelled; real slippage varies with size and liquidity.
- "No edge detected" is a valid and useful result."""

SetupVerdict = Literal["edge detected", "no edge", "insufficient sample"]

MIN_CALIBRATION_BIN = 20  # a band with fewer rows than this is not worth quoting in prose


class IntegritySummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    sessions_validated: int
    verdict_counts: dict[str, int]
    research_sessions: int
    symbols: int
    per_symbol_non_research: dict[str, int]
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
    study: Study
    backtests: dict[str, BacktestSummary]
    setup_verdicts: dict[str, SetupVerdict]
    gate_closed: bool
    n_sessions: int
    first_date: str
    last_date: str
    forecast: ForecastScore | None  # None when the history is too short to forecast on
    money: dict[str, object] | None  # rupee summary of the first setup, None if it has no trades
    bottom_line: tuple[str, ...]


# ---- pieces ---------------------------------------------------------------------------


def integrity_summary(
    verdicts: dict[tuple[str, object], SessionVerdict], daily_verdicts: dict[tuple[str, object], SessionVerdict],
    fetch_log: list[dict[str, object]], index_symbol: str,
) -> IntegritySummary:
    equities = {k: v for k, v in verdicts.items() if k[0] != index_symbol}
    counts = Counter(v.verdict.value for v in equities.values())
    non_research: Counter[str] = Counter()
    for (sym, _), v in equities.items():
        if v.verdict not in RESEARCH_VERDICTS:
            non_research[sym] += 1
    runs = [r for r in fetch_log if r.get("event") == "fetch_run"]
    return IntegritySummary(
        sessions_validated=len(equities),
        verdict_counts={v.value: counts.get(v.value, 0) for v in Verdict},
        research_sessions=sum(1 for v in equities.values() if v.verdict in RESEARCH_VERDICTS),
        symbols=len({k[0] for k in equities}),
        per_symbol_non_research=dict(sorted(non_research.items())),
        daily_rows=len(daily_verdicts),
        daily_corrupt=sum(1 for v in daily_verdicts.values() if v.verdict is Verdict.CORRUPT),
        last_fetch=runs[-1] if runs else None,
    )


def setup_verdict(summary: BacktestSummary) -> SetupVerdict:
    """edge detected: the base variant beats random with a CI above zero AND has positive
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
    return {
        "n": float(r["n"]), "SUSTAINED_pct": float(r["SUSTAINED_pct"]),
        "BUSTED_pct": float(r["BUSTED_pct"]), "NEITHER_pct": float(r["NEITHER_pct"]),
    }


# ---- build -------------------------------------------------------------------------------


def build_report(config: Config) -> Report:
    data_dir = config.data_dir
    store = BarStore(data_dir, config.interval)
    daily_store = BarStore(data_dir, config.daily_interval)
    verdicts = store.load_verdicts()
    if not verdicts:
        raise FileNotFoundError(f"{store.verdicts_path} is empty; run update first")
    integrity = integrity_summary(verdicts, daily_store.load_verdicts(), read_jsonl(store.fetch_log_path), config.index_symbol)

    breakouts = load_breakouts(data_dir)
    by_month = breakouts.assign(month=breakouts["session_date"].astype(str).str.slice(0, 7))

    study_path = data_dir / STUDY_FILE
    if not study_path.exists():
        raise FileNotFoundError(f"{study_path} not found; run study first")
    study = Study.model_validate_json(study_path.read_text(encoding="utf-8"))

    backtests: dict[str, BacktestSummary] = {}
    gate_closed = study.verdict != "signal"
    for name in SETUPS:
        p = data_dir / f"backtest_{name}.json"
        if name == GATED_SETUP and gate_closed:
            # A result left over from a run when the gate was open is not a current result.
            continue
        if p.exists():
            backtests[name] = load_backtest(name, data_dir)
        elif name != GATED_SETUP:
            raise FileNotFoundError(f"{p} not found; run study first")

    dates = sorted(breakouts["session_date"].astype(str).unique())
    verdicts_by_setup = {name: setup_verdict(s) for name, s in backtests.items()}
    try:
        forecast = score_forecast(load_predictions(data_dir), config)
    except (FileNotFoundError, ValueError):
        forecast = None
    money = _money_summary(backtests, data_dir, config)
    return Report(
        generated_at=datetime.now(tz=IST),
        config=json.loads(config.model_dump_json()),
        integrity=integrity,
        n_breakouts=len(breakouts),
        base_overall=_rates(breakouts),
        base_by_month={m: _rates(g) for m, g in by_month.groupby("month")},
        study=study,
        backtests=backtests,
        setup_verdicts=verdicts_by_setup,
        gate_closed=gate_closed,
        n_sessions=len(dates),
        first_date=dates[0] if dates else "-",
        last_date=dates[-1] if dates else "-",
        forecast=forecast,
        money=money,
        bottom_line=_bottom_line(study, verdicts_by_setup, money, forecast),
    )


def _money_summary(
    backtests: dict[str, BacktestSummary], data_dir: Path, config: Config
) -> dict[str, object] | None:
    """Per-trade and total rupees for the first setup, read from its results table."""
    if not backtests:
        return None
    name = next(iter(backtests))
    path = data_dir / f"backtest_{name}.parquet"
    if not path.exists():
        return None
    table = pd.read_parquet(path)
    base = table[(table["role"] == "strategy") & (table["variant"] == "base")]
    twins = table[(table["role"] == "random") & (table["variant"] == "base")]
    if base.empty:
        return None
    rows = []
    for bps in sorted(base["slippage_bps"].unique()):
        g = base[base["slippage_bps"] == bps]
        rows.append({"bps": int(bps), "per_trade": float(g["net_pnl"].mean()), "total": float(g["net_pnl"].sum())})
    mid_bps = int(sorted(base["slippage_bps"].unique())[len(rows) // 2])
    g = base[base["slippage_bps"] == mid_bps]
    t = twins[twins["slippage_bps"] == mid_bps]
    edge = next(
        (v.edge for v in backtests[name].variants if v.variant == "base" and v.slippage_bps == mid_bps), None
    )
    return {
        "setup": name,
        "position": float(config.position_inr),
        "n_trades": int(len(g)),
        "by_slippage": rows,
        "mid": {
            "bps": mid_bps,
            "wins_in_ten": round(float((g["net_pnl"] > 0).mean()) * 10),
            "cost": float(g["cost_total"].mean()),
            "random": float(t["net_pnl"].mean()) if not t.empty else 0.0,
            "same_as_random": bool(edge.indistinguishable_from_zero) if edge else False,
        },
    }


def _bottom_line(
    study: Study, verdicts: dict[str, SetupVerdict], money: dict[str, object] | None,
    forecast: ForecastScore | None,
) -> tuple[str, ...]:
    """Two or three sentences a reader can act on: what was found, and what it means."""
    lines: list[str] = []
    traded_badly = [n for n, v in verdicts.items() if v == "no edge"]
    if traded_badly:
        lines.append(f"This setup ({', '.join(traded_badly)}) lost money on this data, at every slippage level.")

    # The forecast is the stronger evidence about predictability, so it speaks first.
    if forecast is not None and forecast.verdict == "informative":
        cost = f"Rs {money['mid']['cost']:,.0f}" if money else "the cost per trade"  # type: ignore[index]
        lines.append(
            "Failures ARE partly predictable: forecasts made before each session beat the base rate. "
            f"The separation is just worth less than {cost} a round trip, so it does not become profit."
        )
    elif forecast is not None:
        lines.append("Forecasts were made before each session and did not beat the base rate.")
    elif study.verdict == "signal":
        held = ", ".join(f.feature for f in study.findings if f.holds)
        lines.append(f"{held} did separate the failures, but separating them is not the same as profiting from them.")
    elif study.verdict == "no_signal":
        lines.append("Nothing here tells you which breakouts to avoid: the warning signs did not work.")
    else:
        lines.append("There was not enough data to test the warning signs, so nothing is claimed about them.")

    lines.append("Do not trade this on the strength of this report. It measures the past; it does not promise the future.")
    return tuple(lines)


# ---- render ------------------------------------------------------------------------------


def _headline(report: Report) -> list[str]:
    """The four sentences. Everything below them is supporting detail."""
    i = report.integrity
    o = report.base_overall
    lines = [
        f"1. Data: {i.research_sessions} usable sessions out of {i.sessions_validated} across {i.symbols} symbols. "
        f"{i.sessions_validated - i.research_sessions} rejected, {i.daily_corrupt} bad daily rows quarantined.",
        f"2. Of {report.n_breakouts} breakouts: {o['BUSTED_pct']:.0f}% failed, {o['SUSTAINED_pct']:.0f}% ran, "
        f"{o['NEITHER_pct']:.0f}% did neither"
        + (f", across {len(report.base_by_month)} months ("
           + ", ".join(f"{m} {r['BUSTED_pct']:.0f}%" for m, r in report.base_by_month.items()) + " failed)." if report.base_by_month else "."),
    ]
    # Lead with a feature the study could actually test. An untestable feature can show a
    # huge gap on a handful of failures, and putting that first would sell a non-finding.
    testable = [f for f in report.study.findings if f.testable]
    if testable:
        best = max(testable, key=lambda f: f.overall.gap_pct)
        lo, hi = best.overall.thirds[0], best.overall.thirds[-1]
        lines.append(
            f"3. Failure rate by {best.feature}:  low {lo.bust_rate_pct:.0f}%  |  "
            f"mid {best.overall.thirds[1].bust_rate_pct:.0f}%  |  high {hi.bust_rate_pct:.0f}%   "
            f"(overall {best.overall.base_rate_pct:.0f}%)."
            + (" The gap holds in both halves." if best.holds else " The gap does not hold in both halves.")
            + f" Verdict: {report.study.verdict.replace('_', ' ')}."
        )
        untestable = [f.feature for f in report.study.findings if not f.testable]
        if untestable:
            lines.append(
                f"   ({', '.join(untestable)} could not be tested: fewer than {report.study.min_half_busts} "
                "failures in a half where the feature exists. Any gap they show is not a finding.)"
            )
    else:
        lines.append(
            f"3. No feature could be tested: none has {report.study.min_half_busts} failed breakouts in both "
            f"halves where it exists. Verdict: {report.study.verdict.replace('_', ' ')}."
        )
    for name, s in report.backtests.items():
        mid = next((v for v in s.variants if v.variant == "base" and v.slippage_bps == 10), None)
        if mid is None or mid.overall is None:
            lines.append(f"4. Trading {name}: not enough trades for an expectancy.")
            continue
        lines.append(
            f"4. Trading {name}: {mid.overall.expectancy_r:+.2f}R per trade at 10bps. "
            f"Random entries: {mid.edge.mean_random_r:+.2f}R. "
            + ("The rule adds nothing." if mid.edge.indistinguishable_from_zero
               else f"The rule adds {mid.edge.edge_r:+.2f}R.")
            + f" {mid.overall.breakeven_failure_rate * 100:.0f}% of trades never reach +0.5R."
        )
    return lines


def render_plain_summary(study: Study, console: Console, min_sample: int) -> None:
    """Human-readable conclusion, printed BEFORE the statistical tables.

    The tables exist to audit the result. This block exists to read it.
    A reader who stops here must still have the correct conclusion.
    """
    colour = {"insufficient_sample": "yellow", "no_signal": "red", "signal": "green"}[study.verdict]
    console.print()
    if study.verdict == "insufficient_sample":
        console.print(f"[bold {colour}]Not enough data to answer the question yet.[/bold {colour}]")
        console.print(
            f"  No feature has {min_sample} failed breakouts in both halves of the period, so none could be "
            "tested either way. Failures per half, where each feature exists:"
        )
        for f in study.findings:
            first = f.first_half.busts if f.first_half else 0
            second = f.second_half.busts if f.second_half else 0
            console.print(f"    {f.feature}: {first} then {second}")
        console.print("  Collect more sessions, then run this again.")
        console.print("  The tables below describe what happened. They are not evidence of what happens next.")
    elif study.verdict == "no_signal":
        console.print(f"[bold {colour}]No predictive signal was found.[/bold {colour}]")
        console.print(
            f"  Each feature was cut into thirds and the failure rate counted in each. To count, a feature's "
            f"low-third to high-third gap must be at least {MIN_GAP_PCT:.0f} percentage points in both halves "
            f"of the period - the second half being data the pattern was not chosen on."
        )
        untestable = [f.feature for f in study.findings if not f.testable]
        if untestable:
            console.print(
                f"  No feature managed that. {', '.join(untestable)} could not be tested at all: fewer than "
                f"{min_sample} failures in a half where the feature exists."
            )
        else:
            console.print("  No feature managed that.")
        console.print(
            f"  Of {study.n_breakouts} breakouts, {study.base_rate_pct:.1f}% failed. Knowing the feature "
            f"values does not tell you which ones."
        )
        console.print(f"  [bold {colour}]This is a real result: do not trade this setup.[/bold {colour}]")
    else:
        held = ", ".join(f.feature for f in study.findings if f.holds)
        console.print(f"[bold {colour}]A signal was found.[/bold {colour}]")
        console.print(
            f"  Of {study.n_breakouts} breakouts, {study.base_rate_pct:.1f}% failed overall. {held} still "
            f"separates failures from the rest in the second half of the period - data the pattern was not "
            f"chosen on."
        )
        console.print("  Strongest separation first:")
        for f in sorted(study.findings, key=lambda f: -abs(f.overall.gap_pct))[:3]:
            lo, hi = f.overall.thirds[0], f.overall.thirds[-1]
            direction = "lower" if f.overall.gap_pct > 0 else "higher"
            console.print(
                f"    {f.feature}: {direction} values fail more - low third {lo.bust_rate_pct:.0f}%, high "
                f"third {hi.bust_rate_pct:.0f}% (gap {f.overall.gap_pct:+.0f} points)"
            )
        console.print(
            "  A signal in the data is not by itself a tradeable edge: this measures separation only, and "
            "does not account for costs or execution."
        )
    console.print()
    console.print(f"  verdict: [bold {colour}]{study.verdict}[/bold {colour}]")
    console.print(f"  {study.statement}")


def render_report(report: Report, console: Console) -> None:
    c = report.config
    console.rule("[bold]Intraday research report - NSE opening-range breakouts")
    console.print(
        f"generated {report.generated_at:%Y-%m-%d %H:%M} IST | source {c['source']} | interval {c['interval']} | "
        f"opening range {c['opening_range_minutes']}m | fail < {c['bust_extension_atr']} ATR, run >= "
        f"{c['sustain_extension_atr']} ATR | slippage {c['slippage_bps']} bps | min sample {c['min_sample']} | "
        f"stop {c['stop_atr_multiple']} ATR | position Rs {c['position_inr']:,.0f}"
    )

    render_plain_answer(report, console)
    render_plain_summary(report.study, console, int(c["min_sample"]))
    console.print("[dim]Detail below - for auditing the result above.[/dim]")

    console.print("\n[bold]In four sentences[/bold]")
    for line in _headline(report):
        console.print(f"  {line}")

    console.print("\n[bold]Verdicts[/bold]")
    console.print(f"  does anything separate failed breakouts?  [bold]{report.study.verdict}[/bold]")
    for name, v in report.setup_verdicts.items():
        console.print(f"  setup {name}:  [bold]{v}[/bold]")
    if report.gate_closed:
        console.print(f"  setup {GATED_SETUP}:  [bold]gate closed[/bold] (needs a 'signal' verdict above)")

    i = report.integrity
    console.print("\n[bold]1. Data integrity[/bold]")
    console.print(
        f"  {i.sessions_validated} equity sessions across {i.symbols} symbols: "
        + "  ".join(f"{k} {v}" for k, v in i.verdict_counts.items())
        + f"  -> research set {i.research_sessions}"
    )
    if i.per_symbol_non_research:
        console.print("  rejected per symbol: " + ", ".join(f"{k} {v}" for k, v in i.per_symbol_non_research.items()))
    console.print(f"  daily bars: {i.daily_rows} rows, {i.daily_corrupt} CORRUPT (quarantined)")
    if i.last_fetch:
        lf = i.last_fetch
        console.print(f"  last fetch: {str(lf.get('ts'))[:16]} source={lf.get('source')} days={lf.get('days')} failed={lf.get('failed')}")

    console.print(f"\n[bold]2. Base rates[/bold]  ({report.n_breakouts} breakouts)")
    t = Table()
    for col in ("period", "n", "ran %", "failed %", "neither %"):
        t.add_column(col, justify="left" if col == "period" else "right")
    o = report.base_overall
    t.add_row("all", f"{o['n']:.0f}", f"{o['SUSTAINED_pct']:.1f}", f"{o['BUSTED_pct']:.1f}", f"{o['NEITHER_pct']:.1f}")
    for m, r in report.base_by_month.items():
        t.add_row(m, f"{r['n']:.0f}", f"{r['SUSTAINED_pct']:.1f}", f"{r['BUSTED_pct']:.1f}", f"{r['NEITHER_pct']:.1f}")
    console.print(t)
    console.print(
        "  [dim]How to read: shares of every labelled breakout, overall and by month. 'failed' and 'ran' "
        "mean the ATR thresholds in the header line above, not profit or loss.[/dim]"
    )
    console.print(f"  breakeven failure rate (Bulkowski) for this definition: {o['BUSTED_pct']:.1f}%")

    console.print("\n[bold]3. What separates a failed breakout[/bold]")
    render_study(report.study, console)

    if report.forecast is not None:
        console.print("\n[bold]4. Forecast: predictions made before the outcome, then scored[/bold]")
        render_forecast(report.forecast, console)
        console.print("\n[bold]5. Cost of trading it[/bold]")
    else:
        console.print("\n[bold]4. Cost of trading it[/bold]")
    for name, s in report.backtests.items():
        render_backtest(s, console)
    if report.gate_closed:
        console.print(f"\n  {GATED_SETUP}: not run - the gate is closed (verdict {report.study.verdict}).")

    console.print("\n[bold]6. Caveats[/bold]" if report.forecast is not None else "\n[bold]5. Caveats[/bold]")
    console.print(CAVEATS, markup=False, highlight=False)


def render_study(study: Study, console: Console) -> None:
    console.print(
        f"  {study.n_breakouts} breakouts, {study.base_rate_pct:.1f}% failed overall. "
        + (f"Halves split at {study.split_date}; second half holds {study.second_half_busts} failures."
           if study.split_date else "Too few dates to split into halves.")
    )
    for f in study.findings:
        t = Table(title=f"{f.feature} - {FEATURE_MECHANISM[f.feature]}")
        for col in ("rows", "third", "range", "n", "failed", "fail rate", "vs overall"):
            t.add_column(col, justify="left" if col in ("rows", "third", "range") else "right")
        for table in (f.overall, f.first_half, f.second_half):
            if table is None:
                continue
            for third in table.thirds:
                t.add_row(
                    table.scope, third.name, f"{third.lower:.2f} - {third.upper:.2f}", str(third.n), str(third.busts),
                    f"{third.bust_rate_pct:.1f}%", f"{third.bust_rate_pct - table.base_rate_pct:+.1f}",
                )
        console.print(t)
        console.print(
            "  [dim]How to read: each third holds a third of the breakouts, cut by feature value. "
            "'vs overall' is the column that matters - a fail rate is only meaningful against the rate "
            "for these rows, so 25% against a 20% base is nearly nothing. The gap must hold in BOTH "
            "halves to count.[/dim]"
        )
        colour = "green" if f.holds else ("red" if not f.testable else "yellow")
        console.print(f"  [{colour}]{f.statement}")
    console.print(f"\n  [bold]{study.statement}")
    console.print(
        f"  (a feature counts only if it has {study.min_half_busts} failures in each half AND a low-vs-high "
        f"gap of at least {MIN_GAP_PCT:.0f} points in both)"
    )


def save_report_text(console: Console, data_dir: Path) -> Path:
    path = data_dir / "report.txt"
    path.write_text(console.export_text(), encoding="utf-8")
    return path


# ---- the short version ---------------------------------------------------------------

FEATURE_PLAIN_NAME: dict[str, str] = {
    "rvol_open_15m": "quiet first 15 minutes",
    "rvol_breakout_bar": "thin volume on the breakout bar",
    "bar_body_ratio": "small candle body (price stalling)",
}


def render_plain_answer(report: Report, console: Console) -> None:
    """The whole result in rupees and plain questions, for a reader who wants the answer
    and not the statistics. Everything here is also in the tables below."""
    i = report.integrity
    o = report.base_overall
    money = report.money
    n = int(o["n"])

    console.print("=" * 66)
    console.print("  [bold]THE SHORT VERSION[/bold]")
    console.print("=" * 66)
    console.print()
    console.print("  [bold]Question:[/bold]  Do opening-range breakouts make money on these stocks?")
    verdict = "No." if all(v == "no edge" for v in report.setup_verdicts.values()) else "See below."
    console.print(f"  [bold]Answer:[/bold]    [bold]{verdict}[/bold]")
    console.print()

    console.print("  [bold]What was tested[/bold]")
    console.print(f"    {i.symbols} NSE stocks, {report.n_sessions} trading days ({report.first_date} to {report.last_date})")
    console.print(f"    {n} breakouts found" + (f", {money['n_trades']} of them tradeable" if money else ""))
    console.print()

    console.print(f"  [bold]What happened to those {n} breakouts[/bold]")
    counts = {
        "failed": (o["BUSTED_pct"], "reversed back through the range"),
        "ran": (o["SUSTAINED_pct"], "went the distance"),
        "neither": (o["NEITHER_pct"], "drifted, resolved neither way"),
    }
    for label, (pct, meaning) in counts.items():
        console.print(f"    {round(n * pct / 100):4d}  {label:<8}({pct:.0f}%)".ljust(26) + f"- {meaning}")
    console.print()

    if money:
        console.print(f"  [bold]If you had traded all {money['n_trades']} with Rs {money['position']:,.0f} a time[/bold]")
        for row in money["by_slippage"]:
            console.print(
                f"    at {row['bps']:2d} bps slippage:   Rs {row['per_trade']:>5,.0f} per trade"
                f"      Rs {row['total']:>9,.0f} in total"
            )
        mid = money["mid"]
        console.print()
        console.print(f"    {mid['wins_in_ten']} trades in 10 made money.")
        console.print(f"    Costs alone were Rs {mid['cost']:,.0f} per trade at {mid['bps']} bps.")
        console.print(
            f"    Entering at random times instead lost Rs {abs(mid['random']):,.0f} - "
            + ("the same." if mid["same_as_random"] else "different.")
        )
        console.print()

    console.print("  [bold]Could you have known in advance which breakouts would fail?[/bold]")
    fc = report.forecast
    if fc is not None and fc.verdict == "informative":
        console.print(
            f"    A little. Forecasts made before each session, scored on {fc.n} of them, beat simply "
            f"saying \"{report.base_overall['BUSTED_pct']:.0f}%\" every time."
        )
        solid = [b for b in fc.calibration if b.n >= MIN_CALIBRATION_BIN]
        if len(solid) >= 2:
            lo, hi = solid[0], solid[-1]
            console.print(
                f"    The ones it called safest failed {lo.observed_rate:.0%} of the time; the ones it "
                f"called riskiest failed {hi.observed_rate:.0%}."
            )
        console.print("    That is real, and it is still not enough to cover costs - see below.")
    elif fc is not None:
        console.print(f"    No. Forecasts were made and scored on {fc.n} breakouts; they did not beat the base rate.")
    else:
        console.print("    Not yet - too little history to make and score a forecast.")
    any_held = any(f.holds for f in report.study.findings)
    console.print(
        f"    {'Partly.' if any_held else 'Taken one at a time,'} the three warning signs were checked "
        "separately, which is a harder test than the forecast above:"
    )
    for f in report.study.findings:
        name = FEATURE_PLAIN_NAME.get(f.feature, f.feature)
        if not f.testable:
            outcome = "not enough data to judge"
        elif f.holds:
            outcome = "worked - see the tables below"
        else:
            outcome = "checked, did not work"
        console.print(f"      {name:<34} - {outcome}")
    console.print()

    console.print("  [bold]So:[/bold]")
    for line in report.bottom_line:
        console.print(f"    {line}")
    console.print()
    console.print("=" * 66)


def render_forecast(score: ForecastScore, console: Console) -> None:
    """How the forecasts were graded. Brier is mean squared error on the probability."""
    console.print(
        f"  {score.n} breakouts forecast before their session, {score.failures} of which failed. "
        f"Each forecast used a rule built only on earlier sessions."
    )
    console.print(
        f"  Brier score {score.brier:.4f} against {score.brier_base:.4f} for always quoting the base rate "
        f"-> skill {score.skill:+.1%} (95% CI {score.skill_ci[0]:+.1%} to {score.skill_ci[1]:+.1%})."
    )
    table = Table(title="Calibration: what it said against what happened")
    for col in ("forecast band", "n", "average forecast", "actually failed", "difference"):
        table.add_column(col, justify="left" if col == "forecast band" else "right")
    for b in score.calibration:
        table.add_row(
            f"{b.lower:.0%} - {b.upper:.0%}", str(b.n), f"{b.mean_forecast:.1%}",
            f"{b.observed_rate:.1%}", f"{b.observed_rate - b.mean_forecast:+.1f} pp",
        )
    console.print(table)
    console.print(
        "  How to read: a well-calibrated forecast has 'actually failed' close to 'average forecast' in "
        "every band, and a useful one has the bands far apart. Skill above zero means it beat the base rate."
    )
    console.print(f"  [bold]{score.statement}")

"""Backtest orchestration for one setup: trades -> results per slippage level -> matched
random benchmark -> expectancy (overall per variant, then segments). Persists a results
table (strategy rows and their random twins) and a JSON summary for the report.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict
from rich.console import Console
from rich.table import Table

from intraday.benchmark import EdgeVsRandom, SessionUniverse, edge_vs_random, matched_random
from intraday.config import IST, Config
from intraday.expectancy import Expectancy, InsufficientSampleError, expectancy, segment_keys, segmented_expectancy
from intraday.setups import SetupRun, run_setup
from intraday.setups.common import VARIANTS
from intraday.store import BarStore
from intraday.trades import TradeResult, evaluate, results_frame
from intraday.trading_calendar import TradingCalendar

SEGMENTS = (("seg_rvol", "RVOL bucket"), ("seg_or_width", "OR-width quartile"), ("seg_half_hour", "entry half-hour"))


class VariantSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    variant: str
    slippage_bps: int
    n: int
    edge: EdgeVsRandom
    overall: Expectancy | None  # None when below the minimum sample
    overall_n: int
    segments: dict[str, dict[str, Expectancy]]
    skipped_segments: dict[str, dict[str, int]]


class BacktestSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    setup: str
    run_at: datetime
    sessions_seen: int
    sessions_without_atr: int
    skipped_late_entries: int
    n_signals: int
    slippage_bps: tuple[int, ...]
    variants: tuple[VariantSummary, ...]


def _evaluate_all(run: SetupRun, bps: int, config: Config) -> list[TradeResult]:
    return [evaluate(t, bps, config) for t in run.trades]


def run_backtest(
    setup: str,
    slippage_bps: tuple[int, ...],
    store: BarStore,
    daily_store: BarStore,
    features: pd.DataFrame,
    calendar: TradingCalendar,
    config: Config,
    data_dir: Path,
) -> tuple[BacktestSummary, pd.DataFrame]:
    run = run_setup(setup, store, daily_store, features, calendar, config, data_dir)
    if not run.trades:
        raise ValueError(f"{setup}: no trades generated")
    universe = SessionUniverse(store, daily_store, calendar, config)
    window = (config.first_entry_time, config.last_entry_time)
    frames: list[pd.DataFrame] = []
    variants: list[VariantSummary] = []
    for bps in slippage_bps:
        results = _evaluate_all(run, bps, config)
        twins = matched_random(results, universe, window, config, config.benchmark_seed)
        frames.append(results_frame(results).assign(role="strategy"))
        frames.append(results_frame(twins).assign(role="random"))
        for variant in VARIANTS:
            s = [r for r in results if r.trade.variant == variant]
            t = [tw for r, tw in zip(results, twins) if r.trade.variant == variant]
            edge = edge_vs_random(s, t, config, config.benchmark_seed)
            df = segment_keys(results_frame(s), config)
            try:
                overall: Expectancy | None = expectancy(
                    df["r_net"], df["mfe_r"], df["session_date"],
                    f"{setup}/{variant} @ {bps}bps", config, config.benchmark_seed,
                )
            except InsufficientSampleError:
                overall = None
            segs: dict[str, dict[str, Expectancy]] = {}
            skipped: dict[str, dict[str, int]] = {}
            for col, _ in SEGMENTS:
                segs[col], skipped[col] = segmented_expectancy(df, col, config, config.benchmark_seed)
            variants.append(VariantSummary(
                variant=variant, slippage_bps=bps, n=len(s), edge=edge, overall=overall, overall_n=len(s),
                segments=segs, skipped_segments=skipped,
            ))
    summary = BacktestSummary(
        setup=setup, run_at=datetime.now(tz=IST), sessions_seen=run.sessions_seen,
        sessions_without_atr=run.sessions_without_atr, skipped_late_entries=run.skipped_late_entries,
        n_signals=len(run.trades) // len(VARIANTS), slippage_bps=tuple(slippage_bps), variants=tuple(variants),
    )
    return summary, pd.concat(frames, ignore_index=True)


def save_backtest(summary: BacktestSummary, table: pd.DataFrame, data_dir: Path) -> tuple[Path, Path]:
    p_json = data_dir / f"backtest_{summary.setup}.json"
    p_parq = data_dir / f"backtest_{summary.setup}.parquet"
    p_json.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    table.to_parquet(p_parq, index=False)
    return p_json, p_parq


def load_backtest(setup: str, data_dir: Path) -> BacktestSummary:
    p = data_dir / f"backtest_{setup}.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run backtest --setup {setup} first")
    return BacktestSummary.model_validate_json(p.read_text(encoding="utf-8"))


# ---- rendering ---------------------------------------------------------------------------


def render_backtest(summary: BacktestSummary, console: Console) -> None:
    console.rule(f"Backtest: {summary.setup}")
    console.print(
        f"{summary.n_signals} signals over {summary.sessions_seen} sessions "
        f"({summary.sessions_without_atr} without ATR, {summary.skipped_late_entries} signals after last entry time); "
        f"slippage levels {list(summary.slippage_bps)} bps; position Rs 50,000; stop = ATR multiple"
    )

    lead = Table(title="Edge vs matched random benchmark (net R, paired bootstrap 95% CI) - THE HEADLINE")
    for col in ("variant", "bps", "n", "strategy R", "random R", "edge R", "CI low", "CI high", "verdict"):
        lead.add_column(col, justify="left" if col in ("variant", "verdict") else "right")
    for v in summary.variants:
        e = v.edge
        lead.add_row(
            v.variant, str(v.slippage_bps), str(e.n), f"{e.mean_strategy_r:+.3f}", f"{e.mean_random_r:+.3f}",
            f"{e.edge_r:+.3f}", f"{e.ci_low:+.3f}", f"{e.ci_high:+.3f}",
            "[yellow]indistinguishable from random" if e.indistinguishable_from_zero
            else ("[green]above random" if e.edge_r > 0 else "[red]below random"),
        )
    console.print(lead)

    exp = Table(title="Expectancy (net of costs)")
    for col in ("variant", "bps", "n", "win rate", "avg win R", "avg loss R", "expectancy R", "CI low", "CI high", "breakeven fail", "verdict"):
        exp.add_column(col, justify="left" if col in ("variant", "verdict") else "right")
    for v in summary.variants:
        o = v.overall
        if o is None:
            exp.add_row(v.variant, str(v.slippage_bps), str(v.overall_n), "-", "-", "-", "-", "-", "-", "-", "[red]insufficient sample")
            continue
        exp.add_row(
            v.variant, str(v.slippage_bps), str(o.n), f"{o.win_rate:.1%}", f"{o.avg_win_r:+.2f}", f"{o.avg_loss_r:+.2f}",
            f"{o.expectancy_r:+.3f}", f"{o.ci_low:+.3f}", f"{o.ci_high:+.3f}", f"{o.breakeven_failure_rate:.1%}",
            "[yellow]indistinguishable from zero" if o.indistinguishable_from_zero
            else ("[green]positive" if o.expectancy_r > 0 else "[red]negative"),
        )
    console.print(exp)

    for col, title in SEGMENTS:
        seg = Table(title=f"Segments by {title} (base variant)")
        for c in ("bps", "segment", "n", "expectancy R", "CI low", "CI high", "breakeven fail", "verdict"):
            seg.add_column(c, justify="left" if c in ("segment", "verdict") else "right")
        for v in summary.variants:
            if v.variant != "base":
                continue
            for name, e in v.segments[col].items():
                seg.add_row(
                    str(v.slippage_bps), name, str(e.n), f"{e.expectancy_r:+.3f}", f"{e.ci_low:+.3f}", f"{e.ci_high:+.3f}",
                    f"{e.breakeven_failure_rate:.1%}", "indistinguishable from zero" if e.indistinguishable_from_zero else "distinguishable",
                )
            for name, n in v.skipped_segments[col].items():
                seg.add_row(str(v.slippage_bps), name, str(n), "-", "-", "-", "-", "[red]insufficient sample")
        console.print(seg)

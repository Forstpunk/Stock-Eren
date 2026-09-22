"""Command-line entry point. Each subcommand maps to one pipeline stage."""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from pathlib import Path

from rich.console import Console

from intraday.config import DEFAULT_CONFIG, Config

Handler = Callable[[argparse.Namespace, Config], int]


def cmd_fetch(args: argparse.Namespace, config: Config) -> int:
    from intraday.fetch import fetch_universe, print_tally, read_universe
    from intraday.sources import make_source
    from intraday.store import BarStore
    from intraday.trading_calendar import TradingCalendar

    console = Console()
    symbols = read_universe(Path(args.universe))
    if config.index_symbol not in symbols:
        symbols.append(config.index_symbol)
    source = make_source(config)
    store = BarStore(config.data_dir, config.interval)
    daily_store = BarStore(config.data_dir, config.daily_interval)
    calendar = TradingCalendar.from_csv(config.data_dir / "nse_holidays.csv")
    console.print(
        f"source={source.name} interval={config.interval} days={args.days} "
        f"daily={config.daily_interval} x {config.daily_history_days}d symbols={len(symbols)}"
    )
    summary = fetch_universe(symbols, args.days, config, source, store, daily_store, calendar, console)
    print_tally(summary, console)
    return 1 if summary.failed else 0


def cmd_label(args: argparse.Namespace, config: Config) -> int:
    from intraday.labelling import base_rates, label_universe, save_breakouts
    from intraday.store import BarStore

    from intraday.trading_calendar import TradingCalendar

    console = Console()
    store = BarStore(config.data_dir, config.interval)
    daily_store = BarStore(config.data_dir, config.daily_interval)
    calendar = TradingCalendar.from_csv(config.data_dir / "nse_holidays.csv")
    df, skipped = label_universe(store, daily_store, calendar, config)
    path = save_breakouts(df, config.data_dir)
    sessions = store_session_count(store, config)
    n_skipped = sum(len(v) for v in skipped.values())
    console.print(
        f"{len(df)} breakout events over {sessions - n_skipped} labelled research sessions "
        f"({df['session_date'].nunique()} dates, {df['symbol'].nunique()} symbols) -> {path}"
    )
    console.print(
        f"thresholds: sustain >= {config.sustain_extension_atr} ATR, bust < {config.bust_extension_atr} ATR "
        f"and close through opposite boundary before {config.bust_cutoff}"
    )
    if n_skipped:
        console.print(f"[yellow]{n_skipped} sessions not labelled (no prior-day ATR): "
                      + ", ".join(f"{s} x{len(d)}" for s, d in skipped.items()))
    console.print(f"sessions with no breakout: {sessions - df.groupby(['symbol', 'session_date']).ngroups}")
    _print_rates(console, "Base rates - overall", base_rates(df))
    _print_rates(console, "By direction", base_rates(df, "direction"))
    df = df.assign(month=df["session_date"].astype(str).str.slice(0, 7))
    _print_rates(console, "By month", base_rates(df, "month"))
    df = df.assign(half_hour=(df["minutes_since_open"] // 30 * 30).map(lambda m: f"+{m:03d}m"))
    _print_rates(console, "By minutes since open (30m buckets)", base_rates(df, "half_hour"))
    _print_rates(console, "By symbol", base_rates(df, "symbol"))
    return 0


def store_session_count(store, config: Config) -> int:  # type: ignore[no-untyped-def]
    return sum(len(store.research_sessions(s)) for s in store.symbols() if s != config.index_symbol)


def _print_rates(console: Console, title: str, rates) -> None:  # type: ignore[no-untyped-def]
    from rich.table import Table

    table = Table(title=title)
    table.add_column("group")
    for col in ("n", "SUSTAINED", "BUSTED", "NEITHER", "SUSTAINED_pct", "BUSTED_pct", "NEITHER_pct"):
        table.add_column(col.replace("_pct", " %"), justify="right")
    for group, row in rates.iterrows():
        table.add_row(
            str(group), str(int(row["n"])), str(int(row["SUSTAINED"])), str(int(row["BUSTED"])), str(int(row["NEITHER"])),
            f"{row['SUSTAINED_pct']:.1f}", f"{row['BUSTED_pct']:.1f}", f"{row['NEITHER_pct']:.1f}",
        )
    console.print(table)


def cmd_features(args: argparse.Namespace, config: Config) -> int:
    import pandas as pd

    from intraday.features import FEATURE_MECHANISM, FEATURE_NAMES, build_feature_table, save_features
    from intraday.labelling import load_breakouts
    from intraday.store import BarStore
    from intraday.trading_calendar import TradingCalendar

    console = Console()
    breakouts = load_breakouts(config.data_dir)
    store = BarStore(config.data_dir, config.interval)
    daily_store = BarStore(config.data_dir, config.daily_interval)
    calendar = TradingCalendar.from_csv(config.data_dir / "nse_holidays.csv")
    df = build_feature_table(breakouts, store, daily_store, calendar, config)
    path = save_features(df, config.data_dir)
    console.print(f"{len(df)} breakouts featurised -> {path}")
    feats = df[list(FEATURE_NAMES)]
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        console.print(feats.describe().T.to_string())
    console.print()
    console.print("non-null rows per feature:")
    for name in FEATURE_NAMES:
        n = int(feats[name].notna().sum())
        console.print(f"  {name:<30} {n:>5} / {len(df)}   {FEATURE_MECHANISM[name]}")
    complete = feats.notna().all(axis=1)
    console.print()
    console.print(f"rows with every feature present: {int(complete.sum())} / {len(df)}")
    console.print("label mix among complete rows: " + df.loc[complete, "label"].value_counts().to_dict().__repr__())
    return 0


def cmd_diagnose(args: argparse.Namespace, config: Config) -> int:
    from intraday.diagnostic import run_diagnostic
    from intraday.diagnostic_render import render
    from intraday.features import build_feature_table, load_features
    from intraday.labelling import load_breakouts
    from intraday.store import BarStore
    from intraday.trading_calendar import TradingCalendar

    console = Console()
    if args.rvol_lookback == config.rvol_lookback_sessions:
        df = load_features(config.data_dir)
        title = f"Diagnostic (rvol lookback {config.rvol_lookback_sessions} sessions, as specified)"
    else:
        alt = config.model_copy(update={"rvol_lookback_sessions": args.rvol_lookback})
        store = BarStore(alt.data_dir, alt.interval)
        calendar = TradingCalendar.from_csv(alt.data_dir / "nse_holidays.csv")
        df = build_feature_table(load_breakouts(alt.data_dir), store, BarStore(alt.data_dir, alt.daily_interval), calendar, alt)
        title = f"Diagnostic SENSITIVITY RUN (rvol lookback {args.rvol_lookback} sessions, not the specified 20)"
    report = run_diagnostic(df, df["label"], config)
    render(report, console, title)
    out = config.data_dir / f"diagnostic_rvol{args.rvol_lookback}.json"
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"report saved -> {out}")
    return 0


def cmd_backtest(args: argparse.Namespace, config: Config) -> int:
    from intraday.backtest import render_backtest, run_backtest, save_backtest
    from intraday.features import load_features
    from intraday.store import BarStore
    from intraday.trading_calendar import TradingCalendar

    console = Console()
    store = BarStore(config.data_dir, config.interval)
    daily_store = BarStore(config.data_dir, config.daily_interval)
    calendar = TradingCalendar.from_csv(config.data_dir / "nse_holidays.csv")
    features = load_features(config.data_dir)
    summary, table = run_backtest(
        args.setup, tuple(args.slippage_bps), store, daily_store, features, calendar, config, config.data_dir
    )
    render_backtest(summary, console)
    p_json, p_parq = save_backtest(summary, table, config.data_dir)
    console.print(f"saved -> {p_json}, {p_parq}")
    return 0


def cmd_report(args: argparse.Namespace, config: Config) -> int:
    from intraday.report import build_report, render_report, save_report_text

    console = Console(record=True, width=150)
    report = build_report(config)
    render_report(report, console)
    path = save_report_text(console, config.data_dir)
    console.print(f"report text saved -> {path}")
    return 0


HANDLERS: dict[str, Handler] = {
    "fetch": cmd_fetch,
    "label": cmd_label,
    "features": cmd_features,
    "diagnose": cmd_diagnose,
    "backtest": cmd_backtest,
    "report": cmd_report,
}


def _parse_bps(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(","))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m intraday",
        description=(
            "Research pipeline measuring whether intraday setups on NSE equities have edge after costs. "
            "Emits verdicts only - never signals, ratings or predictions."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="fetch and validate bars for a universe")
    fetch.add_argument("--universe", required=True, help="text file, one NSE symbol per line")
    fetch.add_argument("--days", type=int, required=True, help="calendar days of history to request")

    sub.add_parser("label", help="label opening-range breakouts on research sessions")
    sub.add_parser("features", help="compute the feature table for every labelled breakout")
    diagnose = sub.add_parser("diagnose", help="run the bust-vs-sustain diagnostic study")
    diagnose.add_argument(
        "--rvol-lookback", type=int, default=DEFAULT_CONFIG.rvol_lookback_sessions,
        help="sessions in the RVOL lookback; a value other than the config default is a labelled sensitivity run",
    )

    backtest = sub.add_parser("backtest", help="run a setup with the matched random benchmark")
    backtest.add_argument("--setup", required=True, choices=["orb", "failed_orb"])
    backtest.add_argument(
        "--slippage-bps", type=_parse_bps, default=DEFAULT_CONFIG.slippage_bps,
        help="comma-separated slippage levels in bps (default: 5,10,20)",
    )

    sub.add_parser("report", help="render the full research report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return HANDLERS[args.command](args, DEFAULT_CONFIG)


if __name__ == "__main__":
    sys.exit(main())

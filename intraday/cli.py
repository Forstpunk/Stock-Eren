"""Command-line entry point. Two commands:

    update   fetch bars and validate every session
    study    label breakouts, measure them, backtest the setups, write the report
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from rich.console import Console

from intraday.config import DEFAULT_CONFIG, Config

Handler = Callable[[argparse.Namespace, Config], int]


def cmd_update(args: argparse.Namespace, config: Config) -> int:
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
    if summary.failed:
        return 1
    console.print("\nnow run: python -m intraday study")
    return 0


def cmd_study(args: argparse.Namespace, config: Config) -> int:
    from intraday.analysis import run_study
    from intraday.forecast import save_predictions, walk_forward
    from intraday.backtest import run_backtest, save_backtest
    from intraday.features import build_feature_table, save_features
    from intraday.labelling import label_universe, save_breakouts
    from intraday.report import build_report, render_plain_answer, render_plain_summary, render_report, save_report_text
    from intraday.setups import SETUPS
    from intraday.setups.failed_orb import GateClosedError
    from intraday.setups.failed_orb import NAME as GATED_SETUP
    from intraday.setups.failed_orb import STUDY_FILE
    from intraday.store import BarStore
    from intraday.trading_calendar import TradingCalendar

    console = Console()
    store = BarStore.for_config(config, config.interval)
    daily_store = BarStore.for_config(config, config.daily_interval)
    calendar = TradingCalendar.from_csv(config.data_dir / "nse_holidays.csv")
    if config.holdout_from is not None:
        if store.sealed:
            console.print(f"[green]holdout sealed: sessions from {config.holdout_from} are not readable")
        else:
            console.print(
                f"[red]HOLDOUT UNSEALED: sessions from {config.holdout_from} are included. "
                "Everything below is in-sample from here on."
            )

    console.print("labelling breakouts ...")
    breakouts, skipped = label_universe(store, daily_store, calendar, config)
    save_breakouts(breakouts, config.data_dir)
    n_skipped = sum(len(v) for v in skipped.values())
    console.print(f"  {len(breakouts)} breakouts" + (f"; {n_skipped} sessions unlabelled (no prior-day ATR)" if n_skipped else ""))

    console.print("measuring them ...")
    features = build_feature_table(breakouts, store, daily_store, calendar, config)
    save_features(features, config.data_dir)
    study = run_study(features, config)
    (config.data_dir / STUDY_FILE).write_text(study.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"  verdict: {study.verdict}")

    console.print("forecasting (walk-forward, each session predicted from earlier ones only) ...")
    try:
        predictions = walk_forward(features, config)
        save_predictions(predictions, config.data_dir)
        console.print(f"  {len(predictions)} predictions made and saved")
        # The other combination on identical predictions, so the report can show both.
        other = "average" if config.forecast_combination == "logodds" else "logodds"
        walk_forward(features, config, combination=other).to_parquet(
            config.data_dir / f"predictions_{other}.parquet", index=False
        )
        console.print(f"  also scored with the {other} combination, for comparison")
    except ValueError as exc:
        console.print(f"  [yellow]skipped: {exc}")

    for name in SETUPS:
        console.print(f"backtesting {name} ...")
        try:
            summary, table = run_backtest(
                name, tuple(args.slippage_bps), store, daily_store, features, calendar, config, config.data_dir
            )
        except GateClosedError as exc:
            console.print(f"  [yellow]skipped: {exc}")
            for stale in (config.data_dir / f"backtest_{name}.json", config.data_dir / f"backtest_{name}.parquet"):
                if stale.exists():
                    stale.unlink()
                    console.print(f"  [yellow]removed stale {stale.name} from a run when the gate was open")
            continue
        save_backtest(summary, table, config.data_dir)
        console.print(f"  {summary.n_signals} signals")

    report_console = Console(record=True, width=args.width)
    full_report = build_report(config)
    render_report(full_report, report_console)
    path = save_report_text(report_console, config.data_dir)
    console.print(f"\nreport saved -> {path}")
    if not args.quiet:
        if args.plain:
            render_plain_answer(full_report, console)
        elif args.summary:
            render_plain_summary(full_report.study, console, config.min_sample)
        else:
            console.print(report_console.export_text(), markup=False, highlight=False)
    return 0


HANDLERS: dict[str, Handler] = {"update": cmd_update, "study": cmd_study}


def _parse_bps(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(","))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m intraday",
        description=(
            "Measures whether intraday opening-range breakouts on NSE equities have edge after costs. "
            "Emits verdicts only - never signals, ratings or predictions."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    update = sub.add_parser("update", help="fetch bars and validate every session")
    update.add_argument("--universe", default="universe.txt", help="text file, one NSE symbol per line")
    update.add_argument("--days", type=int, default=59, help="calendar days of history (yfinance caps 5m at 59)")

    study = sub.add_parser("study", help="label, measure, backtest and report")
    study.add_argument(
        "--slippage-bps", type=_parse_bps, default=DEFAULT_CONFIG.slippage_bps,
        help="comma-separated slippage levels in bps (default: 5,10,20)",
    )
    study.add_argument("--width", type=int, default=150, help="report width in characters")
    study.add_argument(
        "--unseal-holdout", action="store_true",
        help="read the sealed holdout period too. Meant to be used once, at the end, on a frozen "
             "configuration: after this the holdout is no longer out-of-sample.",
    )
    study.add_argument("--quiet", action="store_true", help="write the report file without printing it")
    study.add_argument(
        "--plain", action="store_true",
        help="print only the short version: the answer in rupees and plain questions, no statistics",
    )
    study.add_argument(
        "--summary", action="store_true",
        help="print only the plain-language conclusion and verdict, no tables (report.txt stays complete)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = DEFAULT_CONFIG
    if getattr(args, "unseal_holdout", False):
        config = config.model_copy(update={"holdout_unsealed": True})
    return HANDLERS[args.command](args, config)


if __name__ == "__main__":
    sys.exit(main())

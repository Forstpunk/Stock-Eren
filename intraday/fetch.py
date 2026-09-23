"""Fetch orchestration: source -> split into sessions -> validate -> store -> tally.

A symbol whose source fetch fails is recorded as a failure and reported; it is never
retried against another source. The run exits non-zero if any symbol failed.
"""
from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from rich.console import Console
from rich.table import Table

from intraday.config import IST, Config
from intraday.sources import BarSource, DataUnavailableError, SymbolNotResolvable
from intraday.store import BarStore
from intraday.trading_calendar import TradingCalendar
from intraday.validate import Verdict, drop_settling_tail, split_sessions, validate_daily_row, validate_session


class SymbolOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    expected_sessions: int
    received_sessions: int
    verdicts: dict[str, int]
    missing_dates: tuple[date, ...]
    missing_slot_counts: dict[str, int]
    daily_rows: int
    daily_corrupt: int
    daily_suspect: int
    error: str | None
    unresolvable: bool = False


class FetchSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    started_at: datetime
    start: datetime
    end: datetime
    expected_sessions: tuple[date, ...]
    outcomes: tuple[SymbolOutcome, ...]

    @property
    def failed(self) -> list[SymbolOutcome]:
        return [o for o in self.outcomes if o.error is not None]

    @property
    def verdict_totals(self) -> dict[str, int]:
        totals: Counter[str] = Counter()
        for o in self.outcomes:
            totals.update(o.verdicts)
        return {v.value: totals.get(v.value, 0) for v in Verdict}


def read_universe(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"universe file not found: {path}")
    symbols = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not symbols:
        raise ValueError(f"universe file {path} is empty")
    dupes = [s for s, n in Counter(symbols).items() if n > 1]
    if dupes:
        raise ValueError(f"universe file {path} has duplicate symbols {dupes}")
    return symbols


def fetch_daily(
    symbol: str,
    end: datetime,
    config: Config,
    source: BarSource,
    daily_store: BarStore,
    calendar: TradingCalendar,
) -> tuple[int, int, int]:
    """Fetch, validate per row and store daily bars. Returns (rows, CORRUPT, SUSPECT).
    Raises DataUnavailableError like any other fetch."""
    fetched_at = datetime.now(tz=IST)
    start = end - timedelta(days=config.daily_history_days)
    daily = source.fetch(symbol, config.daily_interval, start, end)
    validated = []
    previous_close: float | None = None
    for session_date, row in split_sessions(daily).items():
        if session_date == fetched_at.date():
            continue  # in-progress daily bar
        verdict = validate_daily_row(
            row, symbol, session_date, calendar, fetched_at, previous_close=previous_close, config=config
        )
        validated.append((verdict, row))
        previous_close = float(row["close"].iloc[0])
    daily_store.put_sessions(symbol, validated)
    corrupt = sum(1 for v, _ in validated if v.verdict is Verdict.CORRUPT)
    suspect = sum(1 for v, _ in validated if v.verdict is Verdict.SUSPECT)
    return len(validated), corrupt, suspect


def fetch_symbol(
    symbol: str,
    start: datetime,
    end: datetime,
    expected: list[date],
    config: Config,
    source: BarSource,
    store: BarStore,
    daily_store: BarStore,
    calendar: TradingCalendar,
) -> SymbolOutcome:
    fetched_at = datetime.now(tz=IST)
    try:
        bars = source.fetch(symbol, config.interval, start, end)
        daily_rows, daily_corrupt, daily_suspect = fetch_daily(
            symbol, end, config, source, daily_store, calendar
        )
    except SymbolNotResolvable as exc:
        return SymbolOutcome(
            symbol=symbol, expected_sessions=len(expected), received_sessions=0, verdicts={},
            missing_dates=tuple(expected), missing_slot_counts={}, daily_rows=0, daily_corrupt=0,
            daily_suspect=0, error=exc.reason, unresolvable=True,
        )
    except DataUnavailableError as exc:
        return SymbolOutcome(
            symbol=symbol, expected_sessions=len(expected), received_sessions=0, verdicts={},
            missing_dates=tuple(expected), missing_slot_counts={}, daily_rows=0, daily_corrupt=0,
            daily_suspect=0, error=exc.reason,
        )

    validated = []
    for session_date, frame in split_sessions(bars).items():
        frame, dropped = drop_settling_tail(frame, session_date, fetched_at)
        if frame.empty:
            continue
        verdict = validate_session(
            frame, symbol, session_date, config, calendar, fetched_at, dropped,
            expect_volume=symbol != config.index_symbol,
        )
        validated.append((verdict, frame))
    store.put_sessions(symbol, validated)

    received = {v.session_date for v, _ in validated}
    counts = Counter(v.verdict.value for v, _ in validated)
    slots: Counter[str] = Counter()
    for v, _ in validated:
        if v.verdict in (Verdict.PARTIAL, Verdict.TAIL_COLLAPSED):
            slots.update(v.missing_slots)
    return SymbolOutcome(
        symbol=symbol,
        expected_sessions=len(expected),
        received_sessions=len(received),
        verdicts={v.value: counts.get(v.value, 0) for v in Verdict},
        missing_dates=tuple(d for d in expected if d not in received),
        missing_slot_counts=dict(slots),
        daily_rows=daily_rows,
        daily_corrupt=daily_corrupt,
        daily_suspect=daily_suspect,
        error=None,
    )


def fetch_universe(
    symbols: list[str],
    days: int,
    config: Config,
    source: BarSource,
    store: BarStore,
    daily_store: BarStore,
    calendar: TradingCalendar,
    console: Console,
) -> FetchSummary:
    if days <= 0:
        raise ValueError(f"days must be positive, got {days}")
    started_at = datetime.now(tz=IST)
    end = started_at
    start = end - timedelta(days=days)
    expected = calendar.sessions_between(start.date(), end.date())

    outcomes: list[SymbolOutcome] = []
    for symbol in symbols:
        outcome = fetch_symbol(symbol, start, end, expected, config, source, store, daily_store, calendar)
        outcomes.append(outcome)
        if outcome.error:
            console.print(f"  {symbol:<12} [red]FAILED: {outcome.error}")
        else:
            console.print(
                f"  {symbol:<12} " + " ".join(f"{k}={v}" for k, v in outcome.verdicts.items())
                + f"  | daily rows={outcome.daily_rows} corrupt={outcome.daily_corrupt}"
            )

    summary = FetchSummary(
        started_at=started_at, start=start, end=end, expected_sessions=tuple(expected), outcomes=tuple(outcomes)
    )
    store.log_run({
        "source": source.name, "interval": config.interval, "days": days,
        "symbols": symbols, "expected_sessions": len(expected),
        "failed": [o.symbol for o in summary.failed],
        "verdict_totals": summary.verdict_totals,
    })
    return summary


def print_tally(summary: FetchSummary, console: Console) -> None:
    n_symbols = len(summary.outcomes)
    attempted = len(summary.expected_sessions) * n_symbols
    totals = summary.verdict_totals
    received = sum(o.received_sessions for o in summary.outcomes)
    console.print()
    console.rule("Fetch verdict tally")
    console.print(
        f"window {summary.start:%Y-%m-%d} -> {summary.end:%Y-%m-%d %H:%M} IST | "
        f"{len(summary.expected_sessions)} expected sessions x {n_symbols} symbols = {attempted} attempted"
    )
    console.print(
        f"received {received} | CLEAN {totals['CLEAN']}  TAIL_COLLAPSED {totals['TAIL_COLLAPSED']}  "
        f"PARTIAL {totals['PARTIAL']}  SUSPECT {totals['SUSPECT']}  CORRUPT {totals['CORRUPT']} "
        f"| missing {attempted - received} | research set {totals['CLEAN'] + totals['TAIL_COLLAPSED']}"
    )

    table = Table(title="Per symbol")
    for col in ("symbol", "expected", "received", "CLEAN", "TAIL_COLL", "PARTIAL", "SUSPECT", "CORRUPT", "missing dates"):
        table.add_column(col, justify="left" if col in ("symbol", "missing dates") else "right")
    for o in summary.outcomes:
        if o.error:
            table.add_row(o.symbol, str(o.expected_sessions), "0", "-", "-", "-", "-", "-", f"[red]FAILED: {o.error}")
            continue
        missing = ", ".join(d.strftime("%m-%d") for d in o.missing_dates) or "-"
        table.add_row(
            o.symbol, str(o.expected_sessions), str(o.received_sessions),
            str(o.verdicts["CLEAN"]), str(o.verdicts["TAIL_COLLAPSED"]), str(o.verdicts["PARTIAL"]), str(o.verdicts["SUSPECT"]),
            str(o.verdicts["CORRUPT"]), missing,
        )
    console.print(table)

    # Calendar cross-check: a session missing for every symbol is an unlisted holiday
    # or a source-wide gap, not a per-symbol quality problem. Report both separately.
    ok = [o for o in summary.outcomes if o.error is None]
    if ok:
        everywhere = set(summary.expected_sessions)
        for o in ok:
            everywhere &= set(o.missing_dates)
        if everywhere:
            console.print(
                "[yellow]sessions missing for ALL symbols (unlisted holiday or source gap?): "
                + ", ".join(d.isoformat() for d in sorted(everywhere))
            )
        lost = [o for o in ok if set(o.missing_dates) - everywhere]
        if lost:
            console.print("symbols that lost sessions other symbols have:")
            for o in lost:
                own = sorted(set(o.missing_dates) - everywhere)
                console.print(f"  {o.symbol:<12} " + ", ".join(d.isoformat() for d in own))
        else:
            console.print("no symbol lost a session that other symbols have")

    slots: Counter[str] = Counter()
    for o in ok:
        slots.update(o.missing_slot_counts)
    if slots:
        console.print(
            "slots missing across PARTIAL/TAIL_COLLAPSED sessions: " + ", ".join(f"{k} x{n}" for k, n in slots.most_common(8))
        )
    daily_rows = sum(o.daily_rows for o in ok)
    daily_corrupt = sum(o.daily_corrupt for o in ok)
    daily_suspect = sum(o.daily_suspect for o in ok)
    console.print(
        f"daily bars: {daily_rows} rows stored across {len(ok)} symbols, {daily_corrupt} CORRUPT, "
        f"{daily_suspect} SUSPECT (possible unadjusted corporate actions)"
    )

    unresolvable = [o for o in summary.outcomes if o.unresolvable]
    if unresolvable:
        console.print()
        console.print(
            f"[yellow]symbols not resolvable (possible survivorship bias): "
            + ", ".join(o.symbol for o in unresolvable)
        )
        console.print(
            "  The source has no instrument for these today. They may be delisted, renamed or "
            "merged. They are excluded from the study and never substituted, so any result below "
            "describes only the names that still exist."
        )
    other_failures = [o for o in summary.failed if not o.unresolvable]
    if other_failures:
        console.print(
            f"[red]{len(other_failures)} symbol(s) failed to fetch: "
            + ", ".join(o.symbol for o in other_failures)
        )

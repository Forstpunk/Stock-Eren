"""Parquet bar store and JSONL logs.

Layout under the data root:
    bars/{interval}/{SYMBOL}/{YYYY-MM}.parquet        research set: CLEAN + TAIL_COLLAPSED
    quarantine/{interval}/{SYMBOL}/{YYYY-MM}.parquet  PARTIAL / SUSPECT / CORRUPT sessions
    session_verdicts.jsonl                            one line per validated session, append-only
    fetch_log.jsonl                                   one line per add/replace and per fetch run

Writes are at session granularity. Re-fetching a session replaces that session's rows
in whichever tier they lived in, logs the replacement, and never touches other sessions.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from intraday.config import IST
from intraday.sources import BAR_COLUMNS
from intraday.validate import RESEARCH_VERDICTS, SessionVerdict, Verdict

_TIER_OF = {v: ("bars" if v in RESEARCH_VERDICTS else "quarantine") for v in Verdict}
_TIERS = ("bars", "quarantine")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.reset_index().to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _read_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(IST)
    df = df.set_index("ts").sort_index()
    return df[list(BAR_COLUMNS)]


class BarStore:
    def __init__(self, root: Path, interval: str) -> None:
        self.root = root
        self.interval = interval
        self.verdicts_path = root / "session_verdicts.jsonl"
        self.fetch_log_path = root / "fetch_log.jsonl"

    # ---- paths -------------------------------------------------------------------

    def _month_path(self, tier: str, symbol: str, month: str) -> Path:
        return self.root / tier / self.interval / symbol / f"{month}.parquet"

    def _months(self, tier: str, symbol: str) -> list[str]:
        folder = self.root / tier / self.interval / symbol
        if not folder.exists():
            return []
        return sorted(p.stem for p in folder.glob("*.parquet"))

    # ---- writes ------------------------------------------------------------------

    def put_sessions(
        self, symbol: str, sessions: list[tuple[SessionVerdict, pd.DataFrame]]
    ) -> list[dict[str, Any]]:
        """Write validated sessions, replacing any prior copy of the same session in either tier."""
        if not sessions:
            return []
        dates = [v.session_date for v, _ in sessions]
        if len(set(dates)) != len(dates):
            raise ValueError(f"{symbol}: duplicate session dates in one put_sessions call")
        for verdict, frame in sessions:
            if verdict.symbol != symbol:
                raise ValueError(f"verdict for {verdict.symbol} passed to store for {symbol}")
            if set(frame.index.date) != {verdict.session_date}:
                raise ValueError(f"{symbol} {verdict.session_date}: frame contains other dates")

        events: list[dict[str, Any]] = []
        months = sorted({d.strftime("%Y-%m") for d in dates})
        by_month: dict[str, list[tuple[SessionVerdict, pd.DataFrame]]] = {m: [] for m in months}
        for verdict, frame in sessions:
            by_month[verdict.session_date.strftime("%Y-%m")].append((verdict, frame))

        for month, items in by_month.items():
            incoming_dates = {v.session_date for v, _ in items}
            previous: dict[date, tuple[str, int]] = {}
            for tier in _TIERS:
                path = self._month_path(tier, symbol, month)
                existing = _read_parquet(path) if path.exists() else None
                if existing is not None:
                    hit = existing[pd.Index(existing.index.date).isin(incoming_dates)]
                    for d, g in hit.groupby(hit.index.date):
                        previous[d] = (tier, len(g))
                    kept = existing[~pd.Index(existing.index.date).isin(incoming_dates)]
                else:
                    kept = None
                new_frames = [f for v, f in items if _TIER_OF[v.verdict] == tier]
                parts = ([kept] if kept is not None and not kept.empty else []) + new_frames
                if not parts:
                    if path.exists():
                        path.unlink()
                    continue
                merged = pd.concat(parts).sort_index()
                if not merged.index.is_unique:
                    raise ValueError(f"{symbol} {month} {tier}: duplicate timestamps after merge")
                _atomic_write_parquet(merged, path)

            for verdict, frame in items:
                prior = previous.get(verdict.session_date)
                event = {
                    "event": "replaced" if prior else "added",
                    "ts": datetime.now(tz=IST).isoformat(),
                    "interval": self.interval,
                    "symbol": symbol,
                    "session_date": verdict.session_date.isoformat(),
                    "tier": _TIER_OF[verdict.verdict],
                    "verdict": verdict.verdict.value,
                    "rows": len(frame),
                    "previous_tier": prior[0] if prior else None,
                    "previous_rows": prior[1] if prior else None,
                }
                append_jsonl(self.fetch_log_path, event)
                append_jsonl(self.verdicts_path, {**verdict.model_dump(mode="json"), "interval": self.interval})
                events.append(event)
        return events

    def log_run(self, record: dict[str, Any]) -> None:
        append_jsonl(self.fetch_log_path, {
            "event": "fetch_run", "ts": datetime.now(tz=IST).isoformat(), "interval": self.interval, **record,
        })

    # ---- reads -------------------------------------------------------------------

    def _read_tier(self, tier: str, symbol: str) -> pd.DataFrame:
        months = self._months(tier, symbol)
        if not months:
            return pd.DataFrame(columns=list(BAR_COLUMNS), index=pd.DatetimeIndex([], tz=IST, name="ts"))
        return pd.concat(_read_parquet(self._month_path(tier, symbol, m)) for m in months).sort_index()

    def read_research(self, symbol: str) -> pd.DataFrame:
        return self._read_tier("bars", symbol)

    def read_quarantine(self, symbol: str) -> pd.DataFrame:
        return self._read_tier("quarantine", symbol)

    def research_sessions(self, symbol: str) -> list[date]:
        bars = self.read_research(symbol)
        return sorted(set(bars.index.date))

    def symbols(self) -> list[str]:
        folder = self.root / "bars" / self.interval
        if not folder.exists():
            return []
        return sorted(p.name for p in folder.iterdir() if p.is_dir())

    def load_verdicts(self) -> dict[tuple[str, date], SessionVerdict]:
        """Latest verdict per (symbol, session_date) for THIS interval. The log is shared
        across intervals and append-only; last line wins within the interval. A record
        without an interval field is from an older format and is refused."""
        out: dict[tuple[str, date], SessionVerdict] = {}
        for n, rec in enumerate(read_jsonl(self.verdicts_path), start=1):
            if "interval" not in rec:
                raise ValueError(f"{self.verdicts_path} line {n} has no interval field; delete the log and re-fetch")
            if rec["interval"] != self.interval:
                continue
            v = SessionVerdict.model_validate({k: x for k, x in rec.items() if k != "interval"})
            out[(v.symbol, v.session_date)] = v
        return out

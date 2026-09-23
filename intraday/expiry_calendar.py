"""NSE derivative expiry dates, loaded from data/nse_expiries.csv.

The file is maintained by hand, like the holiday calendar. Expiry rules have changed
several times (monthly last-Thursday, then weekly, then a shifting weekday), so the dates
are listed explicitly rather than derived from a weekday rule that would be silently wrong
for older data.

Outside the range the file covers, ``is_expiry`` returns None and the feature built on it
is NaN. An empty file therefore means "unknown everywhere", never "no expiries".
"""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

EXPIRIES_FILE = "nse_expiries.csv"


class ExpiryCalendar:
    def __init__(self, expiries: set[date]) -> None:
        self._expiries = set(expiries)
        self.first: date | None = min(self._expiries) if self._expiries else None
        self.last: date | None = max(self._expiries) if self._expiries else None

    @classmethod
    def from_csv(cls, path: Path) -> ExpiryCalendar:
        if not path.exists():
            raise FileNotFoundError(f"expiry calendar not found at {path}")
        expiries: set[date] = set()
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or "date" not in reader.fieldnames:
                raise ValueError(f"{path} must have a 'date' column, got {reader.fieldnames}")
            for n, row in enumerate(reader, start=2):
                raw = (row.get("date") or "").strip()
                if not raw:
                    continue
                try:
                    expiries.add(date.fromisoformat(raw))
                except ValueError as exc:
                    raise ValueError(f"{path} line {n}: {raw!r} is not an ISO date") from exc
        return cls(expiries)

    @property
    def covered(self) -> tuple[date, date] | None:
        """The range the file speaks for, or None when it is empty."""
        return None if self.first is None or self.last is None else (self.first, self.last)

    def describe(self) -> str:
        if self.covered is None:
            return f"no expiry dates listed; is_expiry_day is NaN everywhere (fill in data/{EXPIRIES_FILE})"
        return f"{len(self._expiries)} expiry dates covering {self.first} to {self.last}"

    def is_expiry(self, day: date) -> bool | None:
        """True/False inside the covered range, None outside it (the file cannot say)."""
        if self.covered is None:
            return None
        if not self.first <= day <= self.last:  # type: ignore[operator]
            return None
        return day in self._expiries

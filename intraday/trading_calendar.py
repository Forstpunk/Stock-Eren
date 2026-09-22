"""NSE trading calendar loaded from data/nse_holidays.csv.

Coverage is the set of calendar years present in the file. Asking about a date
outside coverage raises: the pipeline never assumes an unknown date is a trading day.
"""
from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path


class CalendarRangeError(Exception):
    """A date was requested outside the years the holiday file covers."""


class TradingCalendar:
    def __init__(self, holidays: dict[date, str]) -> None:
        if not holidays:
            raise ValueError("holiday calendar is empty")
        self._holidays = dict(holidays)
        years = sorted({d.year for d in holidays})
        if years != list(range(years[0], years[-1] + 1)):
            raise ValueError(f"holiday calendar years are not contiguous: {years}")
        self.first_year = years[0]
        self.last_year = years[-1]

    @classmethod
    def from_csv(cls, path: Path) -> TradingCalendar:
        if not path.exists():
            raise FileNotFoundError(f"holiday calendar not found at {path}")
        holidays: dict[date, str] = {}
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or "date" not in reader.fieldnames or "name" not in reader.fieldnames:
                raise ValueError(f"{path} must have 'date' and 'name' columns, got {reader.fieldnames}")
            for row in reader:
                d = date.fromisoformat(row["date"])
                if d in holidays:
                    raise ValueError(f"{path}: duplicate holiday {d}")
                holidays[d] = row["name"]
        return cls(holidays)

    def _check_range(self, d: date) -> None:
        if not (self.first_year <= d.year <= self.last_year):
            raise CalendarRangeError(
                f"{d} is outside the holiday calendar coverage {self.first_year}-{self.last_year}"
            )

    def is_trading_day(self, d: date) -> bool:
        self._check_range(d)
        return d.weekday() < 5 and d not in self._holidays

    def holiday_name(self, d: date) -> str | None:
        self._check_range(d)
        return self._holidays.get(d)

    def sessions_between(self, start: date, end: date) -> list[date]:
        """Trading days in [start, end], inclusive."""
        if start > end:
            raise ValueError(f"start {start} is after end {end}")
        out: list[date] = []
        d = start
        while d <= end:
            if self.is_trading_day(d):
                out.append(d)
            d += timedelta(days=1)
        return out

"""The short version: it answers the question, uses rupees, and never becomes a forecast.

This block is what a reader who skips everything else will see. These tests pin the parts
that carry the meaning, and guard against it drifting into advice.
"""
from __future__ import annotations

import re
from datetime import date, datetime

import pytest
from rich.console import Console

from intraday.config import IST
from intraday.report import Report, render_plain_answer
from tests.test_report import _edge, _exp, _summary, _variant
from tests.test_summary_render import make_study

WIDTH = 100
FORBIDDEN = ("buy", "sell", "target price", "entry at", "should trade", "recommend")


def make_forecast(verdict: str = "informative"):  # type: ignore[no-untyped-def]
    from intraday.forecast import CalibrationBin, Score

    return Score(
        n=408, failures=73, brier=0.1434, brier_base=0.1481, skill=0.032, skill_ci=(0.010, 0.051),
        calibration=(
            CalibrationBin(lower=0.1, upper=0.2, n=228, mean_forecast=0.164, observed_rate=0.118),
            CalibrationBin(lower=0.2, upper=0.3, n=179, mean_forecast=0.236, observed_rate=0.257),
        ),
        verdict=verdict, statement="…",
    )


def make_report(
    setup_verdict: str = "no edge", study_verdict: str = "no_signal", with_money: bool = True,
    forecast=None,  # type: ignore[no-untyped-def]
) -> Report:
    from intraday.report import IntegritySummary

    integrity = IntegritySummary(
        sessions_validated=820, verdict_counts={"CLEAN": 100, "TAIL_COLLAPSED": 584, "PARTIAL": 136, "SUSPECT": 0, "CORRUPT": 0},
        research_sessions=684, symbols=20, per_symbol_non_research={}, daily_rows=2894, daily_corrupt=80, last_fetch=None,
    )
    money = {
        "setup": "orb", "position": 50_000.0, "n_trades": 783,
        "by_slippage": [
            {"bps": 5, "per_trade": -112.0, "total": -87_575.0},
            {"bps": 10, "per_trade": -161.0, "total": -125_899.0},
            {"bps": 20, "per_trade": -259.0, "total": -202_545.0},
        ],
        "mid": {"bps": 10, "wins_in_ten": 3, "cost": 150.0, "random": -155.0, "same_as_random": True},
    } if with_money else None
    summary = _summary([_variant(5, _edge(-0.05, 0.03), _exp(-0.20, -0.09)), _variant(20, _edge(-0.05, 0.03), _exp(-0.32, -0.25))])
    return Report(
        generated_at=datetime(2026, 9, 23, tzinfo=IST),
        config={"min_sample": 30},
        integrity=integrity,
        n_breakouts=840,
        base_overall={"n": 840.0, "SUSTAINED_pct": 22.0, "BUSTED_pct": 20.0, "NEITHER_pct": 58.0},
        base_by_month={"2026-08": {"n": 431.0, "SUSTAINED_pct": 23.0, "BUSTED_pct": 20.0, "NEITHER_pct": 57.0}},
        study=make_study(study_verdict),
        backtests={"orb": summary},
        setup_verdicts={"orb": setup_verdict},
        gate_closed=study_verdict != "signal",
        n_sessions=40,
        first_date="2026-07-27",
        last_date="2026-09-21",
        forecast=forecast,
        money=money,
        bottom_line=(
            "This setup (orb) lost money on this data, at every slippage level.",
            "Nothing here tells you which breakouts to avoid: the warning signs did not work.",
            "Do not trade this on the strength of this report. It is a measurement, not a forecast.",
        ),
    )


def render(report: Report) -> str:
    console = Console(record=True, width=WIDTH, force_terminal=False)
    render_plain_answer(report, console)
    return console.export_text()


def test_leads_with_the_question_and_a_one_word_answer() -> None:
    text = render(make_report())
    assert "Question:" in text and "Answer:" in text
    answer_line = next(ln for ln in text.splitlines() if "Answer:" in ln)
    assert answer_line.strip().endswith("No.")


def test_reports_money_in_rupees_at_every_slippage_level() -> None:
    text = render(make_report())
    for bps, per_trade in ((5, "-112"), (10, "-161"), (20, "-259")):
        assert f"at {bps:2d} bps slippage" in text
        assert per_trade in text
    assert "-125,899" in text, "the total must be there, not just the per-trade figure"
    assert "Rs 50,000" in text, "the position size the figures assume must be stated"


def test_says_costs_and_the_random_comparison() -> None:
    text = render(make_report())
    assert "Costs alone were Rs 150" in text
    assert "random" in text.lower() and "the same" in text


def test_counts_add_up_to_the_breakout_total() -> None:
    text = render(make_report())
    counts = [int(m) for m in re.findall(r"^\s+(\d+)\s+(?:failed|ran|neither)", text, re.M)]
    assert len(counts) == 3
    assert sum(counts) == 840


def test_names_each_warning_sign_in_plain_english() -> None:
    text = render(make_report())
    for phrase in ("quiet first 15 minutes", "thin volume on the breakout bar", "small candle body"):
        assert phrase in text
    assert "rvol_open_15m" not in text, "the short version must not use internal feature names"


def test_no_money_section_when_there_are_no_trades() -> None:
    text = render(make_report(with_money=False))
    assert "per trade" not in text
    assert "Question:" in text and "warning sign" in text.lower()


def test_ends_with_the_bottom_line() -> None:
    text = render(make_report())
    assert "measurement, not a forecast" in text


@pytest.mark.parametrize("study_verdict", ["insufficient_sample", "no_signal", "signal"])
def test_never_becomes_advice(study_verdict: str) -> None:
    """The block a reader trusts most is the one that must never recommend a trade."""
    text = render(make_report(study_verdict=study_verdict)).lower()
    for term in FORBIDDEN:
        assert term not in text, f"{study_verdict} short version contains {term!r}"


def test_signal_does_not_promise_profit() -> None:
    text = render(make_report(study_verdict="signal"))
    assert "worked - see the tables below" in text
    # a separation is still not an edge, and the bottom line has to say so
    assert "forecast" in text


def test_forecast_result_answers_the_prediction_question() -> None:
    """When forecasts were scored, the short version must report what they achieved."""
    text = render(make_report(forecast=make_forecast("informative")))
    assert "A little." in text
    assert "408" in text, "the number of scored forecasts must appear"
    assert "12%" in text and "26%" in text, "the safest and riskiest observed rates must appear"
    assert "not enough to cover costs" in text


def test_forecast_that_failed_is_reported_as_such() -> None:
    text = render(make_report(forecast=make_forecast("no_better_than_base_rate")))
    assert "did not beat the base rate" in text
    assert "A little." not in text


def test_without_a_forecast_it_says_so() -> None:
    text = render(make_report(forecast=None))
    assert "too little history" in text

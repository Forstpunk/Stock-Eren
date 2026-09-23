"""The plain-language summary: every verdict renders, says the right thing, and never
drifts into telling the reader what to trade.

The tables are the audit trail; this block is the answer. A reader who stops after it
must still have the correct conclusion, so these tests pin the wording that carries it.
"""
from __future__ import annotations

import re
from datetime import date

import pytest
from rich.console import Console

from intraday.analysis import MIN_GAP_PCT, FeatureFinding, FeatureTable, Study, Third, Verdict
from intraday.features import FEATURE_NAMES
from intraday.report import render_plain_summary, render_study

WIDTH = 150

# Language that would turn a measurement into a recommendation. Guard against drift,
# including by a future contributor.
FORBIDDEN = ("buy", "sell", "target price", "entry at")


def _table(feature: str, scope: str, low: float, high: float, base: float = 20.0) -> FeatureTable:
    thirds = tuple(
        Third(name=name, lower=float(k), upper=float(k + 1), n=100, busts=int(rate), bust_rate_pct=rate)
        for k, (name, rate) in enumerate(zip(("low", "mid", "high"), (low, (low + high) / 2, high)))
    )
    return FeatureTable(feature=feature, scope=scope, n=300, busts=int(base * 3), base_rate_pct=base, thirds=thirds)


def _finding(
    feature: str, low: float, high: float, holds: bool, testable: bool = True, two_sided: bool = False
) -> FeatureFinding:
    return FeatureFinding(
        feature=feature,
        overall=_table(feature, "all", low, high),
        first_half=_table(feature, "first half", low, high),
        second_half=_table(feature, "second half", low, high),
        testable=testable,
        holds=holds,
        two_sided=two_sided,
        statement=f"{feature}: low third fails {low:.1f}%, high third {high:.1f}%.",
    )


def make_study(verdict: Verdict, second_half_busts: int = 52) -> Study:
    """A Study with the shape each verdict actually produces."""
    if verdict == "signal":
        findings = tuple(
            _finding(f, 35.0, 10.0, holds=True) if k == 0 else _finding(f, 21.0, 19.0, holds=False)
            for k, f in enumerate(FEATURE_NAMES)
        )
        statement = f"SIGNAL: {FEATURE_NAMES[0]} separates failed breakouts by at least {MIN_GAP_PCT:.0f} points."
    elif verdict == "no_signal":
        findings = tuple(_finding(f, 21.0, 19.0, holds=False) for f in FEATURE_NAMES)
        statement = "NO SIGNAL: no feature separates failed breakouts in both halves."
    else:
        # "nothing could be tested" is what insufficient_sample now means
        findings = tuple(_finding(f, 21.0, 19.0, holds=False, testable=False) for f in FEATURE_NAMES)
        statement = f"INSUFFICIENT SAMPLE: no feature has enough failures in both halves ({second_half_busts})."
    return Study(
        n_breakouts=840, base_rate_pct=20.0, split_date=date(2026, 9, 2),
        second_half_busts=second_half_busts, min_half_busts=30,
        n_tested=len(FEATURE_NAMES), n_directional=len(FEATURE_NAMES), findings=findings,
        verdict=verdict, statement=statement,
    )


def render(study: Study, min_sample: int = 30) -> str:
    console = Console(record=True, width=WIDTH, force_terminal=False)
    render_plain_summary(study, console, min_sample)
    return console.export_text()


@pytest.mark.parametrize("verdict", ["insufficient_sample", "no_signal", "signal"])
def test_every_verdict_renders(verdict: Verdict) -> None:
    text = render(make_study(verdict, second_half_busts=7 if verdict == "insufficient_sample" else 52))
    assert text.strip(), f"{verdict} rendered nothing"
    assert verdict in text, "the verdict itself must appear"


@pytest.mark.parametrize("verdict", ["insufficient_sample", "no_signal", "signal"])
def test_summary_is_under_twelve_lines(verdict: Verdict) -> None:
    text = render(make_study(verdict, second_half_busts=7 if verdict == "insufficient_sample" else 52))
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) < 12, f"{verdict} summary is {len(lines)} lines:\n{text}"


def test_no_signal_says_not_to_trade_the_setup() -> None:
    assert "do not trade this setup" in render(make_study("no_signal"))


def test_insufficient_sample_reports_the_shortfall_not_a_score() -> None:
    text = render(make_study("insufficient_sample", second_half_busts=7), min_sample=30)
    assert "30 failed breakouts in both halves" in text
    # The summary stays short, so it names the worst few and points at the tables for the rest.
    from intraday.features import FEATURE_PLAIN
    named = [f for f in FEATURE_NAMES if FEATURE_PLAIN.get(f, f) in text]
    assert named, "the thinnest features' shortfalls must be shown"
    assert "in the tables below" in text or len(named) == len(FEATURE_NAMES)
    # No AUC, and no other single summary score dressed up as a finding.
    assert "auc" not in text.lower()
    assert not re.search(r"0\.\d\d", text), f"summary quotes a score-like figure:\n{text}"


def test_insufficient_sample_makes_no_claim_either_way() -> None:
    text = render(make_study("insufficient_sample", second_half_busts=7))
    assert "either way" in text
    assert "do not trade this setup" not in text  # that verdict is not what was measured


def test_signal_names_exactly_three_features() -> None:
    text = render(make_study("signal"))
    named = [f for f in FEATURE_NAMES if f in text]
    assert len(named) == 3, f"expected all three features named, got {named}"
    assert text.count(FEATURE_NAMES[0]) >= 1


def test_signal_gives_a_direction_for_each_feature() -> None:
    text = render(make_study("signal"))
    assert text.count("values fail more") == 3


def test_signal_warns_that_separation_is_not_an_edge() -> None:
    text = render(make_study("signal"))
    assert "not by itself a tradeable edge" in text
    assert "costs" in text


@pytest.mark.parametrize("verdict", ["insufficient_sample", "no_signal", "signal"])
def test_summary_contains_no_price_target_or_buy_sell_language(verdict: Verdict) -> None:
    """A guard against drift. This tool measures; it must never recommend."""
    text = render(make_study(verdict, second_half_busts=7 if verdict == "insufficient_sample" else 52)).lower()
    for term in FORBIDDEN:
        assert term not in text, f"{verdict} summary contains {term!r}:\n{text}"
    assert not re.search(r"(?:rs|inr|₹)\s*[\d,]+", text), "summary quotes a rupee price"


@pytest.mark.parametrize("verdict", ["insufficient_sample", "no_signal", "signal"])
def test_summary_is_strictly_shorter_than_the_tables(verdict: Verdict) -> None:
    """What --summary suppresses: the detail block is much longer and is all tables."""
    study = make_study(verdict, second_half_busts=7 if verdict == "insufficient_sample" else 52)
    summary = render(study)

    detail_console = Console(record=True, width=WIDTH, force_terminal=False)
    render_study(study, detail_console)
    detail = detail_console.export_text()

    assert len(summary.splitlines()) < len(detail.splitlines())
    # and the summary draws no tables at all
    assert not set(summary) & set("┌─│└┬┴┼┐┘")
    assert set(detail) & set("─"), "the detail block should still be drawing tables"

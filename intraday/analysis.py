"""Does any feature separate failed breakouts from the rest?

One method, countable by hand: cut a feature into thirds, count busts in each third,
compare to the overall bust rate. No model, no fitted coefficients.

Stability is checked the same way the split always worked: earliest 70% of dates are
the first half, the rest the second. A feature "holds" only if the low-vs-high gap
points the same way in both halves and is at least MIN_GAP_PCT in each.

The sample floor is applied PER FEATURE PER HALF, not globally: a feature is testable
only if both halves hold at least ``config.min_sample`` failures among the rows where
that feature is present. This matters because a feature can be missing from most rows
(RVOL needs 20 prior sessions, so it barely exists early in a short history), and a
gap measured on a handful of failures is exactly the caveated claim this pipeline
refuses to make.

The verdict:
- insufficient sample: no feature is testable
- signal: at least one testable DIRECTIONAL feature holds
- no signal: at least one feature was testable and none held

Two-sided features (RESEARCH.md pre-registers which) are measured and reported but can
never produce a signal on their own: with no direction predicted in advance, "it came out
significant" is a coin flip presented as a finding.

Testing more features raises the chance that one passes by luck, so the study records how
many were tested and the report states it. A signal resting on one feature out of seven is
weaker evidence than the same signal resting on one out of one.
"""
from __future__ import annotations

import math
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from intraday.config import Config
from intraday.features import FEATURE_EXPECTATION, FEATURE_NAMES, REPORT_ONLY_NAMES, TESTED_NAMES
from intraday.labelling import Label

MIN_GAP_PCT = 10.0  # percentage points between the low and high third to count as a gap
TRAIN_FRACTION = 0.7
THIRD_NAMES = ("low", "mid", "high")
Verdict = Literal["signal", "no_signal", "insufficient_sample"]


class Third(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str  # low / mid / high
    lower: float  # feature range covered by this third
    upper: float
    n: int
    busts: int
    bust_rate_pct: float


class FeatureTable(BaseModel):
    """The bust rate in each third of one feature, over one set of rows."""

    model_config = ConfigDict(frozen=True)

    feature: str
    scope: str  # "all" / "first half" / "second half"
    n: int
    busts: int
    base_rate_pct: float
    thirds: tuple[Third, ...]

    @property
    def gap_pct(self) -> float:
        """Low third's bust rate minus the high third's. Positive = low values fail more."""
        return self.thirds[0].bust_rate_pct - self.thirds[-1].bust_rate_pct


class FeatureFinding(BaseModel):
    model_config = ConfigDict(frozen=True)

    feature: str
    overall: FeatureTable | None  # None when the feature cannot be split at all
    first_half: FeatureTable | None
    second_half: FeatureTable | None
    testable: bool  # both halves clear the per-feature failure floor
    holds: bool  # testable AND the gap is at least MIN_GAP_PCT in both halves
    two_sided: bool  # pre-registered without a direction; can never produce a signal alone
    statement: str


class Study(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_breakouts: int
    base_rate_pct: float
    split_date: date | None
    second_half_busts: int
    min_half_busts: int  # the per-feature-per-half floor that was applied
    n_tested: int  # how many features were examined at all
    n_directional: int  # of those, how many could produce a signal
    findings: tuple[FeatureFinding, ...]
    verdict: Verdict
    statement: str


def split_date_at(dates: pd.Series, fraction: float = TRAIN_FRACTION) -> date | None:
    """First date of the second half, splitting on a date boundary so no session straddles it."""
    unique = sorted(pd.Series(dates).unique())
    if len(unique) < 2:
        return None
    counts = pd.Series(dates).value_counts().reindex(unique)
    cumulative = counts.cumsum() / counts.sum()
    first_half = [d for d, share in cumulative.items() if share < fraction] or [unique[0]]
    return unique[len(first_half)] if len(first_half) < len(unique) else None


def _buckets_for(values: np.ndarray) -> list[tuple[str, float, float, np.ndarray]] | None:
    """(name, lower, upper, mask) per bucket: thirds by value, or one bucket per distinct
    value when the feature only takes a few (a 0/1 flag cannot be cut into thirds)."""
    distinct = np.unique(values)
    if len(distinct) < 2:
        return None  # every row identical: nothing to compare
    if len(distinct) <= len(THIRD_NAMES):
        names = ("low", "high") if len(distinct) == 2 else THIRD_NAMES
        return [
            (name, float(v), float(v), values == v) for name, v in zip(names, distinct)
        ]
    edges = [float(np.quantile(values, q)) for q in (1 / 3, 2 / 3)]
    if edges[0] == edges[1]:
        return None  # more than a third of the rows share one value; thirds are not defined
    return [
        ("low", float(values.min()), edges[0], values <= edges[0]),
        ("mid", edges[0], edges[1], (values > edges[0]) & (values <= edges[1])),
        ("high", edges[1], float(values.max()), values > edges[1]),
    ]


def bust_rate_by_third(rows: pd.DataFrame, feature: str, scope: str) -> FeatureTable | None:
    """Count failures in each bucket of ``feature``. Buckets are thirds by value, or the
    distinct values themselves for a flag. None when the feature cannot be split at all."""
    usable = rows[rows[feature].notna()]
    if len(usable) < 3:
        return None
    values = usable[feature].to_numpy(dtype="float64")
    buckets = _buckets_for(values)
    if buckets is None:
        return None
    is_bust = (usable["label"] == Label.BUSTED.value).to_numpy()

    thirds = []
    for name, lo, hi, mask in buckets:
        n = int(mask.sum())
        if n == 0:
            return None
        busts = int(is_bust[mask].sum())
        thirds.append(Third(name=name, lower=lo, upper=hi, n=n, busts=busts, bust_rate_pct=busts / n * 100))
    return FeatureTable(
        feature=feature, scope=scope, n=len(usable), busts=int(is_bust.sum()),
        base_rate_pct=float(is_bust.mean() * 100), thirds=tuple(thirds),
    )


def _finding(
    rows: pd.DataFrame, feature: str, split: date | None, min_half_busts: int
) -> FeatureFinding:
    overall = bust_rate_by_third(rows, feature, "all")
    if overall is None:
        usable = int(rows[feature].notna().sum())
        return FeatureFinding(
            feature=feature, overall=None, first_half=None, second_half=None,
            testable=False, holds=False, two_sided=feature in REPORT_ONLY_NAMES,
            statement=(
                f"{feature}: NOT MEASURABLE on this data - {usable} rows have a value and they do not "
                "take enough distinct values to split. Nothing is claimed about it either way."
            ),
        )
    first = second = None
    if split is not None:
        dates = pd.to_datetime(rows["session_date"]).dt.date
        first = bust_rate_by_third(rows[dates < split], feature, "first half")
        second = bust_rate_by_third(rows[dates >= split], feature, "second half")

    two_sided = feature in REPORT_ONLY_NAMES
    testable = (
        first is not None and second is not None
        and first.busts >= min_half_busts and second.busts >= min_half_busts
    )
    # A two-sided feature has no pre-registered direction, so a gap in either direction is
    # measured and shown but never counts as holding.
    holds = (
        testable and not two_sided and first is not None and second is not None
        and first.gap_pct >= MIN_GAP_PCT and second.gap_pct >= MIN_GAP_PCT
    )
    lo, hi = overall.thirds[0], overall.thirds[-1]
    statement = (
        f"{feature}: low third fails {lo.bust_rate_pct:.1f}% ({lo.busts}/{lo.n}), "
        f"high third {hi.bust_rate_pct:.1f}% ({hi.busts}/{hi.n}), against {overall.base_rate_pct:.1f}% overall. "
    )
    if two_sided and first is not None and second is not None:
        statement += (
            f"Two-sided, reported only: gaps {first.gap_pct:+.1f} then {second.gap_pct:+.1f} points. "
            "No direction was predicted in advance, so this cannot count as a finding."
        )
    elif first is None or second is None:
        statement += "Not enough dates to check both halves."
    elif not testable:
        statement += (
            f"NOT TESTABLE: the halves hold {first.busts} and {second.busts} failures where this feature "
            f"exists, and {min_half_busts} are needed in each. No claim is made about it either way."
        )
    elif holds:
        statement += (
            f"The gap holds in both halves ({first.gap_pct:+.1f} then {second.gap_pct:+.1f} points). "
            + FEATURE_EXPECTATION[feature].capitalize() + " - and they do."
        )
    else:
        statement += f"The gap does not hold in both halves ({first.gap_pct:+.1f} then {second.gap_pct:+.1f} points)."
    return FeatureFinding(
        feature=feature, overall=overall, first_half=first, second_half=second,
        testable=testable, holds=holds, two_sided=two_sided, statement=statement,
    )


def run_study(features: pd.DataFrame, config: Config) -> Study:
    """Bust rate by third for every feature, with the two-halves stability check."""
    missing = [f for f in TESTED_NAMES if f not in features.columns]
    if missing:
        raise ValueError(f"feature table lacks {missing}")
    if "label" not in features.columns or "session_date" not in features.columns:
        raise ValueError("feature table needs 'label' and 'session_date' columns")
    unknown = set(features["label"].unique()) - {l.value for l in Label}
    if unknown:
        raise ValueError(f"unknown labels {unknown}")

    dates = pd.to_datetime(features["session_date"]).dt.date
    split = split_date_at(dates)
    second_half_busts = (
        int(((dates >= split) & (features["label"] == Label.BUSTED.value)).sum()) if split is not None else 0
    )
    findings = tuple(_finding(features, f, split, config.min_sample) for f in TESTED_NAMES)
    base_rate = float((features["label"] == Label.BUSTED.value).mean() * 100)

    held = [f.feature for f in findings if f.holds]
    testable = [f.feature for f in findings if f.testable and not f.two_sided]
    if not testable:
        verdict: Verdict = "insufficient_sample"
        thin = ", ".join(
            f"{f.feature} ({f.first_half.busts if f.first_half else 0}/{f.second_half.busts if f.second_half else 0})"
            for f in findings
        )
        statement = (
            f"INSUFFICIENT SAMPLE: no feature has {config.min_sample} failed breakouts in both halves of the "
            f"period, so none can be tested. Failures per half where each feature exists: {thin}. "
            "The tables below describe what happened; they are not evidence that it will happen again."
        )
    elif held:
        verdict = "signal"
        statement = (
            f"SIGNAL: {', '.join(held)} separates failed breakouts from the rest by at least "
            f"{MIN_GAP_PCT:.0f} points in both halves of the period. "
            f"{len(FEATURE_NAMES)} directional features were tested, so roughly a "
            f"{_family_error(len(FEATURE_NAMES)):.0%} chance one passes by luck alone; weigh a single "
            "survivor accordingly. Still one regime; confirm on longer history."
        )
    else:
        verdict = "no_signal"
        untestable = [f.feature for f in findings if not f.testable]
        statement = (
            f"NO SIGNAL: of the features with enough failures to test ({', '.join(testable)}), none separates "
            f"failed breakouts by at least {MIN_GAP_PCT:.0f} points in both halves of the period."
        )
        if untestable:
            statement += f" {', '.join(untestable)} could not be tested at all on this sample."
        else:
            statement += " On this data, that angle is closed."
    return Study(
        n_breakouts=len(features), base_rate_pct=base_rate, split_date=split,
        second_half_busts=second_half_busts, min_half_busts=config.min_sample,
        n_tested=len(TESTED_NAMES), n_directional=len(FEATURE_NAMES),
        findings=findings, verdict=verdict, statement=statement,
    )


def _family_error(n_features: int, per_feature: float = 0.05) -> float:
    """Chance that at least one of ``n_features`` independent tests passes by luck."""
    return 1.0 - (1.0 - per_feature) ** n_features

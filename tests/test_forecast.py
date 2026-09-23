"""Forecasting: the rule sees only the past, the arithmetic is checkable by hand, a real
pattern scores as informative, noise scores as no better than the base rate."""
from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from intraday.config import IST, Config
from intraday.forecast import (
    MIN_TRAIN_FAILURES,
    fit_rule,
    predict_one,
    score,
    walk_forward,
)
from intraday.features import FEATURE_NAMES


def make_features(n_sessions: int, per_session: int, signal: float, seed: int, fail_rate: float = 0.25) -> pd.DataFrame:
    """Breakouts across sessions. ``signal`` shifts rvol_open_15m down for failures."""
    rng = np.random.default_rng(seed)
    rows = []
    start = date(2026, 1, 5)
    for s in range(n_sessions):
        day = start + timedelta(days=s)
        for k in range(per_session):
            fails = rng.random() < fail_rate
            row = {f: float(rng.normal()) for f in FEATURE_NAMES}
            row["rvol_open_15m"] = float(rng.normal() - signal * fails)
            row.update({
                "symbol": f"S{k % 10}", "session_date": day, "direction": "long" if k % 2 else "short",
                "breakout_time": datetime.combine(day, datetime.min.time(), tzinfo=IST),
                "label": "BUSTED" if fails else ("SUSTAINED" if rng.random() < 0.4 else "NEITHER"),
            })
            rows.append(row)
    return pd.DataFrame(rows)


# ---- the rule --------------------------------------------------------------------------


def test_rule_is_readable_and_counts_by_hand(config: Config) -> None:
    df = make_features(40, 20, signal=1.5, seed=1)
    rule = fit_rule(df, config)
    assert rule.n_train == 800
    assert rule.base_rate == pytest.approx((df["label"] == "BUSTED").mean())
    assert "rvol_open_15m" in rule.usable_features
    for feature in rule.usable_features:
        buckets = [b for b in rule.buckets if b.feature == feature]
        assert [b.third for b in buckets] == ["low", "mid", "high"]
        assert sum(b.n for b in buckets) == int(df[feature].notna().sum())
        for b in buckets:
            assert b.rate == pytest.approx(b.failures / b.n)
    text = "\n".join(rule.describe())
    assert "base rate" in text and "rvol_open_15m" in text


def test_rule_ignores_features_with_thin_buckets(config: Config) -> None:
    df = make_features(20, 6, signal=1.0, seed=2)  # 120 rows -> 40 per third, still fine
    df.loc[df.index[:110], "bar_body_ratio"] = np.nan  # only 10 rows survive
    rule = fit_rule(df, config)
    assert "bar_body_ratio" not in rule.usable_features
    assert "rvol_open_15m" in rule.usable_features


def test_prediction_is_the_average_of_its_buckets(config: Config) -> None:
    df = make_features(40, 20, signal=1.5, seed=3)
    rule = fit_rule(df, config)
    row = df.iloc[0]
    p = predict_one(row, rule)
    assert p.contributions, "a complete row should hit a bucket for each usable feature"
    assert p.p_fail == pytest.approx(float(np.mean(list(p.contributions.values()))))
    assert set(p.contributions) <= set(rule.usable_features)
    assert p.outcome is None, "a prediction must not carry the answer"


def test_prediction_falls_back_to_base_rate_when_nothing_applies(config: Config) -> None:
    df = make_features(40, 20, signal=1.5, seed=4)
    rule = fit_rule(df, config)
    blank = df.iloc[0].copy()
    for f in FEATURE_NAMES:
        blank[f] = np.nan
    p = predict_one(blank, rule)
    assert p.contributions == {} and p.p_fail == rule.base_rate


# ---- walk forward ----------------------------------------------------------------------


def test_walk_forward_uses_only_earlier_sessions(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every rule must be fitted on dates strictly before the session it predicts."""
    df = make_features(60, 20, signal=1.5, seed=5)
    seen: list[tuple[date, date]] = []
    import intraday.forecast as fc

    real_fit = fc.fit_rule

    def spy(train: pd.DataFrame, cfg: Config):  # type: ignore[no-untyped-def]
        rule = real_fit(train, cfg)
        seen.append(rule.fitted_on)
        return rule

    monkeypatch.setattr(fc, "fit_rule", spy)
    preds = fc.walk_forward(df, config)
    assert len(seen) > 0
    for (_, train_end), target in zip(seen, sorted(preds["session_date"].unique())):
        assert train_end < target, f"rule trained through {train_end} used to predict {target}"


def test_walk_forward_refuses_a_short_history(config: Config) -> None:
    with pytest.raises(ValueError, match="need more than"):
        walk_forward(make_features(10, 20, signal=1.0, seed=6), config)


def test_walk_forward_refuses_when_training_failures_are_thin(config: Config) -> None:
    df = make_features(30, 3, signal=1.0, seed=7, fail_rate=0.05)  # ~4 failures in total
    with pytest.raises(ValueError, match=f"{MIN_TRAIN_FAILURES} failures"):
        walk_forward(df, config)


# ---- scoring ---------------------------------------------------------------------------


def test_real_pattern_scores_as_informative(config: Config) -> None:
    preds = walk_forward(make_features(80, 20, signal=2.0, seed=8), config)
    s = score(preds, config)
    assert s.verdict == "informative"
    assert s.skill > 0 and s.brier < s.brier_base
    assert s.skill_ci[0] > 0, "informative requires the whole interval above zero"
    assert "beat the base rate" in s.statement


def test_a_small_positive_skill_with_a_straddling_interval_is_not_informative(config: Config) -> None:
    """The guard that stops a lucky sliver of skill being sold as a finding."""
    preds = walk_forward(make_features(80, 20, signal=0.25, seed=11), config)
    s = score(preds, config)
    if s.skill > 0 and s.skill_ci[0] <= 0:
        assert s.verdict == "no_better_than_base_rate"
        assert "within noise" in s.statement


def test_noise_scores_as_no_better_than_base_rate(config: Config) -> None:
    preds = walk_forward(make_features(80, 20, signal=0.0, seed=9), config)
    s = score(preds, config)
    assert s.verdict == "no_better_than_base_rate"
    assert s.skill_ci[0] <= 0 <= s.skill_ci[1]
    assert "within noise" in s.statement


def test_brier_by_hand() -> None:
    preds = pd.DataFrame({
        "p_fail": [0.0, 1.0, 0.5, 0.5],
        "base_rate": [0.5, 0.5, 0.5, 0.5],
        "outcome": ["NEITHER", "BUSTED", "BUSTED", "NEITHER"],
    })
    s = score(preds, Config(min_sample=1))
    assert s.brier == pytest.approx((0 + 0 + 0.25 + 0.25) / 4)
    assert s.brier_base == pytest.approx(0.25)
    assert s.skill == pytest.approx(1 - 0.125 / 0.25)
    assert s.n == 4 and s.failures == 2


def test_calibration_bins_report_forecast_against_observed(config: Config) -> None:
    preds = walk_forward(make_features(80, 20, signal=2.0, seed=10), config)
    s = score(preds, config)
    assert s.calibration
    assert sum(b.n for b in s.calibration) == s.n
    for b in s.calibration:
        assert 0.0 <= b.observed_rate <= 1.0
        assert b.lower <= b.mean_forecast <= b.upper


def test_thin_sample_is_not_graded(config: Config) -> None:
    preds = pd.DataFrame({
        "p_fail": [0.2] * 50, "base_rate": [0.2] * 50,
        "outcome": ["BUSTED"] * 5 + ["NEITHER"] * 45,
    })
    s = score(preds, config)
    assert s.verdict == "insufficient_sample"
    assert "cannot be graded" in s.statement

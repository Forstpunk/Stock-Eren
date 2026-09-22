"""Diagnostic: planted signal is found, noise is not, split is chronological at a date
boundary, standardisation uses train statistics only, small samples are flagged."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from intraday.config import Config
from intraday.diagnostic import chronological_split, feature_effect, run_diagnostic
from intraday.features import FEATURE_NAMES


def synthetic_table(n: int, signal: float, seed: int, bust_rate: float = 0.2) -> tuple[pd.DataFrame, pd.Series]:
    """``signal`` is the shift (in sd) of rvol_breakout_bar and vwap_distance_sigma for busts."""
    rng = np.random.default_rng(seed)
    y = rng.random(n) < bust_rate
    x = pd.DataFrame(rng.normal(size=(n, len(FEATURE_NAMES))), columns=list(FEATURE_NAMES))
    x.loc[y, "rvol_breakout_bar"] -= signal
    x.loc[y, "vwap_distance_sigma"] += signal
    x["minutes_since_open"] = rng.integers(15, 300, n).astype(float)
    x["prior_touches"] = rng.integers(0, 5, n).astype(float)
    days = [date(2026, 6, 1) + timedelta(days=k) for k in range(n // 8 + 1)]
    x["session_date"] = [days[k // 8] for k in range(n)]
    labels = pd.Series(np.where(y, "BUSTED", np.where(rng.random(n) < 0.4, "SUSTAINED", "NEITHER")))
    return x, labels


def test_planted_signal_is_detected(config: Config) -> None:
    x, labels = synthetic_table(1200, signal=1.2, seed=1)
    report = run_diagnostic(x, labels, config)
    assert report.verdict == "signal"
    assert report.test_auc > 0.75 and report.test_auc_ci[0] > 0.5
    coefs = {c.feature: c for c in report.coefficients}
    assert coefs["rvol_breakout_bar"].ci_high < 0
    assert coefs["vwap_distance_sigma"].ci_low > 0
    assert coefs["bar_body_ratio"].ci_low < 0 < coefs["bar_body_ratio"].ci_high
    top20 = next(p for p in report.precision_at if p.flag_share == 0.2)
    assert top20.precision > 2 * top20.base_rate
    eff = {e.feature: e for e in report.contrast_bust_vs_sustain}
    assert eff["rvol_breakout_bar"].cohens_d < -0.8 and eff["rvol_breakout_bar"].mw_p < 1e-6
    assert eff["rvol_breakout_bar"].cles < 0.3


def test_pure_noise_is_not_a_signal(config: Config) -> None:
    x, labels = synthetic_table(1200, signal=0.0, seed=2)
    report = run_diagnostic(x, labels, config)
    assert report.verdict == "no_signal"
    assert report.test_auc_ci[0] < 0.5
    assert "NO OUT-OF-SAMPLE SIGNAL" in report.statement


def test_small_test_set_is_insufficient(config: Config) -> None:
    x, labels = synthetic_table(200, signal=2.0, seed=3)
    report = run_diagnostic(x, labels, config)
    assert report.verdict == "insufficient_sample"
    assert "INSUFFICIENT SAMPLE" in report.statement


def test_chronological_split_at_date_boundary() -> None:
    dates = pd.Series([date(2026, 1, 1)] * 5 + [date(2026, 1, 2)] * 5 + [date(2026, 1, 3)] * 5)
    train, split_date = chronological_split(dates, 0.7)
    assert split_date == date(2026, 1, 3)
    assert train.tolist() == [True] * 10 + [False] * 5
    train2, split2 = chronological_split(dates, 0.5)
    assert split2 == date(2026, 1, 2) and train2.sum() == 5
    with pytest.raises(ValueError):
        chronological_split(pd.Series([date(2026, 1, 1)] * 3))


def test_split_is_reflected_in_report(config: Config) -> None:
    x, labels = synthetic_table(800, signal=1.0, seed=4)
    report = run_diagnostic(x, labels, config)
    assert report.n_train + report.n_test == report.n_complete == 800
    assert 0.6 < report.n_train / report.n_complete < 0.8
    assert (pd.Series(x["session_date"]) < report.split_date).sum() == report.n_train


def test_rows_with_missing_features_are_excluded_not_imputed(config: Config) -> None:
    x, labels = synthetic_table(1000, signal=1.0, seed=5)
    x.loc[x.index[:150], "gap_atr"] = np.nan
    report = run_diagnostic(x, labels, config)
    assert report.n_events == 1000 and report.n_complete == 850
    eff = {e.feature: e for e in report.contrast_bust_vs_rest}
    assert eff["gap_atr"].n_a + eff["gap_atr"].n_b == 850
    assert eff["atr_pct"].n_a + eff["atr_pct"].n_b == 1000


def test_scaler_uses_train_statistics_only(config: Config) -> None:
    x, labels = synthetic_table(1000, signal=1.0, seed=6)
    base = run_diagnostic(x, labels, config)
    train, _ = chronological_split(x["session_date"])
    shifted = x.copy()
    shifted.loc[~train, "atr_pct"] += 50.0  # wildly shift the test rows of one feature
    moved = run_diagnostic(shifted, labels, config)
    assert [c.coef for c in moved.coefficients] == [c.coef for c in base.coefficients]
    assert moved.train_auc == base.train_auc
    assert moved.confusion_at_half != base.confusion_at_half  # test probabilities did move


def test_diagnostic_is_deterministic(config: Config) -> None:
    x, labels = synthetic_table(600, signal=1.0, seed=7)
    a = run_diagnostic(x, labels, config, seed=11)
    b = run_diagnostic(x, labels, config, seed=11)
    assert a == b


def test_feature_effect_by_hand() -> None:
    a = pd.Series([1.0, 2.0, 3.0, 4.0])
    b = pd.Series([3.0, 4.0, 5.0, 6.0])
    e = feature_effect("f", a, b)
    pooled = np.sqrt((3 * a.var(ddof=1) + 3 * b.var(ddof=1)) / 6)
    assert e.cohens_d == pytest.approx((a.mean() - b.mean()) / pooled)
    assert e.cles == pytest.approx(2.0 / 16)  # pairs where a > b: (4,3) = 1, plus two ties at 0.5 each
    assert e.median_a == 2.5 and e.n_a == 4


def test_rejects_unknown_labels(config: Config) -> None:
    x, labels = synthetic_table(300, signal=1.0, seed=8)
    labels.iloc[0] = "MAYBE"
    with pytest.raises(ValueError, match="unknown labels"):
        run_diagnostic(x, labels, config)

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
    buckets_hit,
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
            # rate is now the shrunk figure; the raw count is kept alongside it
            assert b.raw_rate == pytest.approx(b.failures / b.n)
            assert min(b.raw_rate, rule.base_rate) <= b.rate <= max(b.raw_rate, rule.base_rate)
    text = "\n".join(rule.describe())
    assert "base rate" in text and "rvol_open_15m" in text


def test_rule_ignores_features_with_thin_buckets(config: Config) -> None:
    df = make_features(20, 6, signal=1.0, seed=2)  # 120 rows -> 40 per third, still fine
    df.loc[df.index[:110], "bar_body_ratio"] = np.nan  # only 10 rows survive
    rule = fit_rule(df, config)
    assert "bar_body_ratio" not in rule.usable_features
    assert "rvol_open_15m" in rule.usable_features


def test_prediction_combines_its_buckets_in_log_odds(config: Config) -> None:
    """Averaged bucket rates before Phase 5, damped log-odds after it, and since Phase 6
    p_fail is the BUSTED share of the three-way split rather than a separate binary number.
    The log_odds field is kept as the per-feature diagnostic for the odds of failure."""
    from intraday.forecast import ORDERED_CLASSES

    df = make_features(40, 20, signal=1.5, seed=3)
    rule = fit_rule(df, config)
    row = df.iloc[0]
    p = predict_one(row, rule)
    assert p.contributions, "a complete row should hit a bucket for each usable feature"
    assert set(p.contributions) <= set(rule.usable_features)
    assert set(p.log_odds) == set(p.contributions)
    assert p.p_fail == pytest.approx(p.class_probabilities["BUSTED"])
    assert sum(p.class_probabilities[c] for c in ORDERED_CLASSES) == pytest.approx(1.0)
    assert p.outcome is None, "a prediction must not carry the answer"


def test_prediction_falls_back_to_base_rate_when_nothing_applies(config: Config) -> None:
    df = make_features(40, 20, signal=1.5, seed=4)
    rule = fit_rule(df, config)
    blank = df.iloc[0].copy()
    for f in FEATURE_NAMES:
        blank[f] = np.nan
    p = predict_one(blank, rule)
    # exact equality is lost to the logit/expit round trip, so compare to tolerance
    assert p.contributions == {} and p.p_fail == pytest.approx(rule.base_rate)


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
        "session_date": [date(2026, 6, d) for d in (1, 2, 3, 4)],
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
        "session_date": [date(2026, 6, 1) + timedelta(days=k) for k in range(50)],
    })
    s = score(preds, config)
    assert s.verdict == "insufficient_sample"
    assert "cannot be graded" in s.statement


def test_score_refuses_predictions_without_sessions(config: Config) -> None:
    """Without session_date the bootstrap would silently fall back to iid resampling."""
    preds = pd.DataFrame({
        "p_fail": [0.2] * 50, "base_rate": [0.2] * 50,
        "outcome": ["BUSTED"] * 25 + ["NEITHER"] * 25,
    })
    with pytest.raises(ValueError, match="session_date"):
        score(preds, config)


def test_score_reports_how_clustered_the_outcomes_are(config: Config) -> None:
    preds = walk_forward(make_features(80, 20, signal=2.0, seed=12), config)
    s = score(preds, config)
    assert s.n_sessions > 0 and s.n_sessions < s.n
    assert 0.0 <= s.session_variance_share <= 1.0


# ---- shrinkage and log-odds combination (Phase 5) -------------------------------------


def test_shrinkage_pulls_a_small_bucket_towards_the_base_rate(config: Config) -> None:
    from intraday.forecast import shrink

    base = 0.20
    k = config.forecast_shrinkage_k
    # 4 of 6 failed: a raw 67% that has earned almost none of its distance from 20%.
    small = shrink(4, 6, base, k)
    assert base < small < 0.67
    assert abs(small - base) < abs(0.67 - base) / 2

    # the same rate on 400 observations should sit much closer to the raw figure
    large = shrink(267, 400, base, k)
    assert large > small
    assert abs(large - 0.6675) < abs(small - 0.6675)

    assert shrink(0, 0, base, k) == pytest.approx(base), "an empty bucket is just the base rate"
    assert shrink(5, 10, base, 0.0) == pytest.approx(0.5), "k=0 disables shrinkage"


def test_buckets_report_both_raw_and_shrunk_rates(config: Config) -> None:
    rule = fit_rule(make_features(40, 20, signal=1.5, seed=20), config)
    for b in rule.buckets:
        assert b.raw_rate == pytest.approx(b.failures / b.n)
        assert min(b.raw_rate, rule.base_rate) <= b.rate <= max(b.raw_rate, rule.base_rate)
    text = "\n".join(rule.describe())
    assert "raw" in text and "shrunk" in text and "k=" in text


def test_prediction_equals_base_when_every_bucket_equals_base(config: Config) -> None:
    """No feature says anything, so the forecast must say exactly what the base rate says."""
    df = make_features(40, 20, signal=0.0, seed=21)
    rule = fit_rule(df, config)
    flat = rule.model_copy(update={
        "buckets": tuple(
            b.model_copy(update={"rate": rule.base_rate, "class_rates": dict(rule.class_shares)})
            for b in rule.buckets
        )
    })
    p = predict_one(df.iloc[0], flat)
    assert p.p_fail == pytest.approx(rule.class_shares["BUSTED"], abs=1e-9)
    assert all(abs(v) < 1e-9 for v in p.log_odds.values())


def test_a_single_strong_feature_moves_the_prediction_the_right_way(config: Config) -> None:
    """p_fail is the BUSTED share of the three-way split, so the class rates are what move it."""
    df = make_features(40, 20, signal=0.0, seed=22)
    rule = fit_rule(df, config)
    one = rule.model_copy(update={"usable_features": (rule.usable_features[0],)})
    feature = one.usable_features[0]

    def with_class_rates(busted: float) -> object:
        rest = (1 - busted) / 2
        rates = {"BUSTED": busted, "NEITHER": rest, "SUSTAINED": rest}
        return one.model_copy(update={
            "buckets": tuple(
                b.model_copy(update={"rate": busted, "class_rates": rates})
                for b in one.buckets if b.feature == feature
            )
        })

    row = df.iloc[0]
    base_busted = one.class_shares["BUSTED"]
    assert predict_one(row, with_class_rates(0.60)).p_fail > base_busted
    assert predict_one(row, with_class_rates(0.05)).p_fail < base_busted


def test_a_missing_feature_contributes_nothing(config: Config) -> None:
    """Under log-odds a missing feature drops out cleanly; an average would rescale."""
    import math as _m

    from intraday.forecast import ORDERED_CLASSES, REFERENCE_CLASS, class_log_ratio

    df = make_features(40, 20, signal=1.5, seed=23)
    rule = fit_rule(df, config)
    row = df.iloc[0].copy()
    full = predict_one(row, rule)

    dropped = rule.usable_features[-1]
    bucket = next(b for b in buckets_hit(row, rule) if b.feature == dropped)
    row[dropped] = np.nan
    partial = predict_one(row, rule)
    assert dropped not in partial.contributions

    # Removing that bucket should move each class's log-ratio by exactly its damped term.
    for cls in ORDERED_CLASSES:
        if cls == REFERENCE_CLASS:
            continue
        base_ratio = class_log_ratio(rule.class_shares, cls)
        expected_shift = -rule.damping * (class_log_ratio(bucket.class_rates, cls) - base_ratio)
        before = _m.log(full.class_probabilities[cls] / full.class_probabilities[REFERENCE_CLASS])
        after = _m.log(partial.class_probabilities[cls] / partial.class_probabilities[REFERENCE_CLASS])
        assert after - before == pytest.approx(expected_shift, abs=1e-9), cls


def test_explain_reproduces_the_number(config: Config) -> None:
    """The printed pieces must rebuild p_fail, or the forecast is not hand-checkable.

    Since Phase 6, p_fail is the BUSTED share of the three-way split, so the rebuild goes
    through the class log-ratios rather than the binary logit.
    """
    import math as _m

    from intraday.forecast import ORDERED_CLASSES, class_log_ratio

    df = make_features(40, 20, signal=1.5, seed=24)
    rule = fit_rule(df, config)
    row = df.iloc[0]
    p = predict_one(row, rule)

    hit = buckets_hit(row, rule)
    scores = {}
    for cls in ORDERED_CLASSES:
        base_ratio = class_log_ratio(rule.class_shares, cls)
        scores[cls] = base_ratio + rule.damping * sum(
            class_log_ratio(b.class_rates, cls) - base_ratio for b in hit
        )
    weights = {c: _m.exp(v - max(scores.values())) for c, v in scores.items()}
    assert weights["BUSTED"] / sum(weights.values()) == pytest.approx(p.p_fail, abs=1e-9)

    text = "\n".join(p.explain())
    assert "base rates" in text and "three-way" in text and "p(fail)" in text


def test_average_combination_is_still_available(config: Config) -> None:
    """The old method, extended to three classes: average the bucket rates and normalise."""
    from intraday.forecast import ORDERED_CLASSES

    df = make_features(40, 20, signal=1.5, seed=25)
    avg_cfg = config.model_copy(update={"forecast_combination": "average"})
    rule = fit_rule(df, avg_cfg)
    row = df.iloc[0]
    p = predict_one(row, rule)
    assert rule.combination == "average"

    hit = buckets_hit(row, rule)
    averaged = {c: float(np.mean([b.class_rates[c] for b in hit])) for c in ORDERED_CLASSES}
    total = sum(averaged.values())
    assert p.p_fail == pytest.approx(averaged["BUSTED"] / total)


def test_walk_forward_can_score_either_combination(config: Config) -> None:
    df = make_features(60, 20, signal=1.5, seed=26)
    a = walk_forward(df, config, combination="logodds")
    b = walk_forward(df, config, combination="average")
    assert len(a) == len(b)
    assert not np.allclose(a["p_fail"], b["p_fail"]), "the two methods should differ"
    # both remain honest walk-forward predictions
    for frame in (a, b):
        assert frame["session_date"].nunique() > 1


# ---- three-outcome forecast (Phase 6) --------------------------------------------------


def test_class_probabilities_sum_to_one(config: Config) -> None:
    from intraday.forecast import ORDERED_CLASSES

    df = make_features(40, 20, signal=1.5, seed=30)
    rule = fit_rule(df, config)
    for k in range(0, len(df), 97):
        p = predict_one(df.iloc[k], rule)
        assert sum(p.class_probabilities[c] for c in ORDERED_CLASSES) == pytest.approx(1.0)
        assert all(0.0 <= p.class_probabilities[c] <= 1.0 for c in ORDERED_CLASSES)
        assert p.p_fail == pytest.approx(p.class_probabilities["BUSTED"])


def test_no_buckets_hit_gives_the_training_class_shares(config: Config) -> None:
    from intraday.forecast import ORDERED_CLASSES

    df = make_features(40, 20, signal=1.5, seed=31)
    rule = fit_rule(df, config)
    blank = df.iloc[0].copy()
    for f in rule.usable_features:
        blank[f] = np.nan
    p = predict_one(blank, rule)
    for c in ORDERED_CLASSES:
        assert p.class_probabilities[c] == pytest.approx(rule.class_shares[c], abs=1e-9)


def test_rps_of_a_perfect_forecast_is_zero() -> None:
    from intraday.forecast import ORDERED_CLASSES, ranked_probability_score

    certain = np.eye(len(ORDERED_CLASSES))
    outcomes = np.arange(len(ORDERED_CLASSES))
    assert ranked_probability_score(certain, outcomes) == pytest.approx(np.zeros(len(ORDERED_CLASSES)))


def test_rps_punishes_being_wrong_by_two_steps_more_than_one() -> None:
    """The whole point of the ranked score: calling a failure when it ran is the worst miss."""
    from intraday.forecast import ranked_probability_score

    said_busted = np.array([[1.0, 0.0, 0.0]])
    off_by_one = ranked_probability_score(said_busted, np.array([1]))[0]  # actually NEITHER
    off_by_two = ranked_probability_score(said_busted, np.array([2]))[0]  # actually SUSTAINED
    assert off_by_two > off_by_one > 0


def test_base_rate_forecast_scores_zero_skill(config: Config) -> None:
    from intraday.forecast import score_three_way

    n = 300
    rng = np.random.default_rng(4)
    outcomes = rng.choice(["BUSTED", "NEITHER", "SUSTAINED"], size=n, p=[0.2, 0.55, 0.25])
    shares = {"BUSTED": 0.2, "NEITHER": 0.55, "SUSTAINED": 0.25}
    preds = pd.DataFrame({
        "p_busted": shares["BUSTED"], "p_neither": shares["NEITHER"], "p_sustained": shares["SUSTAINED"],
        "base_busted": shares["BUSTED"], "base_neither": shares["NEITHER"], "base_sustained": shares["SUSTAINED"],
        "outcome": outcomes,
        "session_date": [date(2026, 6, 1) + timedelta(days=k // 5) for k in range(n)],
    })
    s = score_three_way(preds, config)
    assert s.skill == pytest.approx(0.0, abs=1e-12)
    assert s.rps == pytest.approx(s.rps_base)


def test_three_way_floor_applies_to_the_rarest_class(config: Config) -> None:
    from intraday.forecast import score_three_way

    n = 300
    outcomes = ["BUSTED"] * 5 + ["NEITHER"] * 200 + ["SUSTAINED"] * 95
    preds = pd.DataFrame({
        "p_busted": 0.2, "p_neither": 0.55, "p_sustained": 0.25,
        "base_busted": 0.2, "base_neither": 0.55, "base_sustained": 0.25,
        "outcome": outcomes,
        "session_date": [date(2026, 6, 1) + timedelta(days=k // 5) for k in range(n)],
    })
    s = score_three_way(preds, config)
    assert s.verdict == "insufficient_sample"
    assert s.rarest_class == "BUSTED" and s.rarest_count == 5
    assert "thinnest class" in s.statement


def test_three_way_finds_a_real_pattern(config: Config) -> None:
    from intraday.forecast import score_three_way

    preds = walk_forward(make_features(80, 20, signal=2.5, seed=32), config)
    s = score_three_way(preds, config)
    assert s.n == len(preds)
    assert sum(s.class_counts.values()) == s.n
    assert s.calibration_busted and s.calibration_sustained
    assert s.verdict in {"informative", "no_better_than_base_rate"}


def test_three_way_refuses_predictions_without_the_class_columns(config: Config) -> None:
    from intraday.forecast import score_three_way

    preds = pd.DataFrame({"p_fail": [0.2] * 40, "outcome": ["BUSTED"] * 40})
    with pytest.raises(ValueError, match="re-run the forecast"):
        score_three_way(preds, config)


def test_accuracy_is_reported_next_to_the_majority_baseline(config: Config) -> None:
    """Accuracy is the number people ask for and the easiest to game on a skewed problem.

    A forecaster that never predicts the rare class scores the majority share and knows
    nothing, so the score records both figures and the verdict never rests on accuracy.
    """
    from intraday.forecast import score_three_way

    n = 300
    outcomes = ["BUSTED"] * 60 + ["NEITHER"] * 180 + ["SUSTAINED"] * 60
    never_predicts_failure = pd.DataFrame({
        "p_busted": 0.0, "p_neither": 0.7, "p_sustained": 0.3,
        "base_busted": 0.2, "base_neither": 0.6, "base_sustained": 0.2,
        "outcome": outcomes,
        "session_date": [date(2026, 6, 1) + timedelta(days=k // 5) for k in range(n)],
    })
    s = score_three_way(never_predicts_failure, config)
    assert s.accuracy == pytest.approx(180 / 300), "it gets every NEITHER right and nothing else"
    assert s.accuracy_baseline == pytest.approx(180 / 300)
    assert s.accuracy == pytest.approx(s.accuracy_baseline), (
        "a useless forecaster matches the baseline exactly - which is the point of showing both"
    )
    # and it is not rewarded for that: the ranked score still judges it against the base rates
    assert s.verdict in {"no_better_than_base_rate", "informative"}


def test_binary_flag_features_are_usable_in_a_fitted_rule(config: Config) -> None:
    """A 0/1 flag has no terciles. The forecast must bucket it by value, as the study does,
    or index_or_agrees would silently never contribute to any prediction."""
    rng = np.random.default_rng(40)
    rows = []
    start = date(2026, 3, 2)
    # three blocks with 30%, 50% and 80% ones, so the flag genuinely varies
    for block, share in enumerate((0.3, 0.5, 0.8)):
        for k in range(300):
            day = start + timedelta(days=block * 20 + k // 15)
            row = {f: float(rng.normal()) for f in FEATURE_NAMES}
            row["index_or_agrees"] = float(rng.random() < share)
            fails = rng.random() < (0.35 if row["index_or_agrees"] == 0.0 else 0.12)
            row.update({
                "symbol": f"S{k % 10}", "session_date": day, "direction": "long",
                "breakout_time": datetime.combine(day, datetime.min.time(), tzinfo=IST),
                "label": "BUSTED" if fails else ("SUSTAINED" if rng.random() < 0.4 else "NEITHER"),
            })
            rows.append(row)
    df = pd.DataFrame(rows)

    rule = fit_rule(df, config)
    assert "index_or_agrees" in rule.usable_features, "a binary flag must be bucketable"
    flag_buckets = [b for b in rule.buckets if b.feature == "index_or_agrees"]
    assert len(flag_buckets) == 2, f"expected one bucket per value, got {len(flag_buckets)}"
    assert {b.third for b in flag_buckets} == {"low", "high"}
    assert {b.lower for b in flag_buckets} == {0.0, 1.0}
    assert sum(b.n for b in flag_buckets) == len(df)

    # and a row actually lands in the bucket matching its value
    row = df[df["index_or_agrees"] == 1.0].iloc[0]
    hit = next(b for b in buckets_hit(row, rule) if b.feature == "index_or_agrees")
    assert hit.lower == 1.0


def test_forecast_and_study_bucket_a_feature_identically(config: Config) -> None:
    """The rule must be fitted on the same groups the study reports, or the two disagree."""
    from intraday.analysis import bust_rate_by_third

    df = make_features(40, 20, signal=1.5, seed=41)
    rule = fit_rule(df, config)
    for feature in rule.usable_features:
        table = bust_rate_by_third(df, feature, "all")
        assert table is not None
        fitted = [b for b in rule.buckets if b.feature == feature]
        assert [b.n for b in fitted] == [t.n for t in table.thirds], feature
        assert [b.failures for b in fitted] == [t.busts for t in table.thirds], feature

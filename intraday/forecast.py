"""Forecasting: state a probability before the outcome, then score it.

A backtest is a summary of what happened. A forecast is a claim about what will happen,
made without the answer in hand. The difference is procedural, not statistical, so this
module enforces the procedure:

1. ``fit_rule`` builds a lookup table from a TRAILING WINDOW of finished breakouts:
   for each feature, the observed failure rate in each third. Nothing else.
2. ``predict_one`` applies that table to a new breakout and returns a probability.
   It never sees the breakout's own outcome, and the rule it uses was built only from
   sessions strictly before that date.
3. ``walk_forward`` replays the whole history that way - refit, predict, step - so the
   predictions are honest out-of-sample statements even though we make them in bulk.
4. ``score`` grades them: Brier score against the base rate, and a calibration table.

The scoring answer is the only claim this module makes. If the Brier skill score is not
positive, the forecast carries no information and the report says so.

Why a lookup table and not a model: it can be checked by hand. Every number that goes
into a prediction is printed - the base rate, each feature's bucket, its raw and shrunk
failure rate, and the log-odds each contributes - so the result can be reproduced with a
calculator.

Two refinements over a plain average of bucket rates, both pre-registered:

- Shrinkage. A bucket's rate becomes (failures + k * base) / (n + k). A bucket holding
  six breakouts, four of which failed, is not evidence of a 67% failure rate; shrinkage
  pulls it back towards the base rate until it has earned its distance from it.
- Log-odds combination. Probabilities are combined as
      logit(p) = logit(base) + damping * sum over features of (logit(bucket) - logit(base))
  rather than averaged. A missing feature contributes exactly zero, so a prediction no
  longer changes scale just because a feature is absent - which an average does, silently.
  Damping below 1 stops three agreeing features from compounding into false certainty.

Each prediction also carries the full three-way split - p_busted, p_sustained, p_neither -
because "will it fail?" throws away the difference between a breakout that runs and one
that drifts, and those are not the same trade. The three are built the same way: shrunk
class counts per bucket, combined as log-ratios against NEITHER, then normalised.
``p_fail`` remains exactly ``p_busted`` so nothing downstream changes.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from intraday.analysis import THIRD_NAMES
from intraday.config import IST, Config
from intraday.features import FEATURE_NAMES
from intraday.labelling import Label
from intraday.stats import session_bootstrap, session_variance_share

PREDICTIONS_FILE = "predictions.parquet"
MIN_TRAIN_FAILURES = 30  # the same floor used everywhere else
MIN_BUCKET_ROWS = 20  # below this a bucket's rate is too noisy to quote
PROB_CLIP = (0.001, 0.999)  # logit is undefined at 0 and 1

# Ordered worst to best for the ranked probability score: being wrong by two steps
# (calling a failure when it ran) should cost more than being wrong by one.
ORDERED_CLASSES: tuple[str, ...] = (Label.BUSTED.value, Label.NEITHER.value, Label.SUSTAINED.value)
REFERENCE_CLASS = Label.NEITHER.value  # log-ratios are taken against this one
CALIBRATION_BINS = ((0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.01))


class Bucket(BaseModel):
    model_config = ConfigDict(frozen=True)

    feature: str
    third: str  # low / mid / high
    lower: float
    upper: float
    n: int
    failures: int
    raw_rate: float  # failures / n, before shrinkage
    rate: float  # (failures + k * base) / (n + k): what the forecast actually uses
    class_counts: dict[str, int]  # BUSTED / NEITHER / SUSTAINED counts in this bucket
    class_rates: dict[str, float]  # the same, shrunk towards the training class shares


class Rule(BaseModel):
    """What the forecaster knows, fitted on a closed window. Readable in full."""

    model_config = ConfigDict(frozen=True)

    fitted_on: tuple[date, date]  # inclusive window of session dates
    n_train: int
    train_failures: int
    base_rate: float
    class_shares: dict[str, float]  # training-window share of each class
    buckets: tuple[Bucket, ...]
    usable_features: tuple[str, ...]  # features with enough data to contribute

    shrinkage_k: float
    damping: float
    combination: str

    def describe(self) -> list[str]:
        lines = [
            f"fitted on {self.n_train} breakouts, {self.fitted_on[0]} to {self.fitted_on[1]}, "
            f"{self.train_failures} of them failed (base rate {self.base_rate:.1%})",
            f"combination {self.combination}, shrinkage k={self.shrinkage_k:g}, damping {self.damping:g}",
        ]
        for f in self.usable_features:
            parts = [
                f"{b.third} {b.failures}/{b.n} raw {b.raw_rate:.0%} -> shrunk {b.rate:.0%}"
                for b in self.buckets if b.feature == f
            ]
            lines.append(f"  {f}: " + "  |  ".join(parts))
        if not self.usable_features:
            lines.append("  no feature had enough data; the rule falls back to the base rate")
        return lines


class Prediction(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    session_date: date
    direction: str
    breakout_time: datetime
    made_at: datetime  # when the forecast was produced
    p_fail: float  # identical to class_probabilities[BUSTED]; kept so callers do not break
    class_probabilities: dict[str, float]  # BUSTED / NEITHER / SUSTAINED, summing to 1
    base_rate: float  # what the naive forecast would have said
    base_class_shares: dict[str, float]
    contributions: dict[str, float]  # feature -> shrunk bucket rate used
    log_odds: dict[str, float]  # feature -> logit(bucket) - logit(base), before damping
    outcome: str | None = None  # filled in only by scoring, never by prediction

    def explain(self) -> list[str]:
        """Every number behind the prediction, so it can be checked with a calculator.

        The binary line shows how far each feature moves the odds of failure; the
        three-way line is what ``p_fail`` is actually taken from.
        """
        lines = [
            f"base rates: "
            + ", ".join(f"{c.lower()} {self.base_class_shares.get(c, 0.0):.1%}" for c in ORDERED_CLASSES)
        ]
        for feature, rate in self.contributions.items():
            lines.append(
                f"  {feature}: failure bucket {rate:.1%} -> logit {_logit(rate):+.4f}, "
                f"contributes {self.log_odds[feature]:+.4f} to the odds of failure"
            )
        lines.append(
            "three-way: "
            + ", ".join(f"{c.lower()} {self.class_probabilities[c]:.1%}" for c in ORDERED_CLASSES)
        )
        lines.append(f"p(fail) {self.p_fail:.1%} (the BUSTED share of the three-way split)")
        return lines


def _bucket_for(value: float, buckets: Iterable[Bucket]) -> Bucket | None:
    ordered = sorted(buckets, key=lambda b: b.lower)
    if not ordered or math.isnan(value):
        return None
    for b in ordered[:-1]:
        if value <= b.upper:
            return b
    return ordered[-1]


def _logit(p: float) -> float:
    p = min(max(p, PROB_CLIP[0]), PROB_CLIP[1])
    return math.log(p / (1 - p))


def _expit(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def shrink(failures: int, n: int, base: float, k: float) -> float:
    """Pull a bucket's rate towards the base rate until it has earned its distance."""
    if n + k <= 0:
        return base
    return (failures + k * base) / (n + k)


def buckets_hit(row: pd.Series, rule: Rule) -> list[Bucket]:
    """The buckets this row lands in, one per usable feature it has a value for.

    Exposed so that callers, tests and the report all select buckets the same way rather
    than each re-deriving the boundary rule and disagreeing at the edges.
    """
    found: list[Bucket] = []
    for feature in rule.usable_features:
        b = _bucket_for(float(row[feature]), [x for x in rule.buckets if x.feature == feature])
        if b is not None:
            found.append(b)
    return found


def fit_rule(train: pd.DataFrame, config: Config) -> Rule:
    """Failure rate per third of each feature, over the rows given. Training data only."""
    if train.empty:
        raise ValueError("cannot fit a rule on an empty window")
    dates = pd.to_datetime(train["session_date"]).dt.date
    is_fail = (train["label"] == Label.BUSTED.value).to_numpy()
    base = float(is_fail.mean())
    labels = train["label"].to_numpy()
    shares = {c: float((labels == c).mean()) for c in ORDERED_CLASSES}

    buckets: list[Bucket] = []
    usable: list[str] = []
    for feature in FEATURE_NAMES:
        rows = train[train[feature].notna()]
        if len(rows) < 3 * MIN_BUCKET_ROWS:
            continue
        values = rows[feature].to_numpy(dtype="float64")
        edges = [float(np.quantile(values, q)) for q in (1 / 3, 2 / 3)]
        if edges[0] == edges[1]:
            continue
        fails = (rows["label"] == Label.BUSTED.value).to_numpy()
        masks = [values <= edges[0], (values > edges[0]) & (values <= edges[1]), values > edges[1]]
        bounds = [(float(values.min()), edges[0]), (edges[0], edges[1]), (edges[1], float(values.max()))]
        made: list[Bucket] = []
        for third, mask, (lo, hi) in zip(THIRD_NAMES, masks, bounds):
            n = int(mask.sum())
            if n < MIN_BUCKET_ROWS:
                made = []
                break
            failures = int(fails[mask].sum())
            bucket_labels = rows["label"].to_numpy()[mask]
            counts = {c: int((bucket_labels == c).sum()) for c in ORDERED_CLASSES}
            made.append(Bucket(
                feature=feature, third=third, lower=lo, upper=hi,
                n=n, failures=failures, raw_rate=failures / n,
                rate=shrink(failures, n, base, config.forecast_shrinkage_k),
                class_counts=counts,
                class_rates={
                    c: shrink(counts[c], n, shares[c], config.forecast_shrinkage_k)
                    for c in ORDERED_CLASSES
                },
            ))
        if made:
            buckets.extend(made)
            usable.append(feature)

    return Rule(
        fitted_on=(dates.min(), dates.max()),
        n_train=len(train),
        train_failures=int(is_fail.sum()),
        base_rate=base,
        class_shares=shares,
        buckets=tuple(buckets),
        usable_features=tuple(usable),
        shrinkage_k=config.forecast_shrinkage_k,
        damping=config.forecast_logodds_damping,
        combination=config.forecast_combination,
    )


def predict_one(row: pd.Series, rule: Rule, made_at: datetime | None = None) -> Prediction:
    """P(this breakout fails), from its buckets' shrunk training failure rates.

    With ``combination="logodds"`` each feature contributes the distance of its bucket
    from the base rate in log-odds, damped; a missing feature contributes nothing at all.
    With ``"average"`` the old plain mean of bucket rates is used, kept so the two can be
    compared on identical predictions rather than one being chosen quietly.
    """
    hit = buckets_hit(row, rule)
    contributions = {b.feature: b.rate for b in hit}

    base_logit = _logit(rule.base_rate)
    log_odds = {f: _logit(rate) - base_logit for f, rate in contributions.items()}
    if rule.combination == "average":
        p = float(np.mean(list(contributions.values()))) if contributions else rule.base_rate
    else:
        p = _expit(base_logit + rule.damping * sum(log_odds.values()))

    classes = _combine_classes(rule, hit)
    return Prediction(
        symbol=str(row["symbol"]),
        session_date=pd.Timestamp(row["session_date"]).date(),
        direction=str(row["direction"]),
        breakout_time=pd.Timestamp(row["breakout_time"]).to_pydatetime(),
        made_at=made_at or datetime.now(tz=IST),
        p_fail=classes[Label.BUSTED.value],
        class_probabilities=classes,
        base_rate=rule.base_rate,
        base_class_shares=dict(rule.class_shares),
        contributions=contributions,
        log_odds=log_odds,
    )


def class_log_ratio(rates: dict[str, float], cls: str) -> float:
    """log(P(cls) / P(NEITHER)) for one bucket or for the training window."""
    top = max(rates.get(cls, 0.0), PROB_CLIP[0])
    bottom = max(rates.get(REFERENCE_CLASS, 0.0), PROB_CLIP[0])
    return math.log(top / bottom)


def _combine_classes(rule: Rule, hit: list[Bucket]) -> dict[str, float]:
    """Three-way probabilities from the buckets a row landed in.

    Under "logodds", log-ratios against NEITHER are summed and damped exactly as the
    binary case, then exponentiated and normalised. Under "average" the buckets' shrunk
    class rates are averaged and normalised, which is the old behaviour extended to three
    classes so the two methods stay comparable.

    With no buckets hit the answer is the training-window class shares, which is the
    honest "I know nothing extra beyond the base rates" position.
    """
    if not hit:
        total = sum(rule.class_shares.get(c, 0.0) for c in ORDERED_CLASSES)
        if total <= 0:
            return {c: 1 / len(ORDERED_CLASSES) for c in ORDERED_CLASSES}
        return {c: rule.class_shares.get(c, 0.0) / total for c in ORDERED_CLASSES}

    if rule.combination == "average":
        averaged = {
            c: float(np.mean([b.class_rates.get(c, 0.0) for b in hit])) for c in ORDERED_CLASSES
        }
        total = sum(averaged.values())
        if total <= 0:
            return {c: 1 / len(ORDERED_CLASSES) for c in ORDERED_CLASSES}
        return {c: v / total for c, v in averaged.items()}

    scores: dict[str, float] = {}
    for cls in ORDERED_CLASSES:
        base_ratio = class_log_ratio(rule.class_shares, cls)
        total = base_ratio
        for b in hit:
            total += rule.damping * (class_log_ratio(b.class_rates, cls) - base_ratio)
        scores[cls] = total
    largest = max(scores.values())  # subtract the max before exponentiating, for stability
    weights = {c: math.exp(v - largest) for c, v in scores.items()}
    denominator = sum(weights.values())
    return {c: w / denominator for c, w in weights.items()}


def walk_forward(
    features: pd.DataFrame, config: Config, min_train_sessions: int = 20,
    combination: str | None = None,
) -> pd.DataFrame:
    """Replay the history one session at a time: fit on everything strictly earlier,
    predict the session, step forward. Returns predictions with outcomes attached for
    scoring - the outcome is joined AFTER the prediction is made, never before."""
    if features.empty:
        raise ValueError("no features to forecast on")
    df = features.copy()
    df["_date"] = pd.to_datetime(df["session_date"]).dt.date
    sessions = sorted(df["_date"].unique())
    if len(sessions) <= min_train_sessions:
        raise ValueError(
            f"only {len(sessions)} sessions; need more than {min_train_sessions} to hold out any for forecasting"
        )

    rows: list[dict[str, object]] = []
    for target in sessions[min_train_sessions:]:
        train = df[df["_date"] < target]
        if (train["label"] == Label.BUSTED.value).sum() < MIN_TRAIN_FAILURES:
            continue  # the rule would be quoting rates built on too few failures
        rule = fit_rule(train, config if combination is None else config.model_copy(
            update={"forecast_combination": combination}
        ))
        for _, row in df[df["_date"] == target].iterrows():
            p = predict_one(row, rule)
            rows.append({
                "symbol": p.symbol, "session_date": p.session_date, "direction": p.direction,
                "breakout_time": p.breakout_time, "p_fail": p.p_fail, "base_rate": p.base_rate,
                "n_train": rule.n_train, "train_failures": rule.train_failures,
                "features_used": len(p.contributions),
                "p_busted": p.class_probabilities[Label.BUSTED.value],
                "p_neither": p.class_probabilities[Label.NEITHER.value],
                "p_sustained": p.class_probabilities[Label.SUSTAINED.value],
                "base_busted": rule.class_shares[Label.BUSTED.value],
                "base_neither": rule.class_shares[Label.NEITHER.value],
                "base_sustained": rule.class_shares[Label.SUSTAINED.value],
                "outcome": str(row["label"]),  # joined after the fact, for scoring only
            })
    if not rows:
        raise ValueError(
            f"no session had {MIN_TRAIN_FAILURES} failures in its training window; "
            "the history is too short to forecast on"
        )
    return pd.DataFrame(rows)


# ---- scoring -------------------------------------------------------------------------


class CalibrationBin(BaseModel):
    model_config = ConfigDict(frozen=True)

    lower: float
    upper: float
    n: int
    mean_forecast: float
    observed_rate: float


class ThreeWayScore(BaseModel):
    """Ranked probability score over the ordered classes BUSTED < NEITHER < SUSTAINED.

    RPS sums the squared error of the CUMULATIVE probabilities, so being wrong by two
    steps - calling a failure when it ran - costs more than being wrong by one. A perfect
    forecast scores 0; the base-rate forecast scores whatever the class shares imply, and
    skill is measured against that.
    """

    model_config = ConfigDict(frozen=True)

    n: int
    n_sessions: int
    class_counts: dict[str, int]
    rarest_class: str
    rarest_count: int
    rps: float
    rps_base: float
    skill: float
    skill_ci: tuple[float, float]
    # Plain "how often was the most likely outcome the right one", next to the number a
    # forecaster gets for always naming the commonest outcome. Accuracy alone is
    # misleading here: with failures at ~20%, "it never fails" scores ~80% and is useless,
    # which is why the verdict is decided on RPS skill and not on this.
    accuracy: float
    accuracy_baseline: float
    calibration_busted: tuple[CalibrationBin, ...]
    calibration_sustained: tuple[CalibrationBin, ...]
    verdict: str
    statement: str


class Score(BaseModel):
    model_config = ConfigDict(frozen=True)

    n: int
    n_sessions: int
    failures: int
    session_variance_share: float  # how much of the outcome is decided by the day itself
    brier: float  # mean squared error of the probability; lower is better
    brier_base: float  # the same for always forecasting the base rate
    skill: float  # 1 - brier/brier_base; positive means the forecast adds information
    skill_ci: tuple[float, float]  # bootstrap 95% CI on the skill
    calibration: tuple[CalibrationBin, ...]
    verdict: str
    statement: str


def score(predictions: pd.DataFrame, config: Config, seed: int = 0) -> Score:
    """Grade the forecasts. The only claim this module makes lives here."""
    if "outcome" not in predictions.columns:
        raise ValueError("predictions must carry an outcome column to be scored")
    n = len(predictions)
    if n == 0:
        raise ValueError("no predictions to score")
    actual = (predictions["outcome"] == Label.BUSTED.value).to_numpy(dtype="float64")
    p = predictions["p_fail"].to_numpy(dtype="float64")
    base = predictions["base_rate"].to_numpy(dtype="float64")
    brier = float(np.mean((p - actual) ** 2))
    brier_base = float(np.mean((base - actual) ** 2))
    skill = float(1 - brier / brier_base) if brier_base > 0 else 0.0

    # A skill score without an interval is the same mistake as an effect size without one,
    # and an interval built by resampling rows would be far too narrow: breakouts on one
    # session share that day's shock, so whole sessions are resampled instead.
    if "session_date" not in predictions.columns:
        raise ValueError("predictions must carry session_date so the bootstrap can resample sessions")
    sessions = predictions["session_date"].astype(str).to_numpy()

    def skill_stat(idx: np.ndarray) -> float:
        denom = float(np.mean((base[idx] - actual[idx]) ** 2))
        return math.nan if denom <= 0 else 1 - float(np.mean((p[idx] - actual[idx]) ** 2)) / denom

    rng = np.random.default_rng(seed)
    skill_ci_lo, skill_ci_hi, _ = session_bootstrap(sessions, skill_stat, config.bootstrap_n, rng)
    skill_ci = (skill_ci_lo, skill_ci_hi)
    clustering = session_variance_share(actual, sessions)

    bins: list[CalibrationBin] = []
    for lo, hi in CALIBRATION_BINS:
        mask = (p >= lo) & (p < hi)
        if mask.sum() == 0:
            continue
        bins.append(CalibrationBin(
            lower=lo, upper=min(hi, 1.0), n=int(mask.sum()),
            mean_forecast=float(p[mask].mean()), observed_rate=float(actual[mask].mean()),
        ))

    failures = int(actual.sum())
    if failures < config.min_sample:
        verdict = "insufficient_sample"
        statement = (
            f"INSUFFICIENT SAMPLE: only {failures} of the {n} forecast breakouts actually failed, below the "
            f"{config.min_sample} minimum. The forecast cannot be graded yet."
        )
    elif skill_ci[0] > 0:
        verdict = "informative"
        statement = (
            f"The forecast beat the base rate: Brier {brier:.4f} against {brier_base:.4f}, skill "
            f"{skill:+.1%} (95% CI {skill_ci[0]:+.1%} to {skill_ci[1]:+.1%}). The interval is above zero, "
            "so it carries information on this history. Whether that survives into the future is a "
            "separate question no backtest can answer."
        )
    else:
        verdict = "no_better_than_base_rate"
        statement = (
            f"The forecast did NOT reliably beat always saying '{base.mean():.0%}': Brier {brier:.4f} "
            f"against {brier_base:.4f}, skill {skill:+.1%} (95% CI {skill_ci[0]:+.1%} to {skill_ci[1]:+.1%}). "
            "The interval includes zero, so the apparent improvement is within noise. Use the base rate."
        )
    return Score(
        n=n, n_sessions=int(len(np.unique(sessions))), failures=failures,
        session_variance_share=clustering, brier=brier, brier_base=brier_base,
        skill=skill, skill_ci=skill_ci, calibration=tuple(bins), verdict=verdict, statement=statement,
    )


def save_predictions(df: pd.DataFrame, data_dir: Path) -> Path:
    path = data_dir / PREDICTIONS_FILE
    df.to_parquet(path, index=False)
    return path


def load_predictions(data_dir: Path) -> pd.DataFrame:
    path = data_dir / PREDICTIONS_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run study first")
    return pd.read_parquet(path)



def ranked_probability_score(probabilities: np.ndarray, actual_index: np.ndarray) -> np.ndarray:
    """Per-row RPS for ordered classes.

    ``probabilities`` is (rows, classes) in ORDERED_CLASSES order; ``actual_index`` gives
    the observed class position per row. Returns one score per row: the sum over the first
    k-1 cumulative positions of (cumulative forecast - cumulative outcome) squared.
    """
    if probabilities.ndim != 2 or probabilities.shape[1] != len(ORDERED_CLASSES):
        raise ValueError(f"probabilities must be (rows, {len(ORDERED_CLASSES)})")
    if len(probabilities) != len(actual_index):
        raise ValueError("probabilities and outcomes must be aligned")
    cumulative_forecast = np.cumsum(probabilities, axis=1)[:, :-1]
    outcome = np.zeros_like(probabilities)
    outcome[np.arange(len(actual_index)), actual_index] = 1.0
    cumulative_outcome = np.cumsum(outcome, axis=1)[:, :-1]
    return ((cumulative_forecast - cumulative_outcome) ** 2).sum(axis=1)


def _calibration_for(p: np.ndarray, actual: np.ndarray) -> tuple[CalibrationBin, ...]:
    bins: list[CalibrationBin] = []
    for lo, hi in CALIBRATION_BINS:
        mask = (p >= lo) & (p < hi)
        if mask.sum() == 0:
            continue
        bins.append(CalibrationBin(
            lower=lo, upper=min(hi, 1.0), n=int(mask.sum()),
            mean_forecast=float(p[mask].mean()), observed_rate=float(actual[mask].mean()),
        ))
    return tuple(bins)


def score_three_way(predictions: pd.DataFrame, config: Config, seed: int = 0) -> ThreeWayScore:
    """Grade the full three-outcome forecast against the base-rate forecast.

    The sample floor is applied to the RAREST class: a three-way claim is only as strong
    as its thinnest outcome, and quoting skill when one class has barely occurred would be
    the same mistake the binary scorer already refuses to make.
    """
    needed = {"p_busted", "p_neither", "p_sustained", "base_busted", "base_neither",
              "base_sustained", "outcome", "session_date"}
    missing = needed - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions lack {sorted(missing)}; re-run the forecast")
    n = len(predictions)
    if n == 0:
        raise ValueError("no predictions to score")

    forecast = predictions[["p_busted", "p_neither", "p_sustained"]].to_numpy(dtype="float64")
    base = predictions[["base_busted", "base_neither", "base_sustained"]].to_numpy(dtype="float64")
    outcomes = predictions["outcome"].to_numpy()
    position = {c: k for k, c in enumerate(ORDERED_CLASSES)}
    unknown = set(outcomes) - set(position)
    if unknown:
        raise ValueError(f"unknown outcome labels {sorted(unknown)}")
    actual_index = np.array([position[o] for o in outcomes])

    per_row = ranked_probability_score(forecast, actual_index)
    per_row_base = ranked_probability_score(base, actual_index)
    rps = float(per_row.mean())
    rps_base = float(per_row_base.mean())
    skill = float(1 - rps / rps_base) if rps_base > 0 else 0.0

    sessions = predictions["session_date"].astype(str).to_numpy()

    def skill_stat(idx: np.ndarray) -> float:
        denominator = float(per_row_base[idx].mean())
        return math.nan if denominator <= 0 else 1 - float(per_row[idx].mean()) / denominator

    lo, hi, _ = session_bootstrap(sessions, skill_stat, config.bootstrap_n, np.random.default_rng(seed))

    counts = {c: int((outcomes == c).sum()) for c in ORDERED_CLASSES}
    rarest = min(counts, key=lambda c: counts[c])
    predicted = forecast.argmax(axis=1)
    accuracy = float((predicted == actual_index).mean())
    accuracy_baseline = float(max(counts.values()) / n)

    if counts[rarest] < config.min_sample:
        verdict = "insufficient_sample"
        statement = (
            f"INSUFFICIENT SAMPLE: the rarest outcome ({rarest.lower()}) occurred {counts[rarest]} times, "
            f"below the {config.min_sample} minimum. A three-way forecast is only as strong as its "
            "thinnest class, so it cannot be graded yet."
        )
    elif lo > 0:
        verdict = "informative"
        statement = (
            f"The three-way forecast beat the base rates: RPS {rps:.4f} against {rps_base:.4f}, "
            f"skill {skill:+.1%} (95% CI {lo:+.1%} to {hi:+.1%}). The interval is above zero, so it "
            "carries information about which of the three outcomes follows, not just whether it fails."
        )
    else:
        verdict = "no_better_than_base_rate"
        statement = (
            f"The three-way forecast did NOT reliably beat the base rates: RPS {rps:.4f} against "
            f"{rps_base:.4f}, skill {skill:+.1%} (95% CI {lo:+.1%} to {hi:+.1%}). The interval includes "
            "zero, so the apparent improvement is within noise."
        )

    return ThreeWayScore(
        n=n, n_sessions=int(len(np.unique(sessions))), class_counts=counts,
        rarest_class=rarest, rarest_count=counts[rarest],
        rps=rps, rps_base=rps_base, skill=skill, skill_ci=(lo, hi),
        accuracy=accuracy, accuracy_baseline=accuracy_baseline,
        calibration_busted=_calibration_for(
            forecast[:, position[Label.BUSTED.value]], (outcomes == Label.BUSTED.value).astype(float)
        ),
        calibration_sustained=_calibration_for(
            forecast[:, position[Label.SUSTAINED.value]], (outcomes == Label.SUSTAINED.value).astype(float)
        ),
        verdict=verdict, statement=statement,
    )

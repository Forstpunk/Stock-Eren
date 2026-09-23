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

Why a lookup table and not a model: it can be checked by hand. The probability for a
breakout is the failure rate of its bucket in the training window, and both are printed.
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

PREDICTIONS_FILE = "predictions.parquet"
MIN_TRAIN_FAILURES = 30  # the same floor used everywhere else
MIN_BUCKET_ROWS = 20  # below this a bucket's rate is too noisy to quote
CALIBRATION_BINS = ((0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.01))


class Bucket(BaseModel):
    model_config = ConfigDict(frozen=True)

    feature: str
    third: str  # low / mid / high
    lower: float
    upper: float
    n: int
    failures: int
    rate: float  # P(fail) in this bucket over the training window


class Rule(BaseModel):
    """What the forecaster knows, fitted on a closed window. Readable in full."""

    model_config = ConfigDict(frozen=True)

    fitted_on: tuple[date, date]  # inclusive window of session dates
    n_train: int
    train_failures: int
    base_rate: float
    buckets: tuple[Bucket, ...]
    usable_features: tuple[str, ...]  # features with enough data to contribute

    def describe(self) -> list[str]:
        lines = [
            f"fitted on {self.n_train} breakouts, {self.fitted_on[0]} to {self.fitted_on[1]}, "
            f"{self.train_failures} of them failed (base rate {self.base_rate:.1%})"
        ]
        for f in self.usable_features:
            parts = [
                f"{b.third} {b.rate:.0%} ({b.failures}/{b.n})"
                for b in self.buckets if b.feature == f
            ]
            lines.append(f"  {f}: " + "  ".join(parts))
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
    p_fail: float
    base_rate: float  # what the naive forecast would have said
    contributions: dict[str, float]  # feature -> bucket rate that fed the average
    outcome: str | None = None  # filled in only by scoring, never by prediction


def _bucket_for(value: float, buckets: Iterable[Bucket]) -> Bucket | None:
    ordered = sorted(buckets, key=lambda b: b.lower)
    if not ordered or math.isnan(value):
        return None
    for b in ordered[:-1]:
        if value <= b.upper:
            return b
    return ordered[-1]


def fit_rule(train: pd.DataFrame, config: Config) -> Rule:
    """Failure rate per third of each feature, over the rows given. Training data only."""
    if train.empty:
        raise ValueError("cannot fit a rule on an empty window")
    dates = pd.to_datetime(train["session_date"]).dt.date
    is_fail = (train["label"] == Label.BUSTED.value).to_numpy()
    base = float(is_fail.mean())

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
            made.append(Bucket(
                feature=feature, third=third, lower=lo, upper=hi,
                n=n, failures=int(fails[mask].sum()), rate=float(fails[mask].mean()),
            ))
        if made:
            buckets.extend(made)
            usable.append(feature)

    return Rule(
        fitted_on=(dates.min(), dates.max()),
        n_train=len(train),
        train_failures=int(is_fail.sum()),
        base_rate=base,
        buckets=tuple(buckets),
        usable_features=tuple(usable),
    )


def predict_one(row: pd.Series, rule: Rule, made_at: datetime | None = None) -> Prediction:
    """P(this breakout fails), as the average of its buckets' training failure rates.

    Averaging keeps the arithmetic checkable: every contribution is printed. A feature
    that is missing on this row simply does not contribute.
    """
    contributions: dict[str, float] = {}
    for feature in rule.usable_features:
        b = _bucket_for(float(row[feature]), [x for x in rule.buckets if x.feature == feature])
        if b is not None:
            contributions[feature] = b.rate
    p = float(np.mean(list(contributions.values()))) if contributions else rule.base_rate
    return Prediction(
        symbol=str(row["symbol"]),
        session_date=pd.Timestamp(row["session_date"]).date(),
        direction=str(row["direction"]),
        breakout_time=pd.Timestamp(row["breakout_time"]).to_pydatetime(),
        made_at=made_at or datetime.now(tz=IST),
        p_fail=p,
        base_rate=rule.base_rate,
        contributions=contributions,
    )


def walk_forward(features: pd.DataFrame, config: Config, min_train_sessions: int = 20) -> pd.DataFrame:
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
        rule = fit_rule(train, config)
        for _, row in df[df["_date"] == target].iterrows():
            p = predict_one(row, rule)
            rows.append({
                "symbol": p.symbol, "session_date": p.session_date, "direction": p.direction,
                "breakout_time": p.breakout_time, "p_fail": p.p_fail, "base_rate": p.base_rate,
                "n_train": rule.n_train, "train_failures": rule.train_failures,
                "features_used": len(p.contributions),
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


class Score(BaseModel):
    model_config = ConfigDict(frozen=True)

    n: int
    failures: int
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

    # A skill score without an interval is the same mistake as an effect size without one.
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(config.bootstrap_n):
        idx = rng.integers(0, n, n)
        denom = float(np.mean((base[idx] - actual[idx]) ** 2))
        if denom > 0:
            boots.append(1 - float(np.mean((p[idx] - actual[idx]) ** 2)) / denom)
    skill_ci = (
        (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (0.0, 0.0)
    )

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
    return Score(n=n, failures=failures, brier=brier, brier_base=brier_base, skill=skill, skill_ci=skill_ci,
                 calibration=tuple(bins), verdict=verdict, statement=statement)


def save_predictions(df: pd.DataFrame, data_dir: Path) -> Path:
    path = data_dir / PREDICTIONS_FILE
    df.to_parquet(path, index=False)
    return path


def load_predictions(data_dir: Path) -> pd.DataFrame:
    path = data_dir / PREDICTIONS_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run study first")
    return pd.read_parquet(path)

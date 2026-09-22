"""The diagnostic study: what separates busts from sustains, and does it hold out of sample.

1. Per-feature distributions by class with Cohen's d, Mann-Whitney U and the
   common-language effect size. The primary contrast is BUSTED vs SUSTAINED.
2. Logistic regression (only), standardised inputs, target BUSTED vs not-BUSTED.
   No imputation: rows with any missing feature are excluded and counted.
3. Chronological split at a date boundary: earliest ~70% of events train, latest test.
4. Train and test AUC side by side, with a bootstrap CI on the test AUC.
5. Precision at flag-share thresholds against the test base rate.

Verdict criteria (stated, not tuned):
- insufficient_sample: fewer than ``config.min_sample`` busts in the test set
- signal: the lower bound of the bootstrap 95% CI of the test AUC is above 0.5
  (a CI entirely below 0.5 is an anti-fit, i.e. noise, not a signal)
- no_signal: otherwise
"""
from __future__ import annotations

import math
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from intraday.config import Config
from intraday.features import FEATURE_NAMES
from intraday.labelling import Label

TRAIN_FRACTION = 0.7
BOOTSTRAP_N = 500
FLAG_SHARES = (0.10, 0.20, 0.30, 0.50)
Verdict = Literal["signal", "no_signal", "insufficient_sample"]


class FeatureEffect(BaseModel):
    model_config = ConfigDict(frozen=True)

    feature: str
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    median_a: float
    median_b: float
    cohens_d: float  # (mean_a - mean_b) / pooled sd
    cles: float  # P(value in A > value in B), from the Mann-Whitney U
    mw_p: float


class Coefficient(BaseModel):
    model_config = ConfigDict(frozen=True)

    feature: str
    coef: float  # per standard deviation of the feature, log-odds of BUSTED
    ci_low: float
    ci_high: float


class PrecisionAt(BaseModel):
    model_config = ConfigDict(frozen=True)

    flag_share: float
    threshold: float
    n_flagged: int
    n_busts_flagged: int
    precision: float
    base_rate: float
    lift: float
    recall: float


class Confusion(BaseModel):
    model_config = ConfigDict(frozen=True)

    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int


class DiagnosticReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    contrast_bust_vs_sustain: tuple[FeatureEffect, ...]
    contrast_bust_vs_rest: tuple[FeatureEffect, ...]
    n_events: int
    n_complete: int
    n_train: int
    n_test: int
    split_date: date  # first test date
    base_rate_train: float
    base_rate_test: float
    intercept: float
    coefficients: tuple[Coefficient, ...]
    train_auc: float
    test_auc: float
    test_auc_ci: tuple[float, float]
    confusion_at_half: Confusion
    confusion_at_top20: Confusion
    precision_at: tuple[PrecisionAt, ...]
    verdict: Verdict
    statement: str


# ---- 1. per-feature contrasts ---------------------------------------------------------


def feature_effect(feature: str, a: pd.Series, b: pd.Series) -> FeatureEffect:
    a = a.dropna().to_numpy(dtype="float64")
    b = b.dropna().to_numpy(dtype="float64")
    if len(a) < 2 or len(b) < 2:
        raise ValueError(f"{feature}: need at least two values per class, got {len(a)} and {len(b)}")
    pooled = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    d = math.nan if pooled == 0 else (a.mean() - b.mean()) / pooled
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return FeatureEffect(
        feature=feature, n_a=len(a), n_b=len(b),
        mean_a=float(a.mean()), mean_b=float(b.mean()),
        median_a=float(np.median(a)), median_b=float(np.median(b)),
        cohens_d=float(d), cles=float(u / (len(a) * len(b))), mw_p=float(p),
    )


def contrast(features: pd.DataFrame, mask_a: pd.Series, mask_b: pd.Series) -> tuple[FeatureEffect, ...]:
    return tuple(feature_effect(f, features.loc[mask_a, f], features.loc[mask_b, f]) for f in FEATURE_NAMES)


# ---- 3. chronological split -------------------------------------------------------------


def chronological_split(dates: pd.Series, fraction: float = TRAIN_FRACTION) -> tuple[np.ndarray, date]:
    """Boolean train mask and the first test date. Splits at a date boundary so no session
    straddles the split; the boundary is the first date at which the cumulative share of
    events reaches ``fraction``."""
    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0, 1)")
    order = pd.Series(dates).sort_values()
    unique_dates = sorted(order.unique())
    if len(unique_dates) < 2:
        raise ValueError("chronological split needs at least two distinct dates")
    counts = order.value_counts().reindex(unique_dates)
    cumulative = counts.cumsum() / counts.sum()
    train_dates = [d for d, share in cumulative.items() if share < fraction]
    if not train_dates:
        train_dates = [unique_dates[0]]
    split_date = unique_dates[len(train_dates)]
    train_mask = pd.Series(dates).to_numpy() < split_date
    return train_mask, split_date


# ---- 2/4/5. regression --------------------------------------------------------------------


def _fit(x: np.ndarray, y: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(C=np.inf, max_iter=5000)  # unregularised
    model.fit(x, y)
    return model


def _bootstrap_coefficients(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    n = len(y)
    for _ in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        out.append(_fit(x[idx], y[idx]).coef_[0])
    return np.array(out)


def _bootstrap_auc(y: np.ndarray, p: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    aucs = []
    n = len(y)
    for _ in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def _confusion(y: np.ndarray, p: np.ndarray, threshold: float) -> Confusion:
    flag = p >= threshold
    return Confusion(
        threshold=float(threshold),
        tp=int((flag & (y == 1)).sum()), fp=int((flag & (y == 0)).sum()),
        fn=int((~flag & (y == 1)).sum()), tn=int((~flag & (y == 0)).sum()),
    )


def _precision_at(y: np.ndarray, p: np.ndarray) -> tuple[PrecisionAt, ...]:
    base = float(y.mean())
    total_busts = int(y.sum())
    out = []
    for share in FLAG_SHARES:
        threshold = float(np.quantile(p, 1 - share))
        flag = p >= threshold
        n_flag = int(flag.sum())
        n_bust = int((flag & (y == 1)).sum())
        precision = n_bust / n_flag if n_flag else math.nan
        out.append(PrecisionAt(
            flag_share=share, threshold=threshold, n_flagged=n_flag, n_busts_flagged=n_bust,
            precision=precision, base_rate=base,
            lift=precision / base if base > 0 and n_flag else math.nan,
            recall=n_bust / total_busts if total_busts else math.nan,
        ))
    return tuple(out)


def run_diagnostic(features: pd.DataFrame, labels: pd.Series, config: Config, seed: int = 0) -> DiagnosticReport:
    """``features`` has FEATURE_NAMES columns plus ``session_date``; ``labels`` is aligned."""
    missing = [f for f in FEATURE_NAMES if f not in features.columns]
    if missing:
        raise ValueError(f"feature table lacks {missing}")
    if "session_date" not in features.columns:
        raise ValueError("feature table needs a session_date column for the chronological split")
    if len(features) != len(labels):
        raise ValueError("features and labels are not aligned")
    labels = labels.astype(str)
    valid = set(l.value for l in Label)
    bad = set(labels.unique()) - valid
    if bad:
        raise ValueError(f"unknown labels {bad}")

    is_bust = labels == Label.BUSTED.value
    is_sust = labels == Label.SUSTAINED.value
    bust_vs_sustain = contrast(features, is_bust, is_sust)
    bust_vs_rest = contrast(features, is_bust, ~is_bust)

    complete = features[list(FEATURE_NAMES)].notna().all(axis=1).to_numpy()
    x_all = features.loc[complete, list(FEATURE_NAMES)].to_numpy(dtype="float64")
    y_all = is_bust.to_numpy()[complete].astype(int)
    dates = pd.to_datetime(features.loc[complete, "session_date"]).dt.date.reset_index(drop=True)
    if len(y_all) < 2 * config.min_sample:
        raise ValueError(f"only {len(y_all)} complete rows; need at least {2 * config.min_sample}")

    train, split_date = chronological_split(dates)
    test = ~train
    if y_all[train].min() == y_all[train].max() or y_all[test].min() == y_all[test].max():
        raise ValueError("a split side contains a single class; cannot fit or score")

    scaler = StandardScaler().fit(x_all[train])  # train statistics only
    x_train, x_test = scaler.transform(x_all[train]), scaler.transform(x_all[test])
    y_train, y_test = y_all[train], y_all[test]

    model = _fit(x_train, y_train)
    p_train = model.predict_proba(x_train)[:, 1]
    p_test = model.predict_proba(x_test)[:, 1]
    train_auc = float(roc_auc_score(y_train, p_train))
    test_auc = float(roc_auc_score(y_test, p_test))
    auc_ci = _bootstrap_auc(y_test, p_test, seed)

    boots = _bootstrap_coefficients(x_train, y_train, seed)
    coefficients = tuple(
        Coefficient(
            feature=f, coef=float(model.coef_[0][k]),
            ci_low=float(np.percentile(boots[:, k], 2.5)), ci_high=float(np.percentile(boots[:, k], 97.5)),
        )
        for k, f in enumerate(FEATURE_NAMES)
    )

    precision = _precision_at(y_test, p_test)
    top20 = next(p for p in precision if p.flag_share == 0.20)
    n_test_busts = int(y_test.sum())

    if n_test_busts < config.min_sample:
        verdict: Verdict = "insufficient_sample"
    elif auc_ci[0] > 0.5:
        verdict = "signal"
    else:
        verdict = "no_signal"

    statement = _statement(verdict, train_auc, test_auc, auc_ci, n_test_busts, config.min_sample, top20)

    return DiagnosticReport(
        contrast_bust_vs_sustain=bust_vs_sustain,
        contrast_bust_vs_rest=bust_vs_rest,
        n_events=len(features),
        n_complete=int(complete.sum()),
        n_train=int(train.sum()),
        n_test=int(test.sum()),
        split_date=split_date,
        base_rate_train=float(y_train.mean()),
        base_rate_test=float(y_test.mean()),
        intercept=float(model.intercept_[0]),
        coefficients=coefficients,
        train_auc=train_auc,
        test_auc=test_auc,
        test_auc_ci=auc_ci,
        confusion_at_half=_confusion(y_test, p_test, 0.5),
        confusion_at_top20=_confusion(y_test, p_test, top20.threshold),
        precision_at=precision,
        verdict=verdict,
        statement=statement,
    )


def _statement(
    verdict: Verdict, train_auc: float, test_auc: float, ci: tuple[float, float],
    n_test_busts: int, min_sample: int, top20: PrecisionAt,
) -> str:
    gap = train_auc - test_auc
    lines = [
        f"Train AUC {train_auc:.3f}, test AUC {test_auc:.3f} (95% bootstrap CI {ci[0]:.3f}-{ci[1]:.3f}); "
        f"gap {gap:+.3f}.",
        f"Flagging the top 20% of test breakouts by predicted bust probability catches {top20.n_busts_flagged} "
        f"of the busts at {top20.precision:.1%} precision against a {top20.base_rate:.1%} base rate "
        f"(lift {top20.lift:.2f}x).",
    ]
    if verdict == "insufficient_sample":
        lines.append(
            f"INSUFFICIENT SAMPLE: {n_test_busts} busts in the test set, below the {min_sample} minimum. "
            "No claim is made either way; the numbers above are descriptive only."
        )
    elif verdict == "no_signal":
        lines.append(
            "NO OUT-OF-SAMPLE SIGNAL: the test AUC confidence interval does not sit above 0.5. Whatever the "
            "training fit found does not survive into unseen time; treat it as noise."
        )
    else:
        lines.append(
            "OUT-OF-SAMPLE SIGNAL: the test AUC confidence interval sits above 0.5. The effect survives into "
            "unseen time on this regime; it still needs multi-year confirmation before any setup is trusted."
        )
    if gap > 0.10:
        lines.append(f"The train-test gap of {gap:.2f} is large: the model is fitting sample-specific structure.")
    return " ".join(lines)

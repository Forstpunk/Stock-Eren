"""Rich rendering of a DiagnosticReport. Pure presentation; no numbers are computed here."""
from __future__ import annotations

from rich.console import Console
from rich.table import Table

from intraday.diagnostic import DiagnosticReport, FeatureEffect
from intraday.features import FEATURE_MECHANISM


def _stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def _effects_table(title: str, effects: tuple[FeatureEffect, ...], a: str, b: str) -> Table:
    table = Table(title=title)
    table.add_column("feature")
    for col in (f"n {a}", f"n {b}", f"median {a}", f"median {b}", "Cohen d", f"P({a}>{b})", "MW p", ""):
        table.add_column(col, justify="right")
    table.add_column("mechanism")
    for e in sorted(effects, key=lambda e: e.mw_p):
        table.add_row(
            e.feature, str(e.n_a), str(e.n_b), f"{e.median_a:.3f}", f"{e.median_b:.3f}",
            f"{e.cohens_d:+.2f}", f"{e.cles:.3f}", f"{e.mw_p:.3g}", _stars(e.mw_p), FEATURE_MECHANISM[e.feature],
        )
    return table


def render(report: DiagnosticReport, console: Console, title: str) -> None:
    console.rule(title)
    console.print(_effects_table("1. Per-feature contrast: BUSTED vs SUSTAINED (primary)", report.contrast_bust_vs_sustain, "bust", "sust"))
    console.print(_effects_table("1b. Per-feature contrast: BUSTED vs all other breakouts", report.contrast_bust_vs_rest, "bust", "rest"))

    console.print(
        f"\n2. Logistic regression, target BUSTED vs not, {report.n_complete} of {report.n_events} events with "
        f"every feature present (no imputation). Chronological split at {report.split_date}: "
        f"train {report.n_train} (bust rate {report.base_rate_train:.1%}), test {report.n_test} "
        f"(bust rate {report.base_rate_test:.1%})."
    )
    coef = Table(title="Coefficients (log-odds of BUSTED per 1 sd of feature; 95% bootstrap CI)")
    coef.add_column("feature")
    for col in ("coef", "ci low", "ci high", "excludes 0"):
        coef.add_column(col, justify="right")
    for c in sorted(report.coefficients, key=lambda c: -abs(c.coef)):
        excl = "yes" if (c.ci_low > 0 or c.ci_high < 0) else ""
        coef.add_row(c.feature, f"{c.coef:+.3f}", f"{c.ci_low:+.3f}", f"{c.ci_high:+.3f}", excl)
    coef.add_row("(intercept)", f"{report.intercept:+.3f}", "", "", "")
    console.print(coef)

    console.print(f"\n4. Train AUC [bold]{report.train_auc:.3f}[/bold]   Test AUC [bold]{report.test_auc:.3f}[/bold]   "
                  f"test 95% CI {report.test_auc_ci[0]:.3f}-{report.test_auc_ci[1]:.3f}")

    cm = Table(title="Confusion matrix on the test set")
    for col in ("threshold", "TP", "FP", "FN", "TN", "flagged", "precision"):
        cm.add_column(col, justify="right")
    for name, c in (("p >= 0.5", report.confusion_at_half), ("top 20% of p", report.confusion_at_top20)):
        flagged = c.tp + c.fp
        cm.add_row(f"{name} ({c.threshold:.3f})", str(c.tp), str(c.fp), str(c.fn), str(c.tn), str(flagged),
                   f"{c.tp / flagged:.1%}" if flagged else "n/a")
    console.print(cm)

    pr = Table(title=f"5. Precision at flag thresholds vs test base rate {report.base_rate_test:.1%}")
    for col in ("flag top", "n flagged", "busts caught", "precision", "base rate", "lift", "recall"):
        pr.add_column(col, justify="right")
    for p in report.precision_at:
        pr.add_row(f"{p.flag_share:.0%}", str(p.n_flagged), str(p.n_busts_flagged), f"{p.precision:.1%}",
                   f"{p.base_rate:.1%}", f"{p.lift:.2f}x", f"{p.recall:.1%}")
    console.print(pr)

    colour = {"signal": "green", "no_signal": "yellow", "insufficient_sample": "red"}[report.verdict]
    console.print(f"\n[bold {colour}]VERDICT: {report.verdict}[/bold {colour}]")
    console.print(report.statement)

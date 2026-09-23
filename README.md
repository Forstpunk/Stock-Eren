# intraday — NSE opening-range breakout research pipeline

A measurement instrument. It answers one question — *what distinguishes an opening-range
breakout that fails from one that runs?* — and emits a verdict: signal / no signal /
insufficient sample. It never emits buy signals, ratings, price targets or stock picks.

## Run

```
.venv/Scripts/python -m intraday update     # fetch bars, validate every session
.venv/Scripts/python -m intraday study      # label, measure, backtest, write the report
.venv/Scripts/python -m pytest
```

Two commands. `update` takes a few minutes (it downloads); `study` takes a few more (it
bootstraps confidence intervals).

For the answer in rupees and plain questions, with no statistics at all:

```
.venv/Scripts/python -m intraday study --plain
```

`--summary` gives the verdict with a short explanation; `--quiet` writes
`data/report.txt` without printing. The full `report.txt` always contains everything.

Use `.venv/Scripts/python`, not bare `python` — the system interpreter has no packages.

## How it answers the question

Every feature is cut into thirds by value, and the failure rate is counted in each third
and compared to the overall rate. That is the whole method; you can check any number in
the report by counting rows in `data/features.parquet`.

```
failure rate by rvol_open_15m       n     failed    rate    vs overall
low third                         112       31     27.7%      +7.7
mid third                         117       21     17.9%      -2.1
high third                        118       13     11.0%      -9.0
```

The period is split by date: the earliest 70% of dates are the first period, the later 30%
the second. A feature counts as a finding only if **both** of these hold:

1. each period contains at least **30 failed breakouts among the rows where
   that feature exists** — otherwise the feature is reported as *not testable* and no
   claim is made about it either way;
2. its low-third-to-high-third gap is at least **10 percentage points in both periods**.

The per-feature floor matters more than it sounds. RVOL needs 20 prior sessions, so on a
short history it exists on only a fraction of the rows and is almost absent early on. A
22-point gap measured on 13 failures is not a finding, and an earlier version of this
rule called it one.

There is no model, no fitted coefficient and no score. An earlier version used logistic
regression; it was removed because a coefficient of "−1.62 log-odds per standard
deviation" cannot be checked by hand, and a number you cannot check is a number you
cannot trust.

## Frozen research definitions

Do not change these between runs. Changing a definition after seeing a result is how
research becomes storytelling.

| item | value | where |
|---|---|---|
| opening range | first 15 min (3 x 5m bars) | `config.opening_range_minutes` |
| breakout | first CLOSE beyond the range, per direction | `labelling.detect_breakout_at` |
| ran (SUSTAINED) | extends >= 0.5 x prior-day ATR beyond the boundary | `config.sustain_extension_atr` |
| failed (BUSTED) | never extends >= 0.25 ATR, then closes through the opposite boundary before 15:15 | `config.bust_extension_atr`, `bust_cutoff` |
| neither | everything else, kept as its own class | |
| features tested (directional) | opening-15m volume, breakout-bar volume, breakout-bar body, relative strength vs index, index breaking the same way, overnight gap | `features.FEATURE_NAMES` |
| features reported but two-sided | yesterday's move, expiry day | `features.REPORT_ONLY_NAMES` |
| split | earliest 70% of dates / later 30% of dates | `analysis.TRAIN_FRACTION` |
| gap to count as a finding | 10 points, in both periods | `analysis.MIN_GAP_PCT` |
| failures needed to test a feature | 30 in each period, where the feature exists | `config.min_sample` |
| stop | 1.0 x prior-day ATR | `config.stop_atr_multiple` |
| position | Rs 50,000 | `config.position_inr` |
| min sample | 30 failures / 30 trades per claim | `config.min_sample` |
| slippage | 5 / 10 / 20 bps each side | `config.slippage_bps` |

`or_width_atr` and `minutes_since_open` are carried for segmentation but are **not**
tested as mechanisms: both are partly definitional. A failure is defined as a move back
through the opposite boundary, which is a width-scaled distance, and a breakout late in
the session has less time to resolve either way.

## What the pipeline refuses to do

- No expectancy below 30 trades — `InsufficientSampleError`, never a caveated number.
- No result without a matched random benchmark: same session, same holding time, same
  costs, random symbol and direction. The report leads with that comparison.
- No imputation. A missing value stays NaN and its row is excluded and counted.
- No fallback data source. If the configured source fails, the run fails.
- The failed-ORB setup (`failed_orb`) does not run unless the study verdict is `signal`.
  There is no override flag.

## Changelog of definitions

Research definitions are frozen; when one changes it is recorded here with the date and
the reason, so a later result can never be quietly explained by a rule that moved.

| date | change | reason |
|---|---|---|
| 2026-09-23 | **`breakout_depth_atr` demoted to context: partly definitional; caught in code review before results were interpreted.** | `labelling.resolve` measures the move from the breakout bar onward, so the breakout bar's own excursion already counts towards the label. A close at depth ≥ `bust_extension_atr` can never be BUSTED, and one at depth ≥ `sustain_extension_atr` is SUSTAINED immediately. On the current data no BUSTED row exceeds depth 0.18 against a 0.25 threshold. Testing it would have measured the definition, not a mechanism. |
| 2026-09-23 | **Forecast combination: average of bucket rates → shrunk log-odds.** | An average let a missing feature drag the prediction towards whatever the other features said, so an incomplete row silently changed scale. Under log-odds a missing feature contributes exactly zero. Shrinkage (k=40) stops a six-observation bucket being quoted as if it were evidence. Both parameters pre-registered in RESEARCH.md before implementation; the old method is kept and reported side by side. |
| 2026-09-23 | **Three-outcome forecast added.** | "Will it fail?" throws away the difference between a breakout that runs and one that drifts, and those are not the same trade. Predictions now carry p_busted, p_sustained and p_neither, scored with the ranked probability score so being wrong by two steps costs more than by one. `p_fail` stays the pre-registered BINARY forecast, scored with Brier; the three-way probabilities are separate columns scored with RPS. An earlier version of this change redefined `p_fail` as the BUSTED share of the three-way split, which quietly swapped the model the Brier score was grading; corrected in code review. |
| 2026-09-23 | **Random twin is now stopped.** | The benchmark twin had no stop, so its losses could run past −1R while the strategy's could not. That flattered the strategy by comparison. The twin now uses the same stop rule, capped at the matched duration. This makes reported edge vs random smaller, and correct. |
| 2026-09-23 | **Gap-through stops fill at the bar's open.** | A bar that opens beyond the stop never offered the stop price. Filling at the stop credited trades with prices that were not available. |
| 2026-09-23 | **GST now applies to the SEBI fee.** | It always did in reality; the model omitted it. Round-trip cost on a ₹50,000 flat trade at 10 bps moves from ₹153.0046 to ₹153.0226. |
| 2026-09-23 | **RVOL lookback: 20 → 14 sessions.** | Taken from Zarattini & Aziz, who define relative volume against the previous 14 days, and pre-registered in RESEARCH.md before implementation. Not chosen from our results. It also lifts the rows where RVOL exists from 347 of 840 to 493, which is what made the feature testable in both periods at all. |
| 2026-09-23 | **Bootstrap unit: trade → session.** Every confidence interval (forecast skill, edge vs random, expectancy) now resamples whole sessions rather than individual trades. | Breakouts on the same session share that day's market-wide shock. Resampling rows treats them as independent, which makes intervals far too narrow and can declare an edge that is not there. Measured on synthetic data with zero true effect and a realistic session shock, row resampling produced a false-positive rate well above the nominal 5%; session resampling stays near it. |

## Data-quality facts learned on yfinance (Sep 2026)

- Yahoo rejects a 60-day span for 5m bars; 59 is the real cap, enforced before the network.
- From 2026-08-03 Yahoo folds 15:15–15:30 into the 15:15 bar for NSE equities, whose close
  equals the official close. Those sessions are `TAIL_COLLAPSED` and are usable.
- The 09:15 bar has zero volume in ~90% of sessions. VWAP and RVOL are NaN there.
- Yahoo's daily feed emits a phantom flat, zero-volume row on NSE holidays; the trading
  calendar catches and quarantines them (80 rows across 20 equities).

## Forecasting, not just measuring

`study` also makes real forecasts and grades them. For every session after the first 20 it
refits a rule on **earlier sessions only**, predicts each breakout's probability of
failing, and only afterwards joins the outcome to score it. The score is a Brier skill
against the naive "always say the base rate" forecast, with a bootstrap interval, plus a
calibration table of what it said against what happened.

The rule is a lookup table — the failure rate of each third of each feature in the
training window — so any prediction can be checked by hand from the printed buckets.

On the current data it scores **+3.2% skill (95% CI +1.0% to +5.1%)** over 408 forecasts:
the breakouts it called safest failed 12% of the time, the ones it called riskiest 26%.
That is a genuine out-of-sample signal, and it is still smaller than the ₹150 round-trip
cost, so it does not turn into profit. See RESEARCH.md.

## Current result

On 41 sessions x 20 symbols: 20% of breakouts failed, stable month to month. The baseline
ORB+VWAP setup is indistinguishable from random entries and loses 0.12–0.29R per trade to
costs at 5–20 bps.

See `data/report.txt` after a run for the current verdict; it moves as the data does.

That is a limit of the data, not of the method. yfinance serves a rolling 60 days, so the
sample cannot grow past this.

`intraday/sources/kite.py` is now a working Kite Connect adapter (~10 years of 5-minute
history). It handles instrument-token lookup, the 100-day request cap, the 3 req/sec limit
and the 06:00 IST token expiry. Credentials come from `KITE_API_KEY` and
`KITE_ACCESS_TOKEN`; nothing is stored in the repo. Run `python -m intraday login` for the
daily login steps.

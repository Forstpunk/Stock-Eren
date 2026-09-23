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

A feature counts as a finding only if **both** of these hold:

1. each half of the period contains at least **30 failed breakouts among the rows where
   that feature exists** — otherwise the feature is reported as *not testable* and no
   claim is made about it either way;
2. its low-third-to-high-third gap is at least **10 percentage points in both halves**.

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
| features tested | opening-15m volume, breakout-bar volume, breakout-bar body | `features.FEATURE_NAMES` |
| gap to count as a finding | 10 points, in both halves | `analysis.MIN_GAP_PCT` |
| failures needed to test a feature | 30 in each half, where the feature exists | `config.min_sample` |
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

## Data-quality facts learned on yfinance (Sep 2026)

- Yahoo rejects a 60-day span for 5m bars; 59 is the real cap, enforced before the network.
- From 2026-08-03 Yahoo folds 15:15–15:30 into the 15:15 bar for NSE equities, whose close
  equals the official close. Those sessions are `TAIL_COLLAPSED` and are usable.
- The 09:15 bar has zero volume in ~90% of sessions. VWAP and RVOL are NaN there.
- Yahoo's daily feed emits a phantom flat, zero-volume row on NSE holidays; the trading
  calendar catches and quarantines them (80 rows across 20 equities).

## Current result

On 41 sessions x 20 symbols: 20% of breakouts failed, stable month to month. The baseline
ORB+VWAP setup is indistinguishable from random entries and loses 0.12–0.29R per trade to
costs at 5–20 bps.

The verdict is `no_signal`. Only `bar_body_ratio` had enough failures in both halves to be
tested, and its gap did not hold. The two volume features could not be tested at all: they
exist on 347 of 840 breakouts, and the first half holds 13 failures where they exist
against a floor of 30.

That is a limit of the data, not of the method. yfinance serves a rolling 60 days, so the
sample cannot grow past this. `intraday/sources/kite.py` documents what a Kite Connect
adapter must handle; set `Config.source = "kite"` and nothing downstream changes.

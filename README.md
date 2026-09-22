# intraday — NSE opening-range breakout research pipeline

A measurement instrument. It answers one question — *what distinguishes an opening-range
breakout that sustains from one that fails?* — and emits verdicts: edge detected / no edge /
insufficient sample. It never emits signals, ratings or predictions.

## Run

```
.venv/Scripts/python -m intraday fetch --universe universe.txt --days 59   # yfinance's real cap is 59
.venv/Scripts/python -m intraday label
.venv/Scripts/python -m intraday features
.venv/Scripts/python -m intraday diagnose            # also: --rvol-lookback 10 (labelled sensitivity run)
.venv/Scripts/python -m intraday backtest --setup orb --slippage-bps 5,10,20
.venv/Scripts/python -m intraday backtest --setup failed_orb   # refuses unless diagnose said "signal"
.venv/Scripts/python -m intraday report              # -> data/report.txt
.venv/Scripts/python -m pytest
```

Every stage reads only what the previous stage persisted under `data/`. `features` refuses if
`fetch` ran after `label` (stored indices would be stale): re-run `label` first.

## Frozen research definitions (do not change between runs)

| item | value | where |
|---|---|---|
| opening range | first 15 min (3 x 5m bars) | `config.opening_range_minutes` |
| breakout | first CLOSE beyond the range per direction | `labelling.detect_breakout_at` |
| SUSTAINED | extends >= 0.5 x prior-day ATR beyond the boundary | `config.sustain_extension_atr` |
| BUSTED | never extends >= 0.25 ATR, closes through the opposite boundary before 15:15 | `config.bust_extension_atr`, `bust_cutoff` |
| NEITHER | everything else (kept as a class) | |
| RVOL lookback | 20 sessions, time-of-day matched | `config.rvol_lookback_sessions` |
| stop | 1.0 x prior-day ATR | `config.stop_atr_multiple` |
| position | Rs 50,000 | `config.position_inr` |
| min sample | 30 trades per claim | `config.min_sample` |
| slippage | 5 / 10 / 20 bps each side | `config.slippage_bps` |

Thresholds moved from OR-width units to ATR units after the yfinance pilot showed width-unit
thresholds made the resolution rate a function of range width. Bust rate still depends on
width because "closes through the opposite boundary" is a width-scaled distance; that is the
concept, not the units.

## Data-quality facts learned on yfinance (Sep 2026)

- Yahoo rejects a 60-day span for 5m bars; 59 is the cap. Enforced before the network.
- From 2026-08-03 Yahoo folds 15:15–15:30 into the 15:15 bar for NSE equities (its close
  equals the official close). Such sessions are `TAIL_COLLAPSED` and enter the research set.
- The 09:15 bar has zero volume in ~90% of sessions (Yahoo artifact). VWAP/RVOL are NaN there.
- Yahoo's daily feed emits a phantom flat, zero-volume row on NSE holidays. The calendar check
  quarantines them (80 rows across 20 equities).
- Sessions arrive complete; no symbol lost a session another had, on three fetches.

## Verdict rules (stated, not tuned)

- Diagnostic: `insufficient_sample` if the test set has < 30 busts; `signal` if the bootstrap
  95% CI of the test AUC lies above 0.5; else `no_signal`.
- Setup: `edge detected` only if the base variant beats matched random with a CI above zero
  AND has positive expectancy with a CI above zero at every slippage level.
- S2 (`failed_orb`) runs only on a `signal` diagnostic verdict. No override flag.

## Kite Connect

`intraday/sources/kite.py` documents what the adapter must handle. Set `Config.source = "kite"`;
nothing else changes. The pilot on yfinance (41 sessions, 20 symbols) reached 14–27 test busts
against a floor of 30, so the diagnostic verdict is `insufficient_sample` and S2 is gated.

# What the sources actually say

Sources consulted: the two relevant books on this machine (Bulkowski, *Encyclopedia of
Chart Patterns*, 3rd ed.; Grimes, *The Art and Science of Technical Analysis*), and the
published ORB literature (Zarattini & Aziz, SSRN 4416622 and 4729284, plus the
QuantConnect replication of their "stocks in play" universe rule).

The other eight books on this machine are fundamental/valuation (Graham & Dodd, Koller,
Schilit, Williams, Tracy) and have no bearing on intraday breakouts.

---

## 1. Grimes: the gap between a measurement and a forecast

> "If we are analyzing actual trading records, this can be as simple as calculating
> summary statistics for historical trades, but the problem is much more complicated on a
> **look-forward basis** because we have to make assumptions about how closely future
> conditions are likely to resemble history."
> — *Art and Science*, p23

And on what an edge is at all:

> "A positive expectancy results when the trader successfully identifies those moments
> where markets are **slightly less random than usual**." — p21

> "Markets are extremely competitive. They are usually very close to efficient, and most
> observed price movements are random. It is therefore exceedingly difficult to derive a
> method that makes superior risk-adjusted profits." — p12

**What this pipeline took from it.** A backtest is a summary statistic. To make a forecast
you must (a) state it before the outcome, (b) use only information available at that
moment, and (c) score it afterwards. That is what `predict` and `score` now do. Grimes's
point is that the step from (a) to a *reliable* forecast is an assumption about regime
stability — which is why the scorer reports calibration over time rather than one number.

## 2. Bulkowski: separate the pattern from the exit rule

Bulkowski refuses to test a pattern with a stop-loss attached, because then you are
testing the stop:

> "Does this mean the double bottom lost $1.36 a share? No. It means the stop-loss order
> lost that much. You tested the stop-loss order, not the double bottom."
> — *Encyclopedia*, 3rd ed., p102

His solution is the **ultimate high / ultimate low**: measure from the breakout to the
best price reached before the move is over, i.e. a perfect exit. His **breakeven failure
rate** is then the share of patterns that never move more than 5% — his assumed round-trip
cost — measured on those perfect trades.

And his own caveat:

> "Are the results realistic? Not really. You likely won't be able to duplicate them in
> real life." — p106

**What this pipeline took from it.** Two things.

1. Our breakeven failure rate is the share of trades whose *maximum favourable excursion*
   never reaches +0.5R. That is Bulkowski's metric with R in place of his flat 5%, which
   is the right adaptation because our costs are modelled per trade rather than assumed.
2. The separation matters. Our headline number mixes the pattern with a 1×ATR stop and an
   EOD exit. The `predict`/`score` path deliberately forecasts the **pattern outcome**
   (does this breakout fail?), not the P&L of one exit rule.

## 3. Zarattini & Aziz: the edge is in the universe, not the trigger

The widely-cited ORB result (1,484% vs 169% for QQQ, 2016–2023; and >1,600% net with a
Sharpe of 2.81 on the stock version) does **not** come from a fixed list of large caps.
Their universe is rebuilt every morning:

| their rule | value |
|---|---|
| relative volume | first 5 min volume today ÷ mean first 5 min volume of the previous **14** days |
| price filter | > $5 |
| volatility filter | 14-day ATR > $0.50 |
| candidate pool | 1,000 most liquid US equities |
| selection | **top 20 by relative volume, that day** |
| direction | first 5-minute bar's close vs open |
| stop | 2 × 14-day ATR |
| exit | market close |
| sizing | risk 1% of allocated capital |

**This is the single most important difference from what we tested.** We ran a fixed list
of 20 large caps and asked "does ORB work?". They ran the top 20 *stocks in play* out of
1,000 and asked the same question. Those are different experiments, and the literature's
own claim is that the edge lives in the selection step.

Our own data points the same way and could not confirm it: breakouts in the **highest
third** of opening-15-minute relative volume failed 6.0% of the time against 28.2% in the
lowest third — but the feature exists on only 347 of 840 rows and 13 failures in the first
half, so the pipeline correctly refused to call it a finding.

### The cost assumption that should worry you

The CXO Advisory summary of the QQQ paper records the cost model as commission of
$0.0005/share and:

> "no bid-ask spread, no impact of trading (slippage) and no other execution price
> uncertainty."

Our pipeline charges 20.6–50.6 bps of turnover per round trip, all in, and that is what
turned a roughly break-even rule into −₹161 per trade. A published ORB result computed
with zero slippage is not comparable to ours, and the difference is not a detail: on our
data **costs were 93% of the total loss**.

No comparable peer-reviewed NSE/India study surfaced. The Indian material found was
commercial screeners (StockeZee, BottomStreet, PKScreener) offering ORB scans with volume
filters — products, not evidence. Common thresholds quoted there (breakout-bar volume
≥ 1.5× the opening-range rate; first-15-minute RVOL ≥ 2×) are consistent in direction with
Zarattini & Aziz but carry no published validation.

---

## What follows for this project, in order of expected effect

1. **Select the universe daily by relative volume** instead of using a fixed list. This is
   the literature's actual claim and the one we have not tested. Needs a wider candidate
   pool (~200 liquid NSE names) and therefore more data than yfinance's 60 days.
2. **Use a 14-day RVOL lookback**, per the source. On 41 sessions this alone lifts the
   rows where RVOL exists from 347/840 to a usable fraction earlier in the sample.
3. **Stop at 2 × ATR**, per the source, rather than 1 × ATR.
4. **Keep costs as modelled.** They are the reason our answer differs from the paper's,
   and they are the honest part.
5. **Forecast and score, don't just backtest.** Predictions are logged before the outcome
   with a trailing-window rule, then scored against the base rate. See `forecast.md`.

Items 1–3 are pre-registered here **before** being run, so that if they are tried and fail
the record shows they were chosen from the literature and not from our own data.

---

# Pre-registered features, round 2

**Written before implementation, on 2026-09-23.** The point of writing these down first is
that a feature chosen after seeing which one worked is not evidence. Each entry states the
exact definition, the mechanism it is supposed to capture, and the direction expected of
the failure rate. If a feature comes out the other way, that is a result, not a reason to
re-label the expectation.

All are computed at the close of the breakout bar, from bars at positions `<= i` only, and
are NaN when undefined. Signing by direction means multiplying by −1 for short breakouts,
so a positive value always means "in the direction of the breakout".

| name | definition | mechanism | expected |
|---|---|---|---|
| `rel_strength_vs_index` | (stock close_i / stock session open − 1) − (index close at the same timestamp / index session open − 1), signed by breakout direction | a stock moving with its own demand rather than the market's tide has real buyers behind it | higher → fails less |
| `index_or_agrees` | 1 if the index (`config.index_symbol`) has closed beyond its own opening range in the same direction at or before the breakout timestamp; 0 if not; NaN if index bars are missing | a breakout fighting the index is a breakout fighting every correlated seller | 1 → fails less |
| `gap_atr_signed` | `gap_pct` converted to price terms ÷ prior-day ATR, signed by breakout direction (positive = gap in the breakout direction) | a gap in the breakout's direction means the move began before the session and has overnight commitment behind it | higher → fails less |
| `prior_day_return_atr_signed` | (prior-day close − prior-day open) ÷ prior-day ATR, signed by breakout direction | continuation versus exhaustion — both stories are told, and neither is obviously right | **two-sided: report only** |
| ~~`breakout_depth_atr`~~ **WITHDRAWN** | abs(breakout close − broken boundary) ÷ prior-day ATR | ~~a decisive close through the level is harder to reclaim than a marginal one~~ | **moved to context, not tested — see note below** |
| `is_expiry_day` | 1 if the session date is in `data/nse_expiries.csv`, else 0; NaN if the date is outside the file's covered range | expiry-day positioning and pinning distort intraday ranges | **two-sided: report only** |
| `day_of_week` | 0–4 | segmentation only | not tested as a mechanism |

### Withdrawn before interpretation: `breakout_depth_atr`

Pre-registered above as directional, then withdrawn in code review on 2026-09-23 **before
any result involving it was read**. The original entry is struck through rather than
deleted, because a pre-registration that can be quietly edited is not a pre-registration.

The reason it is unusable: `labelling.resolve` measures the excursion from the breakout
bar onward, and the breakout bar's own close is part of that. So a breakout whose close
already sits `bust_extension_atr` beyond the boundary can never be labelled BUSTED, and one
at `sustain_extension_atr` beyond it is SUSTAINED on that bar. The feature is partly a
restatement of the label rather than an independent signal about it. On the current data no
BUSTED row has a depth above 0.18 against a threshold of 0.25, exactly as that implies.

It is kept as a context column for segmentation, where being tied to the thresholds does no
harm.

**Two-sided features are shown in the report but can never on their own produce a
`signal` verdict.** There is no honest direction to predict for them, so "it came out
significant" would be a coin flip dressed up as a finding.

## Parameter changes pre-registered with them

- **`rvol_lookback_sessions` 20 → 14.** Taken from Zarattini & Aziz, whose relative-volume
  definition uses the previous 14 days. Already proposed as item 2 of the list above. It is
  a definition change and will be logged in the README changelog when implemented. It is
  chosen from the source, not from our results.
- **Forecast shrinkage `k = 40`** and **log-odds damping `0.6`**, fixed here in advance.
  These are not to be tuned against study or forecast output. `k = 40` means a bucket needs
  about 40 observations before it is trusted as much as the base rate; damping 0.6 keeps a
  three-feature agreement from compounding into a false certainty.

## Multiple testing

The directional features are whatever `features.FEATURE_NAMES` holds; after
`breakout_depth_atr` was withdrawn (see above) that is **six**:

    rvol_open_15m, rvol_breakout_bar, bar_body_ratio,
    rel_strength_vs_index, index_or_agrees, gap_atr_signed

Under the two-period rule each feature has its own chance of passing by luck. Treating them
as independent, the chance that **at least one** of `n` passes is `1 − 0.95ⁿ` for an
individual false-positive rate of 5%:

| n directional features | 1 − 0.95ⁿ |
|---|---|
| 1 | 5% |
| 3 | 14% |
| 6 | **26%** |
| 7 | 30% |

Six features therefore carry about a 26% chance that one survives the two-period rule on
noise alone. The features are correlated (the two RVOL measures especially), so the true
figure is lower than the independent bound, but it is not small.

The report states how many features were tested and quotes this figure, because a `signal`
verdict resting on one feature out of six is weaker evidence than the same verdict resting
on one out of one, and the reader cannot judge that without the count. `analysis._family_error`
computes it, so the number in the report and the number here cannot drift apart.

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

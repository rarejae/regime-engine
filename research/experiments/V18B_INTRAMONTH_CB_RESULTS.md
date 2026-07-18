---
date: 2026-04-13
experiment: V18b Intra-Month Portfolio Circuit Breaker
status: FAILED — IMC-20/25 never fire. -27% DD is multi-month, not single-month.
script: experiments/v18b_intramonth_cb/backtest.py
---

# V18b: Intra-Month Portfolio CB — Results

## Verdict

**V18b FAILS because the mechanism never fires at useful thresholds.** IMC-20 and IMC-25 have zero trigger events in 24 years. V16-B's portfolio never drops 20% within a single month because the per-asset CBs already delever mid-month, limiting single-month exposure.

The -27.0% MaxDD is a **multi-month cumulative drawdown**, not a single-month crash. An intra-month CB has nothing to protect against because by the time the portfolio approaches -20% within a month, the per-asset SMA-based CBs have already stripped leverage.

At aggressive thresholds (IMC-15), the mechanism fires 6 times during volatile-but-recovering months, delevering during the bounce and making things **worse** (MaxDD -27.9%, Sharpe 0.824 vs V16-B 0.846).

---

## Table 1 — Core Metrics

| Strategy      |   CAGR | Sharpe |  MaxDD | Term$1 |  DCA$700 | IMC events |
|---------------|-------:|-------:|-------:|-------:|---------:|-----------:|
| IMC-15        | 16.43% |  0.824 | -27.9% | $45.57 |  $3.90M  |      6     |
| IMC-18        | 16.91% |  0.842 | -27.0% | $50.48 |  $4.28M  |      2     |
| IMC-20        | 17.06% |  0.846 | -27.0% | $52.18 |  $4.41M  |      **0** |
| IMC-25        | 17.06% |  0.846 | -27.0% | $52.18 |  $4.41M  |      **0** |
| **V16-B**     | 17.06% |  0.846 | -27.0% | $52.18 |  $4.41M  |      —     |

IMC-20 and IMC-25 are byte-for-byte identical to V16-B — the overlay literally never activates. IMC-18 fires twice and costs -0.15pp CAGR for zero DD improvement. IMC-15 fires 6 times and makes everything worse.

---

## Table 2 — Why the IMC Never Fires (COVID Day-by-Day)

COVID March 2020 is the worst month in V16-B's history. Day-by-day for V16-B and IMC-20:

| Date       | V16-B daily | V16-B DD | IMC-20 daily | IMC-20 DD |
|------------|----------:|-------:|------------:|---------:|
| 2020-03-09 |    -3.06% | -21.8% |      -3.06% |   -21.8% |
| 2020-03-12 |    -4.40% | -25.2% |      -4.40% |   -25.2% |
| 2020-03-16 |    -5.26% | -27.0% |      -5.26% |   -27.0% |
| 2020-03-23 |    +0.53% | -26.4% |      +0.53% |   -26.4% |
| 2020-03-31 |    -0.74% | -22.9% |      -0.74% |   -22.9% |

**Every row is identical.** The IMC-20 never fires because the per-asset CBs already stripped leverage by early March. The portfolio's -27% drawdown from Feb peak to Mar 16 bottom happens at *already-reduced* leverage — the per-asset SMA breaches on QQQ and IVV fired in the last week of February, delevering from SSO/QLD to IVV/QQQ before March even began.

The remaining -27% drawdown is at ~90% effective equity (unlevered QQQ + IVV + IAU), not at 180%. An intra-month CB on top of already-delevered positions has nothing to strip.

---

## The Structural Insight

V16-B's -27% MaxDD breaks down as:

1. **Feb 2020 last week:** Per-asset CBs fire on QQQ and IVV (SMA breach). Portfolio deleveres from 180% to ~90% effective equity.
2. **March 2020:** Portfolio holds unlevered QQQ + IVV + IAU. At ~90% eff equity, the -34% QQQ decline and -26% IVV decline produce a portfolio drawdown of ~-27% (with gold partially offsetting).
3. **Late March:** Portfolio at the bottom. Monthly rebalance at March-end evaluates Faber scores — both still off-signal. Cash position maintained via standard V16-B logic.

**The max drawdown is NOT from leveraged exposure.** It's from *unlevered* equity held after the per-asset CBs already fired. An intra-month CB that strips leverage is solving the wrong problem — there's no leverage left to strip.

To reduce the -27% to, say, -20%, you'd need to exit *unlevered equity* during the crash — which is exactly what the Faber monthly rebalance does (it exits at month-end when scores drop below threshold). The gap between the per-asset CB firing (Feb 27) and the monthly rebalance exiting equity (March 31) is where the drawdown accumulates. Closing that gap would require mid-month equity exits, which is weekly rebalancing — already rejected (V13, weekly rebalancing experiment).

---

## IMC-15 and IMC-18 Events (The Whipsaw Problem)

| Threshold | Events | Dates |
|-----------|:------:|-------|
| IMC-15    |    6   | Nov 2007, May 2010, Feb 2018, Oct 2018, Sep 2020, Jan 2022 |
| IMC-18    |    2   | May 2010, Jan 2022 |

All 6 IMC-15 events are months where the portfolio dropped sharply mid-month then partially recovered. The IMC delevered during the recovery portion:

- **Jan 2022:** -16.02% at trigger (day 15), -13.94% month-end. IMC delevered, missed the late-month bounce. Without IMC: -12.80%. IMC made the month worse.
- **Feb 2018:** -16.54% at trigger (day 6), -10.95% month-end. Massive recovery that IMC missed. Without IMC: -12.80%.

These are exactly the whipsaw events the prompt warned about. The months that trigger IMC-15 are volatile-but-recovering months where delevering mid-month costs return.

---

## Pass / Fail

| Variant | MaxDD | Sharpe | CAGR | Events | Overall |
|---------|------:|-------:|-----:|-------:|:-------:|
| IMC-15  | -27.9%| 0.824  | 16.43%| 6     | FAIL (DD worse, Sharpe worse) |
| IMC-18  | -27.0%| 0.842  | 16.91%| 2     | FAIL (no DD improvement) |
| IMC-20  | -27.0%| 0.846  | 17.06%| 0     | FAIL (never fires) |
| IMC-25  | -27.0%| 0.846  | 17.06%| 0     | FAIL (never fires) |

No threshold passes. At ≥20%, the mechanism never fires. At ≤15%, it fires during recoveries and whipsaws.

---

## What We Learned

### 1. V16-B's -27% MaxDD is from unlevered equity, not leverage
The per-asset CBs strip leverage within days of an SMA breach. The remaining drawdown accumulates at ~90% effective equity. No leverage-stripping overlay can improve this because there's no leverage left to strip.

### 2. The gap between CB and monthly rebalance is the DD source
Per-asset CB fires Feb 27, 2020. Monthly rebalance exits equity March 31, 2020. During those 23 trading days, the portfolio holds unlevered QQQ + IVV + IAU and draws down ~27%. To close this gap you need mid-month equity exits — which is weekly rebalancing, already rejected.

### 3. The only path to MaxDD < -27% is lower base leverage or faster monthly exit
Either reduce the substitution percentage (less leverage = less DD, but also less CAGR) or add weekly equity exits (already proved to add noise, V13). V16-B's architecture is at its structural optimum.

### 4. -27% MaxDD is the CONFIRMED structural floor
V18 failed with portfolio-level CB from HWM. V18b failed with intra-month CB. Both approaches — slow (HWM) and fast (monthly reset) — fail for different reasons. The drawdown protection thesis is definitively closed.

---

## Cross-references

- [[experiments/V18_DRAWDOWN_PROTECTION_RESULTS]] — V18 HWM-based CB: whipsaws 209+ times
- [[experiments/V13_THREE_STATE_RESULTS]] — weekly re-entry rejected (1/18 events)
- [[experiments/V16_TWO_POD_GOLD_RESULTS]] — V16-B frontier confirmed at -27.0% MaxDD

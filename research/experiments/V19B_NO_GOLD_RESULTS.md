---
date: 2026-04-13
experiment: V19b V19 Without Gold — 50/50 CB→Cash
status: Gold EARNS its 10%. V19 dominates V19b on Sharpe and MaxDD.
script: experiments/v19b_no_gold/backtest.py
---

# V19b: V19 Without Gold — Results

## Verdict

**Gold earns its 10%.** V19b has higher CAGR (+0.59pp) and terminal (+$7.35), but V19 has higher Sharpe (+0.033) and shallower MaxDD (+1.9pp). Gold's low equity correlation reduces portfolio vol by 1.96pp (20.93% vs 22.89%), which drives the Sharpe improvement. The risk-adjusted return of the gold allocation exceeds the equity return it displaces.

V19 remains the Pareto frontier balanced point. V19b is a valid alternative for CAGR-prioritizing investors but does not displace V19.

**Bonus finding:** CB→cash improvement is independent of gold. V19b vs V15 (same 50/50, but CB→cash vs CB→equity): +0.020 Sharpe, +2.0pp MaxDD, +0.23pp CAGR. The CB→cash mechanism works on its own.

---

## Core Metrics

| Strategy                 |   CAGR |    Vol | Sharpe | Sortino |  MaxDD | Calmar | Term$1 |  DCA$700 |
|--------------------------|-------:|-------:|-------:|--------:|-------:|-------:|-------:|---------:|
| V19b 50/50 CB→Cash       | **17.88%** | 22.89% | 0.834 | 0.957 | -27.0% | 0.71 | **$62.10** | **$5.22M** |
| **V19 45/45/10 CB→Cash** | 17.29% | **20.93%** | **0.867** | **1.013** | **-25.1%** | **0.72** | $54.75 | $4.64M |
| V15 50/50 CB→Equity      | 17.65% | 23.36% |  0.813 |   0.941 | -29.0% |   0.66 | $59.18 |  $4.92M |

V19b wins on CAGR/terminal/DCA. V19 wins on Sharpe/Sortino/MaxDD/Calmar/Vol. They trade off against each other — neither dominates.

---

## Crisis Drawdowns

| Crisis          |  V19b |    V19 |    V15 |     V9 | Baseline |
|-----------------|------:|-------:|-------:|-------:|---------:|
| GFC 2007-09     | -19.4%| **-16.5%** | -18.1%| -30.6% | -9.0% |
| COVID 2020      | -27.0%| **-25.1%** | -29.0%| -37.9% | -18.5% |
| 2022 bear       | -19.2%| **-17.7%** | -21.5%| -23.9% | -13.2% |

Gold improves every crisis: GFC -2.9pp, COVID -1.9pp, 2022 -1.5pp. Gold's crisis alpha (GFC +14.8% in 13/17 months) continues to pay even in V19's CB→cash architecture, because the gold sleeve remains active during the pre-CB window when leveraged equity is still being held.

---

## What This Means

Gold at 10% costs 0.59pp CAGR (from 10% less equity exposure during bull markets) but buys:
- +0.033 Sharpe (vol reduction from low equity correlation)
- +1.9pp MaxDD (crisis alpha during pre-CB leveraged window)
- +0.056 Sortino (downside protection)

For a 40-year accumulator thinking in percentages, the Sharpe and DD improvements justify the CAGR cost. The $580K DCA gap ($5.22M vs $4.64M) is meaningful but the risk-adjusted tradeoff favors gold.

**V19 remains the balanced frontier point.** V19b is available as a "CAGR-tilted" option for investors who prefer $580K more terminal wealth at the cost of 0.033 Sharpe and 1.9pp deeper MaxDD.

---

## Cross-references

- [[experiments/V19_CB_CASH_EXIT_RESULTS]] — V19 confirmed as frontier leader
- [[experiments/V16_TWO_POD_GOLD_RESULTS]] — original gold validation (GFC crisis alpha)

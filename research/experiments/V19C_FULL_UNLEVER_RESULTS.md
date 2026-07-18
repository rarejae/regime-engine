---
date: 2026-04-13
experiment: V19c V19 with 100% unlevered at score 2/3
status: WASH — keep V19's 70/30 (marginally better Sharpe and crisis DDs)
script: experiments/v19c_full_unlever/backtest.py
---

# V19c: 100% Unlevered at Score 2/3 — Results

## Verdict

**Wash.** V19c and V19 are functionally equivalent. Differences are noise-level: -0.002 Sharpe, +0.12pp CAGR, identical MaxDD (-25.1%). Keep V19's 70/30 — it has marginally better Sharpe and slightly better crisis DDs (dot-com -2.3% vs -3.8%, GFC -16.5% vs -17.4%, 2022 -17.7% vs -18.9%).

Score 2/3 occurs in only ~14% of months. The 30% cash buffer during those months provides a small vol cushion that keeps V19's Sharpe fractionally higher.

---

## Core Metrics

| Strategy        |   CAGR | Sharpe | Sortino |  MaxDD | Term$1 |  DCA$700 |
|-----------------|-------:|-------:|--------:|-------:|-------:|---------:|
| V19c (100%)     | 17.41% |  0.865 |   1.019 | -25.1% | $56.19 |  $4.77M  |
| **V19 (70/30)** | 17.29% |  **0.867** | 1.013 | -25.1% | $54.75 |  $4.64M  |

Score 2/3 months: QQQ 16, IVV 26 out of 291 total.

## Frontier Unchanged

V19 at 70/30 remains the balanced point. V19c is not an improvement.

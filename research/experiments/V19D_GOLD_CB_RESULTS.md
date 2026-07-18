---
date: 2026-04-13
experiment: V19d V19 with Circuit Breaker on Gold Sleeve
status: WASH — adopt for design consistency (all risk assets get CB)
script: experiments/v19d_gold_cb/backtest.py
---

# V19d: Gold Circuit Breaker — Results

## Verdict

**Wash.** V19d and V19 are functionally identical: -0.001 Sharpe, -0.02pp CAGR, same MaxDD. The gold CB fires 10 times in 24 years (~0.4/year) with negligible cumulative impact.

**Adopt V19d for design consistency.** All risk assets (QLD, SSO, IAU) now have the same 3/3 SMA breach → cash emergency exit. Consistent design, zero performance cost.

---

## Core Metrics

| Strategy        |   CAGR | Sharpe |  MaxDD | Term$1 | Gold CB events |
|-----------------|-------:|-------:|-------:|-------:|:--------------:|
| V19d (gold CB)  | 17.27% |  0.866 | -25.1% | $54.60 |       10       |
| V19 (no gold CB)| 17.29% |  0.867 | -25.1% | $54.75 |        0       |

Crisis DDs marginally improved: GFC -16.4% vs -16.5%, 2022 -17.6% vs -17.7%. The gold CB catches a few sharp gold reversals but the 10% allocation means the impact is ~0.01% per event.

## Gold CB Events (10 over 24 years)
2007-01, 2009-01, 2011-12, 2012-12, 2014-07, 2015-02, 2018-05, 2021-06, 2022-01, 2022-05

Jan 2022 is notable — gold CB fires on Jan 6, before the equity CBs fire on Jan 21. This provides a small early-warning benefit (the gold sleeve goes to cash 2 weeks before equities), but the 10% allocation limits the impact.

## Decision

**V19d adopted as the production specification.** The gold CB adds design consistency at zero performance cost. The final V19d spec is:

```
45% Pod 1 (QQQ/QLD) — V9 logic, IVV guard, CB → cash
45% Pod 2 (IVV/SSO) — no guard, CB → cash
10% Gold (IAU ≥ 3 or cash) — CB → cash
Monthly rebalance to 45/45/10 (5% drift)
Score 2/3 → 70% underlying + 30% cash
Re-entry: monthly only (no mid-month)
```

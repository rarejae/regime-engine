# Daily vs Weekly Circuit Breaker on Faber-Sweep-40

**Date:** April 6, 2026  
**Status:** Complete  
**Track:** Production Architecture  
**Related:** [[TAA_PROJECT_STATUS]] | [[2026-04-06_faber_sweep_weekly_daily]]

## Purpose

Test whether a daily leverage circuit breaker outperforms the weekly (Friday-only) version. The only change is check frequency — all SMA calculation, Faber logic, and leverage substitution are identical.

## Results

| Strategy | Return | Vol | Sharpe | Sortino | MaxDD | Calmar | Terminal($1) | vs Weekly |
|----------|--------|-----|--------|---------|-------|--------|-------------|-----------|
| S40-Daily-Weekly | 11.1% | 11.7% | 0.946 | 1.155 | -16.2% | 0.69 | $12.46 | baseline |
| **S40-Daily-Daily** | **11.2%** | **11.7%** | **0.958** | **1.176** | **-15.0%** | **0.74** | **$12.74** | **+$0.28** |
| S40-Daily-Daily-Reentry | 11.4% | 11.7% | 0.971 | 1.194 | -15.0% | 0.76 | $13.28 | +$0.82 |
| IVV B&H | 10.8% | 19.0% | 0.570 | 0.713 | -55.2% | 0.20 | $8.92 | — |

**Daily circuit breaker improves BOTH Sharpe AND terminal wealth** vs weekly — the same pattern as switching from monthly to daily SMAs. Each step toward faster signal processing improves the system without introducing tradeoffs.

### Crisis Analysis

| Strategy | GFC (2008-09) | COVID (Feb-Mar 2020) | 2022 Bear |
|----------|--------------|---------------------|-----------|
| S40-Daily-Weekly | +0.6% (DD -1.1%) | -15.0% (DD -16.2%) | -9.7% (DD -10.7%) |
| S40-Daily-Daily | +0.6% (DD -1.1%) | **-13.8% (DD -15.0%)** | -9.7% (DD -10.7%) |
| S40-Daily-Daily-Reentry | +0.6% (DD -1.1%) | -13.8% (DD -15.0%) | -9.7% (DD -10.7%) |

GFC and 2022: identical — the breaker triggered on the same day regardless of check frequency.
COVID: daily breaker triggers **one day earlier** (Feb 27 vs Feb 28), saving 1.2% max DD (15.0% vs 16.2%) and 1.2% total return (-13.8% vs -15.0%).

## Circuit Breaker Diagnostics

| Metric | Weekly | Daily | Daily+Reentry |
|--------|--------|-------|---------------|
| Total events | 14 | 16 | 18 |
| Events/year | 0.6 | 0.7 | 0.8 |
| Additional vs weekly | — | +2 | +4 |
| Daily re-entries | — | — | 6 |

The daily breaker fires only 2 more times than weekly over 24 years — a negligible increase in trigger frequency. The two additional triggers are 2005-03-22 (QQQ) and 2014-10-16 (IVV), both correct responses to trend deterioration.

### COVID Timing

```
Weekly:  first trigger 2020-02-28 (Friday)  — return since Feb 19: -11.7%
Daily:   first trigger 2020-02-27 (Thursday) — return since Feb 19: -10.1%
Savings: 1 day earlier → 1.6% less damage at trigger point
```

The daily breaker caught COVID one trading day earlier because the Thursday close breached all 3 SMAs, but the weekly version had to wait until Friday.

### 2022 Timing

Both triggered on the same day: 2022-01-21 (a Friday). No improvement — the breach happened on a Friday by coincidence.

## Whipsaw Analysis

**S40-Daily-Daily (no re-entry):** 4 of 16 events saw price recover before the next monthly rebalance. These represent potential whipsaw cost — leverage exited, missed the recovery, then re-entered at month-end. But the net effect is still positive (+$0.28 terminal vs weekly).

**S40-Daily-Daily-Reentry:** 6 daily re-entries after 18 exits. 5 of these were rapid exit-reentry cycles (within 5 trading days). These represent whipsaw — leverage exits, price bounces, leverage re-enters. Despite this, the net effect is positive:

```
Impact of daily re-entry (vs daily no-reentry):
  Sharpe: 0.958 → 0.971 (+0.012)
  Return: 11.2% → 11.4%
  Terminal: $12.74 → $13.28 (+$0.54)
  MaxDD: unchanged (-15.0%)
```

The daily re-entry's +$0.54 terminal gain comes from recovering leverage faster after brief dips. The 5 whipsaw cycles cost less than the benefit of faster re-entry in the other cases.

**However:** The daily re-entry breaks the asymmetric design principle (slow entry, fast exit). With only 6 re-entries over 24 years, the 5 whipsaw cycles represent 83% of re-entries being rapid cycling. This creates operational noise — the system signals exit Monday, then re-entry Wednesday, which is impractical for a human-in-the-loop system.

## Decomposition (cumulative from monthly-SMA baseline)

| Step | Sharpe | Terminal | MaxDD |
|------|--------|----------|-------|
| Monthly SMAs, monthly only | 0.878 | $10.44 | -17.1% |
| + Daily SMAs | +0.049 | +$1.94 | flat |
| + Weekly circuit breaker | +0.019 | +$0.08 | +0.9% |
| + **Daily circuit breaker** | **+0.012** | **+$0.28** | **+1.2%** |
| **Cumulative** | **0.958** | **$12.74** | **-15.0%** |
| + Daily re-entry (optional) | +0.012 | +$0.54 | flat |
| **With re-entry** | **0.971** | **$13.28** | **-15.0%** |

## Decision

**S40-Daily-Daily (daily circuit breaker, monthly re-entry) is the new production architecture.** It improves on the weekly version:
- Sharpe: 0.958 vs 0.946 (+0.012)
- MaxDD: -15.0% vs -16.2% (+1.2%)
- Terminal: $12.74 vs $12.46 (+$0.28)
- Event frequency: 0.7/year (vs 0.6/year weekly) — still manageable

**Daily re-entry (S40-Daily-Daily-Reentry) is NOT recommended for production** despite its +$0.54 terminal advantage. The 5 whipsaw cycles out of 6 re-entries (83%) create impractical operational churn for a human-in-the-loop system. The benefit is real but the operational cost of exit-Monday/re-enter-Wednesday signaling is too high.

## Production Architecture (Updated)

```
Faber multi-timeframe trend filter (126/200/252-day DAILY SMA)
  → Monthly allocation: 3/3 = full weight, 2/3 = 70%, 0-1/3 = cash
  → Baseline weights: IVV 45%, QQQ 25%, VGLT 5%, IAU 10%, DBC 5%, Cash 10%
  → Freed capital: to cash
  → Leverage: if BOTH IVV+QQQ at 3/3 → replace 40% of each with SSO/QLD
  → DAILY circuit breaker: if either IVV or QQQ below ALL 3 daily SMAs at close → exit leverage next open
  → Re-entry: next monthly rebalance only (no daily re-entry)
  → Human-in-the-loop: Telegram for monthly rebalance + daily circuit breaker alerts (~0.7/year)
```

**Performance: 11.2% return, 0.958 Sharpe, -15.0% max DD, $12.74 terminal.**
# Daily vs Weekly Circuit Breaker on Faber-Sweep-40

**Date:** April 6, 2026
**Status:** Complete — Production Architecture Updated
**Track:** Production Architecture
**Related:** [[TAA_PROJECT_STATUS]] | [[2026-04-06_faber_sweep_weekly_daily]]

## Result

Daily circuit breaker improves both Sharpe AND terminal wealth vs weekly — consistent pattern across all signal granularity upgrades.

| Strategy | Return | Vol | Sharpe | MaxDD | Terminal | vs Weekly |
|----------|--------|-----|--------|-------|---------|-----------|
| S40-Daily-Weekly | 11.1% | 11.7% | 0.946 | -16.2% | $12.46 | baseline |
| **S40-Daily-Daily** | **11.2%** | **11.7%** | **0.958** | **-15.0%** | **$12.74** | **+$0.28** |
| S40-Daily-Daily-Reentry | 11.4% | 11.7% | 0.971 | -15.0% | $13.28 | +$0.82 |

## Cumulative Improvement From Monthly Baseline

| Step | Sharpe | Terminal | MaxDD |
|------|--------|----------|-------|
| Monthly SMAs, monthly only | 0.878 | $10.44 | -17.1% |
| + Daily SMAs | +0.049 | +$1.94 | flat |
| + Weekly circuit breaker | +0.019 | +$0.08 | +0.9% |
| + Daily circuit breaker | +0.012 | +$0.28 | +1.2% |
| **Total: S40-Daily-Daily** | **0.958** | **$12.74** | **-15.0%** |

## Key Findings

COVID: daily breaker triggers Feb 27 (Thursday) vs Feb 28 (Friday) — 1 day earlier, saves 1.2% max DD.
2022: both triggered same day (Friday coincidence) — no difference.
Total events: 16/year (0.7/year) vs 14 (0.6/year weekly) — 2 additional triggers over 24 years. Negligible.

Daily re-entry REJECTED for production: 5 of 6 re-entries were rapid whipsaw cycles (exit Monday, re-enter Wednesday). +$0.54 benefit doesn't justify operational churn for human-in-the-loop system.

## Final Production Architecture

```
Faber multi-timeframe trend filter (126/200/252-day DAILY SMA)
  → Monthly allocation: 3/3 = full weight, 2/3 = 70%, 0-1/3 = cash
  → Baseline weights: IVV 45%, QQQ 25%, VGLT 5%, IAU 10%, DBC 5%, Cash 10%
  → Freed capital: to cash
  → Leverage: if BOTH IVV+QQQ at 3/3 → replace 40% of each with SSO/QLD
  → DAILY circuit breaker: if either IVV or QQQ below ALL 3 daily SMAs at close → exit leverage next open
  → Re-entry: next monthly rebalance only
  → Telegram: monthly rebalance + daily circuit breaker alerts (~0.7/year)

Performance: 11.2% return | 0.958 Sharpe | -15.0% max DD | $12.74 terminal from $1
```

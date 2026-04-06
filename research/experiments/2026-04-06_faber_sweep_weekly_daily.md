# Faber-Sweep-40: Daily SMAs + Weekly Circuit Breaker

**Date:** April 6, 2026  
**Status:** Complete  
**Track:** Production Architecture  
**Related:** [[TAA_PROJECT_STATUS]] | [[2026-04-05_faber_sweep]]

## Purpose

Test whether switching from monthly SMAs to daily SMAs (126/200/252-day) and adding a weekly leverage circuit breaker improves the Faber-Sweep-40 system. The monthly-only baseline: 10.4% return, 0.878 Sharpe, -17.1% max DD, $10.44 terminal.

## Design

**Daily SMAs:** 126-day (~6mo), 200-day (~10mo), 252-day (~12mo). Computed from daily closing prices. Trend score = number of SMAs the closing price is above (0-3).

**Monthly allocation:** At month-end, use daily SMA scores to determine Faber weights. Same logic as before: 3/3 = full weight, 2/3 = 70%, 0-1/3 = cash. If both IVV+QQQ at 3/3, apply 40% SSO/QLD substitution.

**Weekly circuit breaker:** Every Friday, if either IVV or QQQ closes below ALL 3 daily SMAs → exit leverage Monday. Underlying Faber weights unchanged. Re-entry only at next monthly rebalance. Requires complete trend breakdown (3/3 breach) to prevent false triggers.

## Results

### Full Performance Table

| Strategy | Return | Vol | Sharpe | Sortino | MaxDD | Calmar | Terminal($1) | vs Monthly |
|----------|--------|-----|--------|---------|-------|--------|-------------|-----------|
| **Faber-Sweep-40-Monthly** | **10.4%** | **11.8%** | **0.878** | **1.058** | **-17.1%** | **0.61** | **$10.44** | **baseline** |
| Faber-Sweep-40-Daily-Monthly | 11.1% | 12.0% | 0.927 | 1.128 | -17.2% | 0.65 | $12.38 | +$1.94 |
| **Faber-Sweep-40-Daily-Weekly** | **11.1%** | **11.7%** | **0.946** | **1.155** | **-16.2%** | **0.69** | **$12.46** | **+$2.02** |
| Faber-1x-Daily-Weekly | 8.9% | 9.1% | 0.975 | 1.202 | -13.0% | 0.68 | $7.72 | -$2.71 |
| IVV B&H | 10.8% | 19.0% | 0.570 | 0.713 | -55.2% | 0.20 | $8.92 | — |
| 60/40 | 8.4% | 11.1% | 0.752 | 0.994 | -29.9% | 0.28 | $6.51 | — |

**Faber-Sweep-40-Daily-Weekly improves BOTH Sharpe AND terminal wealth vs the monthly baseline.** This is the first time in the entire project that a system modification has improved both metrics simultaneously.

### Crisis Analysis

| Strategy | GFC (2008-09) | COVID (Feb-Mar 2020) | 2022 Bear |
|----------|--------------|---------------------|-----------|
| S40-Monthly | +0.6% (DD -1.1%) | -15.0% (DD -16.2%) | -8.5% (DD -11.2%) |
| S40-Daily-Monthly | +0.6% (DD -1.1%) | -15.0% (DD -16.2%) | -9.6% (DD -11.2%) |
| S40-Daily-Weekly | +0.6% (DD -1.1%) | -15.0% (DD -16.2%) | -9.7% (DD -10.7%) |
| IVV B&H | -36.9% (DD -46.0%) | -33.4% (DD -33.7%) | -17.7% (DD -24.5%) |

GFC: all variants identical — Faber filter had already exited before the crash.
COVID: all variants identical — the crash happened too fast for even weekly monitoring (Feb 28 trigger, but most damage done by then).
2022: Weekly breaker reduced max DD by 0.5% (10.7% vs 11.2%) via Jan 21, 2022 QQQ trigger.

### Circuit Breaker Diagnostics

**14 events over 24 years (0.6/year)** — identical count to the prior monthly-SMA version.

Key triggers:
- GFC: 2007-11-09 (IVV below all 3) — early warning
- COVID: 2020-02-28 (IVV below all 3) — caught but most damage done by then
- 2022: 2022-01-21 (QQQ below all 3) — caught early
- 2018 Q4: 2018-10-26 (IVV below all 3) — correct trigger

False triggers (outside major crisis periods): 7 of 14.
Average return in month following false triggers: mixed (+4.6%, -0.3%, -2.3%, +4.3%, +0.3%).

### Signal Agreement: Daily vs Monthly SMAs

```
Months where daily and monthly SMAs agree on all assets:   201/259 (78%)
Months where they disagree on at least one asset:          58/259 (22%)
Months where they disagree on leverage (IVV+QQQ 3/3):     7/259 (3%)
```

Daily and monthly SMAs agree on leverage decisions 97% of the time. The 3% disagreement (7 months) is where the return difference comes from — daily SMAs catch trend changes slightly earlier or later than monthly, and this timing difference accumulates to +0.72% annualized return.

### Decomposition

| Effect | Sharpe | Return | MaxDD | Terminal |
|--------|--------|--------|-------|----------|
| Daily SMAs (vs monthly) | **+0.049** | **+0.72%** | -0.0% | **+$1.94** |
| Weekly circuit breaker (vs daily-monthly) | +0.019 | -0.00% | **+1.0%** | +$0.08 |
| **Combined (vs monthly baseline)** | **+0.067** | **+0.72%** | **+0.9%** | **+$2.02** |

**Daily SMAs are the dominant improvement** (+0.049 Sharpe, +$1.94 terminal). The weekly circuit breaker adds a modest Sharpe boost (+0.019) and 1% max DD improvement at essentially zero return cost.

## Interpretation

### Why daily SMAs improve performance

Daily SMAs (126/200/252-day) respond to trend changes faster than monthly SMAs (6/10/12-month). The monthly SMA can only update at month-end boundaries, creating up to ~21 trading days of lag when a trend reversal happens mid-month. Daily SMAs capture the same economic signal (price vs moving average) with finer granularity.

The 22% signal disagreement rate (58/259 months) shows that daily and monthly SMAs frequently disagree on specific assets — but they only disagree on the leverage decision (IVV+QQQ 3/3) 3% of the time. The return improvement comes from both the leverage disagreement months AND from different allocation to non-equity assets (VGLT, IAU, DBC) during the other disagreement months.

### Why the circuit breaker adds only modest value

The circuit breaker fires 14 times over 24 years — the same count as the prior monthly-SMA version. Half are false triggers (7/14). COVID timing shows the limitation: the breaker fires Feb 28, but the crash started Feb 19 — the 9-day delay means most COVID damage is already done. The breaker is most useful for slow-developing crises (2022, 2018 Q4, 2007 GFC buildup) where the weekly check catches the deterioration 1-3 weeks earlier than waiting for month-end.

### This is the first experiment to improve BOTH Sharpe AND terminal

Every prior modification to the Faber system involved a tradeoff:
- VRP sleeve: better Sharpe, worse terminal
- Turbulence layer: much better Sharpe, much worse terminal
- Pro-rata redistribution: better return, worse Sharpe

Daily SMAs + weekly circuit breaker improves Sharpe by +0.067 AND terminal wealth by +$2.02. This is possible because the improvement comes from better signal timing (catching trend changes faster), not from trading off return for lower volatility.

## Decision

**Faber-Sweep-40-Daily-Weekly is the new production architecture.** It improves on Faber-Sweep-40-Monthly in every dimension:
- Return: 11.1% vs 10.4% (+0.72%)
- Sharpe: 0.946 vs 0.878 (+0.067)
- MaxDD: -16.2% vs -17.1% (+0.9%)
- Terminal: $12.46 vs $10.44 (+$2.02)
- Calmar: 0.69 vs 0.61

The improvement is genuine — it comes from better signal granularity, not from parameter optimization or survivorship.

## Production Architecture

```
Faber multi-timeframe trend filter (126/200/252-day SMA)
  → Monthly rebalance: 3/3 full weight, 2/3 partial (70%), 0-1/3 exit to cash
  → Freed capital to cash
  → If BOTH IVV and QQQ at 3/3: replace 40% of each with SSO/QLD (~98% eff equity)
  → Weekly circuit breaker (Friday): if either IVV or QQQ below ALL 3 daily SMAs → exit leverage Monday
  → Re-entry: next monthly rebalance only
  → Human-in-the-loop: Telegram approval for monthly rebalance + circuit breaker alerts
```

## Next Steps

- Implement in `taa/run.py`
- Configure Telegram alerts: monthly rebalance signals + weekly circuit breaker triggers
- The circuit breaker fires ~0.6 times/year — manageable human oversight frequency
# Faber-Sweep-40: Daily SMAs + Weekly Circuit Breaker

**Date:** April 6, 2026
**Status:** Complete — New Production Architecture
**Track:** Production Architecture
**Related:** [[TAA_PROJECT_STATUS]] | [[2026-04-05_faber_sweep]]

## Result Summary

**First experiment in the entire project to improve BOTH Sharpe AND terminal wealth simultaneously.**

| Strategy | Return | Vol | Sharpe | MaxDD | Terminal | vs Monthly |
|----------|--------|-----|--------|-------|---------|-----------|
| S40-Monthly (prior) | 10.4% | 11.8% | 0.878 | -17.1% | $10.44 | baseline |
| S40-Daily-Monthly | 11.1% | 12.0% | 0.927 | -17.2% | $12.38 | +$1.94 |
| **S40-Daily-Weekly** | **11.1%** | **11.7%** | **0.946** | **-16.2%** | **$12.46** | **+$2.02** |
| Faber-1x-Daily-Weekly | 8.9% | 9.1% | 0.975 | -13.0% | $7.72 | — |

## Decomposition

| Effect | Sharpe | Return | MaxDD | Terminal |
|--------|--------|--------|-------|----------|
| Daily SMAs (dominant) | +0.049 | +0.72% | flat | +$1.94 |
| Weekly circuit breaker | +0.019 | flat | +1.0% | +$0.08 |
| **Combined** | **+0.067** | **+0.72%** | **+0.9%** | **+$2.02** |

Daily SMAs are the dominant improvement. The 126/200/252-day daily SMAs capture trend changes ~0-21 days faster than month-end-only SMAs. Agreement rate: 78% of months all assets agree, 97% agreement on leverage decision specifically. The 3% leverage disagreement (7 months) drives most of the return improvement.

## Circuit Breaker

14 events over 24 years (0.6/year) — same count as prior monthly-SMA version.
7/14 were false triggers. GFC/COVID/2022/2018 Q4 all correctly triggered.
COVID timing limitation: breaker fires Feb 28 but crash started Feb 19 — 9-day delay means most damage already done. Most useful for slow-developing crises (2022, 2007 GFC buildup).

## Production Architecture

```
Faber multi-timeframe trend filter (126/200/252-day DAILY SMA)
  → Monthly allocation: 3/3 = full weight, 2/3 = 70%, 0-1/3 = cash
  → Baseline weights: IVV 45%, QQQ 25%, VGLT 5%, IAU 10%, DBC 5%, Cash 10%
  → Freed capital: to cash
  → Leverage: if BOTH IVV+QQQ at 3/3 → replace 40% of each with SSO/QLD
  → Weekly circuit breaker (Friday close): if either IVV or QQQ below ALL 3 daily SMAs → exit leverage Monday
  → Re-entry: next monthly rebalance only
  → Human-in-the-loop: Telegram for monthly rebalance + circuit breaker alerts
```

# Leverage Tiers: 1.5x Intermediate for Mixed Faber Signals

**Date:** April 6, 2026  
**Status:** Complete  
**Track:** Production Architecture — Leverage Optimization  
**Related:** [[TAA_PROJECT_STATUS]] | [[2026-04-06_leverage_sweep_high]] | [[2026-04-06_faber_daily_circuit_breaker]]

## Purpose

Test whether adding a 1.5x leverage tier during mixed-signal months (one equity ETF at 3/3, the other at 2/3) improves on the current binary 2x system (PROD: 2x when both at 3/3, 1x otherwise).

## State Frequency — Why Tiers Can't Help

The experiment immediately explains why tiers add no value:

| State | Description | Months | % |
|-------|-------------|--------|---|
| A | Both 3/3 (full conviction) | 170 | **66%** |
| B | IVV 3/3, QQQ 2/3 | 5 | **2%** |
| C | QQQ 3/3, IVV 2/3 | 10 | **4%** |
| D | Both 2/3 | 4 | **2%** |
| E | Either <2/3 (no leverage) | 70 | 27% |

**States B+C combined = only 15 months out of 259 (6%).** The mixed-signal state almost never occurs. When the daily SMAs show one equity at 3/3 and the other at 2/3, it's a transient state — within a month, either the lagging ETF catches up to 3/3 (enters State A) or the leading one drops to 2/3 or below (enters State D or E).

State D (both at 2/3) occurs only 4 months out of 259 (2%). Combined, all intermediate states (B+C+D) = 19 months (7%).

**Forward returns by state:**
- State A: +1.16%/month (positive 69%)
- State B: +2.38%/month (positive 100%, but n=4 only)
- State C: -0.15%/month (positive 60%)
- State D: +2.14%/month (positive 75%, but n=4 only)
- State E: +1.06%/month (positive 68%)

State B and D look attractive but with only 4-5 observations each, these are noise.

## Results

| Strategy | Return | Vol | Sharpe | MaxDD | Calmar | Terminal($1) | vs PROD | Lev Mo% |
|----------|--------|-----|--------|-------|--------|-------------|---------|---------|
| **PROD** | **14.5%** | **15.8%** | **0.921** | **-18.1%** | **0.80** | **$25.01** | **baseline** | **66%** |
| T1-BC-100 | 14.5% | 15.9% | 0.912 | -18.1% | 0.80 | $24.73 | -$0.27 | 71% |
| T1-BC-100-D25 | 14.5% | 15.9% | 0.913 | -18.1% | 0.80 | $24.83 | -$0.18 | 73% |
| T1-BC-100-D50 | 14.6% | 15.9% | 0.913 | -18.1% | 0.80 | $24.92 | -$0.08 | 73% |
| T2-ALL-2x | 14.5% | 16.1% | 0.901 | -18.1% | 0.80 | $24.42 | -$0.59 | 71% |
| T2-ALL-2x-D50 | 14.5% | 16.1% | 0.902 | -18.1% | 0.80 | $24.60 | -$0.40 | 73% |

**Every tiered strategy underperforms PROD on both Sharpe AND terminal wealth.** The best tiered result (T1-BC-100-D50) loses $0.08 terminal and 0.008 Sharpe — essentially zero difference. The worst (T2-ALL-2x) loses $0.59 terminal and 0.020 Sharpe.

### Crisis Analysis

All strategies produce identical crisis returns — because during GFC, COVID, and 2022, the system was either in State A (all leveraged strategies behave identically) or State E (all at 1x). The intermediate states don't occur during major crises.

### Circuit Breaker Events

| Strategy | Events | Per year |
|----------|--------|---------|
| PROD | 16 | 0.7 |
| T1-BC-100 | 24 | 1.0 |
| T2-ALL-2x | 24 | 1.0 |
| T1-BC-100-D50 | 26 | 1.1 |

Tiered strategies trigger the circuit breaker 50-60% more often (24-26 vs 16 events). This is because leverage is active during more months (71-73% vs 66%), creating more opportunities for the breaker to fire. The additional breaker events during mixed-signal months create unnecessary churn without improving returns.

### Age-65 Projections ($21,000, 40 years)

| Strategy | $21K at 65 | vs PROD |
|----------|-----------|---------|
| PROD | $4,806,016 | baseline |
| T1-BC-100-D50 (best tier) | $4,825,042 | +$19,027 |
| T2-ALL-2x (worst tier) | $4,722,644 | -$83,372 |

Even the best tiered strategy (T1-BC-100-D50) only adds $19K over 40 years — 0.4% improvement. Not meaningful.

## Answers to Key Questions

**Q1. How many months are in B/C states?** 15 out of 259 (6%). Forward returns are +0.57% avg — positive but based on tiny sample (15 obs). Not enough data to justify adding leverage.

**Q2. Does 1.5x during B/C improve terminal?** No. T1-BC-100 loses $0.27 terminal. The 1.5x leverage during mixed months costs more in vol drag than it earns in return.

**Q3. Does full 2x during mixed months help?** No. T2-ALL-2x loses $0.59 terminal and 0.020 Sharpe. Leveraging up during uncertain signals is purely additive vol without compensating return.

**Q4. MaxDD impact?** Zero. All strategies have identical -18.1% max DD because crises don't occur during mixed-signal months.

**Q5. Calmar maintained?** Yes — 0.80 across all strategies. But this is because crises dominate the DD calculation and all strategies behave identically during crises.

**Q6. Is the tiered system worth the complexity?** No. PROD is the best strategy on terminal wealth. Adding tiers increases circuit breaker frequency by 50%, adds implementation complexity (tracking which ETF is at which score, applying different substitution levels), and produces no improvement.

## Decision

**PROD (binary 2x) confirmed as optimal.** No leverage tier improves on the simple rule: both IVV+QQQ at 3/3 → 100% SSO/QLD substitution, otherwise 1x.

The fundamental reason: mixed-signal states (B, C, D) occur only 7% of months. There isn't enough time in these states for any leverage adjustment to materially affect long-term performance. The system spends 66% of its time at full 2x and 27% at 1x — the remaining 7% is a rounding error.

**Complexity verdict:** Adding a 1.5x tier means tracking per-ETF scores separately, applying different substitution levels per month, and handling 50% more circuit breaker events — all for zero improvement. The binary switch is both simpler and better.

# Leverage Tiers: Per-ETF Mixed Signal Test

**Date:** April 6, 2026
**Status:** Complete — Binary switch confirmed optimal
**Track:** Production Architecture

## Key Finding

Mixed signal states (one ETF at 3/3, other at 2/3) occur only 15 months out of 259 (6%). Not enough time in these states for any leverage adjustment to materially affect long-term performance.

| State | Months | % |
|-------|--------|---|
| A: Both 3/3 | 170 | 66% |
| B: IVV 3/3, QQQ 2/3 | 5 | 2% |
| C: QQQ 3/3, IVV 2/3 | 10 | 4% |
| D: Both 2/3 | 4 | 2% |
| E: Either <2/3 | 70 | 27% |

## Performance — All Tiered Strategies Underperform PROD

| Strategy | Sharpe | Terminal | vs PROD |
|----------|--------|---------|---------|
| PROD (binary) | 0.921 | $25.01 | baseline |
| Best tier (T1-BC-100-D50) | 0.913 | $24.92 | -$0.08 |
| Worst tier (T2-ALL-2x) | 0.901 | $24.42 | -$0.59 |

Age-65 difference: best tier adds only $19K over 40 years (0.4%). Not meaningful.

Circuit breaker events increase 50-60% with tiers (24-26 vs 16) — more churn, zero benefit.

## Decision

**Binary switch confirmed as optimal and final.** The idea was worth testing — the per-ETF logic is principled and clean. But the data shows the mixed-signal state is too rare (6% of months) to move the needle. The system spends 93% of its time in either full conviction (State A, 66%) or clearly unleveraged (State E, 27%).

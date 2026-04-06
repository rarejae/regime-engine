# Faber-Sweep: Flat Substitution Tradeoff Characterization

**Date:** April 5, 2026  
**Status:** Complete  
**Track:** Leverage Calibration  
**Related:** [[TAA_PROJECT_STATUS]] | [[2026-04-05_leverage_calibration]]

## Hypothesis

A flat substitution sweep (no tiers, no medians, no weekly checks) characterizes the return/Sharpe/drawdown tradeoff curve for the Faber-only architecture. The principled 40% substitution target (~98% effective equity) — derived from risk tolerance, not optimization — should sit in a reasonable place on this curve.

## Design

**Rule:** Each month-end, if BOTH IVV and QQQ at 3/3 Faber → replace SUB% of each with SSO/QLD. Otherwise full 1x. No tiers, no transitions, monthly rebalance only.

**Substitution levels tested:** 0%, 10%, 20%, 25%, 30%, 40%, 50%, 65%

**Leveraged ETF sim:** `2.0 * daily_return - rfr/252 - expense/252`. SSO expense: 0.0089, QLD expense: 0.0095.

**Signal alignment:** Month T Faber scores applied to month T+1 daily returns. Asserted no look-ahead.

**Daily granularity** throughout, consistent with [[2026-04-05_leverage_calibration]].

Months with leverage condition met (both equities at 3/3): **163/259 (63%)**.

## Results

### Full Performance Table

| Strategy | Return | Vol | Sharpe | Sortino | MaxDD | Calmar | Terminal($1) | vs 60/40 | vs IVV B&H |
|----------|--------|-----|--------|---------|-------|--------|-------------|---------|-----------|
| Faber-1x | 8.4% | 9.0% | 0.935 | 1.147 | -13.0% | 0.65 | $6.92 | +$0.41 | -$2.00 |
| Faber-Sweep-10 | 8.9% | 9.7% | 0.919 | 1.122 | -13.8% | 0.65 | $7.69 | +$1.18 | -$1.24 |
| Faber-Sweep-20 | 9.4% | 10.4% | 0.905 | 1.099 | -14.6% | 0.64 | $8.52 | +$2.01 | -$0.40 |
| Faber-Sweep-25 | 9.6% | 10.7% | 0.898 | 1.088 | -15.0% | 0.64 | $8.97 | +$2.46 | +$0.05 |
| Faber-Sweep-30 | 9.9% | 11.1% | 0.891 | 1.078 | -15.7% | 0.63 | $9.44 | +$2.93 | +$0.51 |
| **Faber-Sweep-40** | **10.4%** | **11.8%** | **0.878** | **1.058** | **-17.1%** | **0.61** | **$10.44** | **+$3.93** | **+$1.51** |
| Faber-Sweep-50 | 10.9% | 12.6% | 0.867 | 1.039 | -18.7% | 0.58 | $11.52 | +$5.01 | +$2.60 |
| Faber-Sweep-65 | 11.6% | 13.7% | 0.851 | 1.015 | -21.1% | 0.55 | $13.33 | +$6.82 | +$4.41 |
| IVV B&H | 10.8% | 19.0% | 0.570 | 0.713 | -55.2% | 0.20 | $8.92 | +$2.42 | — |
| 60/40 | 8.4% | 11.1% | 0.752 | 0.994 | -29.9% | 0.28 | $6.51 | — | -$2.42 |

### Crisis Analysis

| Strategy | GFC (2008-09) | COVID (Feb-Mar 2020) | 2022 Bear |
|----------|---------------|---------------------|-----------|
| Faber-1x | +0.6% (DD -1.1%) | -11.9% (DD -13.0%) | -6.7% (DD -8.1%) |
| Faber-Sweep-25 | +0.6% (DD -1.1%) | -13.9% (DD -15.0%) | -7.8% (DD -10.1%) |
| Faber-Sweep-40 | +0.6% (DD -1.1%) | -15.0% (DD -16.2%) | -8.5% (DD -11.2%) |
| Faber-Sweep-50 | +0.6% (DD -1.1%) | -15.8% (DD -17.1%) | -8.9% (DD -12.0%) |
| Faber-Sweep-65 | +0.6% (DD -1.1%) | -17.0% (DD -18.3%) | -9.6% (DD -13.2%) |
| IVV B&H | -36.9% (DD -46.0%) | -33.4% (DD -33.7%) | -17.7% (DD -24.5%) |
| 60/40 | -17.0% (DD -25.6%) | -16.0% (DD -17.8%) | -24.0% (DD -26.4%) |

**GFC:** All sweep levels identical (+0.6%) — Faber had already exited equity positions before the crash. The filter is the hedge, leverage is irrelevant.

**COVID:** Linear degradation with substitution (13.0% → 18.3% DD across sweep). The crash happened mid-month before Faber could react. This is the known weakness — fast intra-month drawdowns.

**2022 Bear:** Moderate degradation (8.1% → 13.2% DD). Faber exited early enough to limit damage even at higher leverage.

### Tradeoff Curve

| SUB% | Return | Sharpe | MaxDD | Terminal | Sharpe Cost vs 1x | Return Gain vs 1x | ETF Drag (ann) |
|------|--------|--------|-------|----------|-------------------|-------------------|----------------|
| 0% | 8.4% | 0.935 | -13.0% | $6.92 | 0.000 | +0.0% | ~0.00% |
| 10% | 8.9% | 0.919 | -13.8% | $7.69 | -0.016 | +0.5% | ~0.12% |
| 20% | 9.4% | 0.905 | -14.6% | $8.52 | -0.031 | +1.0% | ~0.23% |
| 25% | 9.6% | 0.898 | -15.0% | $8.97 | -0.038 | +1.2% | ~0.29% |
| 30% | 9.9% | 0.891 | -15.7% | $9.44 | -0.045 | +1.5% | ~0.35% |
| **40%** | **10.4%** | **0.878** | **-17.1%** | **$10.44** | **-0.057** | **+2.0%** | **~0.46%** |
| 50% | 10.9% | 0.867 | -18.7% | $11.52 | -0.069 | +2.5% | ~0.58% |
| 65% | 11.6% | 0.851 | -21.1% | $13.33 | -0.084 | +3.2% | ~0.75% |

**Sharpe declines monotonically.** Peak Sharpe is at 0% (no leverage). Every unit of substitution costs Sharpe — this is the leveraged ETF drag (volatility decay + expense ratios + borrowing costs). The drag scales approximately linearly: ~0.46% annualized at 40% sub, ~0.75% at 65%.

**Return and MaxDD scale approximately linearly** with substitution. Each 10% sub buys roughly +0.5% return and costs -1% additional drawdown.

### Principled Target Confirmation

```
Principled target (40% sub, ~98% effective equity):
  Return:              10.4%
  Sharpe:              0.878
  MaxDD:               -17.1%
  Terminal:            $10.44 (vs IVV B&H $8.92, vs 60/40 $6.51)
  Sharpe cost vs 1x:   -0.057
  Return gain vs 1x:   +2.0%
  Position on curve:   above peak (more aggressive than optimal)
```

The 40% sub sits at a reasonable point on the curve. It sacrifices 0.057 Sharpe (6% relative) to gain 2.0% annualized return and $3.52 terminal wealth over 24 years. The 0.878 Sharpe still dominates IVV B&H (0.570) and 60/40 (0.752) by wide margins. The -17.1% max DD is elevated vs unleveraged (-13.0%) but far below IVV B&H (-55.2%) and 60/40 (-29.9%).

## Key Diagnostics

**Consistency with prior experiment.** Faber-1x: 0.935 Sharpe, -13.0% DD — matches [[2026-04-05_leverage_calibration]] exactly. Faber-Sweep-40 here (0.878 Sharpe) matches the Tier1-Only result from the prior experiment (0.878 Sharpe), confirming that the flat 40% sub is equivalent to the Tier 1 in the graduated system.

**Months levered: 63%.** The system is leveraged roughly two-thirds of the time. The remaining one-third (when either IVV or QQQ breaks trend) provides the drawdown protection.

**ETF drag is the dominant cost.** The ~0.46% annualized drag at 40% sub decomposes roughly into: ~0.18% expense ratios + ~0.28% volatility decay (daily compounding drag). This is the unavoidable cost of physical leveraged ETF implementation vs theoretical leverage.

## Interpretation

The sweep confirms three things:

1. **The tradeoff is monotonic and approximately linear.** There is no free lunch — every increment of leverage costs Sharpe and adds drawdown proportionally. No substitution level magically improves risk-adjusted returns.

2. **The 40% principled target is reasonable.** It sits in the middle of the curve, neither timid nor reckless. The 0.057 Sharpe cost buys meaningful absolute return (+2.0%) and terminal wealth (+$3.52 over 24 years). An investor who needs absolute return growth (retirement, compounding) pays a known, moderate price in risk-adjusted terms.

3. **Simplicity wins.** A single flat substitution level (40% when both equities at 3/3, 0% otherwise) captures the same return profile as the more complex graduated tier system from [[2026-04-05_leverage_calibration]], without expanding medians, tier transitions, or weekly circuit breakers. The weekly breaker saves ~2% drawdown but adds operational complexity.

## Decision

Faber-Sweep-40 is the candidate production architecture. The 40% substitution is chosen from the risk tolerance target (98% effective equity), not from backtest optimization, and the sweep confirms it sits at a reasonable point on the tradeoff curve.

## Next Steps

- Implement Faber-Sweep-40 in `taa/run.py` as the production system
- Decision: whether to add weekly circuit breaker (saves ~2% DD, adds operational complexity)
- Configure Telegram alerts for monthly rebalance signals

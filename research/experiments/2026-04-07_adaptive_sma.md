# Adaptive SMA Lookbacks — Volatility Regime Conditional

**Date:** April 7, 2026  
**Status:** Complete  
**Track:** Production Architecture — Signal Enhancement  
**Related:** [[TAA_PROJECT_STATUS]] | [[2026-04-06_faber_daily_circuit_breaker]]

## Result: Adaptive lookbacks HURT performance

| Strategy | Return | Vol | Sharpe | MaxDD | Terminal($1) | vs Baseline |
|----------|--------|-----|--------|-------|-------------|-----------|
| **BASELINE (fixed)** | **14.5%** | **15.8%** | **0.921** | **-18.1%** | **$25.02** | **baseline** |
| ADAPTIVE | 13.4% | 15.5% | 0.864 | -18.6% | $19.06 | **-$5.96** |
| ADAPTIVE-CB | 13.7% | 15.4% | 0.893 | -18.6% | $20.93 | -$4.09 |

**Adaptive loses $5.96 terminal and -0.057 Sharpe.** Fixed 126/200/252-day lookbacks remain optimal.

## Why Adaptive Hurts

### Regime transitions cause whipsaw

129 regime transitions over 259 months (50%). Every other month, on average, the vol regime changes — switching the SMA lookback periods. Each switch changes which SMA the system is checking, potentially creating false signals at the transition boundary.

When going from NORMAL → HIGH VOL: the system switches to 63/126/200-day SMAs. The 63-day SMA is much more responsive and can flip the score from 3/3 to 2/3 or lower during a brief vol spike that the 126-day SMA would have ignored. This premature exit from leverage costs return during the recovery.

When going from HIGH → NORMAL: the system switches back to 126/200/252-day SMAs. The 252-day SMA may still be below price (bullish) when the 200-day SMA was not (the 200-day was part of the HIGH regime set, already checked). The transition can create a brief period where the score artificially jumps.

### The leverage decision disagrees only 3% of months

78% of months, all asset scores agree between adaptive and fixed. 22% disagree on at least one asset. But only 3% disagree on the leverage condition (both IVV+QQQ at 3/3). This means the adaptive lookbacks change the allocation on non-equity assets 19% of the time but rarely change the leverage decision.

The 19% non-equity disagreement is where the damage comes from: the adaptive system enters/exits VGLT, IAU, DBC positions at different times, creating additional turnover without improving timing.

### COVID: slight improvement, not enough

COVID max DD improved from -18.1% to -17.0% (1.1% better). The faster 63-day SMA in HIGH VOL mode caught the trend break slightly earlier. But this single improvement is overwhelmed by the cumulative cost of 129 regime transitions over 24 years.

## Regime Distribution

| Regime | Days | % | Avg Vol |
|--------|------|---|---------|
| HIGH VOL (>75th pct) | 1,431 | 23% | 26.4% |
| NORMAL (25-75th pct) | 2,802 | 46% | 14.6% |
| LOW VOL (<25th pct) | 1,871 | 31% | 9.9% |

When leverage was active (both equities 3/3): 10% in HIGH vol, 54% NORMAL, 36% LOW. The system is mostly leveraged during normal-to-low vol periods, which is when the fixed lookbacks perform best.

## Crisis Analysis

| Strategy | GFC | COVID | 2022 Bear |
|----------|-----|-------|-----------|
| BASELINE | +0.6% (DD -1.1%) | -16.7% (DD -18.1%) | -12.5% (DD -13.8%) |
| ADAPTIVE | +0.4% (DD -1.1%) | -15.7% (DD -17.0%) | -12.1% (DD -13.3%) |

ADAPTIVE is marginally better during COVID (-1.1% DD improvement) and 2022 (-0.5% DD improvement) but marginally worse during GFC. The crisis improvements are real but tiny — and are paid for by worse performance during the 80% of the time that markets are trending normally.

## Key Insight

The Faber SMA signal is robust because it's slow and stable. Fixed lookback periods avoid the regime transition problem entirely. The 126/200/252-day periods capture multi-month trends regardless of current volatility — this is a feature, not a limitation.

Adaptive lookbacks are conceptually appealing (faster signals in fast markets) but the implementation creates a new failure mode: regime transition whipsaw. With 50% of months involving a regime change, the system is constantly adjusting its signal basis, creating inconsistency.

This is the same finding as weekly rebalancing: **the Faber signal works because it's slow and ignores short-term noise.** Making it faster (weekly rebalance) or adaptive (vol-conditional periods) degrades performance.

## Decision

**Fixed 126/200/252-day lookbacks confirmed as optimal.** Adaptive lookbacks rejected — costs $5.96 terminal and -0.057 Sharpe. The regime transition whipsaw (129 transitions over 259 months) creates more damage than the marginal COVID/2022 improvement provides.

Production architecture unchanged.

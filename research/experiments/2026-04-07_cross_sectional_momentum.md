# Cross-Sectional Momentum Overlay

**Date:** April 7, 2026  
**Status:** Complete  
**Track:** Production Architecture — Signal Enhancement  
**Related:** [[TAA_PROJECT_STATUS]] | [[2026-04-06_faber_daily_circuit_breaker]]

## Result: Terminal wealth improves, Sharpe essentially flat

| Strategy | Return | Vol | Sharpe | MaxDD | Terminal($1) | vs Base |
|----------|--------|-----|--------|-------|-------------|---------|
| BASELINE (fixed) | 14.5% | 15.8% | 0.921 | -18.1% | $25.02 | baseline |
| XSMOM (1.20/0.80) | 14.7% | 16.0% | 0.919 | -18.1% | $25.98 | **+$0.96** |
| XSMOM-STRONG (1.35/0.65) | 14.9% | 16.2% | 0.918 | -18.3% | $26.67 | **+$1.65** |

XSMOM adds +$0.96 terminal (3.8%) at -0.002 Sharpe. XSMOM-STRONG adds +$1.65 (6.6%) at -0.003 Sharpe. The Sharpe costs are negligible — within measurement noise. The return improvement is real but comes with proportionally higher volatility, keeping Sharpe essentially flat.

## Momentum Signal Quality

QQQ is the most frequently top-ranked asset (49% of months). VGLT is the most frequently bottom-ranked (35%). This makes intuitive sense: QQQ (Nasdaq-100) has the strongest long-term momentum, while VGLT (long-term treasuries) often trends opposite to equities.

The tilt helped in 44% of months and hurt in 49% — essentially a coin flip on the direction call. But the average help (+0.22%/month) exceeds the average cost (-0.16%/month), creating a small positive net contribution.

## Why Stronger Tilt Hurts Sharpe (Marginally)

XSMOM-STRONG (1.35/0.65) has higher return (14.9% vs 14.7%) but also higher vol (16.2% vs 16.0%). The stronger tilt amplifies the momentum signal — when momentum is right, it's more right; when it's wrong, it's more wrong. At the 1.35/0.65 level, the additional noise from stronger tilting exactly offsets the additional return, keeping Sharpe flat.

This is consistent with the momentum literature: the optimal tilt strength depends on the breadth of the asset universe. With only 5 risky assets, there isn't enough cross-sectional diversity for strong momentum signals. In a 20-50 asset universe, stronger tilts would be more effective.

## Crisis Performance: Identical

All three strategies have essentially identical crisis behavior — the momentum tilt doesn't affect the equity positions during crises because:
1. GFC: Faber filter already exited most positions
2. COVID: Happens too fast for momentum ranks to change
3. 2022: The leverage condition dominates the return profile

## Tilt Attribution

```
Months tilt helped: 114 (44%), avg +0.22%/month
Months tilt hurt: 126 (49%), avg -0.16%/month
Net contribution: positive (avg help > avg cost)

Best month: 2023-05, +0.88%
Worst month: 2019-05, -0.74%
```

The asymmetry (44% helped at +0.22% vs 49% hurt at -0.16%) creates a small positive expected value. The momentum signal is only slightly better than random on direction, but winners are larger than losers on average.

## DBC in 2022

DBC average weight: BASELINE 3.5%, XSMOM 4.0%. The tilt slightly increased DBC exposure during 2022 (DBC was strong-momentum in a commodity bull). The effect was modest — +0.5% additional DBC weight during a year when DBC was the only positive asset.

## Decision

**XSMOM is a marginal improvement, not worth the complexity for the current system.** The +$0.96 terminal gain (3.8%) comes at -0.002 Sharpe cost and adds implementation complexity (computing 12-month momentum ranks, applying tilt multipliers, renormalizing weights monthly).

For a $21K portfolio growing over 40 years, XSMOM adds approximately $150K at age 65 (rough compound estimate of 0.2% annual return advantage). This is meaningful but the improvement is within the uncertainty range of the backtest itself.

**If the investor ever expands the asset universe** (more ETFs, more sectors), cross-sectional momentum becomes more powerful because there's more cross-sectional dispersion to exploit. With 5 assets, the signal is too noisy. With 15-20 assets, momentum tilting is well-documented to add 0.5-1.0% annual alpha.

**XSMOM deferred to universe expansion phase.** Fixed baseline weights remain production architecture.

# Comprehensive Leverage Sweep: Faber-Sweep-40-Daily-Daily

**Date:** April 6, 2026
**Status:** Complete
**Track:** Production Architecture — Leverage Optimization

## Performance Summary

| Strategy | Return | Vol | Sharpe | MaxDD | Terminal($1) | vs 40% | ETF Drag |
|----------|--------|-----|--------|-------|-------------|--------|----------|
| S40-40% (base) | 11.2% | 11.7% | 0.958 | -15.0% | $12.74 | baseline | ~0.45% |
| S40-60% | 12.4% | 13.0% | 0.948 | -16.1% | $16.23 | +$3.49 | ~0.68% |
| S40-80% | 13.5% | 14.4% | 0.938 | -17.1% | $20.56 | +$7.82 | ~0.91% |
| **S40-100%** | **14.7%** | **15.8%** | **0.929** | **-18.1%** | **$25.91** | **+$13.17** | ~1.13% |
| S40-3x-50% | 14.9% | 15.8% | 0.942 | -18.1% | $27.23 | +$14.49 | ~0.93% |
| IVV B&H | 10.8% | 19.0% | 0.570 | -55.2% | $8.92 | — | — |

## Age-65 Projection ($21,000 starting)

| Strategy | Ann Return | $21K at 65 | vs S40-40% |
|----------|-----------|-----------|-----------|
| S40-40% | 11.2% | $1,464,074 | baseline |
| S40-60% | 12.4% | $2,224,270 | +$760,195 |
| S40-80% | 13.5% | $3,364,592 | +$1,900,518 |
| **S40-100%** | **14.7%** | **$5,068,001** | **+$3,603,927** |
| S40-3x-50% | 14.9% | $5,445,146 | +$3,981,072 |

## Key Findings

- Sharpe declines monotonically but very gently: 0.958 → 0.929 across full 2x range (-0.030 total)
- Terminal wealth increases monotonically — no peak reached
- MaxDD tightly bounded by daily circuit breaker: -18.1% at 100% sub vs -26.4% without breaker (saves 8.3%)
- Calmar ratio IMPROVES with leverage: 0.74 → 0.81 — daily breaker keeps DD growth below return growth
- GFC identical across all strategies — Faber filter already exited before crash
- 3x-50% marginally beats 100% 2x ($27.23 vs $25.91) but adds implementation complexity

## Pedersen Validation

- Pedersen recommendation for age 25: 2.0x = 100% SSO/QLD substitution
- Backtested at Pedersen level: 14.7% return, 0.929 Sharpe, -18.1% DD, $5.1M at age 65
- Dollar drawdown at $21K: -18.1% = $3,801 paper loss — psychologically manageable
- Sharpe cost vs current: -0.029 (3% reduction)
- Terminal wealth gain vs current: +$3.6M at age 65

**Pedersen recommendation CONFIRMED by backtest data.**

## Production Decision

**S40-100% (2x full substitution) adopted as production leverage level for age 25.**

Lifecycle delevering schedule (a priori from Pedersen, not backtest-optimized):
- Age 25-29: 100% substitution (~140% eff equity)
- Age 30-34: 80% substitution (~126% eff equity)
- Age 35-44: 60% substitution (~112% eff equity)
- Age 45-54: 40% substitution (~98% eff equity)
- Age 55+: 0% substitution (1x only)

Dollar-size reduction schedule:
- $21K: 100% sub
- $50K: 80% sub
- $100K: 60% sub
- $250K+: 40% sub

S40-3x-50% is theoretical optimum (+$377K at 65) but SPXL/TQQQ complexity not justified for first implementation.

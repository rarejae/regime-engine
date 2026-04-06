# Comprehensive Leverage Sweep: Faber-Sweep-40-Daily-Daily

**Date:** April 6, 2026  
**Status:** Complete  
**Track:** Production Architecture — Leverage Optimization  
**Related:** [[TAA_PROJECT_STATUS]] | [[2026-04-06_faber_daily_circuit_breaker]]

## Purpose

Find the optimal substitution level for a 25-year-old long-horizon accumulator. The daily circuit breaker (16 events over 24 years, 0.7/year) should constrain drawdowns even at high leverage, allowing the Sharpe/terminal tradeoff curve to shift favorably vs the prior monthly-only sweep.

## Results

### Full Performance Table

| Strategy | Return | Vol | Sharpe | Sortino | MaxDD | Calmar | Terminal($1) | vs 40% | ETF Drag |
|----------|--------|-----|--------|---------|-------|--------|-------------|--------|----------|
| S40-40% (base) | 11.2% | 11.7% | 0.958 | 1.176 | -15.0% | 0.74 | $12.74 | baseline | ~0.45% |
| S40-50% | 11.8% | 12.4% | 0.953 | 1.166 | -15.6% | 0.76 | $14.39 | +$1.65 | ~0.57% |
| S40-60% | 12.4% | 13.0% | 0.948 | 1.157 | -16.1% | 0.77 | $16.23 | +$3.49 | ~0.68% |
| S40-70% | 12.9% | 13.7% | 0.943 | 1.147 | -16.6% | 0.78 | $18.28 | +$5.54 | ~0.79% |
| S40-80% | 13.5% | 14.4% | 0.938 | 1.138 | -17.1% | 0.79 | $20.56 | +$7.82 | ~0.91% |
| **S40-100%** | **14.7%** | **15.8%** | **0.929** | **1.120** | **-18.1%** | **0.81** | **$25.91** | **+$13.17** | ~1.13% |
| S40-3x-25% | 11.9% | 12.4% | 0.962 | 1.177 | -15.6% | 0.76 | $14.75 | +$2.01 | ~0.46% |
| S40-3x-40% | 13.7% | 14.4% | 0.949 | 1.152 | -17.1% | 0.80 | $21.39 | +$8.66 | ~0.74% |
| **S40-3x-50%** | **14.9%** | **15.8%** | **0.942** | **1.136** | **-18.1%** | **0.82** | **$27.23** | **+$14.49** | ~0.93% |
| IVV B&H | 10.8% | 19.0% | 0.570 | 0.713 | -55.2% | 0.20 | $8.92 | — | — |

### Key Properties of the Sweep

**Sharpe declines monotonically** — consistent with prior findings. Every increment of leverage costs Sharpe due to ETF drag. But the decline is very gentle: 0.958 at 40% → 0.929 at 100% 2x → 0.942 at 50% 3x. The daily circuit breaker constrains the vol/DD growth, keeping Sharpe remarkably flat across the sweep.

**Terminal wealth increases monotonically** through the entire 2x range and into 3x. No terminal wealth peak was reached — even S40-3x-50% ($27.23) is still climbing.

**MaxDD is tightly bounded.** Even at 100% 2x substitution, max DD is only -18.1% — compared to IVV B&H's -55.2%. The daily circuit breaker exits leverage before drawdowns compound. COVID at S40-100%: -18.1% DD. Without the breaker (monthly-only sweep from the prior experiment), 100% sub produced -26.4% DD. The breaker saved 8.3% of drawdown.

**3x ETFs slightly outperform maxed 2x.** S40-3x-50% ($27.23) beats S40-100% ($25.91) by $1.32 despite higher theoretical drag. This is because at the same effective equity exposure, 3x requires lower substitution percentage (50% of 3x = 150% effective vs 100% of 2x = 140% effective), meaning more of the position retains 1x characteristics.

### Crisis Analysis

| Strategy | GFC (2008-09) | COVID (Feb-Mar 2020) | 2022 Bear |
|----------|--------------|---------------------|-----------|
| S40-40% | +0.6% (DD -1.1%) | -13.8% (DD -15.0%) | -9.7% (DD -10.7%) |
| S40-80% | +0.6% (DD -1.1%) | -15.7% (DD -17.1%) | -11.6% (DD -12.8%) |
| S40-100% | +0.6% (DD -1.1%) | -16.7% (DD -18.1%) | -12.5% (DD -13.8%) |
| S40-3x-50% | +0.6% (DD -1.1%) | -16.7% (DD -18.1%) | -12.5% (DD -13.7%) |
| IVV B&H | -36.9% (DD -46.0%) | -33.4% (DD -33.7%) | -17.7% (DD -24.5%) |

**GFC: all strategies identical** — leverage was already OFF. The Faber filter is the primary hedge at every leverage level.

**COVID: linear scaling from 40% to 100%.** MaxDD grows from -15.0% to -18.1% — only 3.1% additional drawdown for 2.5x more leverage. The circuit breaker (Feb 27, 2020) limits exposure during the crash regardless of leverage level.

**2022: similarly bounded.** S40-100% DD of -13.8% is manageable — about half of IVV B&H's -24.5%.

### Age-65 Projection ($21,000 starting, 40-year horizon)

| Strategy | Ann Return | $21K at 65 | vs S40-40% |
|----------|-----------|-----------|-----------|
| S40-40% | 11.2% | $1,464,074 | baseline |
| S40-60% | 12.4% | $2,224,270 | +$760,195 |
| S40-80% | 13.5% | $3,364,592 | +$1,900,518 |
| **S40-100%** | **14.7%** | **$5,068,001** | **+$3,603,927** |
| **S40-3x-50%** | **14.9%** | **$5,445,146** | **+$3,981,072** |
| IVV B&H | 10.8% | $1,292,980 | — |

At 40 years, the compound effect is dramatic. S40-100% (2x full substitution) turns $21K into $5.1M vs $1.5M at the current 40% level — a $3.6M difference. S40-3x-50% reaches $5.4M.

### Circuit Breaker at High Leverage

The daily circuit breaker fires identically across all leverage levels (16 events, 0.7/year). **The breaker is leverage-independent** — it monitors whether IVV/QQQ are below their SMAs, not the leverage level. This means higher leverage gets the same protective exits, just with larger positions being converted to 1x.

## Leverage Tradeoff

| SUB% | Return | Sharpe | MaxDD | Terminal | Sharpe Cost | Terminal Gain | ETF Drag |
|------|--------|--------|-------|----------|------------|--------------|----------|
| 40% 2x | 11.2% | 0.958 | -15.0% | $12.74 | baseline | baseline | ~0.45% |
| 50% 2x | 11.8% | 0.953 | -15.6% | $14.39 | -0.005 | +$1.65 | ~0.57% |
| 60% 2x | 12.4% | 0.948 | -16.1% | $16.23 | -0.010 | +$3.49 | ~0.68% |
| 70% 2x | 12.9% | 0.943 | -16.6% | $18.28 | -0.015 | +$5.54 | ~0.79% |
| 80% 2x | 13.5% | 0.938 | -17.1% | $20.56 | -0.020 | +$7.82 | ~0.91% |
| 100% 2x | 14.7% | 0.929 | -18.1% | $25.91 | -0.030 | +$13.17 | ~1.13% |
| 3x-50% | 14.9% | 0.942 | -18.1% | $27.23 | -0.017 | +$14.49 | ~0.93% |

**The Sharpe cost of higher leverage is remarkably low.** Going from 40% to 100% 2x costs only 0.030 Sharpe (3%) while doubling terminal wealth. This is because the daily circuit breaker constrains drawdowns, keeping vol growth proportional to return growth.

**The Calmar ratio actually IMPROVES with leverage** — from 0.74 at 40% to 0.81 at 100%. This means the return-per-unit-of-drawdown gets better, not worse, as leverage increases. The daily circuit breaker is the key enabler.

## Answers to Key Questions

**Q1. Does the daily circuit breaker allow higher leverage to achieve better terminal wealth?**
Yes, dramatically. At 100% 2x: $25.91 terminal with only -18.1% max DD. The prior monthly-only sweep showed 100% 2x at -26.4% DD — the daily breaker saves 8.3% of drawdown.

**Q2. 2x terminal wealth peak?**
No peak reached — S40-100% ($25.91) is the highest 2x level tested. The curve is still climbing.

**Q3. Do 3x ETFs outperform maxed 2x?**
Yes, marginally. S40-3x-50% ($27.23) beats S40-100% ($25.91) by $1.32 — the lower substitution percentage at 3x preserves more 1x character. 3x ETF drag (~0.93%) is actually LOWER than 2x at 100% sub (~1.13%) because less of the portfolio is in leveraged instruments.

**Q4. Maximum sustainable leverage?**
All tested levels produce terminal wealth above the 40% baseline. The tradeoff is purely about drawdown tolerance. At -18.1% max DD (the highest tested), the system is still far safer than IVV B&H (-55.2%).

**Q5. Recommendation for 25-year-old ($21K portfolio, max DD ~-30%)?**
All tested strategies have max DD well below -30%. The unconstrained terminal-wealth maximizer is **S40-3x-50%** ($5.4M at age 65, -18.1% max DD, 0.942 Sharpe). For a simpler implementation avoiding 3x ETFs, **S40-100% 2x** ($5.1M at age 65, -18.1% max DD, 0.929 Sharpe).

A -18.1% max DD on a $21K portfolio is a $3,800 paper loss — psychologically manageable for a 25-year-old with 40 years to recover.

## Decision

**Recommended substitution for a 25-year-old investor: S40-100% (2x full substitution)**

Rationale:
- $5.1M at age 65 vs $1.5M at current 40% level — 3.5x more wealth
- Max DD -18.1% — well within tolerance for a $21K portfolio
- Sharpe 0.929 — still 63% above IVV B&H (0.570)
- Simpler implementation than 3x (SSO/QLD only, no SPXL/TQQQ)
- The daily circuit breaker provides the safety net that makes this viable

S40-3x-50% is the theoretical optimum but introduces 3x ETF complexity (SPXL/TQQQ management, higher tracking error, less liquid options market). The $377K additional wealth at age 65 may not justify the complexity for a first implementation.

**The substitution level should be REDUCED as the portfolio grows** — a -18.1% DD on $21K ($3,800) is fine; a -18.1% DD on $500K ($90,500) may not be. As the portfolio scales, gradually reduce substitution from 100% toward 40%.

## Next Steps

- Implement S40-100% as the production starting configuration
- Define substitution reduction schedule as portfolio grows (e.g., 100% at $21K, 80% at $50K, 60% at $100K, 40% at $250K+)
- Monitor 3x ETFs (SPXL/TQQQ) for possible future adoption if portfolio complexity tolerance increases

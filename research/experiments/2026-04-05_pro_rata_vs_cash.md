# Pro-Rata Redistribution vs Cash Baseline

**Date:** April 5, 2026  
**Status:** Complete  
**Track:** Faber Optimization  
**Related:** [[TAA_PROJECT_STATUS]] | [[KRITZMAN_RESEARCH_FINDINGS]] | [[2026-04-04_faber_only_baseline]]

## Hypothesis

Redistributing freed capital pro-rata across eligible assets (instead of parking in cash) would improve absolute returns while maintaining acceptable risk-adjusted performance. A cross-sectional variant weighting by trend strength (3/3 gets 2x vs 2/3) might further improve by concentrating in stronger trends.

## Design

Three strategies, all using identical Faber multi-SMA trend filter (6/10/12 SMA), same baseline weights (IVV 45%, QQQ 25%, VGLT 5%, IAU 10%, DBC 5%, Cash 10%), 1x leverage, 2002-2026 monthly.

- **Faber-Cash:** Freed capital goes to cash. Same as prior [[2026-04-04_faber_only_baseline]].
- **Faber-ProRata:** Freed capital redistributed proportionally across eligible assets by their current baseline weight. 70% per-asset cap.
- **Faber-CrossSectional:** Same as ProRata but redistribution weighted by trend strength (3/3 gets 2x weight vs 2/3). 70% per-asset cap.
- **Harvey (behind Faber):** Harvey inverse-vol capital direction for comparison.
- **IVV B&H** and **60/40** benchmarks.

Signal alignment: trend_df shifted by 1 month (compute_trend_scores PIT). Month T scores use data through T-1, applied to month T returns. Assertions verified weight sums = 1.0 at every timestep.

## Results

| Strategy | Return | Vol | Sharpe | Sortino | MaxDD | Calmar | Terminal |
|----------|--------|-----|--------|---------|-------|--------|----------|
| Faber-Cash | 8.3% | 7.4% | 1.114 | 1.669 | -9.6% | 0.86 | $6.94 |
| Faber-ProRata | 9.6% | 10.0% | 0.962 | 1.590 | -20.8% | 0.46 | $9.12 |
| Faber-CrossSectional | 9.7% | 10.0% | 0.975 | 1.606 | -20.0% | 0.49 | $9.31 |
| Harvey (behind Faber) | 9.3% | 9.4% | 0.990 | 1.589 | -12.7% | 0.73 | $8.48 |
| IVV B&H | 10.1% | 15.0% | 0.673 | 0.910 | -50.8% | 0.20 | $8.72 |
| 60/40 | 7.9% | 10.0% | 0.787 | 1.058 | -25.7% | 0.31 | $5.97 |

**Alpha summary:**
- Pro-rata alpha over Faber-Cash: **-0.151 Sharpe**
- Cross-sectional alpha over Faber-Cash: **-0.139 Sharpe**
- Pro-rata alpha over Harvey: **-0.028 Sharpe**
- Cross-sectional alpha over Harvey: **-0.015 Sharpe**

### Crisis Analysis

| Strategy | GFC (2008-09) | COVID (2020) | 2022 Bear |
|----------|---------------|--------------|-----------|
| Faber-Cash | +1.1% (DD -0.3%) | -5.8% (DD -1.6%) | -6.1% (DD -2.9%) |
| Faber-ProRata | +14.0% (DD -2.3%) | -4.4% (DD -3.5%) | -17.6% (DD -16.8%) |
| Faber-CrossSectional | +14.8% (DD -2.4%) | -4.4% (DD -3.5%) | -16.8% (DD -16.2%) |
| Harvey (behind Faber) | +0.8% (DD -5.1%) | -6.5% (DD -2.9%) | -9.7% (DD -10.1%) |
| IVV B&H | -37.4% (DD -35.6%) | -9.4% (DD -12.7%) | -17.5% (DD -20.4%) |

## Key Diagnostics

**Average weights:**

| Strategy | IVV | QQQ | VGLT | IAU | DBC | Cash |
|----------|-----|-----|------|-----|-----|------|
| Faber-Cash | 33.8% | 18.4% | 2.6% | 5.6% | 2.0% | 37.6% |
| Faber-ProRata | 40.2% | 22.1% | 6.5% | 10.6% | 4.0% | 16.6% |
| Faber-CrossSectional | 39.9% | 22.1% | 6.5% | 10.7% | 4.1% | 16.6% |

Months where pro-rata differs from cash: 254/291 (87%).  
Mean monthly return diff (ProRata - Cash): +0.113%.  
Std of monthly diff: 1.353%.

Cross-sectional weighting is nearly identical to equal pro-rata — the 2x strength multiplier barely changes allocations because eligible assets are already dominated by the baseline weight scaling.

## Interpretation

**Both pro-rata variants destroy Sharpe relative to Faber-Cash.** This is the same pattern observed with every macro engine: redeploying the cash that Faber frees up earns the equity risk premium (higher absolute return: +1.3-1.4% annualized) but at disproportionate volatility (+2.6%) and catastrophic drawdown cost (max DD doubles from -9.6% to -20.8%).

The mechanism is clear from the weight diagnostics: Faber-Cash holds 37.6% cash on average — this IS the hedge. Pro-rata cuts average cash to 16.6%, surrendering most of the defensive posture. The 2022 bear market is the smoking gun: Faber-Cash lost -6.1% while ProRata lost -17.6%, nearly 3x worse.

**Cross-sectional adds negligible value over equal pro-rata** (+0.013 Sharpe). Trend strength (3/3 vs 2/3) is too coarse a signal to meaningfully differentiate redistribution weights when baseline weights already dominate the allocation.

**GFC performance is misleading.** Pro-rata showed +14% during GFC vs Faber-Cash's +1.1%. This is because gold (IAU) and commodities (DBC) rallied while equities crashed, and pro-rata had more capital in those assets. But this is survivorship — the same concentration that helped in 2008 (more in IAU/DBC) destroyed returns in 2022 when all risky assets fell together.

**Confirms [[KRITZMAN_RESEARCH_FINDINGS]] key insight:** The cash freed by Faber's trend filter IS the hedge. Any mechanism that redeploys it — whether macro-driven (Harvey, Kritzman) or rule-based (pro-rata) — is effectively leveraging up the risk budget that Faber deliberately reduced.

## Decision

[[reject_pro_rata_redistribution]] — Pro-rata redistribution destroys Sharpe by -0.151 and doubles max drawdown. The freed capital should remain in cash.

## Next Steps

- [[planned: universe_expansion]] — More independent trend bets via universe expansion (VXUS, more commodities) may be a better path to higher returns than redeploying cash within the existing universe.
- [[planned: leverage_calibration_faber_only]] — Graduated leverage on the simplified Faber-only architecture remains the designated return amplifier.

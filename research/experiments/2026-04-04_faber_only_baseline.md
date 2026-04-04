# Faber-Only Baseline

**Date:** April 4, 2026  
**Status:** Complete  
**Track:** Kritzman Research  
**Related:** [[TAA_PROJECT_STATUS]] | [[KRITZMAN_RESEARCH_FINDINGS]] | [[2026-04-04_kritzman_vs_harvey]]

## Hypothesis

Faber trend filter alone (no macro engine, freed capital to cash) would underperform macro-enhanced variants, justifying the macro engine's inclusion.

## Design

Faber filter with baseline weights. Ineligible assets' weight goes entirely to cash. No Harvey, no Kritzman. No leverage.

## Results

| Strategy | Return | Vol | Sharpe | MaxDD | Terminal |
|----------|--------|-----|--------|-------|----------|
| Faber-Only | 8.6% | 7.4% | 1.164 | -9.6% | $7.37 |
| Harvey (behind Faber) | 9.4% | 9.5% | 0.982 | -17.8% | $8.55 |
| Kritzman-RP (behind Faber) | 10.2% | 9.0% | 1.134 | -11.7% | $10.51 |

**Macro engine alpha over Faber-only:**
- Harvey: **-0.182 Sharpe**
- Kritzman-RP: **-0.030 Sharpe**

## Interpretation

Every macro engine destroys Sharpe relative to Faber-only. Macro engines add absolute return by redeploying cash into risky assets, but at disproportionate volatility and drawdown cost. This is the equity risk premium, not alpha from better allocation decisions.

## Decision

[[reject_harvey_expected_returns]] — Harvey expected returns confirmed as noise.  
[[reject_kritzman_expected_returns]] — Kritzman expected returns also noise.  
Proceed to test whether the covariance matrix specifically adds value via MVO synthesis.

## Next Steps

[[2026-04-04_mvo_ordinal_scaling]] — Test Faber conviction + Kritzman covariance in MVO.

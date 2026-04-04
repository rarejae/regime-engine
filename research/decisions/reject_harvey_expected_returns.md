# Reject Harvey-Mulliner Expected Returns

**Date:** April 4, 2026  
**Decision:** Reject  
**Triggered by:** [[2026-04-04_faber_only_baseline]]  
**Related:** [[TAA_PROJECT_STATUS]] | [[KRITZMAN_RESEARCH_FINDINGS]]

## What Was Evaluated

Harvey-Mulliner similarity engine using Euclidean distance in z-score space, hard 15th-percentile threshold, equal-weight average of forward returns from similar months.

## Evidence

- Harvey behind Faber: 0.982 Sharpe, -17.8% max DD
- Faber-only (no Harvey): 1.164 Sharpe, -9.6% max DD
- Harvey alpha over Faber: **-0.182 Sharpe**
- Harvey adds 0.8% return but nearly doubles drawdown

## Rationale

The expected returns from macro similarity are noise at monthly frequency. Harvey's capital direction redeployed Faber's protective cash into risky assets, earning the equity risk premium but degrading risk-adjusted returns. The cash that Faber frees up IS the hedge — redeploying it defeats the purpose.

Consistent with literature: out-of-sample return predictability from macro signals is essentially zero at monthly frequency (AQR, Moskowitz et al.).

## Implications

Harvey-Mulliner can be removed from the production architecture without loss of risk-adjusted performance. The hierarchical Faber → Harvey → Leverage architecture simplifies to Faber → Leverage.

## Reversibility

Would revisit if: (a) a fundamentally different return forecasting methodology shows positive alpha over Faber on a walk-forward OOS basis, or (b) the asset universe expands enough that capital direction between many uncorrelated assets becomes valuable.

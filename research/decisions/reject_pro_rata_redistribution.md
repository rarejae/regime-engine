# Reject Pro-Rata Redistribution

**Date:** April 5, 2026  
**Decision:** Reject  
**Triggered by:** [[2026-04-05_pro_rata_vs_cash]]  
**Related:** [[TAA_PROJECT_STATUS]] | [[KRITZMAN_RESEARCH_FINDINGS]]

## What Was Evaluated

Two pro-rata redistribution variants for freed capital behind the Faber trend filter:
1. **Faber-ProRata:** Redistribute proportionally by baseline weight across eligible assets (70% per-asset cap).
2. **Faber-CrossSectional:** Same, but weight by trend strength (3/3 gets 2x vs 2/3).

## Evidence

- Faber-Cash: 1.114 Sharpe, -9.6% max DD
- Faber-ProRata: 0.962 Sharpe, -20.8% max DD (alpha: **-0.151 Sharpe**)
- Faber-CrossSectional: 0.975 Sharpe, -20.0% max DD (alpha: **-0.139 Sharpe**)
- Max drawdown more than doubles under pro-rata
- 2022 bear: Faber-Cash -6.1% vs ProRata -17.6% (3x worse)

## Rationale

Pro-rata redistribution is mechanically equivalent to what Harvey and Kritzman macro engines do — redeploy the cash that Faber frees up into risky assets. The absolute return gain (+1.3%) comes at disproportionate volatility (+2.6%) and drawdown cost (max DD doubles). This is the equity risk premium, not alpha from better allocation.

Faber-Cash holds 37.6% cash on average — that cash IS the hedge. Cutting it to 16.6% (as pro-rata does) surrenders the defensive posture that makes Faber valuable.

Cross-sectional weighting adds negligible value (+0.013 Sharpe) because the 3-level trend score (0-1/2/3) is too coarse to meaningfully differentiate within eligible assets.

## Implications

Freed capital should remain in cash. The architecture stays: Faber filter → cash for freed capital → graduated leverage as the return amplifier. Any future approach to utilizing freed capital must demonstrate positive Sharpe alpha over Faber-Cash on a walk-forward basis before adoption.

## Reversibility

Would revisit if: (a) universe expansion creates enough independent trend bets that selective redistribution among uncorrelated assets improves Sharpe, or (b) a finer-grained trend signal enables meaningful cross-sectional discrimination.

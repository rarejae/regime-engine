# Reject Kritzman Expected Returns

**Date:** April 4, 2026  
**Decision:** Reject  
**Triggered by:** [[2026-04-04_faber_only_baseline]] | [[2026-04-04_kritzman_vs_harvey]]  
**Related:** [[TAA_PROJECT_STATUS]] | [[KRITZMAN_RESEARCH_FINDINGS]]

## What Was Evaluated

Kritzman relevance-weighted expected returns using Mahalanobis distance, soft relevance weighting (similarity + informativeness), theoretically grounded in kernel regression and information theory. Direct upgrade to Harvey's methodology.

## Evidence

- Kritzman-InvVol behind Faber: 0.994 Sharpe
- Faber-only: 1.164 Sharpe
- Kritzman-InvVol alpha over Faber: **-0.170 Sharpe**
- Marginal improvement over Harvey (+0.013 Sharpe) but still net negative vs Faber

## Rationale

Better distance metric (Mahalanobis vs Euclidean) and better weighting (relevance vs equal) did not meaningfully improve forward return estimates. The problem is fundamental: one-month-ahead asset returns are dominated by randomness regardless of how precisely you characterize the current macro environment. Better statistical machinery processing the same noisy signal still produces noise.

## Implications

No macro-based expected return methodology tested improves risk-adjusted returns over simple trend following. This finding is consistent with published literature across multiple research groups and centuries of data.

## Reversibility

Same as [[reject_harvey_expected_returns]].

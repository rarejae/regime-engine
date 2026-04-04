# Kritzman vs Harvey — Component Value-Add

**Date:** April 4, 2026  
**Status:** Complete  
**Track:** Kritzman Research  
**Related:** [[TAA_PROJECT_STATUS]] | [[KRITZMAN_RESEARCH_FINDINGS]]

## Hypothesis

Replacing Harvey-Mulliner's Euclidean distance / equal-weight approach with Kritzman's Mahalanobis distance / relevance-weighted approach would improve risk-adjusted performance, particularly through the conditioned covariance matrix.

## Design

Four allocation methods behind the same Faber gate, all at 1x leverage:
- **Harvey:** Euclidean distance, 15th percentile hard threshold, equal-weight, inverse-vol allocation
- **Kritzman-InvVol:** Mahalanobis distance, relevance weighting, inverse-vol allocation (direct Harvey comparison)
- **Kritzman-MV:** Same + mean-variance optimization using conditioned returns AND covariance
- **Kritzman-RP:** Risk parity using conditioned covariance only — discards expected returns

Same 7 macro indicators, same Faber filter, same asset universe.

## Results

| Strategy | Return | Vol | Sharpe | MaxDD | Terminal |
|----------|--------|-----|--------|-------|----------|
| Harvey | 9.4% | 9.5% | 0.982 | -17.8% | $8.55 |
| Kritzman-InvVol | 9.6% | 9.7% | 0.994 | -19.2% | $9.07 |
| Kritzman-MV | 9.7% | 9.5% | 1.017 | -17.2% | $9.27 |
| Kritzman-RP | 10.2% | 9.0% | 1.134 | -11.7% | $10.51 |

### Crisis Analysis

| Strategy | GFC | COVID | 2022 Bear |
|----------|-----|-------|-----------|
| Harvey | +0.2% | -5.4% | -8.0% |
| Kritzman-InvVol | -3.7% | -4.1% | -7.0% |
| Kritzman-MV | +2.6% | -3.3% | -7.3% |
| Kritzman-RP | +5.4% | -3.3% | -7.8% |

## Interpretation

Clear value gradient: returns-only (worst) → returns+covariance (better) → covariance-only (best). Mahalanobis distance barely improved return estimates (+0.013 Sharpe for InvVol vs Harvey). The conditioned covariance matrix has value; the conditioned expected returns are noise.

## Decision

Proceed to test covariance value more directly. See [[2026-04-04_faber_only_baseline]].

## Next Steps

Test Faber-only baseline to measure how much alpha any macro engine actually adds.

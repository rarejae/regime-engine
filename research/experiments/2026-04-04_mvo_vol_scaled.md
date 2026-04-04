# MVO with Vol-Scaled Conviction (Scaling Fixed)

**Date:** April 4, 2026  
**Status:** Complete  
**Track:** Kritzman Research  
**Related:** [[TAA_PROJECT_STATUS]] | [[2026-04-04_mvo_ordinal_scaling]]

## Hypothesis

Scaling conviction scores as μ_i = S_monthly × σ_i_conditioned × trend_modifier (S=0.85 annual / √12 ≈ 0.245 monthly) would put return and risk terms on the same scale, letting the optimizer genuinely trade off trend conviction against diversification.

## Design

Same MVO framework. S=0.85 annual grounded in published trend-following literature (AQR, Faber, Moskowitz). Lambda sweep 1–50. Unconstrained, 60% cap, 40% cap tested.

## Results

**Lambda now varied portfolios** — scaling fix confirmed working.

Best results (unconstrained):
| λ | Vol | Return | Sharpe | MaxDD |
|---|-----|--------|--------|-------|
| 7 | 9.9% | 11.3% | 1.144 | -17.4% |
| 10 | 9.4% | 11.2% | 1.185 | -14.2% |

S sensitivity (at λ=7):
| S_annual | Sharpe | MaxDD |
|----------|--------|-------|
| 0.50 | 1.161 | -13.0% |
| 0.70 | 1.173 | -15.5% |
| 0.85 | 1.144 | -17.4% |
| 1.00 | 1.104 | -19.2% |

## Key Diagnostics

- **Conditioned vol vs realized vol correlation: -0.049** (no predictive value)
- Broken-trend assets held: **0.3% of months** (trend signal dominates)
- Lower S (more diversification) → higher Sharpe (converges to risk parity)

## Interpretation

The scaling fix worked mechanically but revealed the deeper problem: the conditioned covariance matrix has no predictive value for future asset co-movements. The optimizer can't make good risk-aware decisions with a risk model that doesn't predict risk. Best Sharpe (1.185) marginally exceeded Faber-only (1.164) but this is likely noise, not signal. Lower S always improved Sharpe, meaning the optimizer performs best when you turn off the trend signal and let diversification dominate — which converges to risk parity, already tested.

## Decision

[[reject_conditioned_covariance]] — Conditioned covariance matrix adds no value over unconditional risk estimates.

## Next Steps

Faber-only with graduated leverage is the highest-Sharpe implementable system. Test pro-rata redistribution of freed capital vs cash to determine optimal capital deployment within the simplified architecture.

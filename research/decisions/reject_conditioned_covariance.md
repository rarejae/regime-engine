# Reject Conditioned Covariance Matrix

**Date:** April 4, 2026  
**Decision:** Reject  
**Triggered by:** [[2026-04-04_mvo_vol_scaled]]  
**Related:** [[TAA_PROJECT_STATUS]] | [[KRITZMAN_RESEARCH_FINDINGS]]

## What Was Evaluated

Kritzman relevance-weighted conditioned covariance matrix as the risk input to mean-variance optimization, paired with Faber trend-based conviction scores scaled via implied trend Sharpe ratio (S=0.85 annual).

## Evidence

- Best MVO result (unconstrained λ=10): 1.185 Sharpe (vs Faber-only 1.164)
- Conditioned portfolio vol vs realized vol correlation: **-0.049** (no predictive value)
- Broken-trend assets held for diversification: **0.3% of months** (covariance rarely overrode trend)
- Lower S (less trend, more covariance influence) → higher Sharpe (converges to risk parity)
- Position caps did more diversification work than the optimizer in every test

## Rationale

The conditioned covariance matrix from the Kritzman engine does not predict future asset co-movements. At -0.049 correlation with realized volatility, the regime-conditioned risk estimates are no better than random. An optimizer cannot make good risk-aware decisions with a risk model that has no predictive power. The marginal Sharpe improvement (1.185 vs 1.164) is likely noise rather than genuine alpha from covariance conditioning.

Additionally, in every MVO test, position cap constraints improved Sharpe more reliably than optimizer-driven diversification — indicating that forced diversification via simple rules outperforms sophisticated covariance-based optimization with noisy inputs.

## Implications

The conditioned covariance matrix can be removed from the production architecture. Risk management should rely on Faber's trend signal (binary risk-on/risk-off per asset) rather than continuous covariance-based position sizing. Diversification should be achieved through baseline weights and simple position limits, not optimization.

## Reversibility

Would revisit if: (a) higher-frequency covariance estimation (weekly or daily) showed improved prediction, (b) a different set of conditioning variables produced a covariance matrix with meaningful predictive power (correlation with realized vol > 0.3), or (c) the asset universe expanded sufficiently that covariance-aware sizing becomes necessary to manage interaction effects across 10+ assets.

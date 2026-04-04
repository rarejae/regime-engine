# Kritzman Relevance Engine — Research Findings

**Last Updated:** April 4, 2026  
**Status:** Evaluation complete — all components rejected for production use  
**Related:** [[TAA_PROJECT_STATUS]]

## Executive Summary

The Kritzman relevance engine was evaluated as a potential replacement for the Harvey-Mulliner similarity engine in the TAA system. While theoretically superior (Mahalanobis distance, relevance weighting, conditioned covariance), extensive testing across four experiments demonstrated that no component of the Kritzman engine improves risk-adjusted returns over the Faber trend filter alone.

## Background

The Harvey-Mulliner engine used Euclidean distance in z-score space with a hard 15th-percentile threshold and equal-weight averaging of forward returns from similar historical months. Kritzman's approach upgrades every component:

- **Distance metric:** Mahalanobis (accounts for variable correlations) vs Euclidean
- **Weighting:** Soft relevance weights (similarity × informativeness) vs hard threshold + equal weight
- **Risk model:** Conditioned covariance matrix from relevance-weighted outer products
- **Theoretical grounding:** Kernel regression + information theory vs ad hoc similarity

## Component-Level Findings

### Expected Returns (Rejected)

Both Harvey and Kritzman expected returns destroy Sharpe relative to Faber-only:

| Method | Sharpe | Alpha vs Faber |
|--------|--------|----------------|
| Faber-only | 1.164 | — |
| Harvey | 0.982 | -0.182 |
| Kritzman-InvVol | 0.994 | -0.170 |
| Kritzman-MV | 1.017 | -0.147 |

Better statistical machinery (Mahalanobis + relevance weighting) produced only marginal improvement over Harvey (+0.013 Sharpe). The problem is fundamental: one-month-ahead asset returns are dominated by randomness regardless of macro conditioning precision.

**Decision:** [[reject_harvey_expected_returns]], [[reject_kritzman_expected_returns]]

### Conditioned Covariance Matrix (Rejected)

The covariance matrix was the most promising component — Kritzman-RP (risk parity using only conditioned covariance) achieved the best macro-engine Sharpe at 1.134. However:

- Conditioned portfolio vol vs realized vol correlation: **-0.049** (no predictive value)
- Best MVO Sharpe (1.185) only marginally exceeded Faber-only (1.164) — likely noise
- Lower trend influence (lower S parameter) always improved Sharpe → converges to risk parity
- Position caps outperformed optimizer-driven diversification in every test
- Broken-trend assets held: 0.3% of months (covariance rarely overrode trend signal)

**Decision:** [[reject_conditioned_covariance]]

### MVO Synthesis (Abandoned)

Combining Faber conviction scores with Kritzman covariance in MVO required solving a scaling problem: ordinal conviction scores (+1.0/+0.5/-0.5) dwarfed covariance entries (~0.002), making lambda irrelevant. Vol-scaled conviction (μ = S × σ_conditioned × trend_modifier) fixed the scaling, but the underlying covariance matrix's lack of predictive power made the fix academic.

## Key Insight: Why Macro Engines Fail Here

Macro engines redeploy the cash that Faber's trend filter frees up. That cash IS the hedge. Redeploying it into risky assets earns the equity risk premium (higher absolute returns) but at disproportionate volatility and drawdown cost. This is not alpha — it's leveraging up the risk budget that Faber deliberately reduced.

The value gradient across all tests: covariance-only > returns+covariance > returns-only. Discarding return forecasts always helped. But even the covariance-only approach couldn't beat simple trend following with fixed weights, because the conditioned covariance matrix doesn't predict future co-movements.

## Implications for Architecture

1. **Remove macro engine from production** — Harvey-Mulliner adds complexity without improving risk-adjusted returns
2. **Faber trend filter is the sole signal** — binary risk-on/risk-off per asset
3. **Diversification via baseline weights and position limits** — not optimization
4. **Graduated leverage is the return amplifier** — not capital reallocation

## Experiments

- [[2026-04-04_kritzman_vs_harvey]] — Component comparison across four allocation methods
- [[2026-04-04_faber_only_baseline]] — Faber-only baseline reveals macro engines destroy Sharpe
- [[2026-04-04_mvo_ordinal_scaling]] — MVO scaling failure (lambda had zero effect)
- [[2026-04-04_mvo_vol_scaled]] — Vol-scaled fix works mechanically but covariance has no predictive value

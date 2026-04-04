# MVO with Ordinal Conviction Scores (Scaling Failure)

**Date:** April 4, 2026  
**Status:** Complete  
**Track:** Kritzman Research  
**Related:** [[TAA_PROJECT_STATUS]] | [[2026-04-04_faber_only_baseline]]

## Hypothesis

Using Faber trend scores as expected return input (+1.0/+0.5/-0.5) and Kritzman conditioned covariance as risk input in MVO would combine the best of both signals.

## Design

MVO optimizer: maximize w'μ - (λ/2)w'Σw. Conviction scores: 3/3→+1.0, 2/3→+0.5, 0-1/3→-0.5. Lambda sweep from 0.5 to 20.0. Three constraint configurations: unconstrained, 60% cap, 40% cap.

## Results

**Lambda had zero effect.** All 40 lambda values produced identical portfolios at each constraint level.

| Config | Return | Vol | Sharpe | MaxDD |
|--------|--------|-----|--------|-------|
| Unconstrained | 12.3% | 12.7% | 0.964 | -21.1% |
| 60% cap | 12.0% | 10.9% | 1.105 | -17.2% |
| 40% cap | 11.6% | 9.6% | 1.212 | -15.6% |

## Interpretation

Conviction scores (~1.0) were orders of magnitude larger than covariance entries (~0.002). Optimizer ignored covariance entirely, just ranked assets by Faber score and filled to cap. Position caps — not the optimizer — were doing all diversification. Tighter caps improved Sharpe because forced diversification was more reliable than optimizer-driven diversification with noisy covariance.

## Decision

Scaling must be fixed before the MVO synthesis can be properly evaluated. See [[2026-04-04_mvo_vol_scaled]].

## Next Steps

Rescale conviction scores using implied trend Sharpe ratio (S × σ_conditioned × trend_modifier) to match covariance magnitude.

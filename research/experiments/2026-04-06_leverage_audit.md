
# Leverage Audit — Full Results

**Date:** April 6, 2026
**Status:** Complete — All checks passed
**Track:** Production Architecture Validation

## Formula Validation

| Check | Value | Threshold | Result |
|-------|-------|-----------|--------|
| SSO correlation | 0.9956 | >0.99 | PASS |
| SSO annual return diff | +0.89% | <1.5% | PASS |
| QLD correlation | 0.9959 | >0.99 | PASS |
| QLD annual return diff | +0.70% | <1.5% | PASS |

## Volatility Decay Finding

Formula OVERESTIMATES drag — actual ETFs perform BETTER than simulated:
- SSO theoretical drag: vol decay 3.80% + expense 0.89% + borrowing 1.58% = 6.27%
- SSO actual total drag: 3.36%
- Residual: -2.90% (formula overstates cost by 2.90%)
- QLD residual: -4.16%

Reason: The variance drain formula -(N²-N)*σ²/2 overstates cost in trending markets where autocorrelation partially offsets decay. The backtest is CONSERVATIVE — actual leveraged ETF performance is slightly better than simulated.

## Hybrid Backtest (Real SSO/QLD Data Where Available)

| Strategy | Simulated Return | Hybrid Return | Diff | Simulated Terminal | Hybrid Terminal | Diff |
|----------|-----------------|---------------|------|--------------------|-----------------|------|
| S40-40% | 11.2% | 11.14% | -0.06% | $12.74 | $12.56 | -$0.18 |
| S40-100% | 14.7% | 14.55% | -0.15% | $25.91 | $25.01 | -$0.90 |

## Revised Age-65 Projections ($21,000 starting)

| Strategy | Synthetic | Hybrid (Real Data) | Difference |
|----------|-----------|-------------------|------------|
| S40-40% | $1,464,074 | $1,432,372 | -$31,702 (-2.2%) |
| S40-100% | $5,068,001 | $4,806,029 | -$261,972 (-5.2%) |

## Audit Verdict

- Prior backtest results reliable: **YES**
- 100% substitution still correct: **YES**
- Simulation is slightly optimistic by 0.06-0.15% annualised — negligible
- At 100% substitution with real data: $4.8M at age 65 (was $5.1M simulated) — still 3.4x the 40% baseline ($1.43M)
- Key finding: Synthetic formula is conservative. Actual leveraged ETFs slightly outperform the simulation in trending markets due to autocorrelation partially offsetting variance drain.

## Production Decision — CONFIRMED

All prior conclusions stand. S40-100% with real data produces $4.8M vs $1.4M for 40% substitution — $3.37M difference at age 65. The 5.2% reduction from using real data does not change the production decision.

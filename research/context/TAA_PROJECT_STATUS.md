# TAA Project Status

**Last Updated:** April 6, 2026  
**Current Architecture:** Faber-Sweep-40-Daily-Weekly (daily SMAs + weekly circuit breaker + 40% leverage)  
**Active Research Track:** Production implementation

## Current State

The TAA system uses a Faber multi-timeframe trend filter (6/10/12 SMA) as the sole signal for asset eligibility. Baseline weights (IVV 45%, QQQ 25%, VGLT 5%, IAU 10%, DBC 5%, Cash 10%) serve as the neutral allocation. A graduated conviction-based leverage overlay (1.0x/1.25x/1.5x) amplifies returns during confirmed trends.

The Harvey-Mulliner similarity engine was the original capital direction mechanism. Extensive testing of both Harvey and Kritzman relevance-weighted alternatives revealed that no macro engine improves risk-adjusted returns over Faber alone. See [[KRITZMAN_RESEARCH_FINDINGS]] for full details.

## Research Timeline

| Date | Experiment | Key Finding | Decision |
|------|-----------|-------------|----------|
| 2026-04-04 | [[2026-04-04_kritzman_vs_harvey]] | Kritzman-RP best macro variant (1.134 Sharpe) but covariance value marginal | — |
| 2026-04-04 | [[2026-04-04_faber_only_baseline]] | Faber-only 1.164 Sharpe, every macro engine destroys Sharpe vs Faber | [[reject_harvey_expected_returns]], [[reject_kritzman_expected_returns]] |
| 2026-04-04 | [[2026-04-04_mvo_ordinal_scaling]] | Lambda had zero effect — conviction scores dwarfed covariance penalty | — |
| 2026-04-04 | [[2026-04-04_mvo_vol_scaled]] | Vol-scaled fix worked, best Sharpe 1.185 but conditioned cov has -0.049 correlation with realized vol | [[reject_conditioned_covariance]] |
| 2026-04-05 | [[2026-04-05_pro_rata_vs_cash]] | Pro-rata destroys Sharpe by -0.151, doubles max DD. Cash is the hedge. | [[reject_pro_rata_redistribution]] |
| 2026-04-05 | [[2026-04-05_universe_expansion]] | EFA/VNQ/DBA add +0.010 to +0.022 Sharpe — too small to justify complexity. Crisis correlations converge. | — |
| 2026-04-05 | [[2026-04-05_leverage_calibration]] | Graduated-Weekly: 12.1% ret, 0.870 Sharpe, -16.6% DD. Leverage degrades Sharpe (CML drag) but amplifies returns. Daily Sharpe reality: 0.935 vs 1.114 monthly. | — |
| 2026-04-05 | [[2026-04-05_faber_sweep]] | Flat 40% sub: 10.4% ret, 0.878 Sharpe, -17.1% DD, $10.44 terminal. Tradeoff monotonic — each 10% sub costs ~0.016 Sharpe, buys ~0.5% return. Principled target confirmed reasonable. | — |
| 2026-04-05 | [[2026-04-05_vrp_proxy_validation]] | Only XYLD/BXM has usable proxy (corr 0.944). PUT index unavail from yfinance but CSV obtained separately. SVXY ETF-only. QYLD dropped. | — |
| 2026-04-05 | [[2026-04-05_vrp_backtest]] | PUT index standalone: 0.723 Sharpe, -32.7% DD. Combined 80/20 Faber+VRP: 1.058 Sharpe (+0.019). VRP filter >= 0 boosts standalone to 0.860. Marginal improvement, worsens crisis behavior. | — |
| 2026-04-05 | [[2026-04-05_managed_futures_proxy_validation]] | No proxy passes 0.85 threshold (MF too heterogeneous). AQR TSMOM best at 0.632. DBMF 2022: +21.6%, corr -0.586 with IVV — confirmed crisis diversifier. | — |
| 2026-04-05 | [[2026-04-05_two_pod_combined]] | Turbulence layer: 1.333 Sharpe but de-levered 73% of time — essentially "mostly cash." Terminal $7.37 vs Faber-only $10.44. | — |
| 2026-04-05 | [[2026-04-05_two_pod_s40_rerun]] | Threshold 0.80 still de-levers 66%. Both thresholds tested; pods too correlated for any turbulence layer. Best no-turb: S40-20 (1.078 Sharpe, -10.4% DD, $9.38). Faber standalone confirmed as production. | — |
| 2026-04-06 | [[2026-04-06_faber_sweep_weekly_daily]] | Daily SMAs (126/200/252) + weekly circuit breaker: 11.1% ret, 0.946 Sharpe, -16.2% DD, $12.46 terminal. First experiment to improve BOTH Sharpe AND terminal. | — |
| 2026-04-06 | [[2026-04-06_faber_daily_circuit_breaker]] | Daily breaker beats weekly: 0.958 Sharpe (+0.012), -15.0% DD (+1.2%), $12.74 terminal (+$0.28). COVID caught 1 day earlier. Daily re-entry rejected (83% whipsaw). | — |
| 2026-04-06 | [[2026-04-06_leverage_sweep_high]] | Full 2x (100% sub): 14.7% ret, 0.929 Sharpe, -18.1% DD, $25.91 terminal. $21K→$5.1M at age 65. Sharpe cost only -0.030 vs 40%. Daily breaker enables high leverage safely. 3x-50% slightly better ($27.23). | — |

## Key Findings (Cumulative)

1. **Faber trend filter is the dominant value source** — 1.164 Sharpe, -9.6% max DD at 7.4% vol
2. **Macro return forecasts are noise** — Harvey and Kritzman both destroy Sharpe vs Faber-only
3. **Conditioned covariance matrix has no predictive value** — -0.049 correlation with realized portfolio vol
4. **Position caps do more diversification than optimizers** — tighter caps consistently improved Sharpe
5. **Simpler is better** — confirmed across every experiment in the project's history
6. **Pro-rata redistribution destroys Sharpe** — redeploying freed capital (whether via macro engines or pro-rata rules) doubles max DD for marginal return gain
7. **Universe expansion offers negligible improvement** — EFA/VNQ/DBA add +0.010 to +0.022 Sharpe; crisis correlations converge toward 1.0 for all equity-adjacent assets
8. **Leverage degrades Sharpe monotonically** — leveraged ETF drag (vol decay + expenses) costs ~0.12 Sharpe at 2x; leverage is a return amplifier, not a Sharpe amplifier
9. **Daily Sharpe is 16% lower than monthly** — intra-month drawdowns (esp. COVID) invisible at monthly resolution; daily granularity gives 0.935 vs 1.114 monthly
10. **Two-pod turbulence definitively rejected** — threshold 0.65 de-levers 73%, threshold 0.80 de-levers 66%. Pods too correlated for any threshold to selectively target crises. Cash is the mechanism, not crisis detection.
11. **Best two-pod (no turb): S40-20** — 1.078 Sharpe, -10.4% DD, $9.38 terminal. Optional overlay, not the core system.
12. **Daily SMAs improve BOTH Sharpe AND terminal** — first experiment to achieve this. 126/200/252-day SMAs respond faster than monthly; weekly circuit breaker adds +0.019 Sharpe at zero return cost.

## Active Architecture

```
Faber multi-SMA trend filter (6/10/12)
  → Asset eligibility gate (3/3 full, 2/3 partial, 0-1/3 exit)
  → Freed capital to cash
  → Faber-Sweep-40: when both IVV+QQQ at 3/3, replace 40% of each with SSO/QLD (~98% eff equity)
  → Monthly rebalance only
  → Human-in-the-loop approval via Telegram
```

## Open Questions

1. Implement Faber-Sweep-40 in production (`taa/run.py`)
2. Decision: add weekly circuit breaker? (saves ~2% DD, adds complexity)
3. Configure Telegram alerts for monthly rebalance signals
4. Cross-sectional momentum within eligible assets
5. Truly uncorrelated alternatives (managed futures, tail risk hedges) if revisiting universe expansion

## Rejected Approaches

See decisions/ folder for full rationale on each:
- [[reject_harvey_expected_returns]]
- [[reject_kritzman_expected_returns]]
- [[reject_conditioned_covariance]]
- [[reject_pro_rata_redistribution]]
## Production Architecture (Confirmed April 6, 2026)

```
Faber-Sweep-40-Daily-Daily:
  → Daily SMA trend filter (126/200/252-day)
  → Monthly rebalance: 3/3 full weight, 2/3 partial (70%), 0-1/3 exit to cash
  → Freed capital to cash
  → If BOTH IVV and QQQ at 3/3: replace 40% of each with SSO/QLD (~98% eff equity)
  → DAILY circuit breaker: if either IVV or QQQ below ALL 3 daily SMAs → exit leverage next open
  → Re-entry: next monthly rebalance only (no daily re-entry — 83% whipsaw rate)
  → Human-in-the-loop: Telegram for monthly rebalance + daily circuit breaker alerts (~0.7/year)
```

**Performance at 40% sub (risk-tolerance baseline):**
- Return: 11.2% | Vol: 11.7% | Sharpe: 0.958 | MaxDD: -15.0% | Terminal $1: $12.74

**Performance at 100% sub (young accumulator maximum):**
- Return: 14.7% | Vol: 15.8% | Sharpe: 0.929 | MaxDD: -18.1% | Terminal $1: $25.91
- $21K at age 25 → $5.1M at age 65 (40-year projection)
- Sharpe cost vs 40%: only -0.030 | Calmar actually improves (0.81 vs 0.74)

**Substitution schedule by portfolio size (recommended):**
- $21K–$50K: 100% sub (full 2x) — max growth phase
- $50K–$100K: 80% sub
- $100K–$250K: 60% sub
- $250K+: 40% sub (risk-tolerance target, ~98% effective equity)

Circuit breaker fires 0.7x/year (16 events over 24 years) — independent of leverage level.
See [[2026-04-06_leverage_sweep_high]] | [[2026-04-06_faber_daily_circuit_breaker]]


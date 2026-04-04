# TAA Project Status

**Last Updated:** April 4, 2026  
**Current Architecture:** Faber trend filter + baseline weights + graduated leverage  
**Active Research Track:** Kritzman relevance engine evaluation (concluded)

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

## Key Findings (Cumulative)

1. **Faber trend filter is the dominant value source** — 1.164 Sharpe, -9.6% max DD at 7.4% vol
2. **Macro return forecasts are noise** — Harvey and Kritzman both destroy Sharpe vs Faber-only
3. **Conditioned covariance matrix has no predictive value** — -0.049 correlation with realized portfolio vol
4. **Position caps do more diversification than optimizers** — tighter caps consistently improved Sharpe
5. **Simpler is better** — confirmed across every experiment in the project's history

## Active Architecture

```
Faber multi-SMA trend filter (6/10/12)
  → Asset eligibility gate (3/3 full, 2/3 partial, 0-1/3 exit)
  → Freed capital to cash (or pro-rata — needs testing)
  → Graduated leverage overlay (1.0x / 1.25x / 1.5x)
  → Human-in-the-loop approval via Telegram
```

## Open Questions

1. Pro-rata redistribution vs cash for freed capital — [[planned: pro_rata_vs_cash]]
2. Universe expansion (VXUS, more commodities) for more independent trend bets
3. Leverage calibration on simplified Faber-only architecture
4. Cross-sectional momentum within eligible assets

## Rejected Approaches

See decisions/ folder for full rationale on each:
- [[reject_harvey_expected_returns]]
- [[reject_kritzman_expected_returns]]
- [[reject_conditioned_covariance]]

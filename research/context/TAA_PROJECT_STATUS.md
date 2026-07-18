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
| 2026-04-06 | [[2026-04-06_leverage_tiers]] | Mixed-signal states (B+C) occur only 6% of months. All tiered strategies underperform PROD on both Sharpe and terminal. Binary 2x confirmed optimal — tiers add complexity for zero benefit. | — |
| 2026-04-06 | [[2026-04-06_weekly_rebalancing]] | Weekly rebalancing loses $6.22 terminal and -0.058 Sharpe. Faber signal is low-frequency; weekly adds noise not information. 106 leverage switches/24yr vs 48 monthly. Monthly confirmed optimal. | — |
| 2026-04-06 | [[2026-04-06_harvey_spread_backtest]] | Harvey-conditional put spreads during Faber cash periods: 100% win rate on 23 trades, +1.29%/yr on $100K. Harvey filters out catastrophic months (saved $21K Nov 2018 loss). Supplemental pod for portfolios >$50K. | — |
| 2026-04-07 | [[2026-04-07_spreads_unconstrained]] | Standalone: 1.151 Sharpe, 7.14% return, 97% win rate, -0.162 correlation with Faber (goes to -0.82 in crisis). 90/10 combined: 1.169 Sharpe (+0.061), -12.1% DD. Best Pod 2 candidate — genuine structural diversification. | — |
| 2026-04-07 | [[2026-04-07_two_pod_comparison]] | With position management rules: spreads return only 0.8% (0.314 Sharpe). 50% profit target captures small wins but stops allow medium losses. Combined adds +0.001 Sharpe, costs $128K terminal. Pod 2 deferred to $100K+ portfolio. | — |
| 2026-04-07 | [[2026-04-07_pod2_redesign]] | -0.20d/45DTE redesign WORSE: credit still $0.66 (not $1.80 as hypothesized). Avg loss doubled to -$6,201. Standalone Sharpe dropped to 0.141. 5-point spreads fundamentally can't generate enough credit for managed exits. | — |
| 2026-04-07 | [[2026-04-07_iron_condor]] | Iron condor CATASTROPHIC: -0.920 Sharpe, -84.6% DD, $17K terminal from $100K. Call leg at +0.20d gets destroyed during recovery rallies that Harvey signals. 19 call stops avg -$10,452. All managed options Pod 2 designs exhausted and failed. | — |
| 2026-04-07 | [[2026-04-07_adaptive_sma]] | Adaptive lookbacks HURT: -0.057 Sharpe, -$5.96 terminal. 129 regime transitions (50% of months) create whipsaw. Fixed 126/200/252 confirmed optimal — Faber works because it's slow and ignores noise. | — |
| 2026-04-07 | [[2026-04-07_cross_sectional_momentum]] | Momentum tilt: +$0.96 terminal (+3.8%), -0.002 Sharpe (noise). Coin flip on direction (44% helped, 49% hurt) but winners bigger. Deferred to universe expansion — 5 assets too narrow for momentum. |
| 2026-04-10 | [[experiments/V11_BETA_SCALED_RESULTS]] | V11 PASSES all pass criteria. 17.9% CAGR, 0.790 Sharpe, -30.8% DD, $5.25M DCA. Pareto improvement on Baseline AND V9 AND QQQ. Never trails QQQ at any 2013-2026 year-end. Honest cost: 2022 -22.5% (worst year), $2M behind V9 on terminal. | — |
| 2026-04-10 | [[experiments/V12_INDEPENDENT_2X_RESULTS]] | V12 PASSES all criteria. 17.4% CAGR, 0.803 Sharpe, -28.8% DD, $4.80M DCA. Two independent Faber-gated 50/50 sleeves. Matches V11 on return, edges it on Sharpe + MaxDD with 1/8 complexity. V11 retired — V12 dominates it. | — |
| 2026-04-11 | [[experiments/V13_THREE_STATE_RESULTS]] | V13 FAILS all paths. 17.4% CAGR, 0.742 Sharpe, **-42.0% DD** — worse than V9 on every metric. Delever state drag: V13 holds QQQ 1× in months where V9 holds QLD (1.12%/mo drag). Weekly re-entry: 1/18 CB events resolved, nearly useless. CB→cash worse than CB→QQQ. V9 binary design confirmed correct. | [[reject_three_state]], [[reject_weekly_reentry]] |
| 2026-04-12 | [[experiments/V14_DEFENSIVE_ROTATION_RESULTS]] | V14 FAILS all variants. Best is V14-B: 20.4% CAGR, 0.793 Sharpe but **-39.1% DD** (1.2pp worse than V9). Defensive pool earns +0.04%/mo over cash but with 3× vol. Jun-Sep 2022: 100% DBC concentration → -7.50%, -7.04% months. Cash IS the hedge, confirmed for the third time. | [[reject_defensive_rotation]] |
| 2026-04-12 | [[experiments/V9_DCA_REDEPLOYMENT_RESULTS]] | V9-DCA FAILS all step sizes. 56% DCA win rate but GFC kill shot: 9 tranches into -46% IVV decline, exit below avg cost → -6.17% delta vs T-bills. V9-DCA-10 closest (CAGR +0.08pp) but Sharpe -0.001, MaxDD -1.6pp. Cash is the hedge, for the FOURTH time. V9 modifications officially EXHAUSTED. | [[reject_dca_redeployment]] |
| 2026-04-12 | [[experiments/V15_TWO_POD_RESULTS]] | V15 PASSES vs V9. Two-pod (V9 QLD 50% + IVV/SSO 50%): 17.65% CAGR, **0.813 Sharpe** (highest leveraged), -29.0% DD. Pod rebal adds +0.008 Sharpe, +1.8pp DD. V12 displaced from frontier. GFC DD -18.1% (vs V9 -30.6%, V12 -23.8%). | — |
| 2026-04-12 | [[experiments/V16_TWO_POD_GOLD_RESULTS]] | V16 PASSES both variants. 45/45/10 (QLD+SSO+IAU): 17.06% CAGR, **0.846 Sharpe** (ALL-TIME HIGH), -27.0% DD. Gold GFC alpha: +14.8% in 13/17 months. V15 displaced from frontier. Gold correctly excluded by Jun 2022. | — |
| 2026-04-12 | [[experiments/V17_DYNAMIC_REDEPLOYMENT_RESULTS]] | V17 PASSES criteria but does NOT displace V16-B. 17.54% CAGR (+0.48pp), 0.842 Sharpe (-0.004), -27.7% DD (-0.7pp). Redeployment works (+0.14%/mo excess) but adds vol faster than return. V16-B remains frontier balanced point. | — |
| 2026-04-12 | [[experiments/V18_DRAWDOWN_PROTECTION_RESULTS]] | V18 FAILS all mechanisms. Leading indicators: 0/15 pass validation (best VIX slope 27% hit / 73% FP). Portfolio CB: PCB-10 fires 209 times, drops CAGR to 11.4%. -27% MaxDD is STRUCTURAL FLOOR for 180% eff equity. Per-asset CB correctly calibrated. No free DD protection exists. | [[reject_portfolio_cb]], [[reject_leading_indicators]] |
| 2026-04-13 | [[experiments/V18B_INTRAMONTH_CB_RESULTS]] | V18b FAILS. IMC-20/25 never fire (0 events in 24yr). V16-B's -27% DD is multi-month from UNLEVERED equity after per-asset CBs already stripped leverage. Intra-month CB solves wrong problem — no leverage left to strip. DD protection thesis DEFINITIVELY CLOSED. | [[reject_intramonth_cb]] |
| 2026-04-13 | [[experiments/V19_CB_CASH_EXIT_RESULTS]] | **V19 DOMINATES V16-B on EVERY metric.** CB→cash: 17.29% CAGR (+0.23), **0.867 Sharpe** (+0.021), -25.1% DD (+1.9pp), $54.75 term (+$2.57). Post-CB analysis: equity -5.49% cumul vs cash +0.80%. Cash wins 14/27, equity wins 13/27. V16-B displaced. | — |
| 2026-04-13 | [[experiments/V19B_NO_GOLD_RESULTS]] | V19b (no gold, 50/50 CB→cash): 17.88% CAGR, 0.834 Sharpe, -27.0% DD. Gold earns its 10%: V19 beats V19b on Sharpe (+0.033) and MaxDD (+1.9pp). CB→cash improvement confirmed independent of gold (V19b vs V15: +0.020 Sharpe, +2.0pp DD). V19 remains frontier. | — |
| 2026-04-13 | [[experiments/V19C_FULL_UNLEVER_RESULTS]] | V19c (100% unlev at sc2): WASH. 17.41% CAGR (+0.12pp), 0.865 Sharpe (-0.002), -25.1% DD (same). Keep V19's 70/30 — marginally better Sharpe and crisis DDs. Score 2/3 occurs 14% of months. | — |
| 2026-04-13 | [[experiments/V19D_GOLD_CB_RESULTS]] | V19d (gold CB): WASH. -0.001 Sharpe, -0.02pp CAGR, same DD. Gold CB fires 10× in 24yr. **ADOPTED for design consistency** — all risk assets now have 3/3 SMA breach → cash CB. V19d is the FINAL production spec. | — |
| 2026-04-13 | [[experiments/V20_DIRECTIONAL_TRANSITIONS_RESULTS]] | V20 FAILS all variants. Directional hypothesis INVERTED: 3→2 recovers 57% (not declining), 1→2 never reaches 3 (0% recovery). Score-2 direction is noise. V19d's non-directional 70/30 confirmed optimal. | [[reject_directional_transitions]] | — |

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


---

## Leverage Philosophy — Permanent Decision

**Date confirmed:** April 6, 2026

**Investor stance:** Thinks in percentages, not dollar amounts. Large nominal drawdowns are acceptable provided the percentage drawdown is within system parameters. This is the correct framing for a 25-year-old 40-year accumulator.

**Decision:** Run 100% SSO/QLD substitution indefinitely unless a specific, pre-defined condition is met. The Pedersen lifecycle schedule is a reference framework, not a mandate.

**Pre-defined conditions to reduce substitution (decided IN ADVANCE, not reactively):**
1. Portfolio exceeds $500K AND investor has meaningful near-term liquidity needs from this account
2. A sustained multi-year period where the circuit breaker fires more than 4 times in a 12-month window (indicates the trend filter is not working as designed in the current regime)
3. Investor's income situation changes such that the Roth IRA represents the primary emergency reserve (should never happen by design — keep separate emergency fund)

**NOT valid reasons to reduce substitution:**
- Market is down and the drawdown feels large in dollar terms
- Media/commentators are bearish
- The system has underperformed IVV for 1-2 years (expected periodically)
- General anxiety about leverage

**The math justification (locked in):**
- 100% sub at 14.7% vs 40% sub at 11.2% = 3.5% annual return difference
- Over 40 years from $21K: $5.1M vs $1.5M — $3.6M difference
- MaxDD difference: -18.1% vs -15.0% — only 3.1% worse
- Sharpe difference: 0.929 vs 0.958 — only 3% worse
- The tradeoff is unambiguously correct for a percentage-focused 40-year accumulator

---

## Final Production Architecture — Confirmed April 6, 2026

All research complete. All experiments run. All decisions locked.

### System Specification

```
Faber-Sweep-40-Daily-Daily
  Signal:     Daily SMAs (126/200/252-day) on IVV, QQQ, VGLT, IAU, DBC
  Allocation: Monthly rebalance
              3/3 → full baseline weight
              2/3 → 70% baseline weight, 30% to cash
              0-1/3 → full weight to cash
  Weights:    IVV 45%, QQQ 25%, VGLT 5%, IAU 10%, DBC 5%, Cash 10%
  Freed cap:  To cash
  Leverage:   BOTH IVV and QQQ at 3/3 → 100% SSO/QLD substitution
              Either below 3/3 → 1x, no substitution
              Binary switch — no tiers, no intermediate states
  Circuit:    Daily check — if either IVV or QQQ closes below ALL 3 daily SMAs
              → exit leverage next open (SSO→IVV, QLD→QQQ)
              → re-entry at next monthly rebalance only
  Execution:  Human-in-the-loop via Telegram
              Monthly rebalance approval + circuit breaker alerts (~0.7/year)
```

### Validated Performance (Hybrid Real ETF Data)

| Metric | Value |
|--------|-------|
| Annualised return | 14.55% |
| Volatility | 15.8% |
| Sharpe ratio | 0.929 |
| Sortino ratio | 1.120 |
| Max drawdown | -18.1% |
| Calmar ratio | 0.81 |
| Terminal $1 (2002-2026) | $25.01 |
| $21K at age 65 (40yr) | $4,806,029 |
| Leveraged months | ~66% |
| Circuit breaker events | 16 over 24yr (0.7/yr) |

### Key Decisions and Rationale

**Leverage level: 100% SSO/QLD substitution**
Justified by Ayres & Nalebuff (2010) Lifecycle Investing and Frazzini & Pedersen (2014) Betting Against Beta. Age 22-28 recommended leverage: 2.0x. 100% substitution produces ~140% effective equity exposure. Confirmed by audit against real SSO/QLD data — simulation accurate within 0.15% annualised.

**No leverage tiers**
Tested per-ETF independent tiers (50% sub when one ETF at 3/3, other at 2/3). Mixed signal states B+C occur only 15 months out of 259 (6%). Best tiered result added $19K at age 65 while adding 50% more circuit breaker events and significant implementation complexity. Binary switch is simpler and better.

**Daily SMAs over monthly**
126/200/252-day daily SMAs vs 6/10/12-month monthly SMAs. Daily SMAs catch trend changes up to 21 trading days faster. +0.049 Sharpe, +$1.94 terminal over 24 years.

**Daily circuit breaker over weekly**
Daily check vs Friday-only. Caught COVID 1 day earlier (Feb 27 vs Feb 28). +0.012 Sharpe, +$0.28 terminal, +1.2% max DD improvement. 16 events vs 14 weekly (negligible increase).

**Re-entry monthly only**
Daily re-entry tested and rejected. 83% of re-entries were rapid whipsaw cycles (exit Monday, re-enter Wednesday). Impractical for human-in-the-loop system.

**Freed capital to cash**
Pro-rata redistribution tested and rejected — doubles max DD for marginal return. Cash IS the hedge.

**No macro engine (Harvey-Mulliner / Kritzman)**
All macro engines destroy Sharpe relative to Faber-only. Faber trend filter is the dominant source of value. Macro engines add absolute return by deploying more capital but at disproportionate volatility cost.

**No multi-pod**
VRP (PUT index): 0.533 Faber correlation — too equity-adjacent. Turbulence layer: de-levers 66-73% of months regardless of threshold — fundamental correlation problem. BTAL: negative long-term returns, deferred. ILS: ETF too new, liquidity concerns. DBMF: insufficient proxy history. All deferred pending portfolio growth and/or better data.

### Experiment Archive (Complete)

| File | Finding |
|------|---------|
| 2026-04-04_kritzman_vs_harvey | Kritzman-RP: 1.134 Sharpe, marginal covariance value |
| 2026-04-04_faber_only_baseline | Faber-only: 1.164 Sharpe, every macro engine destroys Sharpe |
| 2026-04-05_pro_rata_vs_cash | Pro-rata doubles max DD. Cash is the hedge. |
| 2026-04-05_universe_expansion | EFA/VNQ/DBA: +0.010-0.022 Sharpe only. Crisis correlations converge. |
| 2026-04-05_leverage_calibration | Daily Sharpe 16% lower than monthly — intra-month drawdowns real |
| 2026-04-05_faber_sweep | 40% sub: 0.878 Sharpe, $10.44 terminal |
| 2026-04-05_vrp_backtest | PUT index: 0.723 Sharpe, 0.533 Faber correlation |
| 2026-04-05_two_pod_combined | Turbulence fires 73% of months at 0.65 threshold |
| 2026-04-05_two_pod_s40_rerun | 0.80 threshold still fires 66% — fundamental problem |
| 2026-04-06_faber_sweep_weekly_daily | Daily SMAs dominant: +$1.94 terminal, +0.049 Sharpe |
| 2026-04-06_faber_daily_circuit_breaker | Daily CB: 0.958 Sharpe, $12.74 terminal, -15.0% MaxDD |
| 2026-04-06_leverage_sweep_high | Terminal wealth peaks monotonically through full 2x range |
| 2026-04-06_pedersen_leverage_validation | 100% sub confirmed: $4.8M at 65 from $21K |
| 2026-04-06_leverage_audit | Formula validated. Simulation accurate within 0.15% annualised. Conservative. |
| 2026-04-06_leverage_tiers | Mixed states occur 6% of months. Tiers add no value. Binary switch optimal. |

### Implementation Roadmap

1. Transfer Roth IRA from J.P. Morgan → Schwab (visit branch, wet ink signature)
2. Apply for Schwab Developer API at developer.schwab.com
3. Implement production system in taa/run.py
4. Configure Telegram bot for monthly approvals + daily circuit breaker alerts
5. Phase 1: Observation mode 3 months (no trades)
6. Phase 2: Paper trading 3 months (Schwab paper account)
7. Phase 3: Live at 100% SSO/QLD substitution

### Lifecycle Delevering Schedule (Pedersen — not backtest-optimized)

| Age | Substitution | Eff Equity |
|-----|-------------|------------|
| 25-29 | 100% | ~140% |
| 30-34 | 80% | ~126% |
| 35-44 | 60% | ~112% |
| 45-54 | 40% | ~98% |
| 55+ | 0% | ~70% |

Reduction triggers (pre-committed, not reactive):
- Age milestone OR portfolio crosses $50K (whichever comes first for next reduction)
- NOT valid: market down, media bearish, short-term underperformance

---

## Options Pod Research — Concluded April 7, 2026

### Summary of All Tests

| Design | Sharpe | Return | Verdict |
|--------|--------|--------|---------|
| Unconstrained put spreads (held to expiry) | 1.151 | 7.1% | Works but impractical live |
| Put spread -0.10d/30DTE + position management | 0.314 | 0.8% | Marginal — management kills return |
| Put spread -0.20d/45DTE + position management | 0.141 | 0.8% | Worse — wider stop allows larger losses |
| Iron condor -0.10d put / +0.20d call | -0.920 | -11.5% | Catastrophic — call side contradicts Harvey recovery signal |

### Root Cause of Failure

The VRP from selling spreads is real. Harvey filtering is real (-0.162 correlation with Faber). The failure is structural: 5-point vertical spreads collect $0.65-0.70 net credit. At 50% profit target that banks $0.33. One stop-out at $3,000-6,000 wipes 9-18 winning trades. Position management cannot make the economics work at this spread width.

The only viable version is unconstrained (hold to expiry, accept occasional large loss). This requires sizing positions so max loss per trade < 2% of portfolio.

### Viability Threshold

| Portfolio | Contracts | Annual Income | Max Loss Event | Viable? |
|-----------|-----------|--------------|---------------|---------|
| $21K | 3 | $312 | $1,290 | No |
| $50K | 7 | $728 | $3,010 | Marginal |
| $100K | 16 | $1,618 | $6,880 | Getting there |
| $150K | 24 | $2,426 | $10,320 | YES — threshold |
| $200K+ | 31+ | $3,134+ | $13,330+ | Comfortable |

**Pod 2 activation trigger: portfolio reaches $150,000**

### Pod 2 Production Spec (when activated at $150K+)

```
Underlying: SPY
Signal: Harvey ER > +0.005 AND VIX > 18
Strike: -0.10 delta short put
Width: 5 points
DTE: 30 days
Management: HOLD TO EXPIRY (no stops, no profit targets)
Sizing: max 2% portfolio per contract max loss
  ($150K × 0.02 / $430 = 7 contracts minimum)
Max position: 24 contracts at $150K
Re-entry: monthly, Harvey signal required
```

Milestone plan:
- $50K: Paper trade 1-2 contracts to learn execution
- $100K: Run 1-2 live contracts ($800-1,600/yr — learning phase)
- $150K: Scale to full sizing, pod earns its keep


---

## Architecture Update — April 2026

### Pedersen Lifecycle: DROPPED
100% SSO/QLD substitution (2x leverage, ~140% effective equity) throughout the entire
investment lifetime. The circuit breaker (daily 3/3 SMA breach → exit leverage next open)
is the drawdown management mechanism. No portfolio-size or age-triggered delevering.

Rationale: lifecycle delevering is voluntary terminal wealth destruction. Circuit breaker
limits max drawdown to -18.1% historically. The two systems serve different functions
and should not be conflated.

### Signal-Off Capital: DBMF 50/50 with T-bills (pending backtest validation)
During equity signal-off periods (IVV or QQQ score ≤ 1), freed equity weight splits:
- 50% → DBMF (iMGP DBi Managed Futures Strategy ETF)
- 50% → T-bills

DBMF has no Faber signal, no circuit breaker, no independent treatment. It is purely
a passive cash substitute. When equity signal restores (both IVV + QQQ at 3/3), DBMF
allocation reverts to zero and full SSO/QLD substitution activates.

Architecture rationale: DBMF trend-follows across asset classes including shorting equity.
In 2022, DBMF +20% while IVV -18% and VGLT -28%. The equity Faber signal identifies
exactly the periods when managed futures crisis alpha is most likely to be active.

Status: awaiting Claude Code backtest (DBMF_CASH_SUBSTITUTE_RESULTS.md).
Key test: does 2022 max drawdown worsen? If not, architecture is adopted.


---

## April 9, 2026 — Architecture Finalized

### Pedersen Lifecycle: DROPPED
100% SSO/QLD substitution throughout entire investment lifetime. No delevering at portfolio milestones. Circuit breaker is the sole drawdown mechanism.

### Signal-Off Capital: DBMF 50/50 ADOPTED
Freed equity weight during signal-off periods → 50% DBMF / 50% T-bills.
Validated by 2022 actual data: +3.1% improvement, max DD unchanged.
Full-period backtest inflated by bad proxy (0.249 correlation). Realistic improvement: Sharpe +0.02-0.03.
See [[experiments/DBMF_CASH_SUBSTITUTE_RESULTS]] for full detail.

### Current Production Spec (v5)
- Faber 126/200/252-day daily SMAs
- Score 3/3 → full baseline weight (eligible for leverage)
- Score 2/3 → 70% baseline, 30% freed
- Score 0-1 → 0%, full baseline freed
- Freed equity weight → 50% DBMF / 50% T-bills
- Freed non-equity weight (VGLT/IAU/DBC) → T-bills (unchanged)
- Both IVV + QQQ at 3/3 → 100% SSO/QLD substitution (no lifecycle cap)
- Daily circuit breaker: 3/3 SMA breach → exit leverage next open


---

## April 9, 2026 — Bull Market Survivability Test Complete

See [[experiments/BULL_MARKET_SURVIVABILITY]] for full detail.

**Key honest findings:**
- 14.55% CAGR is NOT a dot-com artifact. Faber CAGR from any start date: 13.9%-21.5%
- Faber Sharpe beats QQQ Sharpe from EVERY start date tested (2002, 2004, 2007, 2010, 2013, 2019)
- During 2013-2021 bull: Faber 20.7% vs QQQ 23.4% — QQQ wins raw return, Faber wins Sharpe
- Maximum DCA dollar gap: -$61,942 at end of 2020 (closed by 2022 bear)
- Longest 12m underperformance streak: 26 consecutive months (Oct 2014 – Nov 2016)
- QQQ beats Faber's trailing 12m in 61% of bull market months — investor must accept this
- The 2020 CB save paid for every whipsaw event in the system's history combined


---

## April 9, 2026 — Equity Sleeve Tilt Under Research

**Idea:** Dynamic IVV/QQQ split within fixed 70% equity sleeve based on QQQ/IVV ratio vs 200-day SMA.

- QQQ in relative uptrend → tilt QQQ (30% IVV / 40% QQQ)
- QQQ in relative downtrend → tilt IVV (55% IVV / 15% QQQ)
- Total equity always 70%, total effective leverage always 140% when signal on

A priori signal: QQQ/IVV ratio vs its own 200-day SMA — zero new parameters beyond tilt magnitude.

**Critical test:** 1999-2000. QQQ was in relative uptrend throughout; this would have tilted toward QQQ right before the -83% crash. Must confirm Faber signal exits equity fast enough that the tilt doesn't materially worsen dot-com performance.

Prompt: EQUITY_SLEEVE_TILT_PROMPT.md | Results pending Claude Code run.


---

## April 9, 2026 — Equity Sleeve Tilt: REJECTED

See [[experiments/EQUITY_SLEEVE_TILT_RESULTS]] for full detail.

Dot-com stress test failed. Max DD worsened -6.9pp (from -27.5% to -34.4%). QQQ/IVV ratio crossed below SMA on May 10, 2000 — too late. April 2000 already cost -15.3% vs baseline -11.6%.

Full-period: tilt adds +0.3% CAGR but costs -0.015 Sharpe and -0.2% MaxDD. Trades risk-adjusted return for raw return — contradicts system design philosophy. Fixed 45/25 equity split confirmed as optimal.


---

## April 9, 2026 — Architecture Rethink: Terminal Wealth Optimization

**Core question:** Are VGLT/IAU/DBC earning their place in the portfolio, or is the circuit breaker doing all the real defensive work and the defensive assets just diluting equity compounding?

**New direction:** Test simplified high-conviction architectures that remove defensive assets entirely, maximizing equity exposure during signal-on periods with circuit breaker as sole risk management. Must beat QQQ buy-and-hold on CAGR from any start date while maintaining materially better drawdown than naked QQQ.

Variants being tested:
1. QLD only + circuit breaker (single asset, max conviction)
2. QQQ/IVV + Faber signal + cash only (no defensive assets)
3. QLD/SSO split + Faber + cash only
4. Full equity sleeve (IVV+QQQ) at varying splits + cash only
5. QQQ-only universe: Faber on QQQ → QLD when 3/3, QQQ when 1-2/3, cash when 0/3

Prompt: TERMINAL_WEALTH_OPTIMIZATION_PROMPT.md | Results pending.


---

## April 9, 2026 — Terminal Wealth Optimization: Complete

See experiments/TERMINAL_WEALTH_OPTIMIZATION.md for full detail.

**Core finding: no variant passes all four criteria simultaneously.**
- Variants beating QQQ CAGR from 2013 (V1, V9) have max DD > -30%
- Variants with max DD < -30% (Baseline, V3-V5) don't beat QQQ from 2013
- Defensive assets confirmed net positive: Baseline $25.62 beats no-defense $20.21

**The honest tradeoff:**
| Goal | Best variant | Cost |
|------|-------------|------|
| Maximum terminal wealth | V9 QLD+IVVguard | -37.9% max DD, $7.37M DCA |
| Maximum Sharpe | Baseline | 17.2% CAGR from 2013 (trails QQQ 18.9%) |
| Balance | ??? | Need a new variant |

**V9 key numbers:** 19.4% CAGR, 0.777 Sharpe, -37.9% DD, $85.25 terminal, $7.37M DCA
**Baseline key numbers:** 13.8% CAGR, 0.914 Sharpe, -18.1% DD, $25.62 terminal, $2.40M DCA

**Strategic decision pending:** The gap between V9 and Baseline is enormous on terminal wealth ($7.37M vs $2.40M DCA). The -37.9% DD is the price. At age 25 with a $21K account and 40-year horizon, the question is whether that DD is acceptable given the lifecycle context.


---

## April 9, 2026 — V10 Dynamic State Architecture Under Research

**Architecture:** No fixed baseline weights. State machine driven entirely by Faber scores on 5 assets.

State A (min score=3, both 3/3): 100% QLD — V9 full conviction
State B (min score=2, one 2/3): 70% QLD + 30% QQQ — partial delever
State C (min score=1, one 1/3): 30% QQQ + 70% defensive — NEW fast re-entry
State D (min score=0, one 0/3): 100% defensive — full exit

Defensive assets: DBMF (unconditional) + VGLT/IAU (Faber-conditioned on their own scores).
Key innovation: State C allows re-entry at 1/3 signal strength (currently mapped to "off").
Key test: 2022 — does Faber-conditioned VGLT correctly exclude VGLT when it's off-signal?

Prompt: V10_DYNAMIC_STATE_PROMPT.md | Results pending Claude Code run.


---

## April 9, 2026 — Beta-Scaled Signal² Composition Formula

**The core allocation formula for IVV/QQQ split within the equity sleeve:**

```
raw_IVV = IVV_score²
raw_QQQ = QQQ_score² × 1.5^(QQQ_score - 2)

w_IVV = raw_IVV / (raw_IVV + raw_QQQ)
w_QQQ = raw_QQQ / (raw_IVV + raw_QQQ)
```

The exponent (score-2) scales the beta effect by confidence level:
- Score 3 → power +1: beta-seeking (amplify QQQ's higher beta in bull markets)
- Score 2 → power  0: beta-neutral (ignore beta at middling signals)
- Score 1 → power -1: beta-averse (penalize high beta in weak signal environments)

Zero free parameters. Beta values (QQQ≈1.5, IVV≈1.0) are a priori from published research.

Key outputs:
- Both 3/3: 60% QQQ / 40% IVV (beta tilt confirmed bull)
- Both 2/3: 50% / 50% (neutral)
- Both 1/3: 40% QQQ / 60% IVV (beta aversive)
- QQQ weak (1) + IVV strong (3): 7% QQQ / 93% IVV (double penalty on QQQ)

This is the COMPOSITION layer only. Still need: (1) total equity % at each min-score level, (2) leverage overlay (QLD/SSO vs QQQ/IVV).


---

## April 9, 2026 — V11 Architecture: Beta-Scaled Dynamic State System

**Core innovation:** No fixed baseline weights. Portfolio state determined by sum of IVV + QQQ Faber scores. Beta-scaled composition formula tilts toward QQQ at high conviction, toward IVV at low conviction.

**Equity caps by sum score:**
- Sum 6: 100% equity, 2× leverage (both SSO+QLD)
- Sum 5: 100% equity, 2× leverage
- Sum 4: 70% equity, 1× unleveraged
- Sum 3: 30% equity, 1×
- Sum 2: 10% equity, 1×
- Sum 1-0: 0% equity, 100% defensive

**Composition formula (beta-scaled signal²):**
```
raw_IVV = IVV_score²
raw_QQQ = QQQ_score² × 1.5^(QQQ_score - 2)
w_QQQ = raw_QQQ / (raw_IVV + raw_QQQ)
```
- Score 3: beta-seeking (1.5^+1 = amplify QQQ)
- Score 2: beta-neutral (1.5^0 = 1)
- Score 1: beta-averse (1.5^-1 = penalize QQQ)

**Key allocation outputs:**
- Both 3/3: 40% SSO + 60% QLD (beta tilt QQQ at full conviction)
- Both 2/2: 35% IVV + 35% QQQ (neutral)
- 3+1 or 1+3: 65% in strong asset, 35% defensive (7% redirected from weak asset to defensives)

**Defensive pool:** DBMF (unconditional) + VGLT/IAU/DBC (Faber-conditioned, own score ≥ 2)
**Circuit breaker:** identical to V9

Status: prompt pending confirmation of final table.


**CORRECTED — Sum 5 leverage is asset-specific:**
- IVV=3, QQQ=2: 69% SSO (2×) + 31% QQQ (1×)
- IVV=2, QQQ=3: 23% IVV (1×) + 77% QLD (2×)
Only the asset at 3/3 gets leverage. The 2/3 asset held unleveraged.


---

## April 9, 2026 — V11 Prompt Written, Pending Claude Code Run

Prompt: V11_BETA_SCALED_PROMPT.md

Pass criteria: V11 must simultaneously improve on BOTH Baseline AND V9.
- vs Baseline: higher CAGR from 2013, smaller DCA gap vs QQQ
- vs V9: lower max DD, higher Sharpe
- vs QQQ: beat CAGR from 2013 start (18.9%), beat max DD (-53.4%)

If V11 achieves max DD between -18% and -38% while beating QQQ CAGR from 2013,
it represents a genuine Pareto improvement — better terminal wealth than Baseline,
better risk-adjusted returns than V9.


---

## April 10, 2026 — V11 Backtest Complete: PARETO IMPROVEMENT CONFIRMED

See [[experiments/V11_BETA_SCALED_RESULTS]] for full detail.

**V11 passes ALL pass criteria simultaneously.**

| Metric | V11 | Baseline | V9 | QQQ B&H |
|---|---|---|---|---|
| CAGR (full)       | 17.90% | 13.79% | 19.37% | 12.57% |
| Sharpe            | 0.790  | 0.910  | 0.777  | 0.634  |
| Max DD            | -30.8% | -18.5% | -37.9% | -53.4% |
| Terminal $1       | $62.46 | $25.58 | $85.25 | $17.58 |
| DCA $700/mo lifetime | $5.25M | $2.40M | $7.37M | $2.33M |
| CAGR from 2013    | 23.98% | 17.19% | 28.51% | 18.94% |
| Peak DCA gap vs QQQ | $0K  | -$65K  | (best) | (base) |

**Headline result:** V11 never trails QQQ at any year-end of the 2013-2026 DCA path. Closes Baseline's -$65K gap entirely.

**Honest weaknesses to flag:**
- 2022 was V11's worst year (-22.5%) — worse than Baseline (-9.9%) and V9 (-15.2%). Jan 2022 alone -14.66% from sum=6 leverage entering a top.
- V11 buys -7.1pp lower max DD vs V9 at the cost of ~$2M in lifetime DCA terminal.
- Per-asset CB fires more often (25 events vs Baseline 16, V9 14). ~1/year is fine.

**Beta tilt validates exactly:** Sum=6 months avg 60% QQQ / 40% IVV. Sum=2 months avg 7.3% IVV / 2.7% QQQ. Composition formula working as designed.

**State occupancy (291 months):** Sum=6 dominates at 65.6%. Intermediate states (4-5) only ~10%. System is mostly binary "full conviction" or "fully defensive".

**Strategic implication:** V11 is now the leading candidate for production. The decision between V11 and V9 reduces to: do you accept -7pp deeper drawdowns (V9) for an extra $2M of lifetime DCA wealth? V11 is the more defensible choice for a 40-year accumulator under both percentage- and dollar-DCA framings.


---

## April 10, 2026 — V12 Independent 2× Test: V11 RETIRED

See [[experiments/V12_INDEPENDENT_2X_RESULTS]] for full detail.

**V12 strips V11 to its essence:** two independent 50/50 sleeves on IVV and QQQ, each Faber-gated, each leveraged to 2× when at 3/3. No defensive pool, no beta formula, no graduated cap table. Two binary switches + cash.

**V12 passes all criteria and makes V11 structurally redundant:**

| Metric | V12 | V11 | V9 | Baseline |
|---|---|---|---|---|
| CAGR (full)       | 17.41% | 17.90% | 19.37% | 13.79% |
| Sharpe            | **0.803**  | 0.790  | 0.777  | 0.910  |
| Max DD            | **-28.8%** | -30.8% | -37.9% | -18.5% |
| Terminal $1       | $56.18 | $62.46 | $85.25 | $25.58 |
| DCA $700/mo       | $4.80M | $5.25M | $7.37M | $2.40M |
| CAGR from 2013    | 23.45% | 23.98% | 28.51% | 17.19% |
| Moving parts      | 2 switches | 16-row table | 2 thresholds | 12 sub-rules |

**V12 dominates V11 on Sharpe (+0.013), MaxDD (+2.0pp), and simplicity.** V11's -$120K DCA advantage is noise-level. V11 can be retired from the frontier.

**Signal divergence analysis (the thing V12 was designed to test):**
- "Both 3/3" (65.9% of months): V12 earns +0.58%/mo vs Baseline — this is where the leverage advantage lives
- "IVV 3, QQQ<3" (4.1%): V12 earns +0.34%/mo — independent gating wins
- "QQQ 3, IVV<3" (5.9%): V12 loses -0.30%/mo — minor asymmetric drag
- "Neither 3/3" (24.1%): V12 basically matches Baseline

Net-net independent gating is approximately neutral in the divergence states. V12's performance edge over Baseline comes almost entirely from the 200% effective equity in the "Both 3/3" state.

**Pareto frontier as of 2026-04-10:**
- **V9** — max wealth ($7.4M DCA), deepest DD (-38%)
- **V12** — balanced ($4.8M DCA, -29% DD), simplest architecture
- **Baseline** — max Sharpe (0.91), shallowest DD (-18%), lowest terminal

V11 is off the frontier. The remaining choice is V9 vs V12 vs Baseline on the risk/reward curve.

**V12's structural weakness:** 200% eff equity two-thirds of the time means monthly-rebalance exposure to peak drawdowns. Jan 2022 took -13.99% before the CB could fire. COVID -28.8% roughly matches QQQ B&H (-28.6%) — the system limited further pain but the initial blow was unprotected.


---

## April 11-12, 2026 — V13 and V14: V9 Modification Attempts EXHAUSTED

See [[experiments/V13_THREE_STATE_RESULTS]] and [[experiments/V14_DEFENSIVE_ROTATION_RESULTS]].

**V13 (three-state + weekly re-entry):** Failed by modifying V9's OFFENSE. Adding intermediate states (QQQ 1×), tightening the IVV guard (score 2 → delever), and weekly re-entry ALL made things worse. MaxDD WORSENED to -42.0%. The 22 delever months averaged -1.12%/mo vs V9. Weekly re-entry resolved 1/18 CB events — essentially useless. V9's binary QLD/cash with loose IVV guard is the correct offense design.

**V14 (defensive rotation during cash):** Failed by modifying V9's DEFENSE. Routing freed capital to Faber-gated IVV/VGLT/IAU/DBC earned +0.04%/mo over T-bills but with 3× the volatility. Jun-Sep 2022: only DBC survived its Faber gate, creating 100% single-asset concentration into crashing commodities (-7.50%, -7.04% months). V14-B was the closest to passing (+1pp CAGR, +0.016 Sharpe, $106 terminal) but MaxDD worsened 1.2pp.

**Conclusion: V9 is terminal.** Every modification to V9 — offense or defense — has been tested and failed. The Pareto frontier is locked:

| Point | Strategy | CAGR | Sharpe | MaxDD | DCA |
|---|---|---|---|---|---|
| Max wealth | **V9** | 19.4% | 0.777 | -37.9% | $7.37M |
| Balanced | **V12** | 17.4% | 0.803 | -28.8% | $4.80M |
| Max Sharpe | **Baseline** | 13.8% | 0.910 | -18.5% | $2.40M |

No further V9 modifications pending.


---

## April 12, 2026 — V15 Two-Pod: NEW FRONTIER POINT

See [[experiments/V15_TWO_POD_RESULTS]] for full detail.

V15 runs V9 unchanged as Pod 1 (50%) and adds IVV/SSO V9-logic as Pod 2 (50%).
Monthly pod rebalancing to 50/50 if drift > 5%. Pod isolation + rebalancing is
the key structural innovation.

| Metric | V15 (rebal) | V9 | V12 | Baseline |
|---|---|---|---|---|
| CAGR | 17.65% | 19.37% | 17.41% | 13.79% |
| Sharpe | **0.813** | 0.777 | 0.803 | 0.910 |
| MaxDD | **-29.0%** | -37.9% | -28.8% | -18.5% |
| DCA | $4.92M | $7.37M | $4.80M | $2.40M |

**V15 has the highest Sharpe of any leveraged variant (0.813).** Pod rebalancing
adds +0.008 Sharpe and +1.8pp MaxDD at -0.07pp CAGR cost. V12 displaced from frontier.

**GFC standout: -18.1% DD** (vs V9 -30.6%, V12 -23.8%). Pod rebalancing + independent
delevering spread the GFC pain across two sequential exits.

**Updated Pareto frontier:**

| Point | Strategy | CAGR | Sharpe | MaxDD | DCA |
|---|---|---|---|---|---|
| Max wealth | **V9** | 19.4% | 0.777 | -37.9% | $7.37M |
| Balanced | **V15** | 17.7% | 0.813 | -29.0% | $4.92M |
| Max Sharpe | **Baseline** | 13.8% | 0.910 | -18.5% | $2.40M |

V12 no longer on frontier — V15 dominates on Sharpe while matching MaxDD and CAGR.


---

## April 12, 2026 — V16 Two-Pod + Gold: ALL-TIME SHARPE HIGH

See [[experiments/V16_TWO_POD_GOLD_RESULTS]] for full detail.

V16-B adds 10% Faber-gated gold (IAU ≥ 3) to V15's two-pod architecture:

| Metric | V16-B | V15 | V9 | Baseline |
|---|---|---|---|---|
| CAGR | 17.06% | 17.65% | 19.37% | 13.79% |
| Sharpe | **0.846** | 0.813 | 0.777 | 0.910 |
| MaxDD | **-27.0%** | -29.0% | -37.9% | -18.5% |
| DCA | $4.41M | $4.92M | $7.37M | $2.40M |
| GFC DD | **-16.2%** | -18.1% | -30.6% | -9.0% |

**Gold's crisis alpha is real and Faber-gated:**
- GFC: 13/17 months active, +14.8% cumulative while equities -53%
- COVID: 3/3 months active, +6.3%
- 2022: correctly excluded by June (score dropped to 0)
- Dot-com: inactive (gold bear market) — no harm, no help

**V16-B's vol (21.33%) is the lowest of any leveraged variant.** Gold's low equity
correlation damps portfolio vol by 2pp vs V15. This vol reduction drives the Sharpe.

**Updated Pareto frontier (FINAL):**

| Point | Strategy | CAGR | Sharpe | MaxDD | DCA |
|---|---|---|---|---|---|
| Max wealth | **V9** | 19.4% | 0.777 | -37.9% | $7.37M |
| Balanced | **V16-B** | 17.1% | 0.846 | -27.0% | $4.41M |
| Max Sharpe | **Baseline** | 13.8% | 0.910 | -18.5% | $2.40M |

V15 and V12 displaced. V11, V13, V14, V9-DCA all previously failed. The frontier
has three clean points. V16-B is now only 0.064 Sharpe below Baseline while
delivering +3.27pp more CAGR and +$2.01M more DCA.


---

## April 13, 2026 — V19: CB→Cash DOMINATES V16-B

See [[experiments/V19_CB_CASH_EXIT_RESULTS]] for full detail.

**V19 is a strict Pareto improvement over V16-B — better on every metric simultaneously.**

| Metric | V19 | V16-B | Delta |
|---|---|---|---|
| CAGR | **17.29%** | 17.06% | +0.23pp |
| Sharpe | **0.867** | 0.846 | +0.021 |
| MaxDD | **-25.1%** | -27.0% | +1.9pp |
| Terminal $1 | **$54.75** | $52.18 | +$2.57 |
| DCA | **$4.64M** | $4.41M | +$230K |
| Vol | **20.93%** | 21.33% | -0.40pp |

**The insight from V18b unlocked this:** V16-B's -27% DD came from unlevered equity held 23 days between CB fire and monthly rebalance. V19 exits to cash instead, eliminating that exposure. Post-CB event analysis across 27 events: equity cumulative -5.49%, cash cumulative +0.80%. Holding unlevered equity post-CB was actively destroying value.

**V13's CB→cash failure was due to confounding changes** (tighter IVV guard, three states, weekly re-entry), not the CB→cash mechanism itself. V19 isolates it cleanly.

**Updated Pareto frontier:**

| Point | Strategy | CAGR | Sharpe | MaxDD | DCA |
|---|---|---|---|---|---|
| Max wealth | **V9** | 19.4% | 0.777 | -37.9% | $7.37M |
| Balanced | **V19** | 17.3% | **0.867** | -25.1% | $4.64M |
| Max Sharpe | **Baseline** | 13.8% | 0.910 | -18.5% | $2.40M |

V16-B no longer on any efficient frontier. V19 is now only 0.043 Sharpe below Baseline (+3.50pp CAGR, +$2.24M DCA).


---

## April 13, 2026 — V19d Robustness Tests Complete, Research Arc Closed

V19c (100% unlev at score 2): WASH. Keep 70/30.
V19d (gold CB): WASH. Adopted for design consistency.
V20 (directional transitions): FAIL. Hypothesis inverted — 3→2 recovers 57%, 1→2 never recovers.

**V19d is the FINAL production specification.** See [[experiments/V9_TO_V19D_RESEARCH_ARC]] for the complete V9→V19d research narrative.

All research tracks exhausted. Implementation track begins.


---

## April 13, 2026 — V19d Final Backtest + QQQ Tilt Test

See [[experiments/V19D_FINAL_BACKTEST]] for definitive production numbers (all 11 tables, CSVs saved).

**V19d-QQQ 60/30/10 tilt:** NOT PREFERRED. +0.57pp CAGR but -3.6pp MaxDD and -0.010 Sharpe. GFC DD worsens from -16.4% to -20.8%, COVID from -25.1% to -28.7%. 45/45/10 is the correct balanced split. See [[experiments/V19D_QQQ_TILT_RESULTS]].

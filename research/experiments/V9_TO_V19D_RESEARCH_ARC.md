# TAA Research Arc: V9 → V19d (Complete)

**Date range:** April 5–13, 2026
**Final system:** V19d (45/45/10 QLD/SSO/IAU, CB → cash, gold CB)
**Status:** Research complete. Production architecture locked.

---

## The Pareto Frontier (Final)

| Point | System | CAGR | Sharpe | MaxDD | DCA ($21K + $700/mo) |
|---|---|---|---|---|---|
| Max wealth | **V9** | 19.4% | 0.777 | -37.9% | $7.37M |
| Balanced | **V19d** | 17.3% | 0.866 | -25.1% | $4.64M |
| Max protection | **[[2026-04-04_faber_only_baseline\|Baseline]]** | 13.8% | 0.910 | -18.5% | $2.40M |

---

## Experiment Log (Chronological)

### V9 — QLD + IVV Guard (origin point)
**Status:** PASS — terminal wealth maximizer
See [[TERMINAL_WEALTH_OPTIMIZATION]]
- 100% QLD when QQQ 3/3 + IVV ≥ 2. Cash otherwise.
- IVV is a guard signal only — never held.
- CB: QLD → QQQ, monthly re-entry.
- 19.37% CAGR, 0.777 Sharpe, -37.9% DD, $85.25 terminal, $7.37M DCA.

### V11 — Beta-Scaled Dynamic State Architecture
**Status:** PASS on criteria but retired — dominated by V12
See [[V11_BETA_SCALED_RESULTS]]
- 16-row allocation table, beta-scaling formula, graduated equity caps, defensive rotation (DBMF/VGLT/IAU/DBC).
- 17.90% CAGR, 0.790 Sharpe, -30.8% DD, $5.25M DCA.
- 15 moving parts for +0.013 Sharpe over V9.
- **Key finding:** Complexity didn't earn its place.

### V12 — Independent Faber-Gated 2× on IVV + QQQ
**Status:** PASS — retired, displaced by V15 then V19d
See [[V12_INDEPENDENT_2X_RESULTS]]
- 50/50 IVV/QQQ, each independently Faber-gated with 100% SSO/QLD substitution.
- Two binary switches + cash. No guards, no coupling.
- 17.41% CAGR, 0.803 Sharpe, -28.8% DD, $4.80M DCA.
- **Key finding:** V11's 16-row table adds nothing over two binary switches.

### V13 — Three-State V9 with Weekly Re-Entry
**Status:** FAIL — all paths. Dominated by V9 on every metric.
See [[V13_THREE_STATE_RESULTS]]
- Added delever state (QQQ at 1×) at score 2/3, weekly re-entry after CB.
- 17.41% CAGR, 0.742 Sharpe, -42.0% DD.
- **Key findings:**
  1. Score 2/3 is NOT a safe intermediate state.
  2. Weekly re-entry is noise (1/18 CB events resolved weekly).
  3. CB → cash is worse than CB → QQQ in V9's architecture (confounded finding — later reversed by V19).
  4. IVV guard at score 2 is too tight — V9's loose guard (IVV ≤ 1) is correct.

### V14 — V9 + Defensive Rotation During Cash Periods
**Status:** FAIL — all variants worsen MaxDD vs V9.
See [[V14_DEFENSIVE_ROTATION_RESULTS]]
- When V9 exits to cash, rotate into Faber-gated IVV/VGLT/IAU/DBC.
- V14-A (IVV≥2): 0.760 Sharpe, -38.2% DD. V14-B (IVV≥3): 0.793 Sharpe, -39.1% DD.
- **Key findings:**
  1. Cash IS the hedge — confirmed definitively.
  2. Faber-gated defensives concentrate into the "last survivor" (100% DBC in Jun-Sep 2022).
  3. Defensive pool earns +0.04%/mo over cash but with 3× the vol. Risk-adjusted return of cash is HIGHER.

### V9-DCA — Cash Redeployment via Stepped IVV Buying
**Status:** FAIL — all step sizes worsen Sharpe and/or MaxDD.
See [[V9_DCA_REDEPLOYMENT_RESULTS]]
- Deploy 10% of cash into IVV at each 5% decline from anchor.
- GFC: 9 tranches deployed into -46% decline, exited below cost. -6.17% delta vs cash.
- **Key findings:**
  1. "Time in the market" doesn't apply when the trend filter says exit.
  2. V9's cash period starts too late for DCA to catch the trough (COVID: 0 tranches deployed).
  3. The GFC scenario dominates — one catastrophic event wipes all gains.

### V15 — Two-Pod Architecture (V9 QLD + IVV/SSO)
**Status:** PASS — displaced by V16 then V19d
See [[V15_TWO_POD_RESULTS]]
- Pod 1: V9 QLD with IVV guard. Pod 2: IVV/SSO, no guard. 50/50 split.
- 17.65% CAGR, 0.813 Sharpe, -29.0% DD, $4.92M DCA.
- Pod rebalancing adds +0.008 Sharpe (contrarian "sell winners" effect).
- **Key finding:** Pod 2 standalone (IVV/SSO no guard) has 0.786 Sharpe — higher than V9. IVV supports V9-style leverage.

### V16 — Two-Pod + Gold (45/45/10)
**Status:** PASS — displaced by V19d
See [[V16_TWO_POD_GOLD_RESULTS]]
- Added 10% Faber-gated IAU (≥ 3 or cash).
- V16-B: 17.06% CAGR, 0.846 Sharpe, -27.0% DD, $4.41M DCA.
- **Key finding:** Gold earns its 10% through genuine crisis alpha (GFC: +14.8% in 13/17 months) and vol dampening (-1.96pp portfolio vol).

### V17 — V16 with Dynamic Gold Redeployment
**Status:** PASS on criteria but does NOT displace V16-B from frontier.
See [[V17_DYNAMIC_REDEPLOYMENT_RESULTS]]
- When gold is off-signal, redeploy 10% to active equity pods.
- 17.54% CAGR, 0.842 Sharpe, -27.7% DD.
- Redeployment earns +0.14%/mo excess — confirms redeploying into confirmed trends works.
- **Key finding:** Gold's idle cash provides marginal vol dampening worth more than the extra equity return.

### V18 — Portfolio CB + Leading Indicator Leverage Scaling
**Status:** FAIL — all mechanisms. -27% DD is the structural floor for V16-B.
See [[V18_DRAWDOWN_PROTECTION_RESULTS]]
- Part 1: Zero leading indicators pass validation (VIX, yield curve, credit spreads, S&P drawdown). Best candidate 27% hit rate, 73% FP rate.
- Part 2A: Portfolio-level CB from HWM fires 209 times at -10% threshold. Catastrophic whipsaw.
- **Key findings:**
  1. Leading indicators do not predict equity crashes — fourth confirmation ([[2026-04-04_kritzman_vs_harvey|Harvey]], Kritzman, VRP, V18).
  2. Per-asset CB (3/3 SMA breach) is correctly calibrated. Portfolio-level threshold has no structural anchor.
  3. -27% DD is structural and cannot be improved by overlays.

### V18b — Intra-Month Portfolio CB
**Status:** FAIL — IMC-20/25 never fire.
See [[V18B_INTRAMONTH_CB_RESULTS]]
- -20% within a single month trigger. Expected to catch COVID.
- IMC-20: 0 trigger events in 24 years. Per-asset CBs already stripped leverage before portfolio hits -20% intra-month.
- **KEY FINDING:** V16-B's -27% DD is from UNLEVERED equity held post-CB, not from leverage. The per-asset CBs fire Feb 27, 2020. The -27% DD accumulates at ~90% eff equity (QQQ + IVV at 1×) during the 23 days between CB and monthly rebalance.

### V19 — V16-B with CB → Full Cash Exit
**Status:** PASS — strict Pareto improvement over V16-B. New frontier point.
See [[V19_CB_CASH_EXIT_RESULTS]]
- One change: CB exits to cash instead of unlevered equity.
- 17.29% CAGR, 0.867 Sharpe, -25.1% DD, $4.64M DCA.
- Post-CB analysis: cash wins 14/27 events (52%), cumulative equity return -5.49% vs cash +0.80%.
- **Key finding:** CB → cash is correct. V13's finding was confounded by three other changes. Isolated on V16-B, cash dominates equity in the post-CB window.

### V19b — V19 Without Gold (50/50)
**Status:** Gold confirmed necessary.
See [[V19B_NO_GOLD_RESULTS]]
- 17.88% CAGR, 0.834 Sharpe, -27.0% DD, $5.22M DCA.
- Gold costs 0.59pp CAGR but buys +0.033 Sharpe, +1.9pp MaxDD.
- **Key finding:** Gold's vol dampening (-1.96pp) drives Sharpe improvement. Crisis alpha is bonus.

### V19c — V19 with 100% Unlevered at Score 2/3
**Status:** WASH — keep V19's 70/30.
See [[V19C_FULL_UNLEVER_RESULTS]]
- 17.41% CAGR, 0.865 Sharpe, -25.1% DD.
- Score 2/3 occurs 14% of months. Treatment barely matters.
- **Key finding:** 70/30 marginally better Sharpe (+0.002) and crisis DDs.

### V19d — V19 with Gold Circuit Breaker
**Status:** WASH — adopt for design consistency. **PRODUCTION SPEC.**
See [[V19D_GOLD_CB_RESULTS]]
- 17.27% CAGR, 0.866 Sharpe, -25.1% DD.
- Gold CB fires 10 times in 24 years with negligible impact.
- **ADOPTED as production spec.** All risk assets now have consistent 3/3 SMA breach → cash CB.

### V20 — Directional State Transitions
**Status:** FAIL — directional hypothesis is empirically inverted.
See [[V20_DIRECTIONAL_TRANSITIONS_RESULTS]]
- 3→2 (falling): 57% recover next month. Being defensive is wrong majority of the time.
- 1→2 (rising): 0% recover to 3/3. Being aggressive is a dead cat bounce trap.
- **Key finding:** Fastest SMA (126-day) produces the most noise. Short-term breaks are false alarms, short-term restores are traps. Non-directional 70/30 is correct.

---

## Cumulative Key Findings

1. **Faber trend filter is the dominant value source** — confirmed across every experiment.
2. **Macro indicators do not predict crashes** — Harvey, Kritzman, VIX, yield curve, credit spreads all fail.
3. **Cash is the hedge** — every attempt to redeploy V9's cash (defensives, DCA, intermediate states) fails.
4. **Simplicity wins** — V12's two binary switches match V11's 16-row table. V19d has 4 components, not 15.
5. **CB → cash is correct** — V19 is a strict Pareto improvement. Post-CB window is predominantly continued decline.
6. **Gold earns its place** — vol dampening + crisis alpha at 10% allocation.
7. **Score 2/3 is noise** — treatment barely matters (V19c wash). Direction is inverted (V20 fail).
8. **Weekly re-entry is dead** — 1/18 events. Monthly is the correct cadence.
9. **Pod structure works** — independent engines on different indexes + rebalancing adds genuine Sharpe.
10. **-25.1% MaxDD is the structural floor** for V19d. Comes from unlevered equity in the CB-to-rebalance window. Cannot be improved without reducing base leverage.

---

## Dead Ends (Do Not Re-Explore)

- Tightening IVV guard to score 2 ([[V13_THREE_STATE_RESULTS|V13]]: costs return without improving DD)
- Weekly re-entry after CB ([[V13_THREE_STATE_RESULTS|V13]]: 1/18 events resolved)
- Faber-gated defensive rotation during cash periods ([[V14_DEFENSIVE_ROTATION_RESULTS|V14]]: concentration into last survivor)
- DCA into declining equities during cash periods ([[V9_DCA_REDEPLOYMENT_RESULTS|V9-DCA]]: GFC kills it)
- Portfolio-level drawdown CB ([[V18_DRAWDOWN_PROTECTION_RESULTS|V18]]: fires during normal volatility)
- Intra-month CB ([[V18B_INTRAMONTH_CB_RESULTS|V18b]]: per-asset CBs already stripped leverage)
- Leading indicator leverage scaling ([[V18_DRAWDOWN_PROTECTION_RESULTS|V18]]: 0 indicators pass validation)
- Dynamic gold redeployment to equity ([[V17_DYNAMIC_REDEPLOYMENT_RESULTS|V17]]: marginal, doesn't beat V16-B Sharpe)
- Directional score-2 treatment ([[V20_DIRECTIONAL_TRANSITIONS_RESULTS|V20]]: hypothesis inverted)

---

## V19d Production Specification

```
Architecture: Two-Pod + Gold, CB → Cash
─────────────────────────────────────────
Pod 1 (45%): QQQ/QLD
  Signal:    126/200/252-day daily SMAs on QQQ
  Score 3/3: 100% QLD (2× Nasdaq)
  Score 2/3: 70% QQQ + 30% cash
  Score 0-1: 100% cash
  Guard:     IVV score ≤ 1 → exit QLD to QQQ (V9 loose guard)
  CB:        QQQ closes below ALL 3 SMAs → exit to CASH next open

Pod 2 (45%): IVV/SSO
  Signal:    126/200/252-day daily SMAs on IVV
  Score 3/3: 100% SSO (2× S&P 500)
  Score 2/3: 70% IVV + 30% cash
  Score 0-1: 100% cash
  Guard:     None (IVV stands alone)
  CB:        IVV closes below ALL 3 SMAs → exit to CASH next open

Gold (10%): IAU
  Signal:    126/200/252-day daily SMAs on IAU
  Score ≥ 3: 100% IAU
  Score < 3: 100% cash
  CB:        IAU closes below ALL 3 SMAs → exit to CASH next open

Rebalance:   Monthly to 45/45/10 targets (5% drift threshold)
Re-entry:    Monthly only (no mid-month re-entry after CB)
Execution:   Human-in-the-loop via Telegram
             Monthly rebalance approval + daily CB alerts (~1.5/year total)
```

### Validated Performance

| Metric | Value |
|--------|-------|
| Annualised return | 17.27% |
| Volatility | 20.94% |
| Sharpe ratio | 0.866 |
| Sortino ratio | 1.010 |
| Max drawdown | -25.1% |
| Calmar ratio | 0.72 |
| Terminal $1 (2002-2026) | $54.60 |
| DCA $21K + $700/mo | $4,640,000 |
| CB events (equity) | 27 over 24yr (1.1/yr) |
| CB events (gold) | 10 over 24yr (0.4/yr) |
| Rebalance events | ~14 over 24yr (0.6/yr) |

---

## Cross-references

- [[TAA_PROJECT_STATUS]] — project-level status document
- [[KRITZMAN_RESEARCH_FINDINGS]] — macro engine evaluation
- [[2026-04-04_faber_only_baseline]] — original Faber baseline establishment
- [[2026-04-06_faber_daily_circuit_breaker]] — daily CB validation
- [[2026-04-06_leverage_sweep_high]] — leverage calibration
- [[BULL_MARKET_SURVIVABILITY]] — QQQ trailing analysis

# Multi-Pod Trading Architecture

**Date:** April 5, 2026
**Status:** Strategic Framework — Pre-Research
**Related:** [[TAA_PROJECT_STATUS]] | [[KRITZMAN_RESEARCH_FINDINGS]]

---

## Motivation

The Millennium insight applied to this system: they achieve ~2.5 portfolio Sharpe not because any single pod has a 2.5 Sharpe, but because 330 uncorrelated pods with individual Sharpe ~0.5–1.0 each combine via diversification. Portfolio-level Sharpe scales with the square root of uncorrelated strategies. The goal here is a small number of genuinely uncorrelated pods running in parallel.

Combined Sharpe approximation (when correlation is low):
**Portfolio Sharpe ≈ √(Sharpe₁² + Sharpe₂² + ... + SharpeN²)**

At Faber ~1.1 + VRP ~1.0 + Managed Futures ~0.8 with low pairwise correlation, combined Sharpe approaches 1.6–1.8. The key is maintaining actual low correlation — especially during stress.

---

## Pod Definitions

### Pod 1 — Faber TAA (existing, production-ready)
- **Strategy:** Multi-timeframe trend filter (6/10/12 SMA), freed capital to cash, graduated leverage overlay
- **Sharpe:** ~1.1 (backtested 2002–2026)
- **Vol:** ~7.4% at 1x, higher with leverage
- **Risk:** Whipsaw in choppy, trendless markets
- **Status:** Complete — [[2026-04-05_leverage_calibration]] pending

### Pod 2 — Volatility Risk Premium (VRP) Harvesting
- **Strategy:** Systematic monthly selling of cash-secured puts on IVV/SPY via CBOE PUT index methodology.
- **Sharpe:** 0.723 standalone (2002-2026), 0.860 with VRP filter (>=0). Literature Sharpe of ~1.0-1.5 reflects pre-2002 era before premium compression.
- **Vol:** 10.4% (comparable to Faber-Sweep-40 at 9.8%)
- **Risk:** Negative skew (-1.65), fat tails (kurtosis 7.30). GFC DD: -85%. COVID DD: -29%. Correlation with IVV: 0.865 — essentially equity-like.
- **Correlation with Faber:** 0.533 full-period, 0.31-0.39 during crises. Rolling 12m mean: 0.635. Low crisis correlation is asymmetric — Faber protects itself while PUT crashes independently.
- **Combined portfolio:** 80/20 Faber+VRP: 1.058 Sharpe (+0.019 vs Faber-only), but -$1.12 terminal. 90/10: 1.053 Sharpe, best DD improvement (-11.8% vs -13.3%).
- **Implementation:** Schwab Roth IRA, Level 2 options. Min position: ~$52K/contract (1 IVV put).
- **Status:** CONDITIONAL — marginal Sharpe improvement, worsens crisis behavior. Recommended max 10% sleeve with VRP filter. See [[2026-04-05_vrp_backtest]]

### Pod 3 — Managed Futures / Long-Short Trend (DBMF or KMLM)
- **Strategy:** CTA-style long/short trend following across equities, bonds, currencies, and commodities via DBMF ETF (Dynamic Beta Engine replicating top-20 CTA exposures).
- **DBMF live performance (2019-2026):** +21.6% in 2022 (IVV -18.2%), +11.5% in 2021, -8.9% in 2023, +7.2% in 2024, +13.8% in 2025. Max DD: -17.3%.
- **Crisis behavior: CONFIRMED genuine diversifier.** 2022: DBMF-IVV correlation = **-0.586**. COVID: +1.4% (modestly positive but correlated 0.526 — fast crashes don't help).
- **Proxy situation:** No proxy passes 0.85 correlation threshold. AQR TSMOM (1985-2025, 481 months) is best available at 0.632 monthly corr (improves to 0.752 in bear markets). Managed futures strategies are too heterogeneous for any single proxy to represent DBMF specifically.
- **Implementation:** DBMF ETF in Roth IRA. No K-1. Liquid.
- **Status:** Proxy validation complete. See [[2026-04-05_managed_futures_proxy_validation]]. Backtest options: (a) DBMF ETF-only from 2019, (b) AQR TSMOM proxy from 2002 with heavy caveats.

### Pod 4 — Merger Arbitrage (MNA or MERFX)
- **Strategy:** Collect the spread between target stock prices and announced acquisition prices. Earn risk premium for bearing deal-failure risk. ~70–90% of announced deals close; the spread compensates for the ~10–30% that don't.
- **Sharpe:** ~1.0–1.5 (Eurekahedge Arbitrage Index: 1.46 over 25 years, 6.7% return at 3.2% vol)
- **Vol:** ~3–4% — very low
- **Risk:** Negatively skewed. Regime-dependent correlation — uncorrelated in normal/rising markets, converges toward equity during severe downturns (deal flow dries up, spreads widen).
- **Implementation:** MNA (IQ Merger Arbitrage ETF) or MERFX mutual fund. Liquid, no individual deal management required.
- **Status:** Phase 4 — after VRP and managed futures running

---

## Pod Correlation Structure

| | Faber TAA | VRP | Managed Futures | Merger Arb |
|---|---|---|---|---|
| **Faber TAA** | 1.0 | Low-Mod | Very Low (neg in crisis) | Very Low |
| **VRP** | — | 1.0 | Low | Moderate |
| **Managed Futures** | — | — | 1.0 | Low |
| **Merger Arb** | — | — | — | 1.0 |

**Critical note:** VRP and Merger Arb both have regime-dependent correlation — they appear independent in normal markets but converge toward equity during crash environments. Managed futures is the genuinely independent pod because it profits from trends in any direction including downtrends.

---

## Phase 5: Cross-Pod Turbulence Risk Management (Kritzman Layer)

### Why This Works Here (and Why It Failed Before)

In prior Kritzman experiments, the conditioned covariance matrix was used to forecast *which asset* should receive more capital — a return prediction problem. The conditioned expected returns were noise (-0.049 correlation with realized vol). That approach failed.

Here, the Kritzman framework is used only for **risk identification**: detecting when normally-uncorrelated pods are becoming correlated. This is a regime detection problem, not a return forecasting problem. The turbulence index was specifically designed for this (Kritzman & Li 2010 — documented Sharpe improvement from ~1.0 to ~2.2 on SPY across 1980–2022).

### Architecture

```
Layer 1 — Pod Signals (independent, run their own logic)
  Faber TAA         → target allocation per existing system
  VRP Harvesting    → target sizing per options strategy
  Managed Futures   → fixed DBMF/KMLM allocation
  Merger Arb        → fixed MNA allocation

Layer 2 — Cross-Pod Covariance Monitor (Kritzman Turbulence)
  Monthly: compute turbulence index from pod return proxies
  Monthly: compute conditioned pairwise correlation between pods
  Flag: are pods that should be uncorrelated converging?

Layer 3 — Risk Scalar (applied uniformly across all pods)
  Normal regime:    pods run at full target sizing
  Elevated turb:    all pods scale to 75%, excess to cash
  High turb:        all pods scale to 50%, excess to cash
  Extreme turb:     all pods scale to 25%, system in defensive mode
```

### Inputs to the Covariance Matrix

Pod return proxies (actual historical returns):
- Faber TAA: actual system backtest returns
- VRP: CBOE PutWrite Index (PUT) monthly returns
- Managed Futures: SG Trend Index or DBMF live returns
- Merger Arb: HFRI Merger Arbitrage Index or MNA returns

Turbulence computed from these pod return series — measuring when their co-movement is statistically unusual relative to their historical covariance structure. Fires before correlations fully converge.

### Honest Caveat

Works for slow-building stress (2008 GFC buildup, 2022 rate cycle). Failure case is instantaneous crashes (March 2020) where correlations spike in days — no monthly signal saves you there. Turbulence layer is a risk reducer, not a crash eliminator. Conservative pod sizing remains the primary defense.

### Why Phase 5 (Not Earlier)

The covariance matrix requires actual pod return history to be meaningful. Cannot model cross-pod co-movement before the pods exist. Phase 5 becomes buildable once Phases 2–4 have 12+ months of live or simulated pod return data.

---

## Build Sequence

| Phase | Description | Gate |
|-------|-------------|------|
| 1 | Faber optimization — leverage calibration | **Complete** — [[2026-04-05_faber_sweep]] |
| 2 | VRP harvesting research + backtest | **Complete** — [[2026-04-05_vrp_backtest]]. CONDITIONAL: marginal +0.019 Sharpe |
| 3 | Add managed futures sleeve (DBMF) | **Proxy validation complete** — [[2026-04-05_managed_futures_proxy_validation]]. No proxy passes threshold; DBMF ETF-only or AQR TSMOM with caveats |
| 4 | Add merger arb sleeve (MNA) | After 6 months Phase 3 observation |
| 5 | Cross-pod turbulence risk management layer (Kritzman) | **Complete** — [[2026-04-05_two_pod_combined]]. 1.333 Sharpe but -29% terminal. De-levered 73% of months. Not recommended for accumulation. |
| 6 | Options layer refinement (VRP scaling, vertical spreads) | Deferred — Faber standalone is production architecture |

---

## Key References

- Kritzman, M., Li, Y. (2010). "Skulls, Financial Turbulence, and Risk Management." FAJ
- Kritzman, M., Li, Y., Page, S., Rigobon, R. (2011). "Principal Components as a Measure of Systemic Risk." JPM
- Koijen, R., Moskowitz, T., Pedersen, L.H. (2018). "Carry." JFE
- Hurst, B., Ooi, Y.H., Pedersen, L.H. (2017). "A Century of Evidence on Trend-Following." AQR
- Mitchell, M., Pulvino, T. (2001). "Characteristics of Risk and Return in Risk Arbitrage."
- CBOE PutWrite Index (PUT) — VRP harvesting benchmark, Sharpe ~1.0

---

## Phase 5: Kritzman Turbulence Layer — Two-Pod Mechanics (Pod 1 + Pod 2)

**Date:** April 5, 2026
**Status:** COMPLETE — see [[2026-04-05_two_pod_combined]]

**Result:** Tested at both 0.65 and 0.80 correlation thresholds. At 0.65: 1.333 Sharpe, de-levered 73%. At 0.80: 1.281 Sharpe, de-levered 66%. Both thresholds produce excessive de-levering — the pods' 3-month rolling correlation exceeds 0.80 for 52% of months as normal operating state. **Turbulence layer definitively rejected for Faber-VRP combination.** The concept requires genuinely uncorrelated pods (e.g., Faber + managed futures). See [[2026-04-05_two_pod_combined]] and [[2026-04-05_two_pod_s40_rerun]].

### The Problem Being Solved

Faber-VRP full-period correlation: 0.533. Rolling mean: 0.635. The pods are not independent — during normal trending markets both earn equity-like returns from the same source. The turbulence layer should detect when this correlation spikes above its historical norm and scale both pods down, preserving capital.

### How the Turbulence Index Works for Two Pods

The Kritzman turbulence index measures the Mahalanobis distance of the current joint return vector from its historical mean and covariance:

```
Turbulence_t = (r_t - μ)ᵀ Σ⁻¹ (r_t - μ)

where:
  r_t = [Faber_return_t, VRP_return_t]  — current month's two-pod return vector
  μ   = rolling historical mean vector   — what returns "normally" look like
  Σ   = rolling historical covariance matrix — how the pods normally co-move
```

A high turbulence reading means the current joint return is statistically unusual relative to the historical relationship between the pods. This fires when:
1. Returns are extreme in magnitude (one or both pods moving a lot)
2. Correlation is unusual — pods moving together when they normally don't, or diverging when they normally move together

### Critical Design Question: What Are We Actually Detecting?

The VRP crisis behavior creates a subtle problem:

| Crisis | PUT Return | Faber Return | Correlation | Turbulence Expected |
|--------|-----------|--------------|-------------|---------------------|
| GFC 2008-09 | -24.7% | +0.6% | 0.371 (LOW) | HIGH — PUT extreme, but pods diverged |
| COVID 2020 | -28.9% | -15.0% | 0.392 (LOW) | HIGH — both hurt, moderate correlation |
| 2022 Bear | -9.7% | -8.5% | 0.313 (LOW) | MODERATE — both hurt modestly |

The GFC case is the problem: turbulence fires because PUT had extreme returns, but Faber was POSITIVE. De-leveraging Faber during GFC when Faber was earning +0.6% would destroy exactly the value Faber was providing.

**This means turbulence on the joint vector does not cleanly separate "good crisis" (Faber protected, PUT hurt) from "bad crisis" (both hurt). It fires on extreme returns regardless of direction.**

### Two Architectural Options

**Option A: Joint Turbulence (standard Kritzman)**
Compute turbulence on the two-pod joint return vector. Scale BOTH pods when turbulence exceeds threshold.

Problem: De-levers Faber during GFC even though Faber was performing correctly. Reduces the diversification benefit of having two uncorrelated pods.

Benefit: Simplest implementation, fully published methodology.

**Option B: Correlation-Based Trigger (modified)**
Instead of raw turbulence on return magnitude, monitor the ROLLING CORRELATION between pods specifically. When 3-month rolling correlation spikes above a threshold (e.g., above 0.70), scale both pods down. This specifically targets the case where the pods are becoming correlated — which is the actual risk — not the case where one pod has extreme returns for independent reasons.

Problem: Less published validation, slightly more complex.
Benefit: Directly targets the problem (correlation convergence) rather than magnitude.

**Option C: Hybrid — Correlation-Gated Turbulence**
Only apply turbulence-based de-levering when:
1. Rolling 3-month correlation between pods exceeds 0.65 (pods are converging), AND
2. Turbulence exceeds 75th percentile threshold

This means: turbulence alone doesn't trigger de-levering (avoids the GFC Faber problem), but when pods are already converging AND turbulence spikes, scale down. Both conditions must be true.

**Recommended: Option C.** It directly targets the actual risk (pod correlation convergence during stress) while avoiding false triggers during independent pod stress events.

### Proposed Mechanics for Backtest

**Inputs computed monthly from trailing 36-month window:**
- μ = [mean_Faber, mean_VRP]
- Σ = [[var_Faber, cov_Faber_VRP], [cov_Faber_VRP, var_VRP]]
- Rolling 3-month correlation between pods
- Turbulence_t = Mahalanobis distance

**Trigger thresholds (first principles, not optimized):**
- Rolling 3-month correlation > 0.65 (pods converging above normal)
- Turbulence > 75th percentile of trailing 36-month turbulence

**Scaling tiers when BOTH conditions met:**
```
Normal (below thresholds):    both pods at 100% target weight
Elevated (one condition):     both pods at 75% target weight, excess to cash
High (both conditions):       both pods at 50% target weight, excess to cash
```

**Why 36-month window:** Short enough to be responsive to regime changes, long enough to have stable covariance estimates (36 monthly observations per Ledoit-Wolf guidelines).

**Why 0.65 correlation threshold:** Mean rolling Faber-VRP correlation is 0.635. A threshold of 0.65 means the trigger fires when correlation is above its own mean — i.e., when pods are more correlated than usual, not just correlated.

**Why 75th percentile turbulence:** Fires approximately 25% of the time. Conservative enough to avoid constant de-levering during normal markets, sensitive enough to catch genuine stress.

### What the Backtest Needs to Show

1. How often does the trigger fire? (target: 15-25% of months)
2. Does it fire during the right periods (both pods stressed simultaneously)?
3. Does it avoid firing during GFC (when Faber was positive)?
4. Does the combined Sharpe improve vs no turbulence layer?
5. Does max drawdown improve?

### Open Question: Lookback for Covariance

The 36-month window means the turbulence system needs 36 months of pod return history before it can operate. In a live 2-pod system this means the turbulence layer activates after 3 years. For the backtest, use the 2002 start date but only activate the turbulence layer from 2005 onward (after 36 months of history). Before 2005, run at full weights with no turbulence adjustment.

---

## Lifecycle Leverage Framework (Pedersen / Ayres-Nalebuff)

**Source:** Ayres & Nalebuff (2010) "Lifecycle Investing" + Frazzini & Pedersen (2014) "Betting Against Beta"

**Core argument:** At 25, human capital (PV of future earnings) vastly exceeds financial capital. Rational diversification across time means taking more investment risk early when human capital provides a natural cushion, then gradually delevering as financial capital grows and human capital depletes.

**Recommended leverage schedule:**

| Age | Recommended Leverage | Rationale |
|-----|---------------------|-----------|
| 22-28 | 2.0x | Maximum human capital, maximum recovery horizon |
| 28-35 | 1.5-2.0x | Still early accumulation, high recovery capacity |
| 35-45 | 1.25-1.5x | Mid-career, significant financial capital building |
| 45-55 | 1.0-1.25x | Approaching peak financial capital, reduce slowly |
| 55-65 | 1.0x | No leverage, capital preservation |
| 65+ | Sub-1x | Decumulation phase |

**Application to current system (age 25):**
- Pedersen-recommended leverage: **2.0x**
- Portfolio equity sleeve: 70% (IVV 45% + QQQ 25%)
- 2.0x effective equity exposure = 140% total portfolio equity exposure
- Maps to: **100% SSO/QLD substitution** (all IVV → SSO, all QQQ → QLD)
- This is the a priori justified substitution level — not backtest-optimized

**Decision:** Use 100% substitution (2.0x effective equity on equity sleeve) as the production leverage level for the investor at age 25, consistent with Pedersen lifecycle framework. Reduce to 80% substitution (~1.65x) at age 30, 60% (~1.3x) at age 35, 40% (~1.0x) at age 45, 0% at age 55.

# Dynamic Equity Sleeve Tilt: IVV/QQQ Relative Momentum

**Date:** April 9, 2026  
**Status:** Complete — REJECTED (dot-com stress test failed)  
**Track:** System Enhancement  
**Related:** [[TAA_PROJECT_STATUS]] | [[BULL_MARKET_SURVIVABILITY]]

## Concept

Tilt the 70% equity sleeve toward QQQ when QQQ/IVV price ratio is above its 200-day SMA (QQQ in relative uptrend), and toward IVV when below. Total equity stays 70%. Leverage substitution unchanged. Only applies during signal-on periods.

Signal: `QQQ_price / IVV_price` vs its 200-day SMA. Zero new parameters — uses the same tool already in the system.

## Phase 1: Dot-Com Stress Test — FAILED

**This is the kill gate. The tilt failed it.**

| Variant | CAGR 1999-2002 | Max DD | Terminal |
|---------|---------------|--------|---------|
| A Baseline | -4.4% | **-27.5%** | $0.84 |
| B Tilt | -6.7% | **-34.4%** | $0.77 |

**Max DD worsened by 6.9 percentage points** (threshold was 3%). The tilt had QQQ overweighted (+40% vs +25%) heading into the April-May 2000 crash. QQQ/IVV ratio crossed below its 200-day SMA on May 10, 2000 — but by then the damage was done. April 2000 alone: Baseline -11.6%, Tilt -15.3%.

The ratio signal is backward-looking — it confirms QQQ's relative strength after QQQ has already been strong. By the time QQQ breaks relative to IVV, the portfolio has already absorbed the excess QQQ drawdown.

**The idea is REJECTED on first principles: momentum-tilting toward the higher-beta asset amplifies crashes that the tilt signal detects too late to avoid.**

---

## Full-Period Results (reported for completeness — idea already rejected)

| Variant | CAGR | Vol | Sharpe | MaxDD | Terminal |
|---------|------|-----|--------|-------|---------|
| A Baseline | 14.2% | 15.8% | **0.921** | **-18.1%** | $24.99 |
| B Binary tilt | 14.5% | 16.4% | 0.906 | -18.3% | $26.48 |
| C Three-state | 14.4% | 16.3% | 0.907 | -18.3% | $25.98 |

The tilt adds +$1.49 terminal (+6%) and +0.3% CAGR. But it **costs -0.015 Sharpe** and **worsens MaxDD by -0.2%**. The 2013-2021 bull market improvement is +0.8% CAGR — real but modest.

### Sub-Period Breakdown

| Period | A | B | B-A |
|--------|---|---|-----|
| Dot-com | -0.9% | -0.9% | 0.0% |
| Pre-GFC bull | 14.5% | 14.5% | 0.0% |
| GFC | -8.1% | -8.6% | **-0.5%** |
| Recovery | 15.2% | 15.3% | +0.1% |
| **2013-2021 bull** | **20.7%** | **21.6%** | **+0.8%** |
| **2022 bear** | **-13.8%** | **-14.7%** | **-0.9%** |
| 2023-2026 | 22.3% | 22.6% | +0.3% |

The tilt helps during bull markets (+0.8% in 2013-2021) and hurts during bears (-0.5% GFC, -0.9% in 2022). This is exactly the wrong tradeoff for a system whose primary value proposition is crisis protection.

### Tilt Diagnostics

Binary tilt (B): QQQ-tilted 51% of months, IVV-tilted 13%, signal-off 36%. Transitions: 1.0/year (low — not a whipsaw problem).

Three-state (C): QQQ 33%, neutral 27%, IVV 4%, off 36%. Transitions: 1.7/year.

### Sensitivity Check

| Config | Sharpe | MaxDD | Terminal |
|--------|--------|-------|---------|
| Small tilt (37/33 vs 52/18) | 0.914 | -18.2% | $25.75 |
| Base tilt (30/40 vs 55/15) | 0.906 | -18.3% | $26.48 |
| Large tilt (20/50 vs 60/10) | 0.893 | -18.5% | $27.46 |
| Baseline (45/25 fixed) | **0.921** | **-18.1%** | $24.99 |

**Monotonic: larger tilt → higher terminal, lower Sharpe, worse DD.** This confirms the tilt is trading risk-adjusted return for raw return — the opposite of what the system is designed to do.

## Why the Idea Fails

1. **The signal is backward-looking.** QQQ/IVV ratio above its SMA means QQQ has ALREADY outperformed. By the time you overweight QQQ, the relative outperformance may be ending.

2. **Higher beta amplifies both directions.** QQQ's higher beta earns more in bull markets AND loses more in crashes. The tilt increases exposure to the losier asset during crashes.

3. **The Faber filter doesn't save you fast enough.** In April 2000, both IVV and QQQ were still above their individual SMAs (both at 3/3). The tilt overweighted QQQ during a period when Faber said "stay leveraged." By the time Faber exited, QQQ had already fallen more than IVV.

4. **The whole point of fixed weights is diversification.** 45/25 was chosen to balance S&P breadth with Nasdaq growth. Tilting defeats this by concentrating in whichever component has been stronger — which is momentum chasing, not diversification.

## Verdict

**REJECTED.** The dot-com stress test shows -6.9% worse max DD. The full-period results confirm: tilt trades Sharpe for terminal, which contradicts the system's design philosophy. Fixed 45/25 equity split is confirmed as optimal.

The 2013-2021 bull market underperformance vs QQQ (+0.8% improvement from tilt) is not worth the crisis-period degradation (-0.5% GFC, -0.9% 2022).

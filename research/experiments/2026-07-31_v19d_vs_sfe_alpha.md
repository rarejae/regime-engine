# V19d vs SFE — Where the Alpha Comes From

**Date:** 2026-07-31  
**Status:** Complete  
**Track:** Principle / interpretation  
**Related:** [[2026-07-22_sfe_simple_faber_equal]] | [[2026-07-22_sfe_45_45_10_gold_cap]] | [[2026-07-22_sfev3_cb]] | [[V19D_PRODUCTION_SPEC]] | [[V9_TO_V19D_RESEARCH_ARC]] | [[TAA_PROJECT_STATUS]]

---

## Framing

**SFE** (Simple Faber Equal) and **V19d** are the same theory at two levels of specification. SFE is the untuned a priori benchmark: equal 1/3 sleeves of QLD / SSO / GLD, classic 10-month Faber binary, cash when off, monthly only. V19d is a tuned overlay on that skeleton — multi-SMA scores, two-pod structure, gold at 10%, and a daily circuit breaker to cash. They do not have different alpha engines; V19d is an exposure-and-crash-path refinement of the same principle.

---

## Shared alpha (the real edge)

Both systems earn vs equity buy-and-hold from the same load-bearing ideas:

1. **Faber trend gate** — hold risk assets only when price confirms a long SMA
2. **Cash is the hedge** — off-signal capital stays cash; no defensive redeploy
3. **Leverage only when ON** — QLD / SSO when gated on; never levered in a broken trend
4. **Fixed sleeves + monthly cadence** — no pro-rata redistribution of freed capital; low-frequency rebalance

SFE alone is enough to prove this. Over 2000-02 → 2026-07:

| | SFE | QQQ | SPY |
|--|----:|----:|----:|
| CAGR | 12.3% | 8.8% | 8.5% |
| Sharpe | **0.70** | 0.45 | 0.52 |
| MaxDD | **-35%** | -83% | -55% |

Crisis behavior is the clearest tell: GFC −21% vs QQQ −53%; 2022 −21% vs −35%. The system exits, sits in cash, and survives. That is the alpha — **temporal leverage of the equity premium under a trend filter**, not a prediction of next month’s return.

In the V19d lock window (2002+), SFE still delivers 13.8% CAGR / 0.79 Sharpe / −29% MaxDD without score tiers, guards, or CB.

---

## Where V19d’s extras come from

V19d’s locked-window edge (~17.3% CAGR / 0.87 Sharpe / −25% MaxDD vs SFE ~13.8% / 0.79 / −29%) is not a second theory. It comes from causal knobs on the same skeleton:

| Knob | What it does | Why it shows up |
|------|--------------|-----------------|
| Multi-SMA 126/200/252 + score-2 partials | Graduated exposure instead of binary on/off | Softens transitions; keeps some equity when trend is mixed |
| Two-pod + IVV guard / 45/45/10 | Equity-heavy when both pods are on; gold capped at 10% | Higher mean effective equity (~131%) vs equal SFE’s heavy gold sleeve |
| Daily CB → cash | Exit mid-month when price is below all three SMAs; re-enter only monthly | Cuts fast-crash path (esp. COVID); post-CB equity was −5.49% cumulative across equity CB events |

**Side notes (not co-equal strategies):** Cap gold at 10% on the SFE engine (45/45/10) raises CAGR and terminal wealth but worsens Sharpe and MaxDD — gold weight is a risk dial. Adding V19d’s CB on that engine (SFEv3) pays in COVID (−27% vs −35% without CB) but charges a whipsaw premium (worse full-sample path; 2026 YTD CB cost ~4pp). CB alone on the SFE skeleton does not close the gap to V19d.

---

## Where the alpha is *not*

Closed research tracks — these are not sources of edge for either system:

- Macro return forecasts (Harvey, Kritzman) — destroy Sharpe vs Faber-only
- Defensive rotation or DCA redeploy of cash — cash is the hedge; defensives concentrate or buy into declines
- Pro-rata redistribution of freed capital — doubles MaxDD for marginal return
- Portfolio-level drawdown CBs and “leading” indicators — whipsaw or fail validation
- A novel prediction model — neither system forecasts; both gate exposure to trend

---

## Cost of each edge

| Edge | Cost |
|------|------|
| Trend gate (shared) | Lag in strong bulls — AI-era catch-up tax vs naked QQQ; DCA can trail buy-and-hold Nasdaq |
| Leverage when ON (shared) | Levered-ETF drag; honest long-history MaxDD floor ~−35% to −40% from 2000 starts |
| Daily CB (V19d / SFEv3) | Whipsaw premium in chop (e.g. Mar–Apr 2026 exit/miss/re-entry); not free insurance |
| Equity-heavy weights (V19d) | Deeper drawdowns when the filter is late; first month of a crash remains unprotected |

---

## Bottom line

**Shared alpha** = risk-managed temporal leverage of the equity premium: Faber gate, cash hedge, leverage only in confirmed uptrends, fixed sleeves.

**SFE** proves the principle without ornaments. **V19d**’s extras are crash-path and exposure tuning (multi-SMA stack, two-pod mix, CB → cash) — not a second alpha engine. Any future complexity must beat SFE on Sharpe and MaxDD with a clear causal mechanism, not by displacing another tuned sibling.

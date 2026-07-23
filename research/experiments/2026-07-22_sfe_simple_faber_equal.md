# Simple Faber Equal (SFE) — A Priori Principle Backtest

**Date:** 2026-07-22  
**Status:** Complete  
**Track:** Implementation / principle validation  
**Related:** [[TAA_PROJECT_STATUS]] | [[V9_TO_V19D_RESEARCH_ARC]] | [[V19D_PRODUCTION_SPEC]]  
**Script:** `experiments/sfe_simple_faber_equal/backtest.py`  
**Data:** `data/raw/yfinance/sfe_universe.parquet` + FRED `DTB3`

## Hypothesis

If the load-bearing ideas behind V19d are real (Faber trend gate, cash as hedge, leverage only when on, fixed sleeves, monthly cadence), then a deliberately untuned equal-weight binary Faber system on QLD/SSO/GLD should deliver better risk-adjusted returns and drawdowns than buy-and-hold equities — without score tiers, IVV guards, daily CBs, or 45/45/10 weights.

## Design (locked a priori)

```
Sleeves:   1/3 QLD | 1/3 SSO | 1/3 GLD
Signal:    month-end price > 10-month SMA (classic Faber)
Gate on:   QQQ → QLD, SPY → SSO, GLD → GLD
OFF:       that sleeve's 1/3 stays cash (no pro-rata redeploy)
Rebalance: monthly to fixed sleeve targets
Excluded:  multi-SMA voting, score 2/3 partials, IVV guard, daily CB
```

Signal alignment: month-T Faber decision applies to month T+1 returns. Asserted in code.

Pre-2006 QLD/SSO returns: `2× underlying − rf − expense`. Actual ETF returns from inception (2006-06-21). Gold sleeve cash until GLD exists (2004-11).

## Results

### Full sample (2000-02 → 2026-07-22)

| Strategy | CAGR | Vol | Sharpe | Sortino | MaxDD | Terminal $1 | DCA |
|----------|-----:|----:|-------:|--------:|------:|------------:|----:|
| **SFE 1/3 Faber** | **12.25%** | 19.04% | **0.703** | 0.838 | **-34.9%** | **$21.18** | $2.48M |
| QQQ B&H | 8.82% | 26.52% | 0.451 | 0.592 | -83.0% | $9.33 | $2.82M |
| SPY B&H | 8.46% | 19.23% | 0.519 | 0.659 | -55.2% | $8.55 | $1.51M |
| 60/40 | 6.10% | 11.35% | 0.578 | 0.734 | -37.0% | $4.78 | $0.83M |
| GLD B&H (cash pre) | 8.97% | 16.46% | 0.605 | 0.727 | -45.6% | $9.68 | $1.16M |

### Start-date sensitivity (SFE only)

| Window | CAGR | Sharpe | MaxDD | Terminal $1 |
|--------|-----:|-------:|------:|------------:|
| Full (2000-02) | 12.25% | 0.703 | -34.9% | $21.18 |
| 2002-01 (V19d lock window) | 13.84% | 0.791 | -28.7% | $23.96 |
| 2006-07 (live ETF) | 14.71% | 0.810 | -28.7% | $15.59 |
| 2013-01 | 16.55% | 0.898 | -28.7% | $7.93 |

### Crisis Analysis

| Crisis | SFE | QQQ | SPY | 60/40 |
|--------|----:|----:|----:|------:|
| Dot-com 00-02 | -34.8% | -83.0% | -47.5% | -29.4% |
| GFC 07-09 | **-21.1%** | -53.4% | -55.2% | -37.0% |
| COVID 2020 | -28.7% | -28.6% | -33.7% | -21.1% |
| 2022 Bear | **-21.0%** | -34.8% | -24.5% | -14.7% |

### State occupancy (318 months)

| Sleeve | % months ON |
|--------|------------:|
| QQQ/QLD | 71.7% |
| SPY/SSO | 74.8% |
| GLD | 55.0% |
| All ON | 41.8% |
| All OFF | 13.8% |
| Mean cash weight | 32.8% |

## Key Diagnostics

1. **Principle works without V19d ornaments.** SFE beats QQQ/SPY on Sharpe and MaxDD over the full sample and over the 2002 window, using only classic Faber + fixed equal sleeves + cash.
2. **No daily CB shows up clearly in COVID.** SFE held QLD through Feb–Mar 2020 (monthly lag) → COVID DD −28.7%, essentially matching naked QQQ. V19d’s CB→cash edge is real relative to this baseline, not free.
3. **Recent bull lag is the cost.** From 2013 and especially vs QQQ on DCA ($2.48M vs $2.82M full sample), SFE trails buy-and-hold Nasdaq on raw terminal in the AI era — expected trend-following tax.
4. **Equal 1/3 gold is heavy.** Gold ON only 55% of months; when off, that sleeve is cash. This is more defensive (and more drag in equity bulls) than V19d’s 10% gold.
5. **Dot-com MaxDD −34.9%** is in the same neighborhood as the Marketstack-extended V19d −40.7% — confirms ~−35 to −40% is the honest long-history floor for levered Faber systems starting near 2000.

## Interpretation

The a priori principle is **validated**: trend-gated leverage + cash hedge + monthly Faber is sufficient to improve risk-adjusted outcomes vs equity buy-and-hold without the V19d specification search.

V19d still looks like a **tuned improvement** on this skeleton (higher CAGR/Sharpe in the locked window, shallower COVID DD via CB), not a different theory. SFE is the honest null for “how much of V19d is principle vs fitting.”

## Decision

No production replacement yet. SFE is the **principle benchmark** going forward. Any future complexity must beat SFE on Sharpe and MaxDD with a clear causal mechanism — not just displace another tuned sibling.

## Next Steps

- Optional: run SFE vs V19d head-to-head on identical 2002→present yfinance path
- Optional: SFE variant with 10% gold / 45/45 equity (still binary Faber) — only if treating weights as investor preference, not optimization
- Keep V19d live track separate; use SFE as the simplicity floor

## Artifacts

- `research/data/sfe_daily_returns.csv`
- `research/data/sfe_monthly_allocations.csv`

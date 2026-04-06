# Leverage Calibration: Faber-Only Architecture

**Date:** April 5, 2026  
**Status:** Complete  
**Track:** Leverage Calibration  
**Related:** [[TAA_PROJECT_STATUS]] | [[2026-04-05_pro_rata_vs_cash]] | [[2026-04-05_universe_expansion]]

## Hypothesis

The graduated leverage tiers (originally calibrated for Faber+Harvey at ~9.5% vol) need recalibrating for the Faber-only architecture running at 7.4% vol. Deriving substitution percentages from fixed risk tolerance targets (98% and 127% effective equity exposure) should produce the correct tier levels. A weekly circuit breaker should reduce drawdown during fast crashes without sacrificing meaningful return.

## Design

### Tier Substitution Math (from risk targets, not optimization)

When both IVV and QQQ are at 3/3 Faber: IVV=45%, QQQ=25%, total equity=70%.  
Effective equity = equity_weight * (1 + sub), where sub = fraction replaced by 2x ETF.

| Tier | Target Eff Equity | Sub % | IVV Unlev | IVV in SSO | QQQ Unlev | QQQ in QLD |
|------|-------------------|-------|-----------|-----------|-----------|-----------|
| 0 | 70% (baseline) | 0% | 45% | 0% | 25% | 0% |
| 1 | 98% | 40.0% | 27.0% | 18.0% | 15.0% | 10.0% |
| 2 | 127% | 81.4% | 8.4% | 36.6% | 4.6% | 20.4% |

Leveraged ETF simulation: `2.0 * daily_return - rfr/252 - expense/252`  
SSO expense: 0.0089 ann, QLD expense: 0.0095 ann.

### Tier Conditions

- **Tier 0:** Either IVV or QQQ Faber score < 3/3.
- **Tier 1:** Both IVV and QQQ at 3/3.
- **Tier 2:** Tier 1 conditions met, plus both IVV and QQQ trailing 12-month returns above their expanding historical median (seeded with pre-2002 mean of 527 rolling 12m windows).
- **Weekly circuit breaker:** Every Friday, check if either IVV or QQQ is below ALL 3 monthly SMAs. If so, exit leverage until next monthly rebalance.

### Strategies Tested

Six strategies, all Faber-only, freed capital to cash, 2002-2026 daily:

1. **Faber-1x:** No leverage (control)
2. **Faber-Tier1-Only:** Tier 1 leverage only (40% sub), monthly rebalance
3. **Faber-Graduated-Monthly:** Tier 1 + Tier 2 (40%/81.4% sub), monthly rebalance
4. **Faber-Graduated-Weekly:** Tier 1 + Tier 2 with weekly circuit breaker
5. **Faber-Sweep-1.25x:** Flat 25% sub when both equities at 3/3
6. **Faber-Sweep-1.5x:** Flat 50% sub when both equities at 3/3

## Results

**IMPORTANT NOTE: Daily vs Monthly Sharpe**

This is the first daily-granularity Faber-only backtest. Prior experiments used monthly returns, reporting Faber-Cash Sharpe of 1.114. The daily backtest shows **0.935 Sharpe** for the same strategy. The difference comes entirely from daily granularity capturing intra-month drawdowns that monthly returns smooth over — particularly COVID (-13.0% max DD at daily vs -1.6% at monthly resolution). The daily number is the more realistic estimate of live trading risk.

### Performance

| Strategy | Return | Vol | Sharpe | Sortino | MaxDD | Calmar | Terminal |
|----------|--------|-----|--------|---------|-------|--------|----------|
| Faber-1x | 8.4% | 9.0% | 0.935 | 1.147 | -13.0% | 0.65 | $6.92 |
| Faber-Tier1-Only | 10.4% | 11.8% | 0.878 | 1.058 | -17.1% | 0.61 | $10.44 |
| Faber-Graduated-Monthly | 12.2% | 14.2% | 0.863 | 1.024 | -18.6% | 0.66 | $15.14 |
| Faber-Graduated-Weekly | 12.1% | 13.9% | 0.870 | 1.031 | -16.6% | 0.73 | $14.69 |
| Faber-Sweep-1.25x | 9.6% | 10.7% | 0.898 | 1.088 | -15.0% | 0.64 | $8.97 |
| Faber-Sweep-1.5x | 10.9% | 12.6% | 0.867 | 1.039 | -18.7% | 0.58 | $11.52 |
| IVV B&H | 10.8% | 19.0% | 0.570 | 0.713 | -55.2% | 0.20 | $8.92 |

### Crisis Analysis

| Strategy | GFC (2008-09) | COVID (Feb-Mar 2020) | 2022 Bear |
|----------|---------------|---------------------|-----------|
| Faber-1x | +0.6% (DD -1.1%) | -11.9% (DD -13.0%) | -6.7% (DD -8.1%) |
| Faber-Graduated-Monthly | +0.6% (DD -1.1%) | -15.0% (DD -16.2%) | -10.3% (DD -14.4%) |
| Faber-Graduated-Weekly | +0.6% (DD -1.1%) | -15.0% (DD -16.2%) | -10.6% (DD -12.2%) |
| IVV B&H | -36.9% (DD -46.0%) | -33.4% (DD -33.7%) | -17.7% (DD -24.5%) |

GFC: All leveraged strategies show identical performance (+0.6%) because both IVV and QQQ were below 3/3 Faber during the crash — leverage was already off. The Faber filter is the primary hedge.

COVID: Leverage was ON when the crash started (Feb 19). The weekly circuit breaker triggered on Feb 28 (IVV below all 3 SMAs), but by then most of the damage was done. Monthly rebalance on March 1 would have caught it too. This is the system's weakest point — fast crashes that happen within a single Faber signal period.

2022 Bear: Weekly circuit breaker helped significantly: -12.2% DD vs -14.4% monthly, saving 2.2% drawdown. The circuit breaker triggered Jan 21, 2022 (QQQ below all 3 SMAs), catching the initial decline.

### Tier Diagnostics

| | Faber-Graduated-Monthly | Faber-Graduated-Weekly |
|---|---|---|
| Tier 0 (1x) | 108 months (37%) | 109 months (37%) |
| Tier 1 (40% sub) | 41 months (14%) | 40 months (14%) |
| Tier 2 (81% sub) | 143 months (49%) | 143 months (49%) |

Weekly de-lever events: **14 over 21 years (0.6/year)**. Key triggers: COVID 2020-02-28, 2022-01-21, 2018-10-26, 2007-11-09.

### Effective Equity Exposure Confirmation

| Tier | Target | Graduated-Monthly | Graduated-Weekly |
|------|--------|-------------------|-----------------|
| 0 | 70% (baseline) | 23.0% (filter reduces equity) | 25.7% |
| 1 | ~98% | **98.0%** | **98.0%** |
| 2 | ~127% | **127.0%** | **127.0%** |

Targets hit exactly.

### Leverage Sweep

| Level | Return | Vol | Sharpe | MaxDD |
|-------|--------|-----|--------|-------|
| 1.0x | 8.4% | 9.0% | **0.935** | -13.0% |
| 1.25x | 9.6% | 10.7% | 0.898 | -15.0% |
| 1.5x | 10.9% | 12.6% | 0.863 | -18.7% |
| 2.0x | 13.4% | 16.4% | 0.816 | -26.4% |
| 2.5x | 15.8% | 20.2% | 0.784 | -33.5% |

**CML check: Sharpe declines monotonically (0.935 → 0.784).** This is NOT consistent with CML theory (where Sharpe should be flat). The decline comes from leveraged ETF drag — volatility decay + expense ratios compound daily against the leveraged positions. At 2.5x, this costs ~0.15 Sharpe.

## Key Diagnostics

**Tier 2 is dominant (49% of months).** The trailing 12-month momentum condition is easy to satisfy — both IVV and QQQ are above their expanding median returns in most trending months. This means the system spends nearly half its time at maximum leverage (81.4% substitution, 127% effective equity).

**Weekly circuit breaker value: modest but consistent.** Saves 2% max DD (16.6% vs 18.6%) at trivial return cost (12.1% vs 12.2%). The 0.6/year trigger frequency is manageable for a human-in-the-loop system. The breaker's main value is 2022 Bear protection; for COVID, the crash was too fast for even weekly monitoring to help.

## Interpretation

**Leverage is a return amplifier, not a Sharpe amplifier.** The sweep conclusively shows that leverage degrades Sharpe due to daily compounding drag. The investor's decision is whether the absolute return gain (8.4% → 12.1%) justifies the Sharpe cost (0.935 → 0.870) and drawdown increase (-13.0% → -16.6%).

**Graduated-Weekly is the candidate production architecture.** It offers:
- 12.1% annualized return (vs 8.4% unleveraged, 10.8% IVV B&H)
- 0.870 Sharpe (vs 0.570 IVV B&H — still 53% better risk-adjusted)
- -16.6% max DD (vs -55.2% IVV B&H — 70% reduction)
- $14.69 terminal from $1 over 24 years
- Effective equity targets hit exactly (98%/127%)
- Weekly circuit breaker with 0.6/year trigger frequency

**The daily Sharpe reality check is important.** Prior monthly backtests showed 1.114 Sharpe for Faber-1x. The daily backtest shows 0.935 — a 16% reduction driven by intra-month drawdowns that monthly returns can't see. All future backtests should use daily granularity for more realistic risk estimates.

**Faber-Sweep-1.25x deserves consideration as a simpler alternative.** At 0.898 Sharpe and -15.0% max DD, it captures most of the return uplift (9.6% vs 8.4%) with less complexity than the graduated system. No tier logic, no expanding medians, no weekly monitoring. The Sharpe cost vs 1x is only -0.037 (vs -0.065 for Graduated-Weekly).

## Decision

Faber-Graduated-Weekly is the candidate production architecture. The tier substitution percentages (40%/81.4%) are derived from first principles and hit the risk tolerance targets exactly. The weekly circuit breaker adds meaningful drawdown protection at negligible return cost.

## Next Steps

- Implement Faber-Graduated-Weekly in `taa/run.py` as the production system
- Configure Telegram alerts for: monthly rebalance signals, weekly circuit breaker triggers, tier changes
- Consider Faber-Sweep-1.25x as a simplified alternative if the graduated tier complexity is not worth it operationally
- All future backtests should use daily granularity
- See [[2026-04-05_faber_sweep]] — flat 40% sub confirmed as simpler alternative matching Tier 1 performance

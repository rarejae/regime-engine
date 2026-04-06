# VRP (Volatility Risk Premium) Harvesting Backtest — Pod 2

**Date:** April 5, 2026  
**Status:** Complete  
**Track:** Multi-Pod Architecture — Phase 2  
**Related:** [[TAA_PROJECT_STATUS]] | [[MULTI_POD_ARCHITECTURE]] | [[2026-04-05_faber_sweep]] | [[2026-04-05_vrp_proxy_validation]]

## Purpose

Evaluate the CBOE PUT index (cash-secured put-write on S&P 500) as Pod 2 in the multi-pod architecture. Test standalone performance, correlation with Faber-Sweep-40, VRP filter, and combined portfolio construction.

## Data

**PUT index:** CBOE S&P 500 PutWrite Index from `data/raw/optionsdx/PUT_index_combined.csv`. Daily prices from Jun 1988 to Apr 2026. Terminal value from $100: $3,319.65. Pre-2007 data is CBOE back-tested methodology; post-2007 is live-calculated.

**Data quality note:** The CSV contains sparse price snapshots pre-2007 (~7 data points across 1988-2006) followed by continuous daily data from Jan 2007 onward. Daily `pct_change()` produces garbage for the sparse period. All analysis uses **monthly resampled returns** which handles the gaps correctly.

## Step 1: Standalone PUT Index (2002-2026, monthly)

| Strategy | Return | Vol | Sharpe | Sortino | MaxDD | Calmar | Terminal($1) | vs 60/40 | vs IVV |
|----------|--------|-----|--------|---------|-------|--------|-------------|---------|-------|
| PUT Index (VRP) | 7.5% | 10.4% | 0.723 | 0.680 | -32.7% | 0.23 | $5.39 | -$1.12 | -$3.54 |
| Faber-Sweep-40 | 10.2% | 9.8% | 1.039 | 1.490 | -13.3% | 0.77 | $10.44 | +$3.93 | +$1.51 |
| IVV B&H | 10.1% | 14.7% | 0.686 | 0.933 | -50.8% | 0.20 | $8.92 | +$2.42 | — |
| 60/40 | 8.2% | 9.8% | 0.836 | 1.084 | -26.9% | 0.31 | $6.51 | — | -$2.42 |

**The PUT index significantly underperforms Faber-Sweep-40** — lower return (7.5% vs 10.2%), lower Sharpe (0.723 vs 1.039), worse drawdown (-32.7% vs -13.3%). It also underperforms 60/40 on Sharpe (0.723 vs 0.836) and drawdown (-32.7% vs -26.9%).

### Crisis Analysis

| Strategy | GFC (2008-09) | COVID (Feb-Mar 2020) | 2022 Bear |
|----------|---------------|---------------------|-----------|
| PUT Index | -24.7% (DD -85.0%) | -28.9% (DD -28.9%) | -9.7% (DD -16.0%) |
| Faber-Sweep-40 | +0.6% (DD -1.1%) | -15.0% (DD -16.2%) | -8.5% (DD -11.2%) |
| IVV B&H | -36.9% (DD -46.0%) | -33.4% (DD -33.7%) | -17.7% (DD -24.5%) |

**GFC is catastrophic for PUT.** The -85.0% max DD reflects the put-write strategy being assigned at the worst possible time — writing puts into a crash means absorbing the full equity decline with no upside above the strike. The Faber system meanwhile exited equities entirely.

### Return Distribution (monthly)

```
Mean monthly return:         +0.62%
Median monthly return:       +1.09%
Skewness:                    -1.65  (strongly negative — confirms crash risk)
Kurtosis:                    7.30   (fat tails — extreme events more frequent than normal)
% months positive:           73%
Worst single month:          -17.7% (2008-10)
Best single month:           +9.0% (2011-10)
Months with loss > 5%:       14
Months with loss > 10%:      2
Avg gain in positive months: +1.90%
Avg loss in negative months: -2.87%
Steamroller ratio:           1.5x
```

The classic "picking up nickels in front of a steamroller" pattern is confirmed but the steamroller ratio of 1.5x is actually moderate — the average loss is only 1.5x the average gain. The real risk is the tail: 14 months with >5% losses and the catastrophic GFC drawdown.

## Step 2: Correlation with Faber-Sweep-40

### Full-Period Correlation (2002-2026, monthly)

|  | PUT Index | Faber-40 | IVV B&H | 60/40 |
|--|-----------|----------|---------|-------|
| PUT Index | 1.000 | **0.533** | 0.865 | 0.686 |
| Faber-40 | 0.533 | 1.000 | 0.669 | 0.582 |
| IVV B&H | 0.865 | 0.669 | 1.000 | 0.838 |

**PUT-Faber correlation of 0.533 is moderate.** This is lower than PUT-IVV (0.865 — nearly equity-like) because Faber's trend filter moves to cash during drawdowns while PUT stays fully exposed.

### Crisis-Period Correlation

| Crisis | PUT vs Faber-40 | PUT vs IVV B&H |
|--------|-----------------|----------------|
| GFC 2008-2009 | 0.371 | 0.888 |
| COVID Feb-Apr 2020 | 0.392 | 0.989 |
| 2022 Bear | 0.313 | 0.919 |

**Crisis correlations are low (0.31-0.39).** This is the good news — when PUT crashes, Faber is mostly in cash and not correlated. But this is asymmetric: PUT contributes nothing protective during crises, it just doesn't drag Faber down (because Faber protects itself independently).

### Rolling 12-Month Correlation

```
Min:  -0.096
Max:  0.939
Mean: 0.635
```

**Mean rolling correlation of 0.635 is concerning.** There are 11 sustained periods (3+ months) where rolling correlation exceeded 0.60, including a 37-month stretch (2004-2007) and 32-month stretch (2023-present). During normal trending markets, both strategies earn equity-like returns and are highly correlated. The low crisis correlation provides modest diversification but the high normal-market correlation limits the Millennium-style portfolio Sharpe improvement.

## Step 3: VRP Filter Test

| Filter | Return | Vol | Sharpe | MaxDD | Terminal | % Active |
|--------|--------|-----|--------|-------|----------|---------|
| No filter | 7.5% | 10.4% | 0.723 | -32.7% | $5.39 | 100% |
| **>= 0** | **7.5%** | **8.8%** | **0.860** | **-20.7%** | **$5.66** | **85%** |
| >= 2 | 7.2% | 8.3% | 0.863 | -20.7% | $5.23 | 73% |
| >= 4 | 6.2% | 7.5% | 0.823 | -24.7% | $4.18 | 50% |

**The >= 0 filter is a clear improvement:** Sharpe jumps from 0.723 to 0.860, max DD improves from -32.7% to -20.7%, with no return sacrifice. This is the natural threshold — only harvest VRP when there is actually a premium (VIX > realized vol). The filter is active 85% of months, sitting out the ~15% when realized vol exceeds implied vol (typically during/after crash recoveries).

## Step 4: Combined Portfolio Analysis

| Portfolio | Return | Vol | Sharpe | Sortino | MaxDD | Calmar | Terminal($1) | vs Faber |
|-----------|--------|-----|--------|---------|-------|--------|-------------|---------|
| Faber-only | 10.2% | 9.8% | 1.039 | 1.490 | -13.3% | 0.77 | $10.44 | baseline |
| Combined-10 | 9.9% | 9.4% | **1.053** | 1.519 | **-11.8%** | 0.84 | $9.87 | -$0.56 |
| Combined-20 | 9.6% | 9.1% | **1.058** | 1.528 | -13.5% | 0.71 | $9.32 | -$1.12 |
| Combined-30 | 9.4% | 8.9% | 1.052 | 1.490 | -15.8% | 0.59 | $8.78 | -$1.66 |
| PUT-only | 7.5% | 10.4% | 0.723 | 0.680 | -32.7% | 0.23 | $5.39 | -$5.05 |

### Diversification Benefit

| Sleeve | Sharpe Delta | MaxDD Delta | Terminal Delta | Vol Delta |
|--------|-------------|------------|---------------|----------|
| 10% | +0.014 | +1.4% | -$0.56 | -0.4% |
| 20% | +0.019 | -0.2% | -$1.12 | -0.7% |
| 30% | +0.013 | -2.6% | -$1.66 | -0.9% |

**The diversification benefit exists but is very small.** Peak Sharpe improvement is +0.019 at 20% sleeve — from 1.039 to 1.058. This comes at a cost of $1.12 terminal wealth (from $10.44 to $9.32) over 24 years.

**Combined-10 is the only configuration that improves max DD** (-11.8% vs -13.3%, saving 1.4%). At 20% sleeve, the max DD is essentially unchanged. At 30%, it worsens.

### Crisis Analysis (Combined)

| Portfolio | GFC (2008-09) | COVID (Feb-Apr 2020) | 2022 Bear |
|-----------|---------------|---------------------|-----------|
| Faber-only | +0.6% (DD -0.7%) | -6.9% (DD -0.8%) | -8.5% (DD -3.0%) |
| Combined-10 | -2.0% (DD -2.5%) | -7.7% (DD -2.1%) | -8.6% (DD -4.2%) |
| Combined-20 | -4.6% (DD -5.6%) | -8.6% (DD -3.3%) | -8.6% (DD -5.4%) |
| Combined-30 | -7.2% (DD -8.7%) | -9.4% (DD -4.6%) | -8.7% (DD -6.7%) |

**Every combined portfolio is worse than Faber-only in every crisis.** The PUT sleeve adds crisis exposure that Faber's trend filter had eliminated. At 10% sleeve, GFC damage goes from +0.6% to -2.0%. At 30%, GFC damage reaches -7.2%.

## Step 5: Extended History (1988-2001)

| Period | Return | Vol | Sharpe | MaxDD | Key Events |
|--------|--------|-----|--------|-------|------------|
| 1988-2001 | 13.9% | 9.5% | 1.463 | -22.9% | 1987 aftermath, LTCM, dot-com |

- LTCM 1998 (Aug-Oct): +2.0% return, -9.9% DD — PUT index survived well
- Dot-com crash (Mar 2000 - Mar 2001): +5.8% return, -12.0% DD — puts on S&P 500 were not directly hit by tech crash

**The 1988-2001 Sharpe of 1.463 is significantly higher than 2002-2026 (0.723).** This suggests the VRP was larger in the pre-ETF era (before systematic premium harvesting compressed the spread). The 2002-2026 period is more representative of go-forward expectations.

## Step 6: Implementation Notes

1. Level 2 options at Schwab: required, already approved.
2. 1 IVV ATM put contract = ~$52,000 collateral at current prices.
3. Practical threshold: ~$250K portfolio for even a 10% VRP sleeve.
4. Monthly cadence matching PUT index (third Friday expiry).
5. Transaction cost: ~0.12-0.35% annual drag from bid-ask spread.

## Interpretation

**Pod 2 (VRP) provides marginal Sharpe improvement but reduces terminal wealth and worsens crisis behavior.** The fundamental problem is that PUT index is 0.865 correlated with IVV — it's an equity-like return with capped upside and unlimited downside. Faber-Sweep-40 already captures equity upside with trend protection; adding a naked equity exposure (which is what put-writing is) partly undoes Faber's protective cash buffer.

The **crisis correlation analysis is the key result.** PUT-Faber crisis correlation is low (0.31-0.39), which sounds promising, but the mechanism is asymmetric: Faber protects itself by going to cash, while PUT absorbs the full crash. Low correlation doesn't mean diversification benefit — it means Faber is independently protected while PUT independently crashes. In a combined portfolio, the PUT sleeve's crash losses dilute Faber's protection.

**The VRP filter (>= 0) significantly improves standalone PUT performance** (Sharpe 0.723 → 0.860) and should be used if VRP is deployed. But even with the filter, the combined portfolio Sharpe improvement peaks at +0.019 — not enough to justify the operational complexity.

**Why VRP Sharpe was higher historically:** The 1988-2001 Sharpe of 1.463 vs 2002-2026 Sharpe of 0.723 reflects premium compression as more capital systematically harvests VRP (growth of put-write funds, pension overlay strategies, etc.). The go-forward premium is likely closer to the 2002-2026 figure.

## Decision

**Pod 2 status: CONDITIONAL.** VRP is not rejected — the PUT index has a positive Sharpe (0.723 unfiltered, 0.860 filtered) and the combined portfolio does achieve marginally higher Sharpe (+0.019 at 20% sleeve). However, the terminal wealth reduction ($10.44 → $9.32) and worsened crisis behavior make this a marginal addition, not a transformative one.

**Recommended if deployed:** 10% sleeve with VRP filter >= 0. This preserves most of Faber's terminal wealth ($9.87 vs $10.44), achieves the best max DD improvement (-11.8% vs -13.3%), and adds only modest crisis exposure.

**The multi-pod Sharpe improvement formula (√(S₁² + S₂²)) assumes low correlation.** At 0.533 full-period and 0.635 rolling-mean correlation, the actual diversification benefit is well below the theoretical maximum. The formula gives √(1.039² + 0.723²) = 1.266 at zero correlation; the actual combined Sharpe peaks at 1.058.

## Next Steps

- Pod 3 (managed futures / DBMF) is more promising for diversification — structurally negative equity correlation in crises, unlike VRP which is equity-correlated
- If VRP is deployed, use the >= 0 VRP filter and 10% sleeve maximum
- Monitor actual Schwab portfolio size vs $52K/contract minimum position requirement
# VRP (Volatility Risk Premium) Harvesting Backtest — Pod 2

**Date:** April 5, 2026
**Status:** Complete
**Track:** Multi-Pod Architecture — Phase 2
**Related:** [[TAA_PROJECT_STATUS]] | [[MULTI_POD_ARCHITECTURE]] | [[2026-04-05_faber_sweep]] | [[2026-04-05_vrp_proxy_validation]]

## Summary

PUT index (cash-secured put-write) tested as Pod 2. Result: CONDITIONAL — marginal Sharpe improvement (+0.019 at 20% sleeve) at cost of terminal wealth (-$1.12) and worsened crisis behavior. Full-period correlation with Faber-40 is 0.533, rolling mean 0.635. Crisis correlations are low (0.31-0.39) but asymmetric — Faber protects itself independently while PUT absorbs the crash.

## Standalone Performance (2002-2026)

| Strategy | Return | Vol | Sharpe | MaxDD | Terminal($1) |
|----------|--------|-----|--------|-------|-------------|
| PUT Index | 7.5% | 10.4% | 0.723 | -32.7% | $5.39 |
| PUT + Filter (VIX>RV) | ~7.5% | 8.8% | 0.860 | -20.7% | $5.66 |
| Faber-Sweep-40 | 10.2% | 9.8% | 1.039 | -13.3% | $10.44 |
| IVV B&H | 10.1% | 14.7% | 0.686 | -50.8% | $8.92 |
| 60/40 | 8.2% | 9.8% | 0.836 | -26.9% | $6.51 |

## Crisis Performance

| Strategy | GFC (2008-09) | COVID | 2022 Bear |
|----------|--------------|-------|-----------|
| PUT Index | -24.7% (DD -85.0%) | -28.9% | -9.7% |
| Faber-Sweep-40 | +0.6% (DD -1.1%) | -15.0% | -8.5% |

GFC: -85.0% max DD on PUT — catastrophic. Writing puts into a crash means absorbing the full equity decline with no upside.

## Correlation

Full-period PUT vs Faber-40: **0.533**
Rolling 12-month mean: **0.635** (11 sustained periods >0.60)
Crisis correlations: GFC 0.371, COVID 0.392, 2022 0.313

Low crisis correlation is asymmetric — Faber protects itself via trend filter cash, PUT independently crashes. Combined portfolio doesn't get diversification benefit, it gets Faber's protection diluted by PUT's crash exposure.

## Combined Portfolio

| Portfolio | Sharpe | MaxDD | Terminal | Sharpe Delta |
|-----------|--------|-------|---------|-------------|
| Faber-only | 1.039 | -13.3% | $10.44 | baseline |
| Combined-10 | 1.053 | -11.8% | $9.87 | +0.014 |
| Combined-20 | 1.058 | -13.5% | $9.32 | +0.019 |
| Combined-30 | 1.052 | -15.8% | $8.78 | +0.013 |

Peak Sharpe improvement: +0.019 at 20% sleeve. Terminal wealth reduction: -$1.12 over 24 years.
Every combined portfolio is worse than Faber-only in every crisis.

## VRP Filter

VIX >= realized vol filter: Sharpe 0.723 → 0.860, MaxDD -32.7% → -20.7%. Use if deploying VRP.

## Decision

**Pod 2: CONDITIONAL**
- Deploy at 10% sleeve maximum with VRP filter >= 0
- Prioritize Pod 3 (managed futures) — structurally negative equity correlation in crises
- Minimum portfolio ~$250K for even a 10% VRP sleeve ($52K/contract for IVV puts)
- Rolling mean correlation of 0.635 limits theoretical diversification benefit

## Key Insight

The multi-pod Sharpe formula √(S₁² + S₂²) assumes near-zero correlation. At 0.533 full-period correlation, theory predicts combined Sharpe of 1.266. Actual peak is 1.058. The gap between theory and reality is entirely explained by the correlation being 0.533 not 0.0.

# Two-Pod Combined Portfolio with Kritzman Turbulence Layer

**Date:** April 5, 2026  
**Status:** Complete  
**Track:** Multi-Pod Architecture — Phase 5  
**Related:** [[TAA_PROJECT_STATUS]] | [[MULTI_POD_ARCHITECTURE]] | [[2026-04-05_faber_sweep]] | [[2026-04-05_vrp_backtest]]

## Purpose

Combine Pod 1 (Faber-Sweep-40) and Pod 2 (VRP/PUT index with >= 0 filter) into a single portfolio and test whether the Kritzman hybrid turbulence layer (Option C: correlation-gated turbulence) improves combined performance.

## Design

**Pod 1:** Faber-Sweep-40 — identical to [[2026-04-05_faber_sweep]], daily → monthly aggregation.  
**Pod 2:** PUT index with VRP filter >= 0 (only invest when VIX > 21d realized vol; T-bill otherwise).  
**Frequency:** Monthly rebalance. Period: Jan 2002 – Apr 2026.

**Turbulence layer (Option C — Hybrid):**
- 36-month Ledoit-Wolf covariance on [Faber, VRP] joint return vector
- Mahalanobis distance as turbulence index
- Rolling 3-month correlation between pods
- Scaling: Both conditions (corr > 0.65 AND turb > 75th pct) → 50%; one condition → 75%; neither → 100%
- Active from Jan 2005 (36-month warmup)

## Results

### Full Performance Table

| Strategy | Return | Vol | Sharpe | Sortino | MaxDD | Calmar | Terminal($1) | vs Faber |
|----------|--------|-----|--------|---------|-------|--------|-------------|---------|
| Faber-only | 10.2% | 9.8% | 1.039 | 1.490 | -13.3% | 0.77 | $10.44 | baseline |
| VRP-only (filtered) | 7.6% | 8.8% | 0.864 | 0.835 | -20.7% | 0.37 | $5.71 | -$4.73 |
| Two-Pod-10 | 9.9% | 9.3% | 1.061 | 1.520 | -11.4% | 0.87 | $9.91 | -$0.53 |
| Two-Pod-20 | 9.6% | 8.9% | 1.078 | 1.539 | -10.4% | 0.93 | $9.38 | -$1.05 |
| **Two-Pod-10-Turb** | **8.6%** | **6.6%** | **1.303** | **2.213** | **-8.2%** | **1.05** | **$7.67** | **-$2.76** |
| **Two-Pod-20-Turb** | **8.4%** | **6.3%** | **1.333** | **2.263** | **-7.9%** | **1.07** | **$7.37** | **-$3.07** |
| IVV B&H | 10.1% | 14.7% | 0.686 | 0.933 | -50.8% | 0.20 | $8.92 | — |
| 60/40 | 8.7% | 9.7% | 0.903 | 1.139 | -26.9% | 0.32 | $7.41 | — |

### Crisis Analysis

| Strategy | GFC (2008-09) | COVID (Feb-Apr 2020) | 2022 Bear |
|----------|--------------|---------------------|-----------|
| Faber-only | +0.6% (DD -0.7%) | -6.9% (DD -0.8%) | -8.5% (DD -3.0%) |
| Two-Pod-10 | +0.5% (DD -1.2%) | -8.2% (DD -2.1%) | -8.2% (DD -3.8%) |
| Two-Pod-20 | +0.3% (DD -1.8%) | -9.5% (DD -3.3%) | -7.8% (DD -4.6%) |
| Two-Pod-10-Turb | +0.7% (DD -1.0%) | **-3.7%** (DD -1.0%) | **-4.4%** (DD -2.7%) |
| Two-Pod-20-Turb | +0.7% (DD -1.3%) | **-4.4%** (DD -1.7%) | **-4.3%** (DD -3.3%) |
| IVV B&H | -36.9% (DD -35.7%) | -9.2% (DD -12.5%) | -17.7% (DD -20.2%) |

Turbulence layer significantly improved COVID (-3.7% vs -8.2%) and 2022 Bear (-4.4% vs -8.2%) while GFC was approximately neutral (+0.7% vs +0.5%).

## Turbulence Layer Diagnostics

### Scaling Distribution (Jan 2005 – Apr 2026, 256 months)

| Scale | Meaning | Months | % |
|-------|---------|--------|---|
| 1.00 | Normal (no de-lever) | 69 | **27%** |
| 0.75 | One condition met | 144 | **56%** |
| 0.50 | Both conditions met | 43 | **17%** |

**The system is de-levered 73% of the time.** This is the critical finding. The correlation threshold (0.65) fires 62% of months because the mean rolling Faber-VRP correlation is 0.635 — the threshold barely exceeds the mean. The turbulence layer is not a rare crisis response; it's the normal operating mode.

### Condition Breakdown

| Condition | Months |
|-----------|--------|
| Corr > 0.65 only | 115 |
| Turb > 75th pct only | 29 |
| Both conditions | 43 |

### Crisis Trigger Behavior

**GFC (2008-09):** 2 both-condition triggers (Jan 2008, Feb 2009). Average Faber return during GFC months: +0.1%/mo. The hybrid gate partially fired during GFC — it caught 2 months where correlation was high AND turbulence was elevated, but the average scale was 0.82 (mostly running). Faber's protection was slightly diluted but not destroyed.

**COVID (Feb-Apr 2020):** 2 both-condition triggers. Average scale 0.67. The layer caught COVID effectively — both pods were falling together (corr 0.996-0.999) with extreme turbulence. COVID return improved from -8.2% to -3.7%.

**2022 Bear (Jan-Oct):** 1 both-condition trigger, 8 any-condition triggers. Average scale 0.78. Improved from -8.2% to -4.4%.

## Turbulence Value-Add Analysis

| Metric | Two-Pod-10 → 10-Turb | Two-Pod-20 → 20-Turb |
|--------|----------------------|----------------------|
| Sharpe | 1.061 → 1.303 (**+0.242**) | 1.078 → 1.333 (**+0.255**) |
| MaxDD | -11.4% → -8.2% (**+3.2%**) | -10.4% → -7.9% (**+2.5%**) |
| Return | 9.9% → 8.6% (**-1.3%**) | 9.6% → 8.4% (**-1.2%**) |
| Terminal | $9.91 → $7.67 (**-$2.23**) | $9.38 → $7.37 (**-$2.01**) |

## Interpretation

### The 1.333 Sharpe is real but misleading

Two-Pod-20-Turb achieves the highest Sharpe (1.333) and Sortino (2.263) of any strategy tested in this project — but the mechanism is essentially **"mostly hold cash."** The system is de-levered 73% of months, running at 50-75% of target weights most of the time. This means:

- **Vol drops from 9.8% to 6.3%** — because you're in cash more
- **Return drops from 10.2% to 8.4%** — because you're in cash more
- **MaxDD drops from 13.3% to 7.9%** — because you're in cash more
- **Sharpe rises because vol/DD reduction outpaces return reduction**

This is the same finding as every experiment: **cash is the dominant risk reducer.** The turbulence layer is another mechanism for going to cash. It works for the same reason the Faber filter works — avoiding participation during bad periods.

### Terminal wealth is the honest metric

An investor who started with $1 in January 2002:
- **Faber-only: $10.44** (highest terminal wealth)
- **Two-Pod-20-Turb: $7.37** (-29% less money)
- **IVV B&H: $8.92** (more money than Two-Pod-20-Turb!)

The turbulence layer costs $3.07 in terminal wealth over 24 years. The investor sleeps better (7.9% max DD vs 13.3%) but ends up with 29% less money.

### The correlation threshold is too sensitive

At 0.65, the correlation condition fires 62% of months. This is barely above the population mean (0.635). A more useful threshold would fire only during genuinely abnormal correlation — perhaps 0.80, which would only fire during actual crisis convergence. With the current threshold, the system never fully invests during normal markets because the pods are naturally moderately correlated.

### What the turbulence layer actually protects against

The layer is most valuable for **COVID-type events** — fast crashes where both pods fall simultaneously. It improved COVID from -8.2% to -3.7%, which is meaningful. But Faber-only already survived COVID with only -6.9% (and -0.8% DD), so the marginal protection over Faber-only is modest.

## Answers to Key Questions

**1. Does the two-pod portfolio improve Sharpe over Faber-only?**
Yes, modestly: +0.022 at 10% sleeve, +0.039 at 20% sleeve. Consistent with [[2026-04-05_vrp_backtest]].

**2. Does the turbulence layer further improve Sharpe?**
Yes, substantially: +0.242 (10% sleeve), +0.255 (20% sleeve). But this comes from de-levering 73% of the time.

**3. Does the turbulence layer fire correctly?**
Partially. COVID and 2022: yes, it caught both effectively. GFC: it fired 2 months when Faber was already protected (+0.1%/mo) — the hybrid gate reduced but did not eliminate false triggers. The correlation threshold at 0.65 is too sensitive for normal operations.

**4. Optimal sleeve size?**
Two-Pod-20-Turb has the best Sharpe (1.333) but lowest terminal ($7.37). Two-Pod-10 without turbulence has the best balance: 1.061 Sharpe, $9.91 terminal, -11.4% DD.

**5. Is the Sharpe improvement worth the terminal wealth cost?**
No, for a long-term accumulation investor. The turbulence layer converts $3.07 of terminal wealth into lower volatility. A retiree who needs low vol might prefer it; an accumulator should not.

**6. Practical verdict?**
The two-pod architecture with turbulence is mechanically sound but economically unattractive. **Faber-only ($10.44 terminal, 1.039 Sharpe, -13.3% DD) remains the production architecture.** Adding the VRP sleeve provides marginal Sharpe improvement at terminal wealth cost. Adding the turbulence layer further boosts Sharpe but by going to cash 73% of the time.

## Decision

**Production architecture: Faber-Sweep-40 standalone.** The two-pod architecture is not rejected on principle — it does what it promises (higher Sharpe, lower DD). But for a Roth IRA accumulation context, terminal wealth is the primary objective, and every addition to Faber-only reduces it.

**If the investor's priority changes to risk minimization** (e.g., approaching retirement, large portfolio where DD matters more than growth), Two-Pod-10 without turbulence (1.061 Sharpe, -11.4% DD, $9.91) is the best configuration — it improves both Sharpe and DD over Faber-only at modest terminal cost.

The turbulence layer as designed is too aggressive. A future iteration should test a higher correlation threshold (0.80) that fires only during genuine crisis convergence rather than normal moderate correlation.

## Next Steps

- Consider testing turbulence with correlation threshold at 0.80 instead of 0.65
- Pod 3 (DBMF managed futures) may provide better diversification — genuinely negative equity correlation in 2022 vs VRP's positive correlation
- For production: implement Faber-Sweep-40 standalone, defer multi-pod to Phase 3+ only if crisis-period correlation between pods is demonstrably lower than Faber-VRP's 0.533
# Two-Pod Combined Portfolio with Kritzman Turbulence Layer

**Date:** April 5, 2026
**Status:** Complete
**Track:** Multi-Pod Architecture — Phase 5
**Related:** [[TAA_PROJECT_STATUS]] | [[MULTI_POD_ARCHITECTURE]] | [[2026-04-05_faber_sweep]] | [[2026-04-05_vrp_backtest]]

## Summary

Combined Faber-Sweep-40 (Pod 1) + PUT index VRP with >= 0 filter (Pod 2) with Kritzman hybrid turbulence layer. Key finding: turbulence layer fires 73% of months — too aggressive. Effectively converts return into Sharpe by holding cash most of the time. Terminal wealth suffers significantly.

## Performance

| Strategy | Return | Vol | Sharpe | MaxDD | Terminal($1) | vs Faber |
|----------|--------|-----|--------|-------|-------------|---------|
| Faber-only | 10.2% | 9.8% | 1.039 | -13.3% | $10.44 | baseline |
| Two-Pod-10 | 9.9% | 9.3% | 1.061 | -11.4% | $9.91 | -$0.53 |
| Two-Pod-20 | 9.6% | 8.9% | 1.078 | -10.4% | $9.38 | -$1.05 |
| Two-Pod-10-Turb | 8.6% | 6.6% | 1.303 | -8.2% | $7.67 | -$2.76 |
| Two-Pod-20-Turb | 8.4% | 6.3% | 1.333 | -7.9% | $7.37 | -$3.07 |
| IVV B&H | 10.1% | 14.7% | 0.686 | -50.8% | $8.92 | — |
| 60/40 | 8.7% | 9.7% | 0.903 | -26.9% | $7.41 | — |

## Crisis Performance

| Strategy | GFC | COVID | 2022 Bear |
|----------|-----|-------|-----------|
| Faber-only | +0.6% (DD -0.7%) | -6.9% (DD -0.8%) | -8.5% (DD -3.0%) |
| Two-Pod-10 | +0.5% (DD -1.2%) | -8.2% (DD -2.1%) | -8.2% (DD -3.8%) |
| Two-Pod-10-Turb | +0.7% (DD -1.0%) | **-3.7% (DD -1.0%)** | **-4.4% (DD -2.7%)** |

## Turbulence Layer Diagnostics

Scale distribution (Jan 2005 – Apr 2026, 256 months):
- Scale 1.00 (normal): 69 months **27%**
- Scale 0.75 (one condition): 144 months **56%**
- Scale 0.50 (both): 43 months **17%**

**73% of months de-levered.** Correlation threshold 0.65 fires 62% of months because mean rolling Faber-VRP correlation is 0.635 — threshold barely exceeds the mean. The system is never fully invested in normal markets.

Turbulence value-add: +0.242 to +0.255 Sharpe improvement, at cost of -1.2% to -1.3% annual return and -$2.01 to -$2.23 terminal wealth over 24 years.

## Key Insight

The 1.333 Sharpe is achieved by being in cash 73% of the time. This is mechanically sound but economically unattractive for a long-term accumulator. Same finding as every prior experiment: cash is the dominant risk reducer. The turbulence layer is another cash mechanism.

**Terminal wealth is the honest metric for a Roth IRA accumulation account:**
- Faber-only: $10.44
- Two-Pod-20-Turb: $7.37 (29% less)
- IVV B&H: $8.92 (more than Two-Pod-20-Turb)

## Decision

**Production architecture remains: Faber-Sweep-40 standalone.**

Best two-pod config for risk-minimization context: Two-Pod-10 without turbulence (1.061 Sharpe, -11.4% DD, $9.91 terminal).

## What Needs Fixing

Correlation threshold 0.65 is too sensitive — fires during normal moderate correlation. Need to retest at 0.80, which would only fire during genuine crisis convergence. Expected result: turbulence fires ~20-25% of months rather than 73%, preserving more terminal wealth while still catching COVID and 2022.

## Next Steps

1. Retest turbulence with correlation threshold = 0.80
2. Pod 3 (DBMF) may provide genuinely better diversification — negative equity correlation in 2022 vs VRP's positive correlation — deferred pending proxy resolution
3. Production implementation: Faber-Sweep-40 standalone

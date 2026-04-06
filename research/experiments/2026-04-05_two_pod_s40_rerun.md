# Two-Pod S40 Rerun: Correlation Threshold 0.80

**Date:** April 5, 2026  
**Status:** Complete  
**Track:** Multi-Pod Architecture — Phase 5  
**Related:** [[TAA_PROJECT_STATUS]] | [[MULTI_POD_ARCHITECTURE]] | [[2026-04-05_two_pod_combined]]

## Purpose

Rerun the two-pod combined portfolio with correlation threshold raised from 0.65 to 0.80 to test whether a less sensitive turbulence trigger produces a better return/Sharpe/terminal tradeoff. The prior run de-levered 73% of months — far too aggressive. This run explicitly confirms Faber-Sweep-40 (40% SSO/QLD leverage) as Pod 1.

## Key Change

Correlation threshold: **0.65 → 0.80**. All other mechanics identical to [[2026-04-05_two_pod_combined]].

## Results

### Full Performance Table

| Strategy | Return | Vol | Sharpe | Sortino | MaxDD | Calmar | Terminal($1) | vs S40-only |
|----------|--------|-----|--------|---------|-------|--------|-------------|-----------|
| Faber-S40-only | 10.2% | 9.8% | 1.039 | 1.490 | -13.3% | 0.77 | **$10.44** | baseline |
| Faber-1x-only | 8.2% | 7.3% | 1.125 | 1.680 | -9.3% | 0.88 | $6.92 | -$3.52 |
| VRP-only (filtered) | 7.6% | 8.8% | 0.864 | 0.835 | -20.7% | 0.37 | $5.71 | -$4.73 |
| Two-Pod-S40-10 | 9.9% | 9.3% | 1.061 | 1.520 | -11.4% | 0.87 | $9.91 | -$0.53 |
| Two-Pod-S40-20 | 9.6% | 8.9% | 1.078 | 1.539 | -10.4% | 0.93 | $9.38 | -$1.05 |
| Two-Pod-S40-30 | 9.4% | 8.6% | **1.088** | 1.526 | -11.3% | 0.83 | $8.87 | -$1.56 |
| Two-Pod-S40-10-Turb | 8.5% | 6.8% | 1.251 | 2.035 | -8.2% | 1.04 | $7.45 | -$2.99 |
| Two-Pod-S40-20-Turb | 8.3% | 6.5% | 1.281 | 2.090 | -7.9% | 1.07 | $7.17 | -$3.27 |
| IVV B&H | 10.1% | 14.7% | 0.686 | 0.933 | -50.8% | 0.20 | $8.92 | — |
| 60/40 | 8.7% | 9.7% | 0.903 | 1.139 | -26.9% | 0.32 | $7.41 | — |

### Crisis Analysis

| Strategy | GFC (2008-09) | COVID (Feb-Apr 2020) | 2022 Bear |
|----------|--------------|---------------------|-----------|
| Faber-S40-only | +0.6% (DD -0.7%) | -6.9% (DD -0.8%) | -8.5% (DD -3.0%) |
| Two-Pod-S40-10 | +0.5% (DD -1.2%) | -8.2% (DD -2.1%) | -8.2% (DD -3.8%) |
| Two-Pod-S40-20 | +0.3% (DD -1.8%) | -9.5% (DD -3.3%) | -7.8% (DD -4.6%) |
| Two-Pod-S40-10-Turb | +0.7% (DD -1.0%) | -3.7% (DD -1.0%) | -4.5% (DD -2.8%) |
| Two-Pod-S40-20-Turb | +0.7% (DD -1.3%) | -4.4% (DD -1.7%) | -4.4% (DD -3.4%) |

### Turbulence Diagnostics (threshold 0.80)

| Scale | Meaning | Months | % |
|-------|---------|--------|---|
| 1.00 | Normal | 86 | **34%** |
| 0.75 | One condition | 134 | 52% |
| 0.50 | Both conditions | 36 | 14% |

**De-levered 66% of months** (was 73% at threshold 0.65). Target was ~20-30%. Still far too aggressive.

Condition breakdown: Corr > 0.80 only: 98 months. Turb > 75th pct only: 36 months. Both: 36 months.

The correlation condition fires 52% of months at threshold 0.80 — the pods spend over half their time with 3-month rolling correlation above 0.80. This is not abnormal convergence; it's the normal operating state for these two equity-correlated pods.

### Comparison with Prior Run (threshold 0.65)

| Metric | Prior (thr 0.65) | This run (thr 0.80) |
|--------|-----------------|-------------------|
| Two-Pod-S40-10-Turb Sharpe | 1.303 | 1.251 |
| Two-Pod-S40-20-Turb Sharpe | 1.333 | 1.281 |
| Two-Pod-S40-10-Turb Terminal | $7.67 | $7.45 |
| Two-Pod-S40-20-Turb Terminal | $7.37 | $7.17 |
| Months de-levered | 73% | 66% |

**Raising the threshold made BOTH Sharpe AND terminal wealth worse.** At 0.65, the system held more cash during periods that turned out to be harmful — the tighter threshold accidentally provided better risk reduction. At 0.80, it stayed invested during some periods that hurt. This confirms the turbulence layer's value comes entirely from holding cash, not from sophisticated crisis detection.

### Sleeve Size Tradeoff (without turbulence)

| Sleeve | Return | Sharpe | MaxDD | Terminal | Sharpe Delta | Terminal Delta |
|--------|--------|--------|-------|----------|-------------|---------------|
| 0% (S40-only) | 10.2% | 1.039 | -13.3% | $10.44 | baseline | baseline |
| 10% | 9.9% | 1.061 | -11.4% | $9.91 | +0.022 | -$0.53 |
| 20% | 9.6% | 1.078 | -10.4% | $9.38 | +0.039 | -$1.05 |
| 30% | 9.4% | 1.088 | -11.3% | $8.87 | +0.049 | -$1.56 |
| 10%+Turb | 8.5% | 1.251 | -8.2% | $7.45 | +0.212 | -$2.99 |
| 20%+Turb | 8.3% | 1.281 | -7.9% | $7.17 | +0.242 | -$3.27 |

## Answers to Key Questions

**Q1. How often does turbulence (threshold 0.80) fire?**
De-levered 66% of months. Target was ~20-30%. The threshold change from 0.65 to 0.80 reduced firing by only 7 percentage points (73% → 66%) because the pods are fundamentally too correlated — their 3-month rolling correlation exceeds 0.80 for 52% of months.

**Q2. Does it fire during crises and avoid normal markets?**
COVID: 2 both-triggers, avg scale 0.67 — effective. 2022: 1 both-trigger, avg scale 0.80 — modest. But it also fires extensively during non-crisis periods (134 months at scale 0.75), which is where the cash drag comes from.

**Q3. Does Faber-Sweep-40 base produce better results than prior Faber-1x?**
The prior run already used Faber-Sweep-40 (confirmed by identical terminal wealth: $10.44). The non-turbulence results are identical. The turbulence results are worse at 0.80 threshold.

**Q4. Is the Sharpe improvement meaningful without sacrificing too much terminal?**
Without turbulence: Two-Pod-S40-20 gains +0.039 Sharpe at -$1.05 terminal cost. This is a reasonable tradeoff — the investor pays $1.05 over 24 years for meaningfully better risk-adjusted returns and -10.4% max DD (vs -13.3%).

With turbulence: the additional +0.203 Sharpe costs an additional $2.21 terminal — too expensive.

**Q5. At what sleeve does the tradeoff become unacceptable?**
- 10%: +0.022 Sharpe, -$0.53 terminal → reasonable
- 20%: +0.039 Sharpe, -$1.05 terminal → reasonable
- 30%: +0.049 Sharpe, -$1.56 terminal → marginal (MaxDD worsens vs 20%)

The 20% sleeve is the sweet spot: best Calmar (0.93), lowest MaxDD (-10.4%), reasonable terminal cost.

**Q6. Final production architecture verdict:**
- **For accumulation (maximize terminal wealth): Faber-Sweep-40 standalone ($10.44)**
- **For risk-adjusted growth: Two-Pod-S40-20 without turbulence (1.078 Sharpe, -10.4% DD, $9.38)**
- **Turbulence layer REJECTED** — de-levers 66% of months regardless of threshold. The pods are too correlated for any Kritzman-style overlay to selectively target crises.

## Decision

**Turbulence layer definitively rejected.** Two threshold values tested (0.65 and 0.80); both produce excessive de-levering (73% and 66%). The problem is fundamental: Faber-Sweep-40 and VRP (PUT index) have rolling 3-month correlation exceeding 0.80 for 52% of months — this is their normal state, not a crisis indicator. No threshold between 0.65 and 1.0 can make the turbulence layer fire only during genuine crises.

**Production architecture: Faber-Sweep-40 standalone** for a Roth IRA accumulation context. If the investor wants to add VRP, the 20% sleeve without turbulence (1.078 Sharpe, -10.4% max DD, $9.38 terminal) is the recommended configuration — but this is an optional overlay, not the core system.

## Next Steps

- Implement Faber-Sweep-40 standalone in `taa/run.py`
- Pod 3 (DBMF managed futures) with -0.586 IVV correlation in 2022 is the better diversification candidate — but proxy validation showed no reliable pre-2019 data, limiting backtest to ~7 years
- The turbulence layer concept is sound but needs genuinely uncorrelated pods (like Faber + managed futures) to be useful. Faber + VRP are too equity-correlated.
# Two-Pod S40 Rerun: Correlation Threshold 0.80

**Date:** April 5, 2026
**Status:** Complete
**Track:** Multi-Pod Architecture — Phase 5
**Related:** [[TAA_PROJECT_STATUS]] | [[MULTI_POD_ARCHITECTURE]] | [[2026-04-05_two_pod_combined]]

## Summary

Raised correlation threshold from 0.65 to 0.80. Result: still de-levered 66% of months (vs 73% at 0.65). Target was 20-30%. Fundamental problem confirmed — Faber-Sweep-40 and VRP have rolling 3-month correlation exceeding 0.80 for 52% of months. This is their normal state, not a crisis indicator. Turbulence layer definitively rejected.

## Performance

| Strategy | Return | Vol | Sharpe | MaxDD | Terminal | vs S40 |
|----------|--------|-----|--------|-------|---------|--------|
| Faber-S40-only | 10.2% | 9.8% | 1.039 | -13.3% | $10.44 | baseline |
| Two-Pod-S40-10 | 9.9% | 9.3% | 1.061 | -11.4% | $9.91 | -$0.53 |
| Two-Pod-S40-20 | 9.6% | 8.9% | 1.078 | -10.4% | $9.38 | -$1.05 |
| Two-Pod-S40-30 | 9.4% | 8.6% | 1.088 | -11.3% | $8.87 | -$1.56 |
| Two-Pod-S40-10-Turb | 8.5% | 6.8% | 1.251 | -8.2% | $7.45 | -$2.99 |
| Two-Pod-S40-20-Turb | 8.3% | 6.5% | 1.281 | -7.9% | $7.17 | -$3.27 |

## Turbulence Diagnostics (threshold 0.80)

- Scale 1.00 (normal): 86 months — **34%**
- Scale 0.75 (one condition): 134 months — **52%**
- Scale 0.50 (both): 36 months — **14%**
- De-levered 66% of months (vs 73% at 0.65). Still far above 20-30% target.
- Correlation condition fires 52% of months at threshold 0.80 — normal operating state.

Raising threshold 0.65 → 0.80 made both Sharpe AND terminal worse. The tighter threshold at 0.65 accidentally provided better risk reduction by holding more cash during harmful periods.

## Sleeve Size Tradeoff (no turbulence)

| Sleeve | Sharpe Delta | Terminal Delta | Max DD |
|--------|-------------|---------------|--------|
| 0% | baseline | baseline | -13.3% |
| 10% | +0.022 | -$0.53 | -11.4% |
| 20% | +0.039 | -$1.05 | **-10.4%** ← sweet spot |
| 30% | +0.049 | -$1.56 | -11.3% (worsens) |

20% sleeve is the sweet spot: best Calmar (0.93), lowest MaxDD, reasonable terminal cost.

## Final Decisions

**Turbulence layer: DEFINITIVELY REJECTED**
- Requires genuinely uncorrelated pods (like Faber + managed futures) to be useful
- Faber + VRP are too equity-correlated for any threshold to selectively target crises only
- The concept is sound but needs Pod 3 (managed futures) to work properly

**Production architecture:**
- Accumulation focus: **Faber-Sweep-40 standalone** ($10.44 terminal, 1.039 Sharpe, -13.3% DD)
- Risk-adjusted focus: **Two-Pod-S40-20 without turbulence** ($9.38 terminal, 1.078 Sharpe, -10.4% DD)

**Path to turbulence layer:** Requires Pod 3 with genuine negative equity correlation (DBMF managed futures, -0.586 IVV correlation in 2022). Deferred until managed futures proxy problem resolved or 3+ more years of DBMF live data available.

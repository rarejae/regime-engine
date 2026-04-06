# Universe Expansion: EFA, VNQ, DBA

**Date:** April 5, 2026  
**Status:** Complete  
**Track:** Faber Optimization  
**Related:** [[TAA_PROJECT_STATUS]] | [[2026-04-05_pro_rata_vs_cash]]

## Hypothesis

Adding EFA (international developed), VNQ (REITs), and DBA (agriculture commodities) to the Faber-only system would improve risk-adjusted performance by providing more independent trend signals — assets that trend differently from the existing US equity-dominated universe.

## Design

### Proxy Validation (Hard Gate)

| Asset | ETF | Proxy | Overlap | Ann Ret Corr | Status |
|-------|-----|-------|---------|-------------|--------|
| EFA | actual | (no proxy needed — ETF starts Aug 2001, before backtest) | N/A | N/A | PASS |
| VNQ | actual | RWR (SPDR DJ Wilshire REIT ETF) | 2004-2026 | 0.991 | PASS |
| DBA | actual | (none available — FRED ticker SPGSAGSP no longer exists) | N/A | N/A | PASS* |

\* DBA passes without proxy gate (no proxy to test). ETF-only from Jan 2007. Pre-2008 months: Faber SMA needs 12-month warmup → DBA ineligible until ~Jul 2007.

**Note on specified proxy sources:** The user-specified FRED tickers (WILLREITIND for VNQ, SPGSAGSP for DBA) no longer exist in FRED. ^EAFE is delisted on yfinance. RWR was substituted as VNQ proxy (annual corr 0.991 — excellent). DBA proceeded with ETF-only data.

**MSCI EAFE structural drift:** EFA represents large/mid-cap developed markets ex US/Canada. Composition has evolved (Japan ~60% in 1990, ~20% now). Not relevant here since ETF predates the backtest start, but important context for any future pre-2001 analysis.

### Correlation Analysis

**Full-period (2002-2026, monthly returns) — pairs with corr > 0.70:**
- IVV-QQQ: 0.903 (existing, known)
- IVV-EFA: 0.850 (high — EFA provides limited diversification from IVV)
- QQQ-EFA: 0.724 (moderate-high)

**Crisis-period (2008-2009) — the one that matters:**
- IVV-EFA: 0.932 (nearly identical — EFA provides zero crisis diversification)
- IVV-VNQ: 0.841 (converges in crisis)
- QQQ-EFA: 0.833 (converges in crisis)
- QQQ-VNQ: 0.724
- EFA-VNQ: 0.820
- DBC-DBA: 0.739

**Key observation:** All equity and REIT assets converge to near-1.0 correlation during the GFC. Only VGLT (-0.085 with IVV full period, +0.005 in crisis) and IAU (0.089 with IVV full, 0.053 in crisis) maintain independence. DBA shows moderate independence (0.272 with IVV full, 0.319 in crisis) but is only eligible 34% of months.

### Baseline Weight Recalibration

| Strategy | IVV | QQQ | EFA | VNQ | VGLT | IAU | DBC | DBA | Cash | Equity | Real |
|----------|-----|-----|-----|-----|------|-----|-----|-----|------|--------|------|
| Original | 45% | 25% | — | — | 5% | 10% | 5% | — | 10% | 70% | 15% |
| Expanded-Full | 30% | 15% | 15% | 10% | 5% | 7% | 3% | 5% | 10% | 70% | 15% |
| Expanded-EFA-VNQ | 32% | 18% | 13% | 7% | 5% | 9% | 4% | — | 12% | 70% | 13% |
| Expanded-EFA-Only | 35% | 20% | 12% | — | 5% | 10% | 5% | — | 13% | 67% | 15% |

## Results

| Strategy | Return | Vol | Sharpe | Sortino | MaxDD | Calmar | Terminal |
|----------|--------|-----|--------|---------|-------|--------|----------|
| Faber-Original | 8.3% | 7.4% | 1.112 | 1.664 | -9.6% | 0.86 | $6.94 |
| Faber-Expanded-Full | 7.4% | 6.6% | 1.122 | 1.621 | -8.7% | 0.86 | $5.75 |
| Faber-Expanded-EFA-VNQ | 7.6% | 6.8% | 1.131 | 1.622 | -8.9% | 0.86 | $6.05 |
| Faber-Expanded-EFA-Only | 7.7% | 6.8% | 1.134 | 1.671 | -9.2% | 0.83 | $6.10 |
| IVV B&H | 10.1% | 15.0% | 0.672 | 0.907 | -50.8% | 0.20 | $8.72 |
| 60/40 | 7.9% | 10.0% | 0.786 | 1.054 | -25.7% | 0.31 | $5.97 |

**Sharpe decomposition:**
- Faber-Original: 1.112
- Faber-Expanded-Full: 1.122 (delta: +0.010)
- Faber-Expanded-EFA-VNQ: 1.131 (delta: +0.019)
- Faber-Expanded-EFA-Only: 1.134 (delta: +0.022)

### Crisis Analysis

| Strategy | GFC (2008-09) | COVID (2020) | 2022 Bear |
|----------|---------------|--------------|-----------|
| Faber-Original | +1.1% (DD -0.3%) | -5.8% (DD -1.6%) | -6.1% (DD -2.9%) |
| Faber-Expanded-Full | +1.2% (DD -0.3%) | -5.2% (DD -0.8%) | -5.5% (DD -2.6%) |
| Faber-Expanded-EFA-VNQ | +1.2% (DD -0.3%) | -5.3% (DD -1.0%) | -5.5% (DD -2.3%) |
| Faber-Expanded-EFA-Only | +1.1% (DD -0.3%) | -5.2% (DD -1.2%) | -4.9% (DD -2.3%) |

### Independent Signal Contribution

| Asset | Pct Months Eligible | Avg Alloc When Eligible | Corr with IVV (full) |
|-------|--------------------|-----------------------|---------------------|
| DBA | 34% | 4.7% | 0.272 |
| EFA | 69% | 14.3% | 0.850 |
| VNQ | 67% | 9.5% | 0.692 |

## Key Diagnostics

Average cash holdings tell the story:
- Original: 37.8% cash
- Expanded-Full: 40.9% cash
- Expanded-EFA-VNQ: 40.5% cash
- Expanded-EFA-Only: 41.0% cash

The expanded strategies hold MORE cash on average than the original, because spreading baseline weight across more assets means each individual asset has a lower weight. When Faber filters out an asset, less capital is freed per asset — but there are more assets that can be filtered out. Net effect: slightly more time in cash.

## Interpretation

**The Sharpe improvements are real but trivially small (+0.010 to +0.022).** They come from two sources:

1. **Lower vol from diluted equity concentration.** Replacing IVV 45% / QQQ 25% with IVV 30% / QQQ 15% / EFA 15% / VNQ 10% reduces US equity concentration. But this is achievable without universe expansion — just lowering the original IVV/QQQ weights would have similar effect.

2. **Marginally more independent Faber signals.** VNQ and DBA trend differently from US equities in some periods. But the effect is tiny because:
   - EFA correlates 0.932 with IVV during crisis — it's not independent when it matters most
   - VNQ correlates 0.841 with IVV during crisis — also converges
   - DBA is only eligible 34% of months (poor trend persistence) with 4.7% average allocation
   - The Faber filter's binary risk-on/risk-off already captures most diversification benefit via the cash buffer

**EFA is the biggest disappointment.** With 0.850 full-period correlation and 0.932 crisis correlation with IVV, it's essentially a noisier version of IVV. The Faber-EFA-Only variant shows the best Sharpe improvement (+0.022), but this comes entirely from the lower IVV concentration (35% vs 45%), not from EFA's independent signal.

**VNQ offers moderate but unreliable diversification.** Full-period IVV correlation of 0.692 is promising, but 0.841 during GFC means it fails precisely when diversification matters most. The RWR proxy validation (0.991 annual correlation) confirmed data quality but VNQ's crisis convergence limits its value.

**DBA is too sparse to matter.** Only eligible 34% of months with a maximum 5% baseline weight. Even when eligible, the 4.7% average allocation contributes negligible return or risk impact.

**The fundamental constraint:** Within asset classes that are all denominated in the same currency and subject to the same monetary policy, crisis correlations converge toward 1.0. Only fundamentally different assets (gold, treasuries) maintain independence. Adding more equity-like assets (EFA, VNQ) doesn't provide the independent trend signals this system needs.

## Decision

No decision record needed — the Sharpe improvements are positive but too small (+0.010 to +0.022) to justify the added complexity of managing 8-9 assets vs 5. The original 5-asset universe with Faber filter remains the production architecture.

**However:** EFA-Only shows a marginal improvement (+0.022 Sharpe, reduced 2022 drawdown from -6.1% to -4.9%) that could be worth revisiting if the architecture is being restructured for other reasons. It's not worth a standalone change, but if a future experiment already modifies the baseline weights, adding EFA at that point has near-zero marginal cost.

## Next Steps

- The universe expansion path offers diminishing returns for this architecture. The Faber filter's main value comes from the cash buffer, and more assets don't fundamentally improve the cash buffer mechanism.
- [[planned: leverage_calibration_faber_only]] — Graduated leverage on the simplified Faber-only architecture is more promising for return improvement than universe expansion.
- If revisiting universe expansion in future, prioritize assets with low crisis correlation to IVV: managed futures (DBMF), tail risk hedges, or truly uncorrelated alternatives — not more equity-adjacent assets.

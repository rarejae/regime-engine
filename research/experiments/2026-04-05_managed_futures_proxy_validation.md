# Managed Futures Proxy Validation — Data Quality Gate (Pod 3)

**Date:** April 5, 2026  
**Status:** Complete  
**Track:** Multi-Pod Architecture — Phase 3  
**Related:** [[TAA_PROJECT_STATUS]] | [[MULTI_POD_ARCHITECTURE]]

## Purpose

Validate proxy data for DBMF (iMGP DBi Managed Futures ETF, inception May 2019) before running the full Pod 3 backtest. DBMF has only ~7 years of live data — we need a reliable pre-2019 proxy to match the 2002-2026 Faber-Sweep-40 window.

## Step 1: ETF Data

| ETF | First Date | Last Date | Months | 2022 Return | Max DD |
|-----|-----------|----------|--------|------------|--------|
| DBMF | 2019-06-01 | 2026-04-01 | 83 | **+21.6%** | -17.3% |
| KMLM | 2021-01-01 | 2026-04-01 | 64 | **+24.2%** | -25.9% |

Both ETFs were strongly positive in 2022 (+21.6% and +24.2% respectively) while IVV fell -18.2%. DBMF has been more consistent overall (KMLM: -5.7% in 2023, -1.7% in 2024, -3.0% in 2025).

## Step 2: Proxy Download Results

| Source | Status | Date Range | Months | Notes |
|--------|--------|-----------|--------|-------|
| **AQR TSMOM** | **SUCCESS** | **1985-01 to 2025-01** | **481** | Diversified TSMOM across 58 instruments. Best coverage. |
| SG Trend Index | FAILED | — | — | DNS resolution failed (sgindex.com). Requires registration. |
| Barclay CTA Index | FAILED | — | — | No HTML tables found on BarclayHedge page |
| BTOP50 Index | FAILED | — | — | No HTML tables found on BarclayHedge page |
| AQMIX (AQR MF Fund) | SUCCESS | 2010-02 to 2026-04 | 195 | +35.5% in 2022. Best DBMF corr among funds. |
| AMFAX (Abbey Capital) | SUCCESS | 2010-09 to 2026-04 | 188 | +75.3% in 2022 (very high — different strategy) |
| MFTFX (Pimco TRENDS) | SUCCESS | 2010-07 to 2026-04 | 190 | +57.9% in 2022. Highest annual corr with DBMF (0.828) |
| RYMFX (Rydex MF) | SUCCESS | 2007-03 to 2026-04 | 230 | +14.7% in 2022. Longest history. |

**AQR TSMOM is the clear winner:** 40 years of data (1985-2025), broadest asset class coverage (equities, bonds, currencies, commodities across 58 instruments), academic pedigree (Moskowitz, Ooi & Pedersen 2012).

## Step 3: Proxy Selection

**Selected: AQR TSMOM**

Rationale: Diversified time-series momentum across 58 liquid instruments covering all four asset classes. This is the closest conceptual match to what DBMF replicates (large CTA exposures across equities, fixed income, currencies, commodities). History from Jan 1985 covers every major market crisis. Saved to `data/raw/aqr_tsmom_monthly.csv`.

## Step 4: Overlap Correlation Validation

| Proxy | Overlap Period | Months | Monthly Corr | Annual Corr | Ann Ret Diff | Tracking Error | Status |
|-------|---------------|--------|-------------|------------|-------------|---------------|--------|
| **AQR TSMOM** | **2019-06 to 2025-01** | **68** | **0.632** | **0.774** | **-5.38%** | **3.34%** | **FAIL** |
| AQMIX | 2019-06 to 2026-04 | 83 | 0.721 | 0.687 | +0.40% | 2.42% | FAIL |
| MFTFX | 2019-06 to 2026-04 | 83 | 0.725 | 0.828 | +1.18% | 4.65% | FAIL |
| RYMFX | 2019-06 to 2026-04 | 83 | 0.721 | 0.599 | -3.37% | 2.29% | FAIL |
| AMFAX | 2019-06 to 2026-04 | 83 | 0.637 | 0.622 | +0.35% | 3.82% | FAIL |
| KMLM | 2021-01 to 2026-04 | 64 | 0.641 | — | — | — | FAIL |

**Every proxy fails the 0.85 correlation threshold.** The best monthly correlation is MFTFX at 0.725 (and the best annual correlation is also MFTFX at 0.828). AQR TSMOM, despite being the conceptual best match, correlates only 0.632 with DBMF on a monthly basis.

**Why all proxies fail:** Managed futures is an extremely heterogeneous category. DBMF uses a specific "Dynamic Beta Engine" — weekly regression estimating top-20 CTA allocations — that produces a return stream distinct from any individual CTA fund, academic factor, or industry index. Every fund/index trades different instruments, different timeframes, different position sizing, and different risk management. The 0.63-0.73 correlation range is not a data quality failure; it reflects genuine strategy heterogeneity.

**AQR TSMOM return shortfall:** The -5.38% annual return difference (proxy understates DBMF) is partly explained by AQR TSMOM being a gross excess return factor, while DBMF returns include the T-bill return on collateral. Adjusting for ~4-5% T-bill yield closes most of the gap.

## Step 5: Regime Stability

| Regime | AQR TSMOM-DBMF Corr | N months | Interpretation |
|--------|---------------------|----------|---------------|
| Bull equity (12m>10%) | 0.556 | 45 | Moderate |
| Bear equity (12m<0%) | 0.752 | 12 | Stable — improves in stress |
| High VIX (>25) | 0.512 | 18 | Moderate |
| Low VIX (<15) | 0.583 | 14 | Moderate |

**Correlation improves during bear markets (0.752 vs 0.556 in bulls).** This is the favorable pattern — when it matters most (equity downturns), AQR TSMOM and DBMF are more aligned. Both strategies tend to be short equities and long bonds during sustained downtrends.

## Step 6: DBMF Crisis Behavior (ETF Period Only)

| Period | DBMF Return | IVV Return | DBMF-IVV Correlation |
|--------|------------|-----------|---------------------|
| COVID (Feb-Apr 2020) | +1.4% | -9.2% | 0.526 |
| **2022 Bear (Jan-Dec)** | **+21.6%** | **-18.2%** | **-0.586** |
| 2025 tariff (Feb-Apr) | -3.9% | -7.6% | 0.307 |

**2022 is the definitive result: DBMF +21.6%, IVV -18.2%, correlation -0.586.** This is genuine negative equity correlation during a sustained bear market — exactly the diversification Pod 1 (Faber) cannot provide on its own.

COVID: DBMF was modestly positive (+1.4%) but positively correlated with IVV (0.526). Fast crashes don't give trend-following strategies time to reposition — consistent with known CTA behavior.

2025 tariff shock: DBMF down -3.9% — less than IVV (-7.6%), correlation 0.307. Modest protection but not the dramatic diversification of 2022.

## Step 7: Final Validation Summary

| Proxy | Source | Overlap Corr | Regime Stable | Date Range | Verdict |
|-------|--------|-------------|--------------|-----------|---------|
| AQR TSMOM | AQR website | 0.632 | Yes (improves in stress) | 1985-01 to 2025-01 | FAIL (but best available) |
| AQMIX | yfinance | 0.721 | Yes | 2010-02 to 2026-04 | FAIL |
| MFTFX | yfinance | 0.725 | Yes | 2010-07 to 2026-04 | FAIL |
| RYMFX | yfinance | 0.721 | Yes | 2007-03 to 2026-04 | FAIL |

### Conclusions

**1. No proxy passes the formal correlation threshold (0.85).** This is not a data quality issue — it's inherent to managed futures. The category is too heterogeneous for any single proxy to reliably represent DBMF specifically. Even KMLM (another managed futures ETF) only correlates 0.641 with DBMF.

**2. DBMF crisis behavior: CONFIRMED as genuine crisis diversifier.**
- 2022: +21.6% return, -0.586 correlation with IVV — strongly negative
- This is the property that makes Pod 3 valuable: negative equity correlation during sustained downtrends

**3. Best available proxy: AQR TSMOM (with heavy caveats)**
- 40 years of data (1985-2025), covers all major crises
- Monthly correlation with DBMF: 0.632 (well below threshold)
- Improves to 0.752 during bear markets
- Annual return correlation: 0.774 (better at longer horizons)
- Return shortfall of ~5.4% partly explained by missing T-bill collateral return
- Usable for directional analysis (does managed futures help in crises?) but NOT for precise Sharpe/terminal wealth calculations

**4. Recommended backtest approach for Pod 3:**
- **Primary:** DBMF ETF-only from May 2019 (~7 years). Short but clean.
- **Extended analysis:** AQR TSMOM from 2002 (or 1985), clearly flagged as proxy with 0.632 monthly correlation. Useful for crisis behavior validation (GFC, dot-com) but not for precise performance metrics.
- **Cross-validation:** AQMIX from 2010 as secondary proxy (0.721 corr, closest return profile to DBMF)

**5. Structural issues:**
- AQR TSMOM is an academic factor (excess returns); DBMF includes T-bill collateral return — add T-bill return to TSMOM for apples-to-apples comparison
- Managed futures style has evolved: more crowding, faster signals, broader universes post-2010
- Pre-2010 TSMOM returns may overstate go-forward CTA returns
- DBMF's Dynamic Beta Engine adds tracking error vs actual CTA returns (~2-3% annual)
# Managed Futures Proxy Validation — Data Quality Gate (Pod 3)

**Date:** April 5, 2026
**Status:** Complete
**Track:** Multi-Pod Architecture — Phase 3
**Related:** [[TAA_PROJECT_STATUS]] | [[MULTI_POD_ARCHITECTURE]]

## Key Finding

No proxy passes the 0.85 correlation threshold. This is NOT a data quality failure — it's inherent to managed futures strategy heterogeneity. DBMF's Dynamic Beta Engine produces a return stream distinct from any individual CTA fund, academic factor, or industry index.

## ETF Data

| ETF | Period | 2022 Return | Max DD |
|-----|--------|------------|--------|
| DBMF | Jun 2019 – Apr 2026 | **+21.6%** | -17.3% |
| KMLM | Jan 2021 – Apr 2026 | **+24.2%** | -25.9% |

## Proxy Results

| Proxy | Source | Monthly Corr | Annual Corr | Date Range | Verdict |
|-------|--------|-------------|------------|-----------|---------|
| AQR TSMOM | AQR website | 0.632 | 0.774 | 1985-2025 | FAIL (best available) |
| AQMIX | yfinance | 0.721 | 0.687 | 2010-2026 | FAIL |
| MFTFX | yfinance | 0.725 | 0.828 | 2010-2026 | FAIL |
| RYMFX | yfinance | 0.721 | 0.599 | 2007-2026 | FAIL |
| KMLM | yfinance | 0.641 | — | 2021-2026 | FAIL |
| SG Trend Index | FAILED (DNS) | — | — | — | — |

## DBMF Crisis Behavior (Live ETF, May 2019+)

| Period | DBMF | IVV | Correlation |
|--------|------|-----|-------------|
| COVID (Feb-Apr 2020) | +1.4% | -9.2% | +0.526 |
| **2022 Bear (full year)** | **+21.6%** | **-18.2%** | **-0.586** |
| 2025 tariff (Feb-Apr) | -3.9% | -7.6% | +0.307 |

**2022 is definitive: DBMF +21.6%, IVV -18.2%, correlation -0.586.** Genuine negative equity correlation during sustained bear market. COVID was fast — trend signals couldn't reposition in time (+1.4%, positive correlation).

## Regime Stability (AQR TSMOM vs DBMF)

| Regime | Correlation | N |
|--------|------------|---|
| Bull equity | 0.556 | 45 |
| Bear equity | 0.752 | 12 |
| High VIX | 0.512 | 18 |
| Low VIX | 0.583 | 14 |

Correlation improves during bear markets (0.752) — favorable pattern. Both strategies trend short equities/long bonds during sustained downturns.

## Backtest Approach for Full Pod 3 Analysis

- **Primary:** DBMF ETF-only (May 2019 – present, ~7 years). Short but clean, includes the definitive 2022 test.
- **Extended (directional only):** AQR TSMOM from 2002, flagged as proxy with 0.632 monthly corr. Valid for crisis behavior questions (GFC, dot-com) but NOT for precise Sharpe/terminal wealth calculations.
- **Cross-validation:** AQMIX from 2010 as secondary proxy (0.721 corr, closest return profile).

## Structural Caveats

1. AQR TSMOM is gross excess return — add T-bill rate for apples-to-apples vs DBMF total return (~4-5% annual gap)
2. Pre-2010 TSMOM returns may overstate go-forward CTA returns (less crowding, wider spreads)
3. DBMF's Dynamic Beta Engine adds ~2-3% annual tracking error vs actual CTA returns
4. Managed futures style has evolved post-2010: faster signals, broader universes, more crowding

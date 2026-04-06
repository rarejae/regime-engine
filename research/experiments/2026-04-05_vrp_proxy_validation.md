# VRP Proxy Validation — Data Quality Gate

**Date:** April 5, 2026  
**Status:** Complete  
**Track:** VRP Harvesting (Phase 2)  
**Related:** [[TAA_PROJECT_STATUS]] | [[MULTI_POD_ARCHITECTURE]]

## Purpose

Validate data quality for the four candidate VRP instruments (SVXY, PUTW, QYLD, XYLD) before running any strategy backtest. Determine which instruments have reliable pre-ETF proxies, what date ranges are trustworthy, and whether any structural breaks contaminate the data.

## Step 1: ETF Data Availability

| ETF | Inception | First Available | Last Date | Months | Div-Adjusted | 2024 Total Return |
|-----|-----------|----------------|-----------|--------|-------------|------------------|
| SVXY | Oct 2011 | 2011-10-01 | 2026-04-01 | 175 | Yes | -3.2% |
| PUTW | Feb 2016 | 2016-02-01 | 2026-04-01 | 123 | Yes | +17.3% |
| QYLD | Dec 2013 | 2013-12-01 | 2026-04-01 | 149 | Yes | +20.1% |
| XYLD | Jun 2013 | 2013-06-01 | 2026-04-01 | 155 | Yes | +19.5% |

All ETFs downloaded with `auto_adjust=True` (yfinance). QYLD/XYLD pay large monthly distributions (~1%/mo); adjusted close captures these correctly.

## Step 2: SVXY Proxy Construction

### Proxy Sources — All Failed

| Source | Ticker | Result |
|--------|--------|--------|
| S&P VIX Short-Term Futures Index | ^SPVXSTR | Delisted on yfinance |
| S&P VIX Short-Term Futures Total Return | ^SPVXSP | Delisted on yfinance |
| CBOE VIX Futures Historical CSV | VX_History.csv | HTTP 403 Forbidden |

**No SVXY proxy available.** All three potential sources returned errors. The CBOE has restricted access to their historical futures data. SVXY is limited to ETF-only data from Oct 2011.

### February 2018 Structural Break

```
SVXY February 2018 Event:
  Feb 5 2018 single-day SVXY return: -32.0%
  Feb 6 2018 single-day SVXY return: -83.0%
  Peak-to-trough drawdown (Jan-Feb 2018): -93.1%
  Restructuring date: Feb 27, 2018 (SVXY reduced from -1x to -0.5x)
  Note: XIV (-1x version) suffered ~-93% and was terminated.
```

This is a critical structural break. Pre-Feb-27-2018 SVXY data reflects a -1x product; post-restructuring reflects a -0.5x product. **Any backtest using SVXY must either:**
1. Use only post-restructuring data (Mar 2018+), or
2. Scale pre-2018 returns by 0.5x to approximate -0.5x behavior

Given the Feb 5-6 event produced -93% drawdown at -1x (vs theoretical ~-47% at -0.5x), pre-2018 data cannot be naively mixed with post-2018 data.

## Step 3: Proxy Index Availability

| Proxy | Source | First Date | Last Date | Months | Data Gaps |
|-------|--------|------------|-----------|--------|-----------|
| PUT (CBOE PutWrite) | NOT AVAILABLE | — | — | — | — |
| BXM (CBOE S&P 500 BuyWrite) | yfinance ^BXM | 2000-01-01 | 2026-04-01 | 315 | None |
| BXN (CBOE Nasdaq-100 BuyWrite) | yfinance BXN | 2014-12-01 | 2015-03-01 | 3 | Unusable |

**PUT Index:** Not available from yfinance (^PUT delisted/not found). CBOE website would need manual CSV download. This blocks PUTW proxy validation.

**BXM Index:** Full 25+ year history available. Clean data, no gaps. This is the only proxy with adequate coverage.

**BXN Index:** Only 3 months of data from yfinance — effectively unusable. This kills QYLD proxy validation.

## Step 4: Overlap Correlation Validation

### PUTW vs PUT Index
**SKIPPED** — PUT Index data unavailable. PUTW restricted to ETF-only (Feb 2016+).

### XYLD vs BXM Index
```
Overlap period: 2013-07 to 2026-04 (154 months)
Monthly return correlation:  0.944
Annual return correlation:   0.967
Mean monthly return ETF:     0.67%
Mean monthly return proxy:   0.62%
Annual return diff (proxy - ETF): -0.66%
Monthly tracking error:      1.08%
Status: MARGINAL (0.90 < 0.944 < 0.95)
```

BXM understates XYLD returns by ~0.66% annualized. The monthly correlation of 0.944 is MARGINAL — above the 0.90 fail threshold but below the 0.95 pass threshold. The annual correlation of 0.967 is stronger, suggesting the monthly differences wash out over longer periods.

### QYLD vs BXN Index
**FAILED** — Only 3 months overlap. Insufficient for any meaningful validation. QYLD dropped.

### SVXY vs Proxy
**SKIPPED** — No proxy data available. SVXY restricted to ETF-only.

## Step 5: Regime Stability Test

| Instrument | Normal (VIX<20) | Elevated (VIX 20-30) | Stress (VIX>30) | Interpretation |
|------------|----------------|---------------------|----------------|---------------|
| XYLD/BXM | 0.881 (n=114) | 0.948 (n=31) | 0.995 (n=9) | Stable — IMPROVES in stress |
| PUTW/PUT | N/A | N/A | N/A | Data unavailable |
| QYLD/BXN | N/A | N/A | N/A | Insufficient overlap |

**XYLD/BXM proxy relationship is strongest during stress** (0.995 at VIX>30 vs 0.881 at VIX<20). This is the favorable pattern — the proxy is most reliable precisely when accurate data matters most (during crisis periods). The lower normal-market correlation (0.881) likely reflects XYLD's monthly distribution timing and fund-specific execution differences, which are less important.

Note: only 9 stress months in the sample — small n, but the directional pattern (improving with VIX) is encouraging.

## Step 6: SVXY Backtest Period Assessment

```
SVXY proxy: NOT AVAILABLE
Fallback: ETF-only from Oct 2011
Coverage gap: Jan 2002 - Oct 2011 (118 months unavailable)
Missing periods: 2002 bear, 2003-2006 recovery, 2008 GFC (critical!), 2009-2011 recovery

WARNING: Missing the 2008 GFC means SVXY backtest cannot validate behavior during the
most important stress event. The Feb 2018 Volmageddon event IS captured in ETF data,
but the 2008 crisis is not. This materially limits confidence in SVXY stress behavior.

For clean -0.5x data: restrict to post-restructuring (Mar 2018+), yielding only ~8 years.
```

## Step 7: Final Validation Summary

| Instrument | Proxy Source | Overlap Corr | Regime Stable? | Reliable From | BT Start | Verdict |
|------------|-------------|-------------|---------------|--------------|---------|---------|
| SVXY | ETF only | N/A | N/A | 2011-10 | 2011-10 | ETF-ONLY |
| PUTW | PUT Index (unavail) | N/A | N/A | 2016-02 | 2016-02 | ETF-ONLY |
| QYLD | BXN Index (unusable) | N/A | N/A | — | — | DROPPED |
| XYLD | BXM Index | 0.944 | Yes (improves in stress) | 2002-01 | 2002-01 | CAUTION |

### Conclusions

**1. Instruments cleared for full VRP backtest (proxy + ETF):**
- **XYLD** via BXM Index proxy, backtest from 2002-01. CAUTION flag: monthly correlation 0.944 (MARGINAL), proxy understates returns by 0.66% ann. Regime stability is good (improves in stress).

**2. Instruments restricted to ETF-only periods:**
- **SVXY**: ETF only from Oct 2011. Post-restructuring clean data from Mar 2018. Missing the 2008 GFC is a significant limitation.
- **PUTW**: ETF only from Feb 2016. PUT proxy index unavailable from yfinance.

**3. Instruments dropped:**
- **QYLD**: BXN proxy has only 3 months of data — no validation possible. Cannot backtest with confidence.

**4. Recommended VRP backtest approach:**
- Primary instrument: **XYLD** (BXM proxy from 2002, ETF from 2013) — longest history, validated proxy
- Secondary: **PUTW** (ETF only from 2016) — valuable for comparing put-write vs buy-write strategies but limited to ~10 years
- Conditional: **SVXY** (ETF only from 2011, or post-restructure 2018) — fundamentally different strategy (short vol vs covered calls), but missing GFC data limits crisis analysis
- Dropped: **QYLD** — no proxy validation possible

**5. For the full VRP backtest:**
- Use BXM index as the primary VRP return series (2002-2013), switching to XYLD ETF returns from 2013 onward
- Run PUTW and SVXY as separate overlay instruments with their shorter ETF-only histories
- Report all instrument entry dates prominently in results
- Flag the 0.66% annual return understatement from BXM vs XYLD in any backtest that uses the proxy

### Data Quality Risks to Flag

1. **BXM proxy understatement (-0.66% ann):** The VRP backtest using BXM pre-2013 will slightly understate returns. This is a conservative bias — real performance would likely be marginally better.

2. **SVXY missing GFC:** Cannot validate short-vol behavior during 2008. The Feb 2018 Volmageddon event is captured (and is arguably more relevant to -0.5x SVXY behavior) but the 2008 omission means crisis analysis is incomplete.

3. **QYLD dropped entirely:** If Nasdaq-100 BuyWrite exposure is desired, revisit by manually downloading BXN data from the CBOE website (not yfinance).

4. **PUT index unavailable:** If cash-secured put-write exposure (PUTW) is a priority, manually download PUT index data from CBOE. The yfinance ticker ^PUT appears to be delisted/unavailable.
# VRP Proxy Validation — Data Quality Gate

**Date:** April 5, 2026
**Status:** Complete
**Track:** VRP Harvesting (Phase 2)
**Related:** [[TAA_PROJECT_STATUS]] | [[MULTI_POD_ARCHITECTURE]]

## Purpose

Validate data quality for the four candidate VRP instruments (SVXY, PUTW, QYLD, XYLD) before running any strategy backtest. Determine which instruments have reliable pre-ETF proxies, what date ranges are trustworthy, and whether any structural breaks contaminate the data.

## Step 1: ETF Data Availability

| ETF | Inception | First Available | Last Date | Months | Div-Adjusted | 2024 Total Return |
|-----|-----------|----------------|-----------|--------|-------------|------------------|
| SVXY | Oct 2011 | 2011-10-01 | 2026-04-01 | 175 | Yes | -3.2% |
| PUTW | Feb 2016 | 2016-02-01 | 2026-04-01 | 123 | Yes | +17.3% |
| QYLD | Dec 2013 | 2013-12-01 | 2026-04-01 | 149 | Yes | +20.1% |
| XYLD | Jun 2013 | 2013-06-01 | 2026-04-01 | 155 | Yes | +19.5% |

## Step 2: SVXY Proxy — All Sources Failed

SPVXSTR delisted on yfinance. CBOE VIX futures CSV returned HTTP 403. No SVXY proxy available.

**Feb 2018 structural break:**
- Feb 5: -32.0% single day. Feb 6: -83.0% single day.
- Peak-to-trough drawdown: -93.1%
- Restructured Feb 27, 2018 from -1x to -0.5x
- Pre-2018 SVXY data reflects a different product. Cannot mix naively with post-2018 data.

## Step 3: Proxy Index Availability

| Proxy | Source | First Date | Last Date | Months | Notes |
|-------|--------|------------|-----------|--------|-------|
| PUT (CBOE PutWrite) | NOT AVAILABLE | — | — | — | ^PUT delisted on yfinance |
| BXM (S&P 500 BuyWrite) | yfinance ^BXM | 2000-01-01 | 2026-04-01 | 315 | Clean, no gaps |
| BXN (Nasdaq BuyWrite) | yfinance BXN | 2014-12-01 | 2015-03-01 | 3 | Effectively unusable |
| SPVXSTR (VIX futures) | Not available | — | — | — | CBOE 403 Forbidden |

## Step 4: Overlap Correlation Validation

**XYLD vs BXM (2013-07 to 2026-04, 154 months):**
- Monthly return correlation: 0.944 — MARGINAL
- Annual return correlation: 0.967
- BXM understates XYLD returns by 0.66% annualized
- Monthly tracking error: 1.08%

**PUTW vs PUT:** Skipped — PUT data unavailable.
**QYLD vs BXN:** Failed — only 3 months overlap.
**SVXY vs proxy:** Skipped — no proxy available.

## Step 5: Regime Stability

| Instrument | Normal VIX<20 | Elevated VIX 20-30 | Stress VIX>30 | Pattern |
|------------|--------------|-------------------|--------------|---------|
| XYLD/BXM | 0.881 (n=114) | 0.948 (n=31) | 0.995 (n=9) | Improves in stress |

XYLD/BXM proxy is MOST reliable during crisis periods — exactly when accuracy matters most. The lower normal-market correlation reflects distribution timing noise, not a systematic proxy failure.

## Final Summary

| Instrument | Proxy | Overlap Corr | Regime Stable | BT Start | Verdict |
|------------|-------|-------------|--------------|---------|---------|
| SVXY | None | N/A | N/A | 2011-10 (ETF only) | ETF-ONLY, missing GFC |
| PUTW | PUT (unavail) | N/A | N/A | 2016-02 (ETF only) | ETF-ONLY, 10yr history |
| QYLD | BXN (3 months) | N/A | N/A | — | DROPPED |
| XYLD | BXM | 0.944 MARGINAL | Yes (improves in stress) | 2002-01 | PROCEED WITH CAUTION |

**Cleared for full backtest (2002+):** XYLD via BXM proxy only.
**ETF-only overlays:** SVXY (2011+), PUTW (2016+) — useful for comparison but not primary.
**Dropped:** QYLD.

**Manual download needed for PUTW:** CBOE website has PUT index CSV. If put-write exposure is a priority, download manually and re-run validation. URL: https://www.cboe.com/us/indices/dashboard/put/
# Purpose
## Purpose

Validate data quality for VRP instruments before running any strategy backtest. Determine which instruments have reliable pre-ETF proxies, what date ranges are trustworthy, and whether any structural breaks contaminate the data.

## Final Outcome

**PUT index acquired manually from investing.com:**
- Combined CSV: `data/PUT_index_combined.csv`
- Date range: Jun 2, 1988 – Apr 2, 2026 (9,524 daily observations, zero gaps)
- Pre-2007: CBOE back-tested methodology (reliable but not live-calculated)
- Post-2007: Live daily calculations
- Reliable daily range: Jan 3, 2007 – Apr 2, 2026 (4,844 observations, ~19 years)
- **This is the primary VRP instrument for the backtest**

**Other instruments validated:**

| Instrument | Proxy | Overlap Corr | Regime Stable | BT Start | Verdict |
|------------|-------|-------------|--------------|---------|---------|
| PUT index | Manual CSV | N/A (primary) | N/A | 1988-06 (2007 reliable) | PRIMARY |
| SVXY | None available | N/A | N/A | 2011-10 ETF only | ETF-ONLY, missing GFC |
| PUTW | PUT index (now available) | TBD | TBD | 2016-02 ETF only | ETF-ONLY, validates vs PUT |
| QYLD | BXN (3 months) | N/A | N/A | — | DROPPED |
| XYLD | BXM | 0.944 MARGINAL | Improves in stress | 2002-01 | SECONDARY only |

**VRP backtest approach:**
- Primary: PUT index returns directly (2002–2026, pre-2007 back-tested)
- The PUT index IS the cash-secured put strategy — no proxy needed, it's the actual strategy return series
- XYLD/BXM retained as secondary comparison only (covered call, different risk profile)
- SVXY deferred — missing GFC, structural break Feb 2018

## Proxy Validation Details

### SVXY — All Proxy Sources Failed

| Source | Ticker | Result |
|--------|--------|--------|
| S&P VIX Short-Term Futures Index | ^SPVXSTR | Delisted on yfinance |
| S&P VIX Short-Term Futures Total Return | ^SPVXSP | Delisted on yfinance |
| CBOE VIX Futures Historical CSV | VX_History.csv | HTTP 403 Forbidden |

**Feb 2018 structural break:**
- Feb 5: -32.0% single day. Feb 6: -83.0% single day
- Peak-to-trough: -93.1%
- Restructured Feb 27, 2018 from -1x to -0.5x
- Cannot mix pre/post data naively

### XYLD vs BXM (154-month overlap)
- Monthly return correlation: 0.944 — MARGINAL
- Annual return correlation: 0.967
- BXM understates XYLD by 0.66% annualized
- Regime stability: 0.881 (VIX<20) → 0.948 (VIX 20-30) → 0.995 (VIX>30) — improves in stress

### PUT Index Data Quality
- CBOE daily download: Jan 2007 onward (4,844 observations)
- Pre-2007 from CBOE: sparse (7 data points, unusable)
- investing.com: full series Jun 1988 – Apr 2026 with pre-2007 back-test
- Combined file covers complete history with zero gaps


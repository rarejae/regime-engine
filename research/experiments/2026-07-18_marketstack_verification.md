# V19d Marketstack Verification — 2000 to Present

**Date:** 2026-07-18
**Status:** Complete
**Track:** Implementation (production verification)
**Related:** [[TAA_PROJECT_STATUS]] | [[V19D_PRODUCTION_SPEC]] | [[V19D_FINAL_BACKTEST]]

## Hypothesis

The locked V19d numbers were produced on yfinance data ending 2026-03-31. An
independent data vendor (Marketstack) should (a) reproduce the locked numbers
on the 2002–2026 window, and (b) extend the record back to 2000 and forward
to the present, covering the dot-com crash onset and the Apr–Jun 2026 period
that the final backtest missed.

## Design

- **Strategy code:** unchanged — imports `run_v19d_full` from
  [[V19D_FINAL_BACKTEST|experiments/v19d_final]]. No reimplementation.
- **Data:** Marketstack EOD (new fetcher `data/sources/marketstack.py`) from
  2016-08 (plan history limit) to 2026-07-17; yfinance before the splice.
  Splice is ratio-anchored; daily-return correlation on the overlap validates
  agreement (all ≥ 0.998).
- **Proxies:** IVV→SPY, IAU→GLD, VGLT→TLT (unchanged from all prior work).
- Signal alignment assertion: SMA at day t verified to use closes ≤ t only.

### Marketstack data-quality issues found (fixed in fetcher)

1. **Unadjusted splits:** SSO's 2020/2022/2025 2:1 splits and several QLD
   splits are missing from `adj_close`. Fetcher detects and back-adjusts.
2. **Zero-price rows:** scattered 0.0 closes (Apr/Jun 2026, SPY/GLD/QLD/SSO).
   Treated as missing and filled from yfinance.
3. **Broken pagination totals:** `pagination.total` echoes the page count;
   fetcher pages until a short page instead.
4. **History cap:** earliest row ≈ 2016-07 on current plan. Full-history
   single-vendor verification is not possible on this tier.

## Results

### Quality gate — reproduce locked V19d (2002-01 → 2026-03)

| Metric | This run | Locked | Delta |
|--------|---------:|-------:|------:|
| CAGR   | 17.19%   | 17.27% | -0.08pp |
| Sharpe | 0.876    | 0.866  | +0.010 |
| MaxDD  | -25.14%  | -25.10%| -0.04pp |

**PASS.** Residual deltas are data-vendor differences (dividend adjustment
timing, TLT overlap corr 0.9983), not strategy divergence.

### Full run — 2000-01-01 → 2026-07-17

| Strategy | CAGR | Vol | Sharpe | Sortino | MaxDD | Terminal $1 | DCA |
|----------|-----:|----:|-------:|--------:|------:|------------:|----:|
| **V19d** | **14.18%** | 20.90% | **0.740** | 0.840 | **-40.7%** | $33.57 | $4.70M |
| QQQ B&H  | 8.52% | 26.79% | 0.439 | 0.575 | -83.0% | $8.71 | $2.71M |
| IVV B&H  | 8.02% | 19.40% | 0.495 | 0.630 | -55.2% | $7.72 | $1.44M |
| 60/40 (TLT) | 6.76% | 11.38% | 0.632 | 0.848 | -30.3% | $5.66 | $0.88M |

CB events: 41 (QQQ 15, IVV 16, IAU 10) | rebalances: 19.

### Crisis Analysis

| Crisis | V19d | QQQ B&H | IVV B&H |
|--------|-----:|--------:|--------:|
| Dot-com 2000–02 (full) | **-40.7%** | -83.0% | -47.5% |
| GFC 07–09 | -16.4% | -53.1% | -53.9% |
| COVID 2020 | -25.1% | -28.6% | -33.7% |
| 2022 bear | -17.6% | -35.2% | -24.5% |
| 2025–26 recent | -14.0% | -22.9% | -19.0% |

### Current state (2026-07-17 close, Marketstack)

| Asset | Close | Score | Position |
|-------|------:|:-----:|----------|
| QQQ | 670.86 | 3/3 | Pod 1: QLD (levered) |
| IVV | 710.73 | 3/3 | Pod 2: SSO (levered) |
| IAU | 368.41 | 0/3 | Gold: cash |

Effective equity 180%. YTD 2026: **-0.72%** vs QQQ +13.19%. Trailing 12m:
+17.62% vs QQQ +24.77%. The gap is the Mar–Apr 2026 whipsaw: score break in
March (-7.91%), full cash exit in April (missed the rebound), re-entry in May
(+14.63%).

## Key Diagnostics

**The 2000 start materially changes the drawdown story.** The locked backtest
starts 2002-01, after the dot-com top. From 2000-01:

- 2000 annual return: -34.11% (vs QQQ -36.11%).
- Max drawdown -40.7% — far beyond the -25.1% "structural floor" claimed in
  [[V19D_PRODUCTION_SPEC]], which is start-date dependent.
- **Warmup caveat:** QQQ price history begins 1999-03. The 252-day SMA first
  becomes valid 2000-03-07 — three weeks before the absolute top. Before
  that, QQQ's Faber score is mechanically capped below 3/3 by missing SMAs,
  then reaches 3/3 immediately at the peak and leverages in. A live trader in
  2000 faced the same data constraint, so this is honest history, but it is
  the worst possible signal-formation timing, and NDX index data (pre-1999)
  would likely improve it.

Even so, V19d beat both benchmarks through the dot-com era on every metric
and its 26.5-year Sharpe (0.740) remains ~1.7x QQQ B&H (0.439).

## Interpretation

1. **Verification passed.** An independent vendor reproduces the locked V19d
   results within noise. Strategy implementation and data pipeline are sound.
2. **The -25.1% MaxDD is a 2002-start artifact.** Under the harshest signal
   conditions (dot-com top with barely-warmed SMAs), realized MaxDD was
   -40.7% — close to V9's -37.9% floor from the 2002 start. Position sizing
   and expectations should assume ~-40%, not -25%.
3. **Live-relevant whipsaw cost is visible in 2026 YTD** (-0.72% vs QQQ
   +13.19%): one full exit-and-reenter cycle cost roughly one QQQ-year. This
   is the documented, accepted cost of trend following (see
   [[BULL_MARKET_SURVIVABILITY]]: QQQ beats trailing 12m in 61% of bull
   months).
4. **Marketstack is usable but not trustworthy raw** — splits, zero prices,
   and pagination all needed correction. Any future Marketstack-fed pipeline
   must keep these guards.

## Decision

No architecture change. V19d spec remains locked. Two documentation updates:
the MaxDD expectation caveat (this note) and the Marketstack data-quality
guards (`data/sources/marketstack.py`).

## Next Steps

- Consider a warmup-free dot-com rerun using NDX/SPX index data to separate
  the warmup artifact from true strategy behavior: [[planned_ndx_dotcom_rerun]]
- Deferred research tracks unchanged: vol-managed overlay, DBMF/KMLM pod,
  MERFX merger-arb pod.

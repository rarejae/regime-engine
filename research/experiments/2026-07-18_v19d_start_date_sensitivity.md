# V19d Start-Date Sensitivity

**Date:** 2026-07-18  
**Status:** Complete  
**Track:** Implementation (robustness / investor timing)  
**Related:** [[TAA_PROJECT_STATUS]] | [[V19D_PRODUCTION_SPEC]] | [[2026-07-18_marketstack_verification]]

## Hypothesis

If V19d's edge is structural (Faber gate + CB→cash + leverage only when trends confirm), then the calendar month an investor starts should not flip the long-horizon verdict vs buy-and-hold. We expected:

1. High win rates vs IVV / 60/40 / 50/50 on CAGR, Sharpe, and MaxDD from almost any start through present.
2. Occasional short-horizon underperformance vs naked QQQ after sharp Nasdaq bottoms / mega-bull recoveries (leverage lag + cash/CB periods).
3. Dot-com starts to look worse on *absolute* MaxDD (as in Marketstack verification) but still dominate QQQ on risk-adjusted and terminal metrics over full history.

## Design

- **Data:** Pre-computed daily/monthly return series from `viz/packages/v19d_marketstack_verification/` (same Marketstack-verified V19d path as [[2026-07-18_marketstack_verification]]). Columns: `v19d`, `ivv_bh`, `qqq_bh`, `blend_5050`, `sixty_forty`. No re-simulation; no look-ahead introduced.
- **Metrics:** `viz/metrics.py` (`metrics_from_daily`, `dca_terminal` vault convention $21k + $700/mo).
- **Start grid:** Every calendar month-start from 2000-01 through 2023-07 (first trading day on/after), requiring ≥3 years of remaining history through 2026-07-17 → **283 starts**.
- **Horizons:**
  - **to_end:** start → series end (primary investor question).
  - **5y / 10y forward:** fixed windows from each eligible start (259 / 199 starts) to catch “looked bad for years then recovered.”
- **Backfire (strict):** from that start through horizon end, V19d has **lower terminal wealth AND lower Sharpe** than QQQ B&H.
- **Soft defs:** trails terminal but beats/ties Sharpe; trails Sharpe but beats/ties terminal; trails CAGR alone.
- CSV: [[research/data/v19d_start_date_sensitivity.csv]]

## Results

### Full sample (reference)

| Strategy | CAGR | Sharpe | MaxDD | Terminal $1 | DCA |
|----------|-----:|-------:|------:|------------:|----:|
| **V19d** | **14.18%** | **0.740** | **-40.7%** | **$33.57** | **$4.70M** |
| QQQ B&H | 8.51% | 0.439 | -83.0% | $8.71 | $2.71M |
| IVV B&H | 8.02% | 0.495 | -55.2% | $7.72 | $1.44M |
| 50/50 | 8.55% | 0.480 | -68.9% | $8.79 | $2.00M |
| 60/40 | 6.93% | 0.646 | -29.9% | $5.90 | $0.89M |

### Distribution across month-starts (to end)

| Metric | Min | P10 | Median | P90 | Max |
|--------|----:|----:|-------:|----:|----:|
| V19d CAGR | 10.42% | 16.60% | 19.23% | 22.01% | 24.09% |
| V19d Sharpe | 0.654 | 0.857 | 0.935 | 1.033 | 1.090 |
| V19d MaxDD | -40.7% | -25.1% | -25.1% | -20.2% | -17.3% |
| Terminal $1 | $1.52 | $2.37 | $15.76 | $49.20 | $51.71 |

Start date **does** move absolute CAGR/terminal (path length + entry level), but the *level* of V19d outcomes stays strong: median CAGR ~19%, median Sharpe ~0.94. The left tail of absolute CAGR is dominated by **late-2021 / early-2022** starts (short remaining sample into a choppy then AI-led Nasdaq bull), not by dot-com.

### Win rates (to end): fraction of starts where V19d beats benchmark

| Benchmark | CAGR | Sharpe | Terminal | MaxDD (less neg.) | DCA |
|-----------|-----:|-------:|---------:|------------------:|----:|
| QQQ B&H | 89.8% | 91.5% | 89.8% | **100%** | 85.5% |
| IVV B&H | 96.8% | 94.3% | 96.8% | **100%** | 92.9% |
| 50/50 | 92.2% | 93.6% | 92.2% | **100%** | 89.8% |
| 60/40 | **100%** | 98.2% | **100%** | 93.3% | **100%** |

**Conditioning on history length:**

| Start set | n | Beat QQQ CAGR | Beat QQQ Sharpe |
|-----------|--:|--------------:|----------------:|
| Pre-2021 | 252 | **99.2%** | 96.4% |
| Pre-2022 | 264 | 96.2% | 96.6% |
| 2021+ only | 31 | 12.9% | 51.6% |
| ≥15 years remaining | 139 | 99.3% | 93.5% |

MaxDD: V19d beats QQQ **and** IVV from **every** to-end start (100%).

Terminal wealth ratio V19d/QQQ: median **1.21×** (p10 ≈ 0.99×; min 0.71× on short recent starts).

### Backfire analysis vs QQQ (to end)

| Definition | Count | Rate |
|------------|------:|-----:|
| **STRICT** (lower Term **and** lower Sharpe) | 16 / 283 | **5.7%** |
| Soft: trails Term, beats/ties Sharpe | 13 / 283 | 4.6% |
| Soft: trails Sharpe, beats/ties Term | 8 / 283 | 2.8% |
| Trails QQQ on CAGR alone | 29 / 283 | 10.2% |

**STRICT starts (all):** `2009-03`, then `2022-05` … `2023-07` (15 consecutive recent months).

- **2009-03** is a hairline miss after the GFC bottom: V19d CAGR 21.06% vs QQQ 21.22% (−0.16pp), Sharpe 0.987 vs 1.030, Term $27.57 vs $28.24. Only pre-2022 strict backfire in the entire 23-year grid.
- **2022-05 → 2023-07** are short samples (~3–4 years) into the post-bear / AI mega-rally where naked QQQ compounded extremely fast while V19d’s CB/cash/leverage path lagged on *growth*, not on drawdown control (MaxDD still better every time).

**Soft (trails terminal, beats Sharpe):** mostly `2021-05`→`2022-04` plus `2019-01` — peak-ish equity entry, then enough CB/cash drag that QQQ pulls ahead on wealth while V19d still wins risk-adjusted.

### Worst starts

**Worst absolute V19d CAGR (to end):**

| Start | V19d CAGR | Sharpe | MaxDD | Term $1 | vs QQQ CAGR |
|-------|----------:|-------:|------:|--------:|------------:|
| 2022-01 | 10.42% | 0.654 | -19.9% | $1.57 | −2.70pp |
| 2021-12 | 11.31% | 0.685 | -20.2% | $1.64 | −1.79pp |
| 2021-11 | 11.33% | 0.681 | -20.2% | $1.66 | −2.00pp |
| 2021-09 | 11.60% | 0.687 | -20.2% | $1.71 | −1.62pp |
| 2000-01 | 14.18% | 0.740 | **-40.7%** | $33.57 | **+5.67pp** |

Starting at the **dot-com top** is painful on MaxDD (−40.7%, confirming [[2026-07-18_marketstack_verification]]), but it is **not** a relative failure vs QQQ — it is one of the *best* relative starts (+5.7pp CAGR, Sharpe +0.30).

**Worst relative vs QQQ (CAGRΔ):** all **2022-H2 / 2023** AI-bull window (e.g. 2023-01: −11.9pp CAGR; 2022-10: −11.1pp). Absolute V19d CAGRs there are still mid-teens to ~20%; QQQ simply printed 23–31% from those troughs.

**Best absolute V19d CAGR:** COVID trough / late-2019 entries (2020-04/05 ≈ 24% CAGR, Sharpe ~1.09).

### By era (to end, median)

| Era | n | Med CAGR | Med Sharpe | Beat QQQ CAGR | Med CAGRΔ vs QQQ |
|-----|--:|---------:|-----------:|--------------:|-----------------:|
| 2000–02 (dot-com) | 36 | 16.9% | 0.867 | **100%** | +4.6pp |
| 2003–07 (pre-GFC) | 60 | 18.2% | 0.898 | **100%** | +2.7pp |
| 2008–09 (GFC) | 24 | 20.6% | 0.972 | 95.8% | +1.6pp |
| 2010–19 (bull) | 120 | 20.8% | 0.980 | 99.2% | +1.3pp |
| 2020–21 | 24 | 17.7% | 0.924 | 66.7% | +1.4pp |
| 2022–23 | 19 | 17.3% | 0.963 | **0%** | −7.0pp |

### Fixed forward windows

| Horizon | n | V19d CAGR med | Beat QQQ CAGR | Beat QQQ Sharpe | STRICT backfire |
|---------|--:|--------------:|--------------:|----------------:|----------------:|
| **5y** | 259 | 17.1% | 83.4% | 62.5% | **41 (15.8%)** |
| **10y** | 199 | 18.1% | **96.5%** | 80.4% | **7 (3.5%)** |

- **5y:** more start-dependent. Worst *absolute* V19d CAGRs are early-2000 (near flat / low single digits) — still crush QQQ, which was deeply negative. Relative underperformance clusters in mid-cycle bulls (2014–15, 2011, post-2002 bounce) where QQQ ran hard for five years.
- **10y:** robust. Strict backfires are tiny-margin cases around GFC bottom / 2010 (e.g. 2009-03 −1.4pp CAGR). Dot-com 10y windows: V19d ~5–9% CAGR vs QQQ deeply negative terminal wealth.

### Crisis note (start-conditional MaxDD)

Worst realized MaxDD for V19d (−40.7%) requires a start **at or before** early-2000. From 2002 onward starts, MaxDD collapses to the familiar ~−25% COVID floor (or better for post-COVID starts). Sizing should still assume ~−40% if the investor’s horizon can include a “start into a nascent SMA regime at a secular tech peak.”

## Key Diagnostics

- Signal alignment preserved: analysis only slices already-computed return series.
- Quarter-start subset (n=95) matches month-grid: ~90% beat QQQ on CAGR/Sharpe; 5 strict backfires (same recent cluster).
- Artifacts: short remaining samples (2022–23) inflate QQQ’s relative CAGR; do not extrapolate “AI bull underperformance” as a multi-decade property.

## Interpretation

**Verdict: robustly positive for multi-year / multi-decade starts; start-dependent mainly in short post-2021 windows vs naked QQQ.**

1. Against IVV, 50/50, and especially 60/40, V19d almost never “backfires” on growth or risk-adjusted metrics through present; MaxDD is uniformly better vs equity B&H.
2. Against QQQ, long-history starts (pre-2021) win ~99% of the time on CAGR/terminal and ~96% on Sharpe. The only historical strict backfire is Mar 2009 (economically negligible).
3. True strict backfires concentrate in **May 2022 – Jul 2023** starts: not strategy collapse, but a short sample where buy-and-hold Nasdaq from a deep trough into an AI melt-up outruns a gated/levered system that sometimes sits in cash.
4. Dot-com entry is the **worst drawdown story** and still a **strong relative story** vs QQQ — consistent with Marketstack verification’s honest −40.7% MaxDD finding.
5. Five-year hold periods can look unfavorable vs QQQ ~16% of the time; ten-year holds almost always restore the edge.

## Decision

No architecture change. Finding informs **investor communication / sizing**:

- Expect MaxDD up to ~−40% if starting near a secular peak with immature SMA history.
- Do not treat 3-year underperformance vs QQQ after a Nasdaq crash bottom as a falsification of V19d.
- For ≥10-year horizons, start-date sensitivity is low and outcomes are robustly favorable.

## Next Steps

- Optional: overlay contribution timing (lump-sum vs DCA start-month grid) — DCA already in CSV columns.
- Live path / watcher work remains owned separately; not in scope here.

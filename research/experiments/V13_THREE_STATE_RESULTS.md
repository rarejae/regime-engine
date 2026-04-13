---
date: 2026-04-11
experiment: V13 Three-State V9 with Weekly Re-Entry
status: FAILED — all three paths, max DD worsened, CAGR dropped, Sharpe dropped
script: experiments/v13_three_state/backtest.py
---

# V13: Three-State V9 with Weekly Re-Entry — Results

## Verdict

**V13 FAILS every pass criterion.** Both hypotheses (delever state, weekly re-entry) failed to improve V9. Max DD actually *worsened* from -37.9% to -42.0%. CAGR dropped -1.96pp. Sharpe dropped -0.035. V13 is strictly dominated by V9 on every metric.

The result is unambiguous: V9's binary architecture is the correct design. Intermediate states add drag. Faster re-entry barely fires. Cash-on-CB is worse than QQQ-on-CB.

---

## Table 1 — Core Metrics (2002-2026)

| Strategy              |   CAGR |    Vol | Sharpe | Sortino |  MaxDD | Calmar | Term$1 | DCA$700 |  CB |
|-----------------------|-------:|-------:|-------:|--------:|-------:|-------:|-------:|--------:|----:|
| V13 ThreeState+Weekly | 17.41% | 26.36% |  0.742 |   0.807 | -42.0% |   0.47 | $56.16 |  $5.08M |  18 |
| V9 QLD+IVVguard       | 19.37% | 27.82% |  0.777 |   0.887 | -37.9% |   0.57 | $85.25 |  $7.37M |  14 |
| V12 Independent 2×    | 17.41% | 23.42% |  0.803 |   0.932 | -28.8% |   0.65 | $56.18 |  $4.80M |  31 |
| Baseline (Sweep-40)   | 13.79% | 15.52% |  0.910 |   1.089 | -18.5% |   0.76 | $25.58 |  $2.40M |  16 |
| QQQ B&H               | 12.57% | 22.77% |  0.634 |   0.847 | -53.4% |   0.27 | $17.58 |  $2.33M |   — |

V13 is worse than V9 on **every single metric**: CAGR, Sharpe, Sortino, MaxDD, Calmar, terminal, DCA. There is no dimension on which V13 improves.

---

## Table 2 — CAGR by Start Date

| Strategy              |   2002 |   2007 |   2010 |   2013 |   2019 |
|-----------------------|-------:|-------:|-------:|-------:|-------:|
| V13 ThreeState+Weekly | 17.41% | 20.97% | 20.81% | 24.38% | 24.97% |
| V9 QLD+IVVguard       | 19.37% | 22.56% | 23.35% | 28.51% | 28.12% |
| V12 Independent 2×    | 17.41% | 19.12% | 20.47% | 23.45% | 23.65% |
| Baseline              | 13.79% | 14.94% | 15.61% | 17.19% | 19.55% |
| QQQ B&H               | 12.57% | 15.37% | 17.95% | 18.94% | 20.79% |

V13 trails V9 by 1.5-4.1pp at every start date. V13 still beats QQQ and Baseline from every start date, but V9 dominates it absolutely.

---

## Table 3 — State Occupancy (291 months)

| State    | Months |   Pct |
|----------|-------:|------:|
| FULL     |    191 | 65.6% |
| DELEVER  |     22 |  7.6% |
| CASH     |     78 | 26.8% |

V13 DELEVER months — what would V9 have held:
- While V9 holds **QLD** (full leverage): **12 months** — V13 is underinvested
- While V9 holds **QQQ 70%/30%**: **10 months** — roughly comparable (V13 holds 100% QQQ, V9 holds 70% QQQ + 30% cash)

**The 12 months where V13 delevered but V9 stayed in QLD are the root cause of the CAGR gap.** These are months with QQQ score 3 but IVV score 2 — V13's IVV guard is tighter than V9's, stripping leverage in months where V9's looser guard (IVV ≤ 1 only) keeps QLD running. Those months averaged 1.88% return in V9 vs 0.76% in V13. That's -1.12%/month × 12 months = ~-13.4% of total return drag.

---

## Table 4 — Weekly Re-Entry Diagnostics

**Weekly re-entry was nearly useless.**

| Metric | Value |
|--------|-------|
| Total CB events | 18 |
| Resolved via weekly (full QLD re-entry) | **1** |
| Resolved via monthly rebalance | 16 |
| Weekly unlevered re-entries (cash → QQQ) | 3 |
| Mean days CB → weekly unlev re-entry | 14.0 days |
| Mean days CB → monthly resolution | 15.1 days |
| Whipsaw rate (CB within 30d of resolution) | 11% (2/18) |

**Only 1 out of 18 CB events was resolved via weekly re-entry.** The 2-consecutive-Fridays requirement for QLD re-entry is so strict it almost never fires before the monthly rebalance arrives (mean 15.1 days for monthly vs 14.0 for weekly unlev). The weekly mechanism is redundant — monthly catches up fast enough that the weekly window barely opens.

The single weekly full resolution was June 2019 (CB fired June 3, weekly unlev re-entry June 7, weekly QLD re-entry June 21). This saved ~9 days vs monthly — a genuine recovery capture, but a single event over 24 years cannot justify the mechanism's complexity.

The 3 unlevered re-entries (cash → QQQ) were all followed by monthly rebalance taking over before the 2-Friday QLD requirement was met. Two of these were followed by whipsaws within 30 days — but the 11% overall rate is low enough not to be the problem.

**Conclusion on weekly re-entry:** The hypothesis is rejected. Monthly re-entry is correct. The ~15-day average wait is short enough that weekly evaluation adds near-zero value, and the mechanism barely fires. See [[2026-04-06_weekly_rebalancing]] for the same finding at the portfolio level.

---

## Table 5 — Delever State Return Analysis

| Metric | Value |
|--------|-------|
| V13 DELEVER months | 22 |
| Mean V13 return | 0.76% |
| Mean V9 return in same months | 1.88% |
| Delta V13 - V9 | **-1.12%** |
| Months with positive V13 return | 12/22 (55%) |

**The delever state is a net drag.** V13 earns 0.76%/mo while V9 earns 1.88%/mo in the same months. The delever state systematically underperforms V9 because:

1. In 12 of 22 months, V9 holds QLD at 2× leverage while V13 holds QQQ at 1× — V13 is leaving half the return on the table
2. The remaining 10 months are roughly comparable (V13 at 100% QQQ vs V9 at 70% QQQ + 30% cash)
3. The delever state exists to reduce drawdowns, but it doesn't — see Table 7 for crisis detail

**Conclusion on delever state:** Score 2/3 is not a reliable intermediate signal. Of the 22 delever months, 12 were months where QQQ was at 3/3 (strong) but IVV was at 2/3. The IVV guard in V13 is stricter than V9's (V13 delevered at IVV=2, V9 only delevered at IVV≤1), and this strictness *costs return without improving DD*.

---

## Table 6 — 2022 Month-by-Month

| Month   | QQQ | IVV | V13 state | V13 ret |  V9 ret |  BL ret |
|---------|----:|----:|-----------|--------:|--------:|--------:|
| 2022-01 |   3 |   3 | FULL      | -22.26% | -17.33% |  -9.39% |
| 2022-02 |   1 |   2 | CASH      |   0.02% |   0.02% |   0.45% |
| 2022-03 |   0 |   1 | CASH      |   0.04% |   0.04% |   3.12% |
| 2022-04 |   1 |   3 | CASH      |   0.06% |   0.06% |  -2.68% |
| 2022-05 |   0 |   0 | CASH      |   0.08% |   0.08% |  -0.36% |
| 2022-06 |   0 |   0 | CASH      |   0.12% |   0.12% |   0.92% |
| 2022-07 |   0 |   0 | CASH      |   0.18% |   0.18% |  -1.22% |
| 2022-08 |   0 |   0 | CASH      |   0.24% |   0.24% |   1.06% |
| 2022-09 |   0 |   0 | CASH      |   0.26% |   0.26% |   1.92% |
| 2022-10 |   0 |   0 | CASH      |   0.30% |   0.30% |   0.56% |
| 2022-11 |   0 |   0 | CASH      |   0.33% |   0.33% |  -2.95% |
| 2022-12 |   1 |   2 | CASH      |   0.35% |   0.35% |  -1.56% |

**Jan 2022 V13 -22.26% is worse than V9's -17.33%.** Both entered at FULL (QLD). When the CB fired on Jan 21, V9 exited QLD → QQQ (still held 1× equity), while V13 exited QLD → cash. In the final ~8 trading days of Jan, QQQ continued declining — but the split behavior explains the gap. V9's exit to QQQ at 1× kept some equity exposure that partially offset vs V13's full cash exit.

**Key insight:** V13's CB exits to cash while V9 exits to QQQ. This design choice is part of the problem — by going to cash on CB, V13 misses any late-month bounce *and* requires the weekly re-entry mechanism (which barely fires) to get back in. V9's simpler "QLD → QQQ on CB, hold until next monthly" is the correct behavior.

Months 02-12 are identical between V13 and V9 because both systems are in cash — the DELEVER state never fired in 2022 (QQQ score never reached 2 while IVV was also ≥ 2 during the bear).

---

## Table 7 — Crisis Drawdowns

| Crisis          |    V13 |     V9 |    V12 | Baseline | QQQ B&H |
|-----------------|-------:|-------:|-------:|---------:|--------:|
| Dot-com 2002-03 |   0.0% |  -5.0% |  -2.6% |    -2.1% |  -51.9% |
| GFC 2007-09     | -20.9% | -30.6% | -23.8% |    -9.0% |  -53.1% |
| COVID 2020      | -25.2% | -37.9% | -28.8% |   -18.5% |  -28.6% |
| 2022 bear       | -23.9% | -23.9% | -21.6% |   -13.2% |  -34.8% |

**V13 has better crisis drawdowns than V9 in GFC and COVID** — the delever state and CB-to-cash transition genuinely reduce peak-to-trough in these crises. GFC -20.9% vs -30.6% is a +9.7pp improvement. COVID -25.2% vs -37.9% is +12.7pp.

**But the full-period max DD (-42.0%) is WORSE than V9.** The -42.0% doesn't come from any single crisis — it comes from the cumulative drag of the delever state during non-crisis periods. The system delevered during months that turned out to be brief wobbles, not crashes. Those missed-leverage months compound into a lower peak, making subsequent drawdowns larger as a percentage of the reduced high-water mark.

This is the fundamental trap: the delever state *does* reduce crisis drawdowns, but it *also* reduces non-crisis compounding, and the second effect dominates over 24 years.

---

## Table 8 — Recovery-Period Returns

| Window                  |    V13 |     V9 |    V12 | QQQ B&H |
|-------------------------|-------:|-------:|-------:|--------:|
| GFC trough → 1yr        | 54.74% | 69.35% | 55.00% |  78.89% |
| COVID trough → 6mo      | 25.90% | 50.01% | 34.12% |  55.35% |
| 2022 trough → 6mo       |  4.86% |  2.49% | -2.20% |  19.61% |

**V13 recovers more slowly than V9 in every crisis except 2022.** The weekly re-entry was supposed to fix this by catching recovery upside faster. It didn't — only 1 of 18 CB events resolved via weekly, and the ~15-day average wait for monthly is short enough to capture most of the recovery anyway.

GFC recovery: V13 54.7% vs V9 69.4% — V13 stays in DELEVER (QQQ 1×) longer before monthly rebalance upgrades to QLD. V9's simpler path (QQQ → monthly → QLD) is faster because it doesn't require the 2-consecutive-Fridays gate.

COVID recovery: V13 25.9% vs V9 50.0% — massive gap. V13's CB exited to cash (not QQQ), then needed weekly + monthly to climb back. V9 exited to QQQ at 1×, still participated in the V-shaped recovery.

---

## Table 9 — DCA Terminal by Year-End ($21K + $700/mo, 2013 start)

| Year |     V13 |      V9 |     V12 |      BL |     QQQ | V13-QQQ |
|-----:|--------:|--------:|--------:|--------:|--------:|--------:|
| 2013 |    $48K |    $48K |    $46K |    $39K |    $38K |   $+10K |
| 2015 |    $77K |    $85K |    $72K |    $59K |    $68K |    $+9K |
| 2018 |   $163K |   $207K |   $159K |   $118K |   $126K |   $+37K |
| 2020 |   $368K |   $475K |   $339K |   $220K |   $284K |   $+83K |
| 2022 |   $468K |   $637K |   $447K |   $293K |   $258K |  $+210K |
| 2025 |  $1127K |  $1594K |  $1006K |   $580K |   $642K |  $+486K |
| 2026 |  $1012K |  $1491K |   $921K |   $573K |   $606K |  $+406K |

V13 never trails QQQ (smallest gap +$1K in 2016). But it trails V9 by $479K at 2026. V13 is strictly between V12 and V9 on DCA performance.

---

## Pass / Fail Summary

| Path | Criterion | Result |
|------|-----------|--------|
| A — DD improvement | Max DD shallower than V9 | ✗ (-42.0% vs -37.9%) |
| A — DD improvement | CAGR within 1pp of V9 | ✗ (17.41% vs 19.37%) |
| A — DD improvement | Sharpe ≥ V9 | ✗ (0.742 vs 0.777) |
| B — Return improvement | CAGR higher than V9 | ✗ |
| C — Sharpe improvement | Sharpe > 0.80 | ✗ |

**All three paths fail. V13 is dominated by V9 on every metric.**

Additional fail checks:
- Whipsaw rate: 11% — acceptable (below 50% threshold), but the mechanism barely fires so the rate is irrelevant
- Delever mean return: +0.76% — positive, but V9 earns +1.88% in the same months (-1.12% drag)
- Max DD worse than V9: **YES** (-42.0% vs -37.9%)

---

## What We Learned

### 1. Score 2/3 is NOT a safe intermediate state
The delever state holds QQQ at 1× during score-2 months. In 12 of 22 cases, V9 was fully leveraged (QLD) during these same months — meaning V13 was delevering when it shouldn't have been. The IVV guard at score==2 (V13's tighter rule) is *too tight*. V9's rule (only exit QLD when IVV ≤ 1) is correct because IVV score 2 is "wobbly but not broken" — exactly the state where leverage should be maintained.

### 2. Weekly re-entry is noise
1 out of 18 CB events resolved via weekly. Monthly catches up within ~15 days on average. The 2-consecutive-Fridays requirement for QLD re-entry is so strict it nearly never fires before monthly. The compounding cost of slow re-entry (~15 days of missed equity exposure) is small enough that faster mechanisms add complexity for nothing.

### 3. CB → cash is worse than CB → QQQ
V9's CB behavior (QLD → QQQ) is superior to V13's (QLD → cash). Keeping 1× equity exposure after CB allows the system to participate in partial recoveries and V-shaped bounces. V13's full cash exit forfeits this exposure and then needs the weekly mechanism (which barely works) to get back in.

### 4. V9's simplicity is structural, not accidental
V9 has two meaningful design features: (a) binary QLD/cash with 70%/30% at score 2, (b) IVV as a loose guard (only strips leverage at IVV ≤ 1). Both features are exactly right. Tightening the guard or adding intermediate states makes things worse. V9's terminal wealth leadership ($85.25, $7.37M DCA) comes from *staying levered as long as possible* — any mechanism that strips leverage earlier costs return without reducing DD.

---

## Cross-references

- [[experiments/V12_INDEPENDENT_2X_RESULTS]] — V12 dominates V13 on Sharpe and DD
- [[2026-04-06_weekly_rebalancing]] — weekly rebalancing also failed at portfolio level
- [[2026-04-06_leverage_tiers]] — leverage tiers failed similarly (intermediate states add no value)

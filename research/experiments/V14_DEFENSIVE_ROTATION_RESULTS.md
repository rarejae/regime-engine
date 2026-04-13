---
date: 2026-04-12
experiment: V14 V9 + Defensive Rotation During Cash Periods
status: FAILED — all variants worsen MaxDD vs V9
script: experiments/v14_defensive_rotation/backtest.py
---

# V14: Defensive Rotation During V9 Cash Periods — Results

## Verdict

**V14 FAILS.** All three variants worsen MaxDD vs V9. The defensive pool earns marginally more than T-bills (+0.04%/mo) but with 3× the volatility — the risk-return tradeoff is terrible. The 2022 bear confirms the mechanism: DBC collapsed -7.50% and -7.04% in months where V9 peacefully sat in cash earning T-bill rates.

**V14-B (IVV ≥ 3) is the closest to passing:** +1.04pp CAGR, +0.016 Sharpe, $105.89 terminal (vs V9 $85.25). But MaxDD worsened from -37.9% to -39.1% — the pass criterion is strict and V14-B fails it by -1.2pp.

**The project's earlier finding is confirmed and reinforced: cash IS the hedge.** Defensive assets during equity off-signal periods are a mirage.

---

## Table 1 — Core Metrics (2002-2026)

| Strategy             |   CAGR |    Vol | Sharpe | Sortino |  MaxDD | Calmar | Term$1 |  DCA$700 |  CB |
|----------------------|-------:|-------:|-------:|--------:|-------:|-------:|-------:|---------:|----:|
| V14-A (IVV≥2)        | 19.28% | 28.65% |  0.760 |   0.944 | -38.2% |   0.57 | $83.64 |   $7.09M |  14 |
| **V14-B (IVV≥3)**    | 20.41% | 28.63% |  0.793 |   0.977 | -39.1% |   0.58 | $105.89|   $8.61M |  14 |
| V14-C (no IVV)       | 20.11% | 28.67% |  0.783 |   0.969 | -41.1% |   0.55 | $99.43 |   $8.34M |  14 |
| V9 (control)         | 19.37% | 27.82% |  0.777 |   0.887 | -37.9% |   0.57 | $85.25 |   $7.37M |  14 |
| V12 Independent 2×   | 17.41% | 23.42% |  0.803 |   0.932 | -28.8% |   0.65 | $56.18 |   $4.80M |  31 |
| Baseline (Sweep-40)  | 13.79% | 15.52% |  0.910 |   1.089 | -18.5% |   0.76 | $25.58 |   $2.40M |  16 |
| QQQ B&H              | 12.57% | 22.77% |  0.634 |   0.847 | -53.4% |   0.27 | $17.58 |   $2.33M |   — |

V14-B has the highest absolute terminal wealth ($105.89) and DCA ($8.61M) of any variant ever tested. The cost: -39.1% MaxDD, which is 1.2pp worse than V9 and ~10pp worse than V12.

---

## Table 2 — CAGR by Start Date

| Strategy             |   2002 |   2007 |   2010 |   2013 |   2019 |
|----------------------|-------:|-------:|-------:|-------:|-------:|
| V14-A (IVV≥2)        | 19.28% | 22.75% | 22.51% | 26.04% | 24.96% |
| V14-B (IVV≥3)        | 20.41% | 23.91% | 23.87% | 27.76% | 26.51% |
| V14-C (no IVV)       | 20.11% | 23.92% | 23.89% | 28.02% | 27.12% |
| V9 (control)         | 19.37% | 22.56% | 23.35% | 28.51% | 28.12% |

V14-B beats V9 from 2002, 2007, and 2010, but trails from 2013 and 2019. The defensive pool earns its keep during the early period (2002-2012 has more off-signal months with good defensive trends), but during the 2013-2021 bull the ~66% time in QLD dominates and defensives are irrelevant — V9's slightly lower vol edges ahead.

V14-A trails V9 from 2013 by 2.5pp — IVV at score≥2 during off-signal months is a drag. IVV's correlation with QQQ means the Faber gate doesn't protect it during equity bear markets.

---

## Table 3 — Defensive Utilization During Off-Signal Months

73 months where V9 holds cash:

**Mean monthly return during off-signal:**

| Variant |  Mean | Vol  | Positive months | Sharpe-like |
|---------|------:|-----:|----------------:|:-----------:|
| V14-A   | +0.29%| 3.92%| 48/73 (66%)    | ~0.07       |
| V14-B   | +0.60%| 3.74%| 51/73 (70%)    | ~0.16       |
| V14-C   | +0.52%| 3.78%| 48/73 (66%)    | ~0.14       |
| V9 cash | +0.25%| 1.33%| 71/73 (97%)    | ~0.19       |

**V9's cash has a *higher* risk-adjusted return than any defensive pool.** The defensive pool earns 0.04-0.35%/mo more than cash, but with 2.8-3× the volatility. Cash has 97% positive months vs 66-70% for defensives. The defensive pool's Sharpe-like ratio (~0.07-0.16) is *lower* than cash's (~0.19).

**Defensive asset activity:**

| Asset     | Months active |   Pct |
|-----------|-------------:|------:|
| IVV       |          17  |  23%  |
| VGLT      |          35  |  48%  |
| IAU       |          34  |  47%  |
| DBC       |          19  |  26%  |
| Cash only |          19  |  26%  |

26% of off-signal months have NO eligible defensives — all four fail their own Faber gates simultaneously. In those months V14 defaults to cash (identical to V9). The Faber gate works correctly but confirms that during the worst equity bear markets, all asset trends break together.

**Pool size distribution:**

| Defensives active | Months |
|:-----------------:|:------:|
| 0 (cash fallback) |   19   |
| 1                 |   16   |
| 2                 |   26   |
| 3                 |   11   |
| 4                 |    1   |

Most off-signal months have 1-2 active defensives. The diversification benefit of equal-weighting is limited when only 1-2 assets qualify.

---

## Table 4 — 2022 Month-by-Month (V14-A)

| Month   | QQQ | IVV | Mode    | Defensives  |  V14-A |     V9 |     BL |
|---------|----:|----:|---------|-------------|-------:|-------:|-------:|
| 2022-01 |   3 |   3 | OFFENSE | QLD         | -17.33%| -17.33%|  -9.39%|
| 2022-02 |   1 |   2 | DEFENSE | IVV+IAU+DBC |  +3.21%|  +0.02%|  +0.45%|
| 2022-03 |   0 |   1 | DEFENSE | IAU+DBC     |  +5.30%|  +0.04%|  +3.12%|
| 2022-04 |   1 |   3 | DEFENSE | IVV+IAU+DBC |  -1.83%|  +0.06%|  -2.68%|
| 2022-05 |   0 |   0 | DEFENSE | IAU+DBC     |  +0.66%|  +0.08%|  -0.36%|
| 2022-06 |   0 |   0 | DEFENSE | DBC         |  **-7.50%**|  +0.12%|  +0.92%|
| 2022-07 |   0 |   0 | DEFENSE | DBC         |  -1.99%|  +0.18%|  -1.22%|
| 2022-08 |   0 |   0 | DEFENSE | DBC         |  -1.49%|  +0.24%|  +1.06%|
| 2022-09 |   0 |   0 | DEFENSE | DBC         |  **-7.04%**|  +0.26%|  +1.92%|
| 2022-10 |   0 |   0 | DEFENSE | cash        |  +0.30%|  +0.30%|  +0.56%|
| 2022-11 |   0 |   0 | DEFENSE | cash        |  +0.33%|  +0.33%|  -2.95%|
| 2022-12 |   1 |   2 | DEFENSE | IVV         |  -5.76%|  +0.35%|  -1.56%|

**This is the kill shot for V14.** Jun-Sep 2022 the ONLY active defensive was DBC (commodities). VGLT, IAU, and IVV all failed their Faber gates. DBC was the sole remaining asset with a healthy trend score — but commodity prices reversed brutally mid-2022. Four consecutive months of DBC-only exposure produced: -7.50%, -1.99%, -1.49%, -7.04%. V9 earned +0.12%, +0.18%, +0.24%, +0.26% in T-bills during the same months.

Dec 2022 is another scar: IVV was the only defensive and returned -5.76% (IVV dropped from score 3 to below 2 during the month).

**The Faber gate correctly excluded VGLT, IAU, and IVV as their trends broke.** But this concentrates the entire portfolio into the last surviving defensive — when DBC was the only asset with a healthy trend, V14 went 100% into commodities. Single-asset concentration during a bear market is the opposite of defensive.

---

## Table 5 — Crisis Drawdowns

| Crisis          | V14-A  | V14-B  | V14-C  |     V9 |    V12 | Baseline | QQQ B&H |
|-----------------|-------:|-------:|-------:|-------:|-------:|---------:|--------:|
| Dot-com 02-03   |  -6.7% |   0.0% |   0.0% |  -5.0% |  -2.6% |    -2.1% |  -51.9% |
| GFC 07-09       | -37.2% | -37.2% | -37.2% | -30.6% | -23.8% |    -9.0% |  -53.1% |
| COVID 2020      | -37.9% | -37.9% | -37.9% | -37.9% | -28.8% |  -18.5%  |  -28.6% |
| 2022 bear       | -32.6% | -27.0% | -24.4% | -23.9% | -21.6% |   -13.2% |  -34.8% |

**GFC is the structural failure.** V14 draws down -37.2% vs V9's -30.6% (+6.6pp worse). During the GFC's off-signal months, V14 held defensive assets that themselves crashed — bonds had spread blowouts, gold was volatile, commodities collapsed. The Faber gate excluded assets sequentially but each one took damage before being removed.

**COVID is identical** (-37.9%) — V14 and V9 hold QLD during the crash and the CB fires at the same point. Off-signal capital behavior doesn't affect COVID because the system re-enters QLD before any meaningful off-signal period.

**2022 worst for V14-A** (-32.6%) — IVV at score ≥ 2 gets held during the broad-market decline, adding 8.7pp of drawdown vs V9. V14-B is slightly better (-27.0%) and V14-C (-24.4%) is the best of the three because it excludes both IVV and the DBC-only concentration (fewer months of single-asset holding).

---

## Table 6 — DCA Terminal by Year-End ($21K + $700/mo, 2013 start)

| Year |   V14-A |   V14-B |   V14-C |      V9 |     QQQ |
|-----:|--------:|--------:|--------:|--------:|--------:|
| 2015 |    $84K |    $84K |    $84K |    $85K |    $68K |
| 2018 |   $191K |   $209K |   $208K |   $207K |   $126K |
| 2020 |   $459K |   $502K |   $498K |   $475K |   $284K |
| 2022 |   $511K |   $611K |   $629K |   $637K |   $258K |
| 2025 |  $1232K |  $1465K |  $1507K |  $1594K |   $642K |
| 2026 |  $1153K |  $1370K |  $1409K |  $1491K |   $606K |

V14-B leads through 2020 ($502K vs V9 $475K) as defensives outperform T-bills in the pre-2022 period. But the 2022 bear wipes out the advantage — V14-B drops from $786K (2021) to $611K (2022), while V9 drops from $745K to $637K. V9 never trails after 2022 because its cash protection preserved more capital through the bear.

---

## Table 7 — Recovery-Period Returns

| Window                 |  V14-A |  V14-B |  V14-C |     V9 | QQQ B&H |
|------------------------|-------:|-------:|-------:|-------:|--------:|
| GFC trough → 1yr       | 72.44% | 72.44% | 72.44% | 69.35% |  78.89% |
| COVID trough → 6mo     | 56.39% | 56.39% | 56.39% | 50.01% |  55.35% |
| 2022 trough → 6mo      |  1.42% |  8.00% |  8.00% |  2.49% |  19.61% |

V14 recoveries are slightly better than V9 in GFC and COVID — defensive assets participate in the initial recovery while V9 sits in cash. But this recovery advantage is small (+3pp GFC, +6pp COVID) and insufficient to offset the crisis DD damage.

---

## Pass / Fail

| Variant | CAGR (within 1pp) | Sharpe ≥ V9 | MaxDD ≤ V9 | Overall |
|---------|:-----------------:|:-----------:|:----------:|:-------:|
| V14-A   | ✓ (19.28%)        | ✗ (0.760)   | ✗ (-38.2%) | FAIL    |
| V14-B   | ✓ (20.41%)        | ✓ (0.793)   | ✗ (-39.1%) | FAIL    |
| V14-C   | ✓ (20.11%)        | ✓ (0.783)   | ✗ (-41.1%) | FAIL    |

**All variants fail on MaxDD.** The defensives add drawdown exposure during periods that are supposed to be safe. Cash's immunity to asset-class declines is worth more than the marginal return from defensive rotation.

---

## What We Learned

### 1. Cash IS the hedge — definitively confirmed
The defensive pool earns +0.29%/mo (V14-A) vs +0.25%/mo (V9 cash) during off-signal months — a +0.04% advantage with 3× the volatility. The risk-adjusted return of cash is *higher* than any defensive pool configuration. This was the project's earliest finding ([[2026-04-05_pro_rata_vs_cash]]) and V14 confirms it with a completely different architecture.

### 2. Faber-gated defensives concentrate into the last survivor
The Faber gate correctly excludes assets as their trends break. But this creates a perverse concentration: during the worst bear markets, only 1 asset survives, and V14 goes 100% into it. Jun-Sep 2022 was 100% DBC; Dec 2022 was 100% IVV. Single-asset concentration during bears is the opposite of defense.

### 3. IVV at score ≥ 2 is too loose for defensive holding
V14-A (-38.2% DD) is the worst variant. IVV at score 2 is "wobbly" — holding it during equity bear markets adds drawdown. V14-B's higher threshold (IVV ≥ 3) avoids the worst of this but can't prevent the DBC/VGLT concentration problem.

### 4. V9's off-signal cash is load-bearing
V9 sits in cash ~34% of months. That cash protects the portfolio's high-water mark during bear markets. Converting cash to defensive assets trades guaranteed protection for uncertain return — and during the moments protection matters most (2022, GFC), the defensive assets fail their own protection guarantee.

### 5. V14-B is the closest to a real improvement
$105.89 terminal (+24% vs V9), $8.61M DCA (+$1.24M vs V9), 0.793 Sharpe (+0.016 vs V9). If the user is willing to accept -39.1% MaxDD (1.2pp worse than V9), V14-B is the highest-terminal-wealth strategy ever tested. But it violates the explicit pass criterion.

---

## Cross-references

- [[2026-04-05_pro_rata_vs_cash]] — "Pro-rata destroys Sharpe by -0.151, doubles max DD. Cash is the hedge." V14 confirms this finding with a completely different mechanism.
- [[experiments/V13_THREE_STATE_RESULTS]] — V13 failed by modifying V9's offense. V14 failed by modifying V9's defense. V9 is correct as-is.
- [[experiments/V11_BETA_SCALED_RESULTS]] — V11's DBMF-based defensive pool had the same concentration problem (Nov 2022 -8.84% from DBMF-only months)

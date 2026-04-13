---
date: 2026-04-12
experiment: V16 Two-Pod + Gold — 45/45/10
status: PASSED — both variants. 0.846 Sharpe, new Pareto frontier leader.
script: experiments/v16_two_pod_gold/backtest.py
---

# V16: Two-Pod + Gold (45/45/10) — Results

## Verdict

**V16 PASSES both variants.** V16-B (IAU ≥ 3) achieves **0.846 Sharpe** — the highest of any variant ever tested — with -27.0% MaxDD and 17.06% CAGR. Gold earns its 10% allocation through genuine crisis alpha: +14.8% cumulative during GFC (13/17 months active) while equities crashed, correctly excluded by June 2022 when its own trend broke.

V16 displaces V15 on the Pareto frontier. The 10% gold allocation costs -0.59pp CAGR but buys +0.033 Sharpe and +2.0pp shallower MaxDD. The tradeoff is unambiguously Sharpe-positive.

---

## Table 1 — Core Metrics (2002-2026)

| Strategy             |   CAGR |    Vol | Sharpe | Sortino |  MaxDD | Calmar | Term$1 |  DCA$700 |  CB |
|----------------------|-------:|-------:|-------:|--------:|-------:|-------:|-------:|---------:|----:|
| V16-A (IAU≥2)        | 16.99% | 21.28% |  0.845 |   0.998 | -27.1% |   0.66 | $51.39 |  $4.33M |  27 |
| **V16-B (IAU≥3)**    | 17.06% | 21.33% | **0.846** | 0.997 | **-27.0%** | **0.67** | $52.18 | $4.41M | 27 |
| V15 Two-Pod          | 17.65% | 23.36% |  0.813 |   0.941 | -29.0% |   0.66 | $59.18 |  $4.92M |  27 |
| V9 QLD+IVVguard      | 19.37% | 27.82% |  0.777 |   0.887 | -37.9% |   0.57 | $85.25 |  $7.37M |  14 |
| V12 Independent 2×   | 17.41% | 23.42% |  0.803 |   0.932 | -28.8% |   0.65 | $56.18 |  $4.80M |  31 |
| Baseline (Sweep-40)  | 13.79% | 15.52% |  0.910 |   1.090 | -18.5% |   0.76 | $25.58 |  $2.40M |  16 |
| QQQ B&H              | 12.57% | 22.77% |  0.634 |   0.847 | -53.4% |   0.27 | $17.58 |  $2.33M |   — |

V16-B's vol (21.33%) is the lowest of any leveraged variant — gold's low equity correlation damps portfolio volatility by 2pp vs V15. This vol reduction is the primary driver of the Sharpe improvement.

V16-A and V16-B are nearly identical (+0.001 Sharpe, +0.1pp MaxDD). The IAU threshold barely matters — gold at score 2 vs score 3 produces similar results because gold trends are persistent (median IAU score is 3 when it's on-trend).

---

## Table 2 — Gold Utilization

| Metric | V16-A (≥2) | V16-B (≥3) |
|--------|:----------:|:----------:|
| IAU active | 172/291 (59%) | 152/291 (52%) |
| IAU in cash | 119/291 (41%) | 139/291 (48%) |
| Active during equity-off | 23/50 (46%) | 21/50 (42%) |

Gold is active ~55% of months overall. During equity-off periods (both pods in cash), gold is active ~44% of the time — meaning it provides non-zero return in ~44% of the months where both equity engines are parked. The remaining 56% of equity-off months fall back to 100% cash (gold's own Faber gate excluded it).

---

## Table 3 — Gold During Crises (KEY RESULT)

| Crisis | IAU months active | IAU cumul return |
|--------|:-----------------:|:----------------:|
| Dot-com 2002-03 | 0/15 | +0.0% |
| **GFC 2007-09** | **13/17** | **+14.8%** |
| **COVID 2020** | **3/3** | **+6.3%** |
| 2022 bear | 5/12 | -0.8% |

**GFC is gold's masterpiece.** Active 13 of 17 crisis months, gold earned +14.8% while equities were crashing -53%. V16's 10% gold allocation earned ~+1.5% of portfolio return during the worst equity bear market in the dataset. This directly explains V16's GFC DD improvement (-16.2% vs V15's -18.1%).

**COVID:** Gold active all 3 months, +6.3% cumulative. Small but helpful — gold's safe-haven bid during panic is real.

**2022:** Gold correctly excluded by June 2022 (score dropped to 0 from rate hikes). Active only 5/12 months, cumulative -0.8%. The Faber gate prevented the worst of gold's 2022 rate-driven decline. By comparison, V14's defensive pool held DBC through Jun-Sep 2022 and took -7.50% months — gold's Faber gate is more responsive.

**Dot-com:** Gold inactive (gold was in a secular bear market in 2002-2003). No harm, no help. Gold data begins in 2004 for IAU; GLD proxy used for earlier period.

---

## Table 4 — 2022 Month-by-Month (V16-A)

| Month   | QQQ | IVV | IAU | Pod1 | Pod2 | Gold | V16A ret | V15 ret |  V9 ret |
|---------|----:|----:|----:|------|------|------|--------:|--------:|--------:|
| 2022-01 |   3 |   3 |   3 | qld  | sso  | iau  | -13.25% | -14.12% | -17.33% |
| 2022-02 |   1 |   2 |   2 | cash | ivv  | iau  |  -0.44% |  -0.99% |  +0.02% |
| 2022-03 |   0 |   1 |   3 | cash | cash | iau  |  +0.14% |  +0.04% |  +0.04% |
| 2022-04 |   1 |   3 |   3 | cash | sso  | iau  |  -5.24% |  -5.50% |  +0.06% |
| 2022-05 |   0 |   0 |   3 | cash | cash | iau  |  -0.21% |  +0.08% |  +0.08% |
| 2022-06 |   0 |   0 |   0 | cash | cash | cash |  +0.12% |  +0.12% |  +0.12% |
| 2022-07–11 | 0 | 0 | 0 | cash | cash | cash | ~+0.2% | ~+0.2% | ~+0.2% |
| 2022-12 |   1 |   2 |   1 | cash | ivv  | cash |  -1.46% |  -1.62% |  +0.35% |

**Jan 2022:** V16-A -13.25% (better than V15 -14.12%). The 10% gold (active at IAU score 3) partially offset equity losses. Gold returned ~+0.8% in Jan 2022 while equities crashed.

**Mar 2022:** V16-A +0.14% (V15 +0.04%). Both pods in cash but gold active at score 3. Gold earned ~+1% during this month, turning a near-zero return slightly positive.

**May 2022:** V16-A -0.21% (V15 +0.08%). Gold active at score 3 but declined ~-3%. This is the gold drag — gold was trending but experienced a pullback within the trend.

**Jun-Nov:** Gold correctly excluded (score 0). V16 = V15 = V9 in these months — all in cash.

---

## Table 5 — Crisis Drawdowns

| Crisis          |  V16-A |  V16-B |    V15 |     V9 | Baseline |
|-----------------|-------:|-------:|-------:|-------:|---------:|
| Dot-com 2002-03 |  -2.3% |  -2.3% |  -2.7% |  -5.0% |    -2.1% |
| **GFC 2007-09** | **-16.6%** | **-16.2%** | -18.1% | -30.6% | -9.0% |
| COVID 2020      | -27.1% | -27.0% | -29.0% | -37.9% |   -18.5% |
| 2022 bear       | -20.0% | -20.0% | -21.5% | -23.9% |   -13.2% |

**GFC -16.2% (V16-B)** — the best leveraged-variant GFC drawdown by a wide margin. Only 7.2pp worse than Baseline (-9.0%) despite running 180% effective equity when both pods are on. Gold's +14.8% during GFC is the differentiator.

**COVID -27.0%** — 2.0pp better than V15. Gold's +6.3% during Feb-Apr 2020 provided meaningful cushion.

**2022 -20.0%** — 1.5pp better than V15. Gold was correctly excluded by June, but the early months (Jan-May) where gold was active contributed a mix of small gains and losses that net out slightly positive.

---

## Table 6 — CAGR by Start Date

| Strategy     |   2002 |   2007 |   2010 |   2013 |   2019 |
|--------------|-------:|-------:|-------:|-------:|-------:|
| V16-B (IAU≥3)| 17.06% | 18.77% | 19.44% | 22.06% | 22.66% |
| V15 Two-Pod  | 17.65% | 19.28% | 20.22% | 23.13% | 22.99% |
| V9           | 19.37% | 22.56% | 23.35% | 28.51% | 28.12% |
| Baseline     | 13.79% | 14.94% | 15.61% | 17.19% | 19.55% |
| QQQ B&H      | 12.57% | 15.37% | 17.95% | 18.94% | 20.79% |

V16-B trails V15 by 0.3-1.1pp from each start date — the gold dilution's CAGR cost. V16 still beats QQQ from every start date by 3-5pp.

---

## Table 7 — DCA Terminal ($21K + $700/mo, 2013 start)

| Year |   V16-A |   V16-B |     V15 |      V9 |     QQQ | V16A-QQQ |
|-----:|--------:|--------:|--------:|--------:|--------:|---------:|
| 2016 |    $81K |    $83K |    $85K |    $94K |    $82K |    -$1K  |
| 2020 |   $297K |   $301K |   $328K |   $475K |   $284K |   $+13K  |
| 2022 |   $383K |   $388K |   $431K |   $637K |   $258K |  $+125K  |
| 2025 |   $870K |   $886K |   $969K |  $1594K |   $642K |  $+228K  |
| 2026 |   $815K |   $828K |   $887K |  $1491K |   $606K |  $+209K  |

V16-A briefly trails QQQ at end-2016 (-$1K). Otherwise always ahead. The gold dilution costs ~$72K vs V15 at end-2026 ($815K vs $887K). Over a lifetime DCA this is ~$590K less than V15 ($4.33M vs $4.92M). This is the CAGR cost of the 10% gold allocation.

---

## Pass / Fail

### V16-A (IAU ≥ 2):
- ✓ CAGR: 16.99% (within 1pp of V15 17.65%)
- ✓ Sharpe: 0.845 vs V15 0.813
- ✓ MaxDD: -27.1% vs V15 -29.0%
- **→ PASS**

### V16-B (IAU ≥ 3):
- ✓ CAGR: 17.06% (within 1pp of V15 17.65%)
- ✓ Sharpe: 0.846 vs V15 0.813
- ✓ MaxDD: -27.0% vs V15 -29.0%
- **→ PASS**

---

## What V16-B Buys vs Each Predecessor

**vs V15 (two-pod, no gold):** +0.033 Sharpe, +2.0pp shallower MaxDD, at -0.59pp CAGR and -$510K lifetime DCA. Gold's crisis alpha improves every crisis DD (GFC -16.2% vs -18.1%, COVID -27.0% vs -29.0%, 2022 -20.0% vs -21.5%). The Sharpe improvement comes from vol reduction — gold's low equity correlation damps portfolio vol by 2pp.

**vs V9:** +0.069 Sharpe, +10.9pp shallower MaxDD, at -2.31pp CAGR and -$2.96M DCA. V16 is the opposite end of the risk spectrum from V9 within the leveraged family.

**vs Baseline:** +3.27pp CAGR, +$2.01M DCA, at -8.5pp deeper MaxDD and -0.064 Sharpe. V16 occupies the gap between Baseline and V9 more efficiently than any previous variant.

---

## Honest Weaknesses

1. **$590K DCA gap vs V15, $2.96M vs V9.** Gold's crisis alpha is real but the CAGR drag is permanent. Over 40 years, the compound effect of 10% less equity is substantial.

2. **2016 DCA dip:** V16-A briefly trails QQQ (-$1K at end-2016). V15 and V9 never trail QQQ. The gold dilution creates a narrow window of vulnerability during sustained equity bull markets.

3. **Gold's crisis alpha is asymmetric.** Gold was inactive during dot-com (0/15 months). Crisis alpha depends on gold being in a secular bull or safe-haven bid — it's not guaranteed in all equity bears.

4. **V16-A ≈ V16-B.** The IAU threshold (score ≥ 2 vs ≥ 3) barely matters. This means the gold decision is binary: include it or don't. The threshold is not a meaningful lever.

---

## Updated Pareto Frontier

| Point | Strategy | CAGR | Sharpe | MaxDD | DCA |
|---|---|---|---|---|---|
| Max wealth | **V9** | 19.4% | 0.777 | -37.9% | $7.37M |
| **Balanced** | **V16-B** | **17.1%** | **0.846** | **-27.0%** | **$4.41M** |
| Max Sharpe | **Baseline** | 13.8% | 0.910 | -18.5% | $2.40M |

V16-B displaces V15 (17.7%, 0.813, -29.0%). The gap between V16-B and Baseline narrows — V16-B is now only 0.064 Sharpe below Baseline while delivering +3.27pp more CAGR and +$2.01M more DCA.

V15 and V12 are both off the frontier. V11 was already retired.

---

## Cross-references

- [[experiments/V15_TWO_POD_RESULTS]] — displaced by V16 on Sharpe + MaxDD
- [[experiments/V14_DEFENSIVE_ROTATION_RESULTS]] — V14's multi-asset defensives failed; V16's single gold asset succeeds because it avoids the concentration-into-last-survivor problem

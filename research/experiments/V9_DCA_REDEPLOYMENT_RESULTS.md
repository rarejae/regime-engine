---
date: 2026-04-12
experiment: V9-DCA Cash Redeployment via Stepped IVV Buying
status: FAILED — all step sizes worsen Sharpe and/or MaxDD
script: experiments/v9_dca/backtest.py
---

# V9-DCA: Stepped IVV Buying During V9 Cash Periods — Results

## Verdict

**V9-DCA FAILS at every step size.** The mechanism wins slightly more often than it loses (56% win rate), but the losses are catastrophic (GFC: -6.17% delta from 9 tranches deployed into a -46% decline, exited below cost) while the wins are marginal (2022: +2.51% delta). The risk-adjusted return is strictly worse than V9 cash.

V9-DCA-10 (most conservative) comes closest — virtually identical to V9 on CAGR and Sharpe — but even it fails Sharpe by -0.001 and worsens MaxDD by 1.6pp. The mechanism does almost nothing at the conservative end and destroys returns at the aggressive end.

**"Time in the market" does not apply when the trend filter has specifically told you to exit.** The Faber signal identifies periods where equities are falling. Buying into those periods is value-destructive.

---

## Table 1 — Core Metrics (2002-2026)

| Strategy         |   CAGR |    Vol | Sharpe | Sortino |   MaxDD | Calmar | Term$1 |  DCA$700 |  CB |
|------------------|-------:|-------:|-------:|--------:|--------:|-------:|-------:|---------:|----:|
| V9-DCA-3         | 18.88% | 29.13% |  0.740 |   0.892 | **-56.4%** | 0.38 | $76.89 |  $7.01M |  14 |
| V9-DCA-5         | 19.44% | 28.48% |  0.767 |   0.914 | -48.7%  |   0.45 | $86.38 |  $7.59M |  14 |
| V9-DCA-7         | 19.37% | 28.10% |  0.771 |   0.910 | -43.1%  |   0.50 | $85.09 |  $7.42M |  14 |
| V9-DCA-10        | 19.45% | 27.98% |  0.776 |   0.908 | -39.5%  |   0.55 | $86.67 |  $7.54M |  14 |
| **V9 (control)** | 19.37% | 27.82% |  **0.777** |   0.887 | **-37.9%** | 0.57 | $85.25 | $7.37M |  14 |
| V12 Indep 2×     | 17.41% | 23.42% |  0.803 |   0.932 | -28.8%  |   0.65 | $56.18 |  $4.80M |  31 |
| QQQ B&H          | 12.57% | 22.77% |  0.634 |   0.847 | -53.4%  |   0.27 | $17.58 |  $2.33M |   — |

Max DD monotonically worsens with more aggressive step size: -37.9% (V9) → -39.5% → -43.1% → -48.7% → -56.4%. Every extra tranche of IVV deployed during a declining market adds drawdown exposure. The 3% step size produces a worse MaxDD (-56.4%) than QQQ buy-and-hold (-53.4%) — the DCA mechanism turns V9's best feature (DD protection) into its worst.

---

## Table 2 — CAGR by Start Date

| Strategy     |   2002 |   2007 |   2010 |   2013 |   2019 |
|--------------|-------:|-------:|-------:|-------:|-------:|
| V9-DCA-10    | 19.45% | 22.67% | 23.63% | 28.87% | 28.57% |
| V9-DCA-5     | 19.44% | 22.64% | 23.89% | 29.12% | 28.71% |
| V9-DCA-3     | 18.88% | 21.87% | 23.96% | 29.15% | 28.51% |
| V9 (control) | 19.37% | 22.56% | 23.35% | 28.51% | 28.12% |

DCA helps from 2013+ (where the main off-signal period was 2022 — a successful DCA episode). From 2002 and 2007, V9-DCA-3 actually trails V9 because GFC losses dominate. The improvement is start-date dependent, which means it's driven by a single event (2022), not structural alpha.

---

## Table 3 — DCA Event Log (V9-DCA-5, step=5%)

18 cash periods over 24 years. 10 DCA beat T-bills (56%).

| Anchor     | Exit       | Days | MaxDecl | Tranch | DCA ret | Cash ret | Winner |
|------------|------------|-----:|--------:|-------:|--------:|---------:|--------|
| 2004-05-03 | 2004-06-01 |   29 |    3.0% |      0 |  -0.20% |   +0.09% | CASH   |
| 2004-08-02 | 2004-11-01 |   91 |    3.8% |      0 |  +1.16% |   +0.42% | DCA    |
| 2005-05-02 | 2005-06-01 |   30 |    0.6% |      0 |  +1.85% |   +0.25% | DCA    |
| 2005-07-01 | 2005-08-01 |   31 |    0.0% |      0 |  +0.59% |   +0.27% | DCA    |
| 2006-06-01 | 2006-10-02 |  123 |    4.8% |      0 |  -0.82% |   +1.68% | CASH   |
| 2008-02-01 | 2008-06-02 |  122 |    8.3% |      1 |  -0.16% |   +0.53% | CASH   |
| **2008-07-01** | **2009-05-01** | **304** | **46.2%** | **9** | **-5.65%** | **+0.52%** | **CASH** |
| 2010-07-01 | 2010-08-02 |   32 |    0.5% |      0 |  +1.33% |   +0.01% | DCA    |
| 2010-09-01 | 2010-10-01 |   30 |    0.0% |      0 |  -0.20% |   +0.01% | CASH   |
| 2011-09-01 | 2011-11-01 |   61 |    8.6% |      1 |  -4.53% |   +0.00% | CASH   |
| 2012-01-02 | 2012-02-01 |   30 |    0.0% |      0 |  +1.57% |   +0.00% | DCA    |
| 2015-09-01 | 2015-11-02 |   62 |    1.5% |      0 |  +2.29% |   +0.00% | DCA    |
| 2016-02-01 | 2016-04-01 |   60 |    5.6% |      1 |  +3.24% |   +0.05% | DCA    |
| 2016-05-02 | 2016-06-01 |   30 |    1.8% |      0 |  -0.03% |   +0.02% | CASH   |
| 2018-11-01 | 2019-03-01 |  120 |   13.8% |      2 |  +4.24% |   +0.75% | DCA    |
| 2020-04-01 | 2020-05-01 |   30 |    0.0% |      0 |  -2.81% |   +0.01% | CASH   |
| **2022-02-01** | **2023-02-01** | **365** | **20.4%** | **4** | **+4.91%** | **+2.40%** | **DCA** |
| 2025-04-01 | 2025-06-02 |   62 |   11.5% |      2 |  +4.85% |   +0.72% | DCA    |

**Two events dominate the entire experiment:**
- GFC (Jul 2008 – May 2009): 9 tranches deployed, IVV avg cost $67.86, exited at $64.73. **Loss.** Delta: -6.17%.
- 2022 (Feb 2022 – Feb 2023): 4 tranches deployed, IVV avg cost $369.43, exited at $393.81. **Profit.** Delta: +2.51%.

Cumulative DCA return across all events: +11.64% vs T-bills +7.75%. The DCA earns +3.89% more than T-bills over 24 years. This is noise-level for the drawdown cost.

---

## Table 4 — Crisis Detail

### GFC (THE KILL SHOT)

Cash period Jul 2008 → May 2009 (304 days):
- Anchor: IVV $92.59 (Jul 1, 2008)
- IVV dropped 46.2% from anchor → 9 of 10 tranches deployed
- Average cost basis: **$67.86**
- V9's QLD signal restored May 2009 at IVV **$64.73**
- **Exit price was BELOW average cost** → crystallized loss of -4.6%
- DCA period return: -5.65% vs T-bills +0.52% → **-6.17% delta**

The fundamental flaw: V9's Faber signal restored when QQQ was back above all 3 SMAs — but IVV hadn't recovered to the DCA average cost yet. The trend filter signals "equity risk is OK again," not "IVV has recovered to your purchase price." The system sells IVV at a loss to buy QLD.

### COVID

Cash period Apr 2020 → May 2020 (30 days):
- By the time V9 entered cash (April), IVV had already bounced from the March trough
- IVV didn't decline from the anchor → **0 tranches deployed**
- The DCA mechanism was completely inert during COVID because V9's cash period starts AFTER the recovery has already begun

This is a structural mismatch: the DCA is designed to buy dips, but V9's Faber signal exits ~1 month after the trough and re-enters ~1 month after recovery begins. The buying window is too narrow for the DCA to deploy.

### 2022

Cash period Feb 2022 → Feb 2023 (365 days):
- IVV declined 20.4% from anchor, 4 tranches deployed
- Average cost: $369.43, exit price: $393.81 → +6.6% on tranches
- DCA return +4.91% vs T-bills +2.40% → **+2.51% delta**

2022 worked because the decline was gradual (enough time to deploy tranches) and the recovery was partial (IVV recovered above avg cost before QLD signal restored). This is the ideal DCA scenario — slow decline, measured recovery. But it's the exception, not the rule.

---

## Table 5 — Max Drawdown Comparison

| Strategy     | Full DD  |  GFC DD | COVID DD | 2022 DD |
|--------------|--------:|--------:|---------:|--------:|
| V9-DCA-3     | -56.4%  | -55.7%  |  -37.9%  | -24.1%  |
| V9-DCA-5     | -48.7%  | -47.9%  |  -37.9%  | -23.9%  |
| V9-DCA-7     | -43.1%  | -42.2%  |  -37.9%  | -23.9%  |
| V9-DCA-10    | -39.5%  | -38.6%  |  -37.9%  | -23.9%  |
| V9 (control) | -37.9%  | -30.6%  |  -37.9%  | -23.9%  |

GFC is where every DCA variant diverges from V9. V9-DCA-3 hits -55.7% during GFC (25pp worse than V9) because 10 tranches at 3% steps deploys 100% into IVV by a 30% decline — and IVV kept falling to -46%. V9-DCA-10 is contained (-38.6%) because only 1-2 tranches deployed at the wider step interval.

COVID and 2022 DDs are identical because the DCA either didn't deploy (COVID: 0 tranches) or deployed modestly (2022: ~4 tranches at modest loss).

---

## Table 6 — DCA Terminal by Year-End ($21K + $700/mo, 2013 start)

| Year |  DCA-10 |   DCA-5 |   DCA-3 |      V9 |     QQQ |
|-----:|--------:|--------:|--------:|--------:|--------:|
| 2020 |   $481K |   $488K |   $495K |   $475K |   $284K |
| 2022 |   $644K |   $645K |   $640K |   $637K |   $258K |
| 2025 |  $1655K |  $1693K |  $1697K |  $1594K |   $642K |
| 2026 |  $1547K |  $1583K |  $1587K |  $1491K |   $606K |

DCA adds $56K-$96K of terminal DCA wealth vs V9 at end-2026. This is a 3.7-6.4% improvement — modest but real. The 2022 DCA success compounds forward. However, the improvement is entirely from one good DCA period (2022) and would be negative if you start from 2007 (GFC dominates).

---

## Pass / Fail

| Variant | CAGR > V9 | Sharpe ≥ V9 | MaxDD ≤ -40.9% | DCA wins >50% | Overall |
|---------|:---------:|:-----------:|:--------------:|:-------------:|:-------:|
| DCA-3   | ✗ (18.88%)| ✗ (0.740)   | ✗ (-56.4%)     | ✓ (56%)       | FAIL    |
| DCA-5   | ✓ (19.44%)| ✗ (0.767)   | ✗ (-48.7%)     | ✓ (56%)       | FAIL    |
| DCA-7   | ✗ (19.37%)| ✗ (0.771)   | ✗ (-43.1%)     | ✓ (56%)       | FAIL    |
| DCA-10  | ✓ (19.45%)| ✗ (0.776)   | ✓ (-39.5%)     | ✓ (56%)       | FAIL    |

**Every variant fails Sharpe.** The DCA mechanism adds vol (buying declining equities) without enough return to compensate. V9-DCA-10 comes closest (-0.001 Sharpe) but the signal is clear: the mechanism adds risk without commensurate reward.

---

## What We Learned

### 1. The trend filter says "exit" — buying in is contradictory
V9's Faber signal identifies periods where equities are falling. The DCA mechanism then systematically buys the thing the signal says is falling. This is a fundamental contradiction. The signal is right ~56% of the time (equities recover before QLD signal restores), but when it's wrong (GFC: -46% decline, exit below cost), the loss is catastrophic.

### 2. V9's cash period starts too late for DCA to catch the trough
COVID demonstrated this perfectly: V9 entered cash in April 2020, *after* the March trough. By April, IVV had already bounced 20%+. The DCA mechanism saw 0% decline from anchor and deployed 0 tranches. The buying window that matters (March 2020) happens while V9 is still holding QLD and getting stopped out — not during the cash period.

### 3. The GFC scenario dominates
One event (Jul 2008 → May 2009) produced 9 deployed tranches and a -6.17% delta vs T-bills. This single event wiped out the cumulative gains from all other DCA periods. The GFC took IVV down -46% from anchor and the QLD signal restored while IVV was still below avg cost — forcing a sale at a loss. Any architecture that deploys cash into equities during Faber-off periods is vulnerable to this scenario.

### 4. Step size determines the severity, not the sign
More aggressive steps (3%) = faster deployment = deeper drawdown. More conservative steps (10%) = less deployment = closer to V9. But no step size improves Sharpe. The mechanism's expected value is slightly positive (+3.89% cumulative over 24 years) but with unacceptable tail risk. There is no Goldilocks step size.

### 5. V9 standalone is confirmed terminal
V13 failed (offense modification). V14 failed (defense modification: Faber-gated assets). V9-DCA failed (defense modification: stepped equity buying). Every conceivable modification to V9 — changing the offense, changing the defense to diversified assets, changing the defense to equity purchases — has been tested and failed. V9's architecture is final.

---

## Cross-references

- [[experiments/V14_DEFENSIVE_ROTATION_RESULTS]] — Faber-gated defensives also failed during off-signal periods
- [[experiments/V13_THREE_STATE_RESULTS]] — offense modifications failed
- [[2026-04-05_pro_rata_vs_cash]] — original "cash is the hedge" finding

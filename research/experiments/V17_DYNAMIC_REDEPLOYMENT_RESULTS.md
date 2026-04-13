---
date: 2026-04-12
experiment: V17 Two-Pod + Gold with Dynamic Redeployment
status: PASSED on criteria but does NOT displace V16-B from frontier
script: experiments/v17_dynamic_redeploy/backtest.py
---

# V17: Dynamic Redeployment — Results

## Verdict

**V17 passes its stated pass criteria** (CAGR ≥ V16, Sharpe within 0.02, MaxDD within 3pp). But V17 does **not displace V16-B from the Pareto frontier** because V16-B has higher Sharpe (0.846 vs 0.842) and shallower MaxDD (-27.0% vs -27.7%).

V17 trades V16-B's risk-adjusted superiority for more raw return (+0.48pp CAGR, +$5.58 terminal, +$360K DCA). This is the same CAGR-vs-Sharpe tradeoff that defines V9-vs-V16, just at a smaller scale.

The redeployment mechanism works (+0.14%/mo excess during redeployment months), but the extra equity exposure adds proportionally more vol than return, slightly diluting V16-B's Sharpe. **Gold's idle cash is not quite load-bearing, but its absence slightly worsens the portfolio's risk profile.**

---

## Table 1 — Core Metrics (2002-2026)

| Strategy              |   CAGR |    Vol | Sharpe | Sortino |  MaxDD | Calmar | Term$1 |  DCA$700 |  CB |
|-----------------------|-------:|-------:|-------:|--------:|-------:|-------:|-------:|---------:|----:|
| V17 Dyn Redeploy      | 17.54% | 22.12% |  0.842 |   0.999 | -27.7% |   0.67 | $57.76 |  $4.77M |  27 |
| **V16-B (45/45/10)**  | 17.06% | 21.33% | **0.846** | 0.997 | **-27.0%** | 0.67 | $52.18 | $4.41M | 27 |
| V15 Two-Pod           | 17.65% | 23.36% |  0.813 |   0.941 | -29.0% |   0.66 | $59.18 |  $4.92M |  27 |
| V9 QLD+IVVguard       | 19.37% | 27.82% |  0.777 |   0.887 | -37.9% |   0.57 | $85.25 |  $7.37M |  14 |
| Baseline (Sweep-40)   | 13.79% | 15.52% |  0.910 |   1.090 | -18.5% |   0.76 | $25.58 |  $2.40M |  16 |

V17 sits between V16-B and V15 on almost every metric. It's a valid middle ground but not a new frontier point.

---

## Table 2 — Redeployment Diagnostics

| Metric | Value |
|--------|-------|
| Redeployment active months | 110/291 (38%) |
| Gold on (no redeployment) | 152 |
| All cash (no redeployment) | 29 |
| V17 mean return during redeployment | +1.84% |
| V16 mean return during redeployment | +1.70% |
| **Delta** | **+0.14%/mo** |
| V17 positive months | 73/110 (66%) |

The redeployment mechanism earns +0.14%/mo excess during the 110 months it's active. 66% of those months have positive V17 returns. The extra equity exposure is net positive — confirming that gold's off-trend signal does NOT predict equity weakness. The equity pods' own Faber gates are the correct risk signal; gold breaking trend is uncorrelated noise from the equity perspective.

---

## Table 3 — State Occupancy (291 months)

| State | Weights | Months | Pct |
|-------|---------|-------:|----:|
| Both eq + gold | 45/45/10 | 117 | 40.2% |
| Both eq, gold→eq | 50/50/0 | 96 | 33.0% |
| Pod1 + gold | 45/0/10 | 8 | 2.7% |
| Pod1 only, gold→Pod1 | 55/0/0 | 3 | 1.0% |
| Pod2 + gold | 0/45/10 | 6 | 2.1% |
| Pod2 only, gold→Pod2 | 0/55/0 | 11 | 3.8% |
| Gold only | 0/0/10 | 21 | 7.2% |
| All cash | 0/0/0 | 29 | 10.0% |

The dominant redeployment state is "Both eq, gold→eq" (33% of months) — gold off-signal, both pods running, portfolio at 50/50 with 200% effective equity. This is the state where V17 diverges most from V16-B (which holds 45/45 with 10% cash in those same months).

---

## Table 4 — Crisis Drawdowns

| Crisis          |    V17 |  V16-B |    V15 |     V9 | Baseline |
|-----------------|-------:|-------:|-------:|-------:|---------:|
| Dot-com 2002-03 |  -3.0% |  -2.3% |  -2.7% |  -5.0% |    -2.1% |
| GFC 2007-09     | -17.8% | -16.2% | -18.1% | -30.6% |    -9.0% |
| COVID 2020      | -27.7% | -27.0% | -29.0% | -37.9% |   -18.5% |
| 2022 bear       | -20.3% | -20.0% | -21.5% | -23.9% |   -13.2% |

V17 is ~0.3-1.6pp worse than V16-B in every crisis. The extra equity exposure during gold-off months means V17 enters crises with slightly higher effective equity, which costs during the initial drawdown. The difference is small but consistently in V16-B's favor.

---

## Table 5 — DCA Terminal ($21K + $700/mo, 2013 start)

| Year |     V17 |   V16-B |     V15 |      V9 | V17-QQQ |
|-----:|--------:|--------:|--------:|--------:|--------:|
| 2020 |   $308K |   $301K |   $328K |   $475K |   $+24K |
| 2022 |   $406K |   $388K |   $431K |   $637K |  $+148K |
| 2025 |   $915K |   $886K |   $969K |  $1594K |  $+273K |
| 2026 |   $855K |   $828K |   $887K |  $1491K |  $+250K |

V17 beats V16-B by ~$27K at 2026. V17 never trails QQQ (smallest gap +$1K at 2016). The DCA improvement is real but modest — $360K of lifetime DCA improvement ($4.77M vs $4.41M) in exchange for -0.004 Sharpe and -0.7pp MaxDD.

---

## Pass / Fail

| Criterion | V17 | V16-B | Result |
|-----------|-----|-------|--------|
| CAGR ≥ V16 | 17.54% | 17.06% | ✓ |
| Sharpe ≥ V16-0.02 | 0.842 | 0.826 floor | ✓ |
| MaxDD ≤ V16+3pp | -27.7% | -30.0% floor | ✓ |

**PASS on stated criteria.**

But V17 does not dominate V16-B:
- Sharpe: V17 0.842 < V16-B 0.846
- MaxDD: V17 -27.7% < V16-B -27.0%

V17 improves CAGR and terminal wealth at the cost of slightly worse risk metrics. The Pareto frontier is unchanged — V16-B remains the balanced-point leader.

---

## The Honest Assessment

V17 proves that redeploying gold's idle cash to confirmed-trend equities is a valid mechanism (positive expected value, +0.14%/mo excess). Unlike V14/V9-DCA (which redeployed into declining markets), V17 adds capital to engines already validated by their own Faber signals. The difference in direction explains why V17 works (+0.14%/mo) while V14 (-0.30%/mo in crisis months) and V9-DCA (-6.17% in GFC) failed.

But "valid mechanism" ≠ "improves the frontier." V16-B's 10% cash buffer when gold is off provides a small but consistent vol reduction that V17 sacrifices for CAGR. The $360K DCA improvement over a lifetime is meaningful, but the -0.004 Sharpe and -0.7pp MaxDD cost are real.

**Bottom line:** V17 is a valid intermediate point between V16-B (Sharpe-optimized) and V15 (CAGR-optimized among diversified variants). For a user who values terminal wealth over Sharpe, V17 is the correct pick over V16-B. For a user who values risk-adjusted returns, V16-B is unchanged as the frontier leader.

---

## Pareto Frontier (UNCHANGED)

| Point | Strategy | CAGR | Sharpe | MaxDD | DCA |
|---|---|---|---|---|---|
| Max wealth | **V9** | 19.4% | 0.777 | -37.9% | $7.37M |
| Balanced | **V16-B** | 17.1% | 0.846 | -27.0% | $4.41M |
| Max Sharpe | **Baseline** | 13.8% | 0.910 | -18.5% | $2.40M |

V17 (17.5%, 0.842, -27.7%) is dominated by V16-B on Sharpe and MaxDD. It's a valid choice for CAGR-prioritizing investors but does not change the frontier.

---

## Cross-references

- [[experiments/V16_TWO_POD_GOLD_RESULTS]] — V16-B remains frontier leader
- [[experiments/V14_DEFENSIVE_ROTATION_RESULTS]] — V14 redeployed INTO declining markets (failed); V17 redeploys INTO confirmed trends (works, but marginally)
- [[experiments/V9_DCA_REDEPLOYMENT_RESULTS]] — V9-DCA redeployed cash INTO equity declines (failed catastrophically in GFC)

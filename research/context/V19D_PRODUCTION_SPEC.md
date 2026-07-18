# V19d Production Specification

**Status:** LOCKED — April 13, 2026
**Research arc:** [[experiments/V9_TO_V19D_RESEARCH_ARC]]
**Final backtest:** [[experiments/V19D_FINAL_BACKTEST]]

---

## Architecture

```
V19d: Two-Pod + Gold, CB → Cash
```

| Component | Weight | Asset | Leverage |
|-----------|--------|-------|----------|
| Pod 1 | 45% | QQQ / QLD | 2× when eligible |
| Pod 2 | 45% | IVV / SSO | 2× when eligible |
| Gold | 10% | IAU | 1× only (never leveraged) |

---

## Signal

126/200/252-day daily SMAs on each asset independently.

**Faber score** = count of SMAs where current price > SMA value (0, 1, 2, or 3).

Scores computed through month-end T. Applied to month T+1 allocations.

---

## Pod 1: QQQ/QLD (45%)

| QQQ Score | IVV Score | Holding | Effective Equity |
|-----------|-----------|---------|:----------------:|
| 3 | ≥ 2 | 100% QLD | 90% (45% × 2) |
| 3 | ≤ 1 | 100% QQQ | 45% |
| 2 | any | 70% QQQ + 30% cash | 31.5% |
| 0–1 | any | 100% cash | 0% |

IVV acts as a **guard signal** for Pod 1 leverage only. IVV is never held in Pod 1.
Guard threshold: IVV ≤ 1 strips leverage (QLD → QQQ). IVV ≥ 2 allows leverage.

---

## Pod 2: IVV/SSO (45%)

| IVV Score | Holding | Effective Equity |
|-----------|---------|:----------------:|
| 3 | 100% SSO | 90% (45% × 2) |
| 2 | 70% IVV + 30% cash | 31.5% |
| 0–1 | 100% cash | 0% |

No guard signal. IVV stands alone.

---

## Gold Sleeve (10%)

| IAU Score | Holding |
|-----------|---------|
| ≥ 3 | 100% IAU |
| < 3 | 100% cash |

Binary: IAU on-trend or cash. No partial positions. No leverage.

---

## Circuit Breaker

**Trigger:** Asset closes below ALL three daily SMAs (126, 200, 252).

**Action:** Exit to CASH at next market open. Not unlevered equity — full cash.

**Applies to:** QQQ (Pod 1), IVV (Pod 2), IAU (Gold) — all three independently.

**Re-entry:** Next monthly rebalance only. No mid-month re-entry.

| Asset | CB fires | Action | Re-entry |
|-------|----------|--------|----------|
| QQQ | QQQ < all 3 SMAs | QLD/QQQ → cash | Monthly |
| IVV | IVV < all 3 SMAs | SSO/IVV → cash | Monthly |
| IAU | IAU < all 3 SMAs | IAU → cash | Monthly |

CB → cash is load-bearing. Post-CB equity returns are net negative (-5.49% cumulative across 27 equity CB events). See [[experiments/V19_CB_CASH_EXIT_RESULTS]].

---

## Rebalancing

**Frequency:** Monthly, at month-end alongside Faber score evaluation.

**Targets:** 45% Pod 1 / 45% Pod 2 / 10% Gold.

**Trigger:** Any component drifts beyond ±5% of target weight.

**Mechanism:** Sell overweight, buy underweight to restore targets. Contrarian rebalancing adds ~+0.008 Sharpe. See [[experiments/V15_TWO_POD_RESULTS]].

---

## Leveraged ETF Simulation

```python
R_leveraged(t) = 2.0 × R_underlying(t) − R_rf(t)/252 − expense/252
```

| ETF | Underlying | Expense Ratio | Inception |
|-----|-----------|:-------------:|-----------|
| QLD | QQQ (Nasdaq-100) | 0.95% | June 2006 |
| SSO | IVV (S&P 500) | 0.89% | June 2006 |

Use actual QLD/SSO prices post-inception (validated: daily correlation 0.996 with simulation). Use simulation formula pre-inception (2002–2006).

---

## Effective Equity Exposure by State

| Pod 1 | Pod 2 | Gold | Eff Equity | Gold | Cash |
|:-----:|:-----:|:----:|:----------:|:----:|:----:|
| QLD | SSO | IAU | 180% | 10% | 0% |
| QLD | SSO | cash | 180% | 0% | 10% |
| QLD | IVV 70% | IAU | 122% | 10% | 14% |
| QLD | cash | IAU | 90% | 10% | 45% |
| QQQ 70% | SSO | IAU | 122% | 10% | 14% |
| cash | SSO | IAU | 90% | 10% | 45% |
| QQQ 70% | IVV 70% | IAU | 63% | 10% | 27% |
| cash | cash | IAU | 0% | 10% | 90% |
| cash | cash | cash | 0% | 0% | 100% |

**Median effective equity:** 180% (65.6% of months)
**Mean effective equity:** 131.2%

---

## Validated Performance (2002-01 → 2026-03)

| Metric | Value |
|--------|-------|
| Annualized return | 17.27% |
| Annualized volatility | 20.94% |
| Sharpe ratio | 0.866 |
| Sortino ratio | 1.010 |
| Maximum drawdown | -25.1% |
| Calmar ratio | 0.72 |
| Terminal $1 | $54.60 |
| DCA $21K + $700/mo | $4,640,000 |
| Leveraged months | 216/291 (74%) |

### Circuit Breaker Activity

| Component | Events | Frequency |
|-----------|:------:|:---------:|
| Pod 1 (QQQ) | 14 | 0.6/year |
| Pod 2 (IVV) | 13 | 0.5/year |
| Gold (IAU) | 10 | 0.4/year |
| **Total** | **37** | **1.5/year** |
| Rebalances | 13 | 0.5/year |

### Crisis Drawdowns

| Crisis | V19d | V9 | Baseline | QQQ B&H |
|--------|-----:|---:|---------:|--------:|
| Dot-com 2002–03 | -2.3% | -5.0% | -2.1% | -51.9% |
| GFC 2007–09 | -16.4% | -30.6% | -9.0% | -53.1% |
| COVID 2020 | -25.1% | -37.9% | -18.5% | -28.6% |
| 2022 bear | -17.6% | -23.9% | -13.2% | -34.8% |

### Pareto Frontier Context

| Point | System | CAGR | Sharpe | MaxDD | DCA |
|-------|--------|-----:|-------:|------:|----:|
| Max wealth | V9 | 19.4% | 0.777 | -37.9% | $7.37M |
| **Balanced** | **V19d** | **17.3%** | **0.866** | **-25.1%** | **$4.64M** |
| Max protection | Baseline | 13.8% | 0.910 | -18.5% | $2.40M |

---

## Execution

### Human-in-the-Loop via Telegram

| Event | Frequency | Action Required |
|-------|:---------:|-----------------|
| Monthly rebalance | 12/year | Review scores, approve trades |
| Equity CB alert | ~1.1/year | Approve QLD→cash or SSO→cash next open |
| Gold CB alert | ~0.4/year | Approve IAU→cash next open |
| Pod rebalance | ~0.5/year | Approve weight adjustment trades |

Total alerts: ~14/year (monthly) + ~1.5/year (CB) + ~0.5/year (rebalance) ≈ **16/year**.

### Monthly Rebalance Checklist

1. Compute QQQ, IVV, IAU Faber scores from 126/200/252-day daily SMAs
2. Determine Pod 1 state (QLD / QQQ / QQQ 70% / cash) with IVV guard check
3. Determine Pod 2 state (SSO / IVV 70% / cash)
4. Determine Gold state (IAU / cash)
5. Check pod weight drift — rebalance to 45/45/10 if drift > 5%
6. Execute trades at market open on first trading day of new month

### Daily CB Check (automated)

1. At market close, check: is QQQ below ALL of SMA-126, SMA-200, SMA-252?
2. Same check for IVV, same check for IAU
3. If any asset breaches all 3: send Telegram alert
4. Execute exit to cash at next market open
5. No re-entry until next monthly rebalance

---

## Key Design Decisions (with evidence)

| Decision | Alternative Tested | Result | Reference |
|----------|-------------------|--------|-----------|
| CB → cash (not equity) | CB → unlevered equity | Cash wins 14/27 events, cumul equity -5.49% | [[V19_CB_CASH_EXIT_RESULTS]] |
| 10% gold | No gold (50/50) | Gold adds +0.033 Sharpe, +1.9pp MaxDD | [[V19B_NO_GOLD_RESULTS]] |
| 45/45/10 split | 60/30/10 QQQ tilt | Tilt costs -3.6pp MaxDD for +0.57pp CAGR | [[V19D_QQQ_TILT_RESULTS]] |
| Score 2 → 70/30 | Score 2 → 100% unlev | Wash, 70/30 marginally better crisis DDs | [[V19C_FULL_UNLEVER_RESULTS]] |
| Gold CB | No gold CB | Wash, adopted for consistency | [[V19D_GOLD_CB_RESULTS]] |
| Non-directional score 2 | Directional (3→2 / 1→2) | Hypothesis inverted, directional is worse | [[V20_DIRECTIONAL_TRANSITIONS_RESULTS]] |
| Monthly re-entry only | Weekly re-entry | 1/18 events resolved weekly — noise | [[V13_THREE_STATE_RESULTS]] |
| IVV guard at ≤ 1 | IVV guard at ≤ 2 | Tighter guard costs -1.12%/mo in 12 months | [[V13_THREE_STATE_RESULTS]] |
| Cash during off-signal | Defensive assets | Defensives concentrate into last survivor | [[V14_DEFENSIVE_ROTATION_RESULTS]] |
| Cash during off-signal | DCA into IVV | GFC: 9 tranches, exit below cost | [[V9_DCA_REDEPLOYMENT_RESULTS]] |
| Per-asset SMA CB | Portfolio drawdown CB | Portfolio CB fires 209× in 24yr (whipsaw) | [[V18_DRAWDOWN_PROTECTION_RESULTS]] |
| Faber filter only | Macro/vol indicators | 0/15 indicators pass validation | [[V18_DRAWDOWN_PROTECTION_RESULTS]] |

---

## Structural Limitations (accepted)

1. **-25.1% MaxDD is the structural floor.** Comes from unlevered equity in the 23-day CB-to-rebalance window. Cannot be improved without reducing leverage or adding weekly equity exits (rejected). See [[V18B_INTRAMONTH_CB_RESULTS]].

2. **First month of a crash is unprotected.** The Faber filter and CB both lag price. At 180% effective equity entering a peak (Jan 2022: -12.8%), the first month of damage is unavoidable.

3. **QQQ/IVV correlation is ~0.90.** Pod diversification helps at the margins (GFC: -16.4% vs V9's -30.6%) but both pods tend to exit simultaneously during systemic crises.

4. **Gold is inactive during secular gold bear markets.** Dot-com 2002–03: IAU score 0 for 15 months. Gold adds nothing when it's not trending.

5. **~1.5 CB alerts/year require human action.** Each CB is a same-day decision to exit at next open. Missed or delayed execution degrades performance.

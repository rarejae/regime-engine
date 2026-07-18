---
date: 2026-04-12
experiment: V18 Drawdown Protection — Portfolio CB + Leading Indicator Leverage Scaling
status: FAILED — all mechanisms. -27% MaxDD is the structural floor for V16-B.
script: experiments/v18_drawdown_protection/backtest.py
---

# V18: Drawdown Protection — Results

## Verdict

**V18 FAILS comprehensively.** Both independent mechanisms and all leading indicators produce worse risk-adjusted results than V16-B.

- **Part 1 (Leading Indicators):** Zero indicators pass the validation criteria. Best candidate (VIX slope ≥10) has 27% hit rate and 73% false positive rate. Yield curve inversion has 100% FP rate — when inverted, QQQ drawdown probability is actually *lower* than baseline.
- **Part 2A (Portfolio CB):** Catastrophic whipsaw. PCB-8 fires 330+ times and drops CAGR from 17.06% to 10.27%. Even PCB-15 fires 132 times, CAGR drops to 12.64%. At 180% effective equity, normal monthly volatility routinely produces 8-15% drawdowns from HWM.
- **Part 2B (Leading Indicators as overlay):** Skipped — no indicators passed Part 1.

**-27% MaxDD is the structural floor for a 180% effective equity portfolio with monthly rebalancing and lagging trend signals.** There is no free drawdown protection. Accept it or reduce leverage.

---

## Part 1 — Leading Indicator Validation (288 months, 2002-2025)

### Indicator Rankings (sorted by hit rate minus false positive rate)

| Indicator       | Thresh |  Active | HitRate |  FP Rate | Crises | Lift |
|-----------------|--------|--------:|--------:|---------:|:------:|-----:|
| VIX_slope       |    ≥10 |      4% |     27% |      73% |   2/4  | +20% |
| VIX_slope       |    ≥15 |      1% |     25% |      75% |   1/4  | +17% |
| sp500_drawdown  |   ≤-10 |     19% |     20% |      80% |   1/4  | +15% |
| VIX_slope       |     ≥5 |      9% |     19% |      81% |   3/4  | +12% |
| sp500_drawdown  |    ≤-5 |     33% |     17% |      83% |   3/4  | +13% |
| VIX             |    ≥20 |     33% |     17% |      83% |   3/4  | +13% |
| credit_spread   |   ≥3.0 |     20% |     12% |      88% |   0/4  |  +5% |
| VIX             |    ≥30 |      9% |     11% |      89% |   1/4  |  +3% |
| yield_curve     |   ≤0.0 |     18% |      0% |     100% |   2/4  |  -10%|

**Indicators passing Part 1 (hit≥30%, FP≤80%, crises≥2): ZERO.**

### Key Findings

**VIX slope is the best candidate but still fails.** VIX_slope ≥ 10 fires only 11 months in 24 years (4% of the time). It has the highest hit rate (27%) and lowest FP rate (73%), but 27% << 30% threshold. Even when VIX spikes by 10+ points in 21 days, there's only a 1-in-4 chance QQQ drops ≥10% in the next 6 months.

**Yield curve inversion is a CONTRARIAN signal.** When inverted (≤ 0.0), P(QQQ -10% in 6mo) drops to 0% vs 10% baseline. Inversion led GFC by 183 days and COVID by 184 days — but the *timing* is so early that by the time equities actually crash, the curve has often steepened again. The inversion predicts *recessions*, not *the timing of equity drawdowns*.

**Credit spreads are noise.** At every threshold, the hit rate approximately equals the no-signal rate. Credit spreads contain zero marginal information about QQQ drawdowns beyond what the Faber filter already captures.

**S&P 500 drawdown from 252-day high is a lagging indicator.** At ≤-5%, it has identical profile to VIX ≥ 20 (both describe "the market is already falling"). Not predictive — reactive.

### Implication for This Project

This confirms the Kritzman/Harvey finding from April 4: **macro indicators do not reliably predict equity drawdowns with actionable lead time.** The Faber filter is the best available risk signal because it responds to *price* — the only leading indicator that is both timely and causal.

---

## Part 2A — Portfolio-Level Drawdown Circuit Breaker

### Core Metrics

| Strategy      |   CAGR |    Vol | Sharpe | Sortino |  MaxDD | Calmar | Term$1 |  DCA$700 | PCB events |
|---------------|-------:|-------:|-------:|--------:|-------:|-------:|-------:|---------:|-----------:|
| PCB-8 (-8%)   | 10.27% | 16.42% |  0.679 |   0.811 | -23.7% |   0.47 | $10.47 |  $1.19M  |     330+   |
| PCB-10 (-10%) | 11.42% | 17.55% |  0.709 |   0.838 | -24.2% |   0.51 | $13.43 |  $1.46M  |     209    |
| PCB-12 (-12%) | 12.57% | 18.65% |  0.744 |   0.870 | -25.7% |   0.53 | $17.49 |  $1.81M  |     157    |
| PCB-15 (-15%) | 12.64% | 19.27% |  0.716 |   0.833 | -27.2% |   0.51 | $17.80 |  $1.86M  |     132    |
| **V16-B**     | 17.06% | 21.33% | **0.846** | 0.997 | -27.0% |   0.67 | $52.18 |  $4.41M  |      —     |

**The PCB is catastrophic.** Every threshold destroys CAGR (5-7pp loss), destroys terminal wealth (3-5× reduction), and destroys Sharpe (0.10-0.17 drop). Even PCB-15 underperforms V16-B on every metric.

### Why the PCB Fails

At 180% effective equity, a 10% portfolio drawdown from HWM is **normal monthly volatility**, not a crash. The portfolio routinely oscillates ±10% from its high during healthy bull markets. The PCB fires during these oscillations, delevering at the bottom of normal chop, then missing the recovery when it re-enters at the next monthly rebalance.

PCB-10 fires **209 times** in 24 years — nearly once per month. The per-asset CBs fire 27 times (14 QQQ + 13 IVV) because the 3/3 SMA breach requirement is correctly calibrated to distinguish "crash" from "chop." The portfolio-level threshold has no such calibration.

### Crisis-Specific Results

| Crisis | PCB-8  | PCB-10 | PCB-12 | PCB-15 |  V16-B |
|--------|-------:|-------:|-------:|-------:|-------:|
| GFC    | -16.6% | -17.0% | -17.1% | -18.6% | -16.2% |
| COVID  | -23.7% | -24.2% | -25.7% | -27.2% | -27.0% |
| 2022   | -16.1% | -18.0% | -18.1% | -19.8% | -20.0% |

**COVID:** PCB-8 catches COVID at -23.7% vs V16-B -27.0% (+3.3pp). This is the ONE scenario where the PCB helps — a fast -30% crash that hits both pods simultaneously. But the 330+ non-crisis firings destroy the mechanism's value.

**GFC:** PCB thresholds actually WORSEN GFC DD vs V16-B (-16.2%). The PCB fires, delevels, then V16-B's per-asset CBs fire independently and produce a better exit sequence than the blunt portfolio-level exit.

**2022:** Modest improvement at aggressive thresholds, but V16-B's per-asset CBs already handle 2022 well.

---

## Part 2B — Leading Indicator Leverage Scaling

**SKIPPED.** Zero indicators passed Part 1 validation criteria. No indicator reliably leads equity crashes with acceptable false positive rate.

---

## Pass / Fail Summary

| Mechanism | DD improvement | Sharpe vs V16-B | CAGR vs V16-B | Overall |
|-----------|:-------------:|:---------------:|:--------------:|:-------:|
| PCB-8     | +3.3pp        | -0.167          | -6.79pp        | FAIL    |
| PCB-10    | +2.8pp        | -0.137          | -5.64pp        | FAIL    |
| PCB-12    | +1.3pp        | -0.102          | -4.49pp        | FAIL    |
| PCB-15    | -0.2pp        | -0.130          | -4.42pp        | FAIL    |
| LI overlay| N/A           | N/A             | N/A            | SKIPPED |

---

## What We Learned

### 1. -27% MaxDD is structural and cannot be improved by overlays
At 180% effective equity with monthly rebalancing and lagging SMA signals, the first month of any fast crash is unprotected. COVID dropped ~27% in a month while both equity pods were fully leveraged. The per-asset CB fired within days but the damage was done. No portfolio-level mechanism can fix this without also firing during normal volatility.

### 2. Leading indicators do not predict equity crashes
This is the fourth time this project has found that macro/vol indicators add no value to the Faber filter:
- April 4: Harvey expected returns → noise
- April 4: Kritzman conditioned covariance → noise  
- April 5: VRP filter → marginal
- **April 12: VIX, yield curve, credit spreads, S&P drawdown → all fail**

The Faber filter responds to *price directly*. Every other indicator either lags price (drawdown from high) or leads by so much that timing is worthless (yield curve leads by 6 months with 100% FP rate).

### 3. The per-asset CB is correctly calibrated
The 3/3 SMA breach requirement fires 27 times in 24 years (~1.1/year). A portfolio-level -10% threshold fires 209 times (~8.7/year). The SMA breach captures genuine trend breakdowns. The portfolio drawdown threshold captures normal volatility. The existing design is optimal.

### 4. The only path to lower MaxDD is lower leverage
If -27% is unacceptable, reduce the substitution percentage (SSO/QLD allocation). At 50% substitution, effective equity drops from 180% to ~135% and MaxDD should proportionally decrease. But CAGR will drop too — this is the leverage/DD tradeoff that defines the entire project. V16-B at 100% substitution is already the Sharpe-optimal point on that curve.

---

## Cross-references

- [[2026-04-04_kritzman_vs_harvey]] — macro return forecasts are noise (confirmed for the fourth time)
- [[2026-04-05_vrp_backtest]] — VRP filter marginal, confirms vol-based signals add little
- [[2026-04-06_faber_daily_circuit_breaker]] — per-asset CB correctly calibrated at 3/3 SMA breach
- [[experiments/V16_TWO_POD_GOLD_RESULTS]] — V16-B's -27.0% MaxDD is the structural floor for its architecture

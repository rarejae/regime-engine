# Terminal Wealth Optimization: Simplified High-Conviction Architectures

**Date:** April 9, 2026  
**Status:** Complete  
**Track:** Architecture Exploration  
**Related:** [[TAA_PROJECT_STATUS]] | [[BULL_MARKET_SURVIVABILITY]]

## Research Question

Are the defensive assets (VGLT 5%, IAU 10%, DBC 5%) earning their place, or is the circuit breaker doing all the defensive work while non-equity assets dilute compounding?

## Answer: NO variant passes all four criteria

The pass/fail framework required: (1) beat QQQ CAGR from 2013, (2) max DD < -30%, (3) Sharpe > QQQ, (4) dot-com DD < -40%. No single variant satisfies all four simultaneously.

**The fundamental constraint:** variants that beat QQQ on CAGR (V1, V6, V9) have max DD > -30%. Variants with max DD < -30% (Baseline, V3-V5) don't beat QQQ CAGR from 2013.

---

## Full-Period Comparison (2002-2026)

| Variant | CAGR | Vol | Sharpe | MaxDD | Terminal $1 | DCA $700/mo |
|---------|------|-----|--------|-------|-----------|------------|
| **Baseline** | **13.8%** | **15.5%** | **0.914** | **-18.1%** | **$25.62** | **$2.40M** |
| V1 QLD-only | 18.9% | 28.2% | 0.756 | -43.5% | $77.18 | $7.00M |
| V2 SSO-only | 15.2% | 20.7% | 0.786 | -30.4% | $34.82 | $2.92M |
| V3 Eq+Cash 45/25 | 12.7% | 15.2% | 0.862 | -18.1% | $20.21 | $1.98M |
| V4 Eq+Cash 35/35 | 13.1% | 15.9% | 0.853 | -19.8% | $21.88 | $2.16M |
| V5 Eq+Cash 25/45 | 13.4% | 16.6% | 0.840 | -21.4% | $23.57 | $2.34M |
| V9 QLD+IVVguard | 19.4% | 27.8% | 0.777 | -37.9% | $85.25 | $7.37M |
| QQQ B&H | 12.6% | 22.8% | 0.634 | -53.4% | $17.58 | $2.33M |
| IVV B&H | 9.4% | 19.0% | 0.569 | -55.2% | $8.85 | $1.21M |

### Key findings from Table 1:

**V1 QLD-only is the terminal wealth monster:** $77.18 from $1 (vs Baseline $25.62). 18.9% CAGR. But -43.5% max DD makes it psychologically brutal. DCA: $7.0M from $21K + $700/mo.

**V9 QLD+IVVguard is marginally better:** $85.25 terminal, -37.9% DD (5.6% better than V1). The IVV guard catches broad market breakdowns that pure QQQ misses.

**Defensive assets ADD value:** Baseline ($25.62) beats V3 no-defense ($20.21) by $5.41. VGLT/IAU/DBC contribute genuine return when their trends are active, not just diversification.

**V5 QQQ-heavy (25/45) is interesting:** $23.57 terminal — close to Baseline — but with lower Sharpe (0.840 vs 0.914). Shifting weight toward QQQ helps return but hurts risk-adjusted.

---

## Start-Date Sensitivity

| Variant | 2002 | 2007 | 2010 | 2013 | 2019 |
|---------|------|------|------|------|------|
| Baseline | 13.8% | 14.9% | 15.6% | 17.2% | 19.6% |
| V1 QLD-only | 18.9% | 22.0% | 23.6% | **29.0%** | 28.9% |
| V5 Eq+Cash 25/45 | 13.4% | 14.9% | 15.9% | 18.1% | 19.7% |
| V9 QLD+IVVguard | 19.4% | 22.6% | 23.3% | **28.5%** | 28.1% |
| QQQ B&H | 12.6% | 15.4% | 18.0% | **18.9%** | 20.8% |

**V1/V9 crush QQQ from every start date** — 29.0% vs 18.9% from 2013. The Faber-gated QLD approach earns QQQ's upside while dodging its worst drawdowns (QQQ DD -53.4% vs V1 -43.5% and V9 -37.9%).

**Baseline trails QQQ from 2013 (17.2% vs 18.9%)** — confirmed from the survivability analysis. But V5 (18.1%) gets close to matching QQQ from 2013.

---

## Sub-Period Breakdown

| Period | Baseline | V1 QLD | V9 QLD+guard | QQQ B&H |
|--------|----------|--------|-------------|---------|
| **Dot-com** | **+0.1%** | +1.7% | +1.7% | **-29.4%** |
| **GFC** | **-6.1%** | -29.5% | **-22.6%** | -34.2% |
| **2013-2021** | 19.7% | **35.5%** | **34.8%** | 23.4% |
| **2022 bear** | -12.9% | -15.2% | -15.2% | **-32.7%** |
| 2023-2026 | 21.1% | 27.8% | 27.8% | 27.9% |

**V1/V9 during 2013-2021: 35.5%/34.8% vs QQQ 23.4%.** They beat QQQ by 10+ percentage points during the bull market — because leveraged QLD (2x) captures the Nasdaq uptrend at double the rate when Faber says "trend is on."

**GFC is the V1 catastrophe:** -29.5%. QLD amplified the crash before Faber could exit. V9 is better (-22.6%) because the IVV guard catches the broad market breakdown earlier.

**Dot-com: both V1 and V9 survived.** QQQ was off-signal for most of 2001-2002, so the system was in cash. The actual dot-com damage was contained.

---

## DCA Dollar Gap vs QQQ (2013-2026)

| Year-end | Baseline | V1 QLD | V9 QLD+guard | QQQ B&H |
|----------|----------|--------|-------------|---------|
| 2015 | $59K | $85K | $85K | $68K |
| 2018 | $118K | $207K | $207K | $126K |
| 2020 | $219K | $498K | $475K | $284K |
| 2022 | $282K | $667K | $637K | $258K |
| 2026 | $575K | $1,560K | $1,491K | $606K |

**V1 NEVER trails QQQ at any year-end.** The Baseline's painful $62K gap at 2020 does not exist for V1/V9 — they're ahead of QQQ from the first year onward.

**V9 at 2026: $1.49M vs QQQ's $606K** — 2.5× more wealth. Even the 2022 drawdown only reduced V9 from $745K to $637K — still massively ahead of QQQ's $258K.

---

## Pass/Fail Summary

| Variant | 2013 CAGR > QQQ? | DD < -30%? | Sharpe > QQQ? | Dot-com < -40%? | PASS? |
|---------|-----------------|-----------|--------------|----------------|-------|
| Baseline | NO (17.2%) | YES (-18.1%) | YES | YES (-2.1%) | **FAIL** |
| V1 QLD-only | YES (29.0%) | NO (-43.5%) | YES | YES (-5.0%) | **FAIL** |
| V5 Eq+Cash 25/45 | NO (18.1%) | YES (-21.4%) | YES | YES (-2.2%) | **FAIL** |
| V9 QLD+IVVguard | YES (28.5%) | NO (-37.9%) | YES | YES (-5.0%) | **FAIL** |

**Every variant fails on exactly ONE criterion:**
- Baseline/V3-V5: fail on CAGR (don't beat QQQ from 2013)
- V1/V6: fail on MaxDD (-43.5%)
- V9: fail on MaxDD (-37.9%)

---

## Verdict

**No variant passes all four criteria.** The constraint set is genuinely binding — you cannot simultaneously beat QQQ's CAGR, keep DD below -30%, maintain higher Sharpe than QQQ, and survive dot-com. The tradeoff is real and irreducible.

**The honest tradeoff map:**

| Want... | Best variant | Sacrifice |
|---------|-------------|----------|
| Maximum terminal wealth | V9 QLD+IVVguard ($85.25, $7.37M DCA) | -37.9% max DD |
| Maximum Sharpe | Baseline (0.914) | 17.2% CAGR from 2013 (trails QQQ 18.9%) |
| Beat QQQ CAGR + best DD | V9 QLD+IVVguard (28.5% from 2013) | -37.9% DD |
| DD < -20% | Baseline or V3 (-18.1%) | 12.7-13.8% CAGR |

**The key insight: the circuit breaker alone does NOT provide adequate drawdown protection for single-asset QLD.** The GFC (-29.5% for V1) and the max DD of -43.5% demonstrate that QLD's 2x leverage amplifies drawdowns beyond what the monthly Faber filter + daily CB can contain. The defensive assets in Baseline don't just diversify — they mechanically reduce the equity exposure during months when they're the only assets with positive Faber scores (e.g., gold trending up while equities are breaking down).

**Defensive assets confirmed as net positive:** Baseline ($25.62) beats V3 no-defense ($20.21). Removing VGLT/IAU/DBC and holding more cash does NOT improve results — the non-equity trends add genuine return when active.

**V9 is the correct high-conviction variant** if the investor can stomach -37.9% max DD. It beats QQQ from every start date, earns 19.4% full-period CAGR (vs 12.6% QQQ), and the IVV guard provides 5.6% DD improvement over pure V1. At $21K + $700/month DCA, V9 reaches $7.37M — 3× the Baseline's $2.40M.

**But -37.9% max DD on a $21K portfolio is a $7,959 paper loss.** Whether that's acceptable is a preference question, not a math question.
# Terminal Wealth Optimization Results

**Date:** April 9, 2026 | **Status:** Complete

## Key Numbers

| Variant | CAGR | Sharpe | MaxDD | Terminal $1 | DCA $700/mo |
|---------|------|--------|-------|-------------|------------|
| Baseline (current) | 13.8% | 0.914 | -18.1% | $25.62 | $2.40M |
| V1 QLD-only | 18.9% | 0.756 | -43.5% | $77.18 | $7.00M |
| V9 QLD+IVVguard | 19.4% | 0.777 | -37.9% | $85.25 | $7.37M |
| V5 Eq+Cash 25/45 | 13.4% | 0.840 | -21.4% | $23.57 | $2.34M |
| QQQ B&H | 12.6% | 0.634 | -53.4% | $17.58 | $2.33M |

## Finding
No variant passes all four criteria (beat QQQ CAGR from 2013 + DD < -30% + Sharpe > QQQ + dot-com < -40%).
Defensive assets confirmed net positive. Circuit breaker alone insufficient for single-asset QLD DD control.
V9 is the terminal wealth maximizer if -37.9% DD is acceptable.

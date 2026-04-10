# Three-Portfolio Direct Comparison: Faber vs Spreads vs Combined

**Date:** April 7, 2026  
**Status:** Complete  
**Track:** Multi-Pod Architecture — Final Evaluation  
**Related:** [[TAA_PROJECT_STATUS]] | [[2026-04-07_spreads_unconstrained]] | [[2026-04-06_harvey_spread_backtest]]

## Purpose

Direct comparison of three portfolios on $100,000 from January 2010 to December 2023, with Pod 2 using fully specified a priori position management rules (profit target, delta stop, premium stop, time exit).

## 1. Full Performance Table

| Portfolio | Return | Vol | Sharpe | Sortino | MaxDD | Calmar | Terminal |
|-----------|--------|-----|--------|---------|-------|--------|---------|
| **A: Faber Only** | **15.8%** | **14.8%** | **1.070** | **1.590** | **-14.1%** | **1.12** | **$774,953** |
| B: Spreads Only | 0.8% | 2.4% | 0.314 | 0.283 | -13.3% | 0.06 | $110,760 |
| C: Combined 90/10 | 14.3% | 13.3% | 1.072 | 1.591 | -13.1% | 1.09 | $646,768 |
| IVV B&H | 13.4% | 14.8% | 0.905 | 1.374 | -23.9% | 0.56 | $554,770 |
| 60/40 | 10.1% | 9.8% | 1.028 | 1.434 | -25.7% | 0.39 | $380,713 |

**The position management rules dramatically changed the spread strategy's character.** The prior unconstrained test (held to expiry) showed 7.14% return and 1.151 Sharpe. With proper risk management (profit target at 50%, delta stop at -0.25, premium stop at 2×), the spreads return only 0.8% with 0.314 Sharpe. The profit target captures wins early (81% of trades, avg 6 days held), but the stops cut losers aggressively — net result is much lower return.

## 2. Annual NAV

| Year | A: Faber | B: Spreads | C: Combined | IVV B&H |
|------|---------|-----------|------------|---------|
| 2010 | $118,148 | $96,524 | $115,942 | $115,056 |
| 2013 | $195,381 | $90,893 | $181,664 | $179,916 |
| 2017 | $309,646 | $88,410 | $274,791 | $281,692 |
| 2020 | $527,852 | $97,523 | $450,670 | $417,424 |
| 2023 | $774,953 | $110,760 | $646,768 | $554,770 |

The spreads-only portfolio barely breaks even — $110,760 from $100,000 over 14 years (0.8% annual return). The 2010-2011 drawdown (-$10K from initial capital) was never fully recovered until 2020. This is a T-bill-plus strategy at best.

## 3. Crisis Analysis

| Period | A: Faber | B: Spreads | C: Combined | IVV B&H |
|--------|---------|-----------|------------|---------|
| 2011 correction | -1.9% (DD -2.4%) | -6.0% (DD -6.0%) | -2.3% (DD -2.7%) | -4.4% (DD -12.1%) |
| 2015-16 vol | -12.8% (DD -6.5%) | -5.8% (DD -6.6%) | -12.1% (DD -6.3%) | -7.0% (DD -6.7%) |
| 2018 Q4 | -12.2% (DD -2.7%) | **+0.9%** | -10.9% (DD -2.4%) | -13.5% (DD -8.8%) |
| COVID Feb-Mar 2020 | -7.5% (DD -0.8%) | **+1.8%** | -6.5% (DD -0.7%) | -19.4% (DD -12.5%) |
| 2022 bear | -12.5% (DD -4.2%) | **+4.8%** | -10.9% (DD -3.7%) | -17.7% (DD -20.2%) |

**Spreads are positive during 3 of 5 crises** (2018 Q4, COVID, 2022). They lose during 2011 and 2015-16. The position management rules kept losses contained: no spread loss exceeded -6.6% in any crisis period.

## 4. Correlation

```
Monthly return correlation (Faber vs Spreads): 0.222
```

Higher than the unconstrained test (-0.162) because the position management rules change the return profile — early profit-taking creates a different timing pattern.

**Crisis correlations are mixed:**
- 2011: 0.90 (highly correlated — both lost)
- 2018 Q4: 0.74 (moderately correlated)
- COVID: **-0.67** (negatively correlated — diversifying)
- 2022: 0.39 (low)

**Loss clustering:** 4 months where both pods lost >1% (May 2010, Aug 2011, Sep 2011, Jan 2016). Not zero — but manageable.

## 5. Position Management Effectiveness

| Exit Rule | Trades | % | Avg Hold | Avg P&L |
|-----------|--------|---|----------|---------|
| **Profit target (Rule 1)** | **26** | **81%** | **6 days** | **+$709** |
| Delta stop (Rule 2) | 2 | 6% | 6 days | -$3,656 |
| Premium stop (Rule 3) | 4 | 12% | 7 days | -$2,962 |
| Time exit (Rule 4) | 0 | 0% | — | — |

**Win rate: 81%.** Average winner: $709. Average loser: -$3,193. Largest loss: -$4,194 (May 2010).

The 50% profit target dominates: 81% of trades close within 6 days at half the max credit. This is extremely conservative — it leaves significant theta decay on the table but ensures consistent small wins.

The delta stop fired twice, the premium stop 4 times. Time exit never triggered because positions always hit one of the other rules first.

**The trade-off:** Position management converts a high-return/high-risk strategy (7.14% unconstrained) into a low-return/low-risk one (0.8%). The 50% profit target captures wins early but the stops cut losers at relatively high cost. Net result: lots of small wins (+$709 avg) and a few medium losses (-$3,193 avg), with the losses dominating the P&L enough to reduce return dramatically.

## 6. Combined Portfolio Value-Add

```
Sharpe improvement (C vs A):     +0.001 (1.070 → 1.072)
Max DD improvement (C vs A):     +1.0% (-14.1% → -13.1%)
Terminal wealth cost (C vs A):   -$128,185
Correlation Faber/Spreads:       0.222
```

The combined portfolio improves Sharpe by a negligible +0.001 and max DD by 1.0%, at a terminal wealth cost of **$128,185** ($774,953 → $646,768). The Sharpe improvement is statistically meaningless — within noise.

## 7. Plain Language Summary

```
Starting with $100,000 in January 2010:

Portfolio A (Faber Only) reached $774,953 by December 2023
Portfolio B (Spreads Only) reached $110,760 by December 2023
Portfolio C (Combined) reached $646,768 by December 2023
IVV Buy & Hold reached $554,770 by December 2023

Faber/Spreads monthly correlation: 0.222
Combined Sharpe vs Faber-only: 1.072 vs 1.070 (+0.001)
Combined Max DD vs Faber-only: -13.1% vs -14.1% (+1.0%)

Position management: profit target fired 26 times (+$709 avg),
delta stop 2 times (-$3,656 avg), premium stop 4 times (-$2,962 avg).
```

## Interpretation

### The position management paradox

The unconstrained test showed 1.151 Sharpe and 7.14% return. With proper risk management, the strategy returns 0.8% with 0.314 Sharpe. What happened?

**The 50% profit target is too aggressive.** By closing at 50% of max credit after ~6 days, the strategy captures only small wins ($709 avg on ~$1,500 max credit). Meanwhile, the stops allow losses of $3,000-$4,000 per trade. The win/loss ratio (81/19) doesn't compensate for the asymmetry ($709 win vs -$3,193 loss).

The unconstrained version held to expiry and let theta decay work over the full 30 days. Most spreads that finished in the money on day 6 would have finished at max profit by day 30. The early exit leaves 50% of the premium uncollected.

### What this means for the two-pod architecture

With proper position management, Pod 2 adds essentially zero Sharpe (+0.001) and costs $128K in terminal wealth. The 1.0% DD improvement is real but small. **Pod 2 does not earn its place with these management rules.**

The fundamental tension: aggressive position management (good for risk control) destroys the strategy's return (bad for portfolio contribution). Without position management, the strategy has great standalone metrics but exposes the portfolio to occasional large losses.

### The right path forward

If Pod 2 is to be deployed, the position management rules need recalibration:
- **Raise profit target to 75%** (close at 75% of max credit, not 50%) — capture more theta decay
- **Widen delta stop to -0.30** (from -0.25) — give trades more room to breathe
- **Remove time exit** (already unused) — simplify
- **Keep premium stop at 2×** — this is the catastrophic loss prevention rule

But this is optimization territory — fitting rules to past data. The safer conclusion: **Pod 2 is not worth the complexity at current portfolio size ($21K). Revisit when portfolio exceeds $100K and a single put spread contract is a reasonable position size.**

## Decision

**Pod 2 with a priori position management rules adds negligible value (+0.001 Sharpe, -$128K terminal).** The position management rules that make the strategy safe also destroy its return. Faber-Sweep-40 standalone remains the production architecture.

**Pod 2 deferred** to portfolio growth phase ($100K+). At that point, revisit with:
1. Higher profit target (75%)
2. Wider delta stop (-0.30)
3. Manual execution during Faber cash periods only (not continuous)

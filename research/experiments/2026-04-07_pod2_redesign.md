# Pod 2 Redesign: -0.20 Delta, 45 DTE

**Date:** April 7, 2026  
**Status:** Complete  
**Track:** Options Pod — Redesign Attempt  
**Related:** [[TAA_PROJECT_STATUS]] | [[2026-04-07_two_pod_comparison]]

## Purpose

Fix the prior test's failure where -0.10 delta / 30 DTE spreads collected only $0.70 credit, making the 50% profit target bank only $0.35 per contract — negative expected value after losses. The hypothesis: moving to -0.20 delta at 45 DTE would collect ~$1.80 credit.

## Result: The hypothesis was wrong

| Metric | Prior (-0.10d, 30DTE) | New (-0.20d, 45DTE) |
|--------|----------------------|---------------------|
| Avg credit collected | $0.70 | **$0.66** |
| 50% profit target banks | $0.35 | **$0.33** |
| Avg winning trade | $709 | $1,589 |
| Avg losing trade | -$3,193 | **-$6,201** |
| Win rate | 81% | 79% |
| Standalone Sharpe | 0.314 | **0.141** |
| Standalone return | 0.8% | 0.8% |
| Combined Sharpe | 1.072 | **1.067** |

**The average credit DECREASED from $0.70 to $0.66.** The fundamental error in the hypothesis: moving the short put closer to ATM (-0.20 vs -0.10 delta) increases the short put's individual premium, but the long put (5 points below) ALSO becomes more expensive because it's closer to ATM too. The net spread credit is bounded by the spread width divided by the probability of being in the money — it doesn't scale with delta the way a naked option would.

## Why It Got Worse

The redesign made everything worse because the wider delta stop (-0.35 vs -0.25) allows larger losses to accumulate before cutting:

- **Avg loss nearly doubled:** -$6,201 vs -$3,193. The delta stop at -0.35 gives the trade "room to breathe" but that room is filled with losses.
- **Win rate slightly lower:** 79% vs 81%. The -0.20 delta short put is closer to ATM, so it gets tested more often.
- **Profit factor ≈ 1.0:** $49,259 gross wins / $49,613 gross losses = 0.99. Break-even before commissions.
- **The delta stop fired 8 times (21%)** vs 2 times (6%) in the prior test. More stop-outs, larger losses per stop.

## Three-Portfolio Comparison

| Portfolio | Return | Vol | Sharpe | Sortino | MaxDD | Calmar | Terminal |
|-----------|--------|-----|--------|---------|-------|--------|---------|
| A: Faber Only | 15.8% | 14.8% | 1.070 | 1.590 | -14.1% | 1.12 | $774,953 |
| B: Spreads (new) | 0.8% | 5.3% | 0.141 | 0.307 | -21.8% | 0.03 | $108,846 |
| C: Combined 90/10 | 14.3% | 13.4% | 1.067 | 1.605 | -13.3% | 1.08 | $646,012 |

Combined: **-0.004 Sharpe** (worse), +0.8% DD, -$128,941 terminal. Strictly worse than the prior test.

## Crisis Behavior

| Period | B: Spreads |
|--------|-----------|
| 2011 correction | **-11.2%** (worse — -0.20d gets hit harder) |
| 2018 Q4 | +2.2% (win) |
| COVID | +4.2% (win) |
| 2022 bear | **-9.8%** (DD -13.3% — 8 months of -0.20d exposure during sustained bear) |

The 2022 drawdown is the smoking gun: at -0.20 delta, the short put is close enough to ATM that a sustained bear market triggers the delta stop repeatedly over multiple months, each time crystallizing a ~$6K loss. The prior -0.10 delta version was far enough OTM to survive most months.

## The Fundamental Problem With Managed Vertical Spreads

After two attempts, the pattern is clear:

**Credit collected scales with probability of loss, not with premium income.** A 5-point spread at -0.10 delta collects ~$0.70 and has ~10% loss probability. A 5-point spread at -0.20 delta collects ~$0.66 and has ~20% loss probability. The credit doesn't increase proportionally to the risk increase because the long put's cost also increases.

**The profit target is inherently destructive for spreads.** Closing at 50% of max credit captures $0.33-0.35 per contract ($33-35 per 100 shares). With $2.60 commission per spread, the cost is 7-8% of the profit. And one stop-out at -$6,200 wipes out 9 winning trades at +$709. The math doesn't work at any delta for a 5-point wide spread.

**To make the 50% profit target economically viable, the spread needs to be ~20-25 points wide** (collecting $3-4 net credit, banking $1.50-2.00 at 50% profit). But this requires $2,000-2,500 margin per contract, allowing only 12-15 contracts on $30K margin — and a single max loss would be $12,000-$15,000. The risk/reward profile doesn't improve.

## Position Management Breakdown

| Exit Rule | Trades | % | Avg Hold | Avg P&L |
|-----------|--------|---|----------|---------|
| Profit target | 31 | 79% | 10 days | +$1,589 |
| Delta stop (-0.35) | 8 | 21% | 13 days | -$6,201 |

Zero premium stops and zero time exits — all positions either hit the profit target or the delta stop. The delta stop at -0.35 is too permissive; at -0.25 (prior test) losses were -$3,656 avg; at -0.35 they're -$6,201.

## Decision

**Pod 2 vertical spread design REJECTED.** Two parameter sets tested:
- -0.10 delta / 30 DTE: 0.314 Sharpe, +0.001 combined Sharpe
- -0.20 delta / 45 DTE: 0.141 Sharpe, -0.004 combined Sharpe

Neither produces meaningful standalone returns. The fundamental economics of 5-point vertical spreads with conservative profit targets don't support a positive-expectancy strategy — the credit collected is too small relative to the max loss, and position management rules that prevent large losses also cap the strategy's return.

**The unconstrained version (held to expiry, no stops) worked** — 1.151 Sharpe, 97% win rate. The position management rules destroy the edge. This is the core tension: the unconstrained strategy has great backtest metrics but would expose the live portfolio to occasional catastrophic losses that the stops are designed to prevent.

**Path forward:** If deploying spreads, hold to expiry with NO position management (accept the occasional large loss) and size positions small enough that a max loss on any single trade is < 2% of portfolio value. This means 1-2 contracts maximum on a $100K portfolio, generating ~$500-1000/year. At that scale, the operational overhead exceeds the return.

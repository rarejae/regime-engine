# Asymmetric Iron Condor Pod 2

**Date:** April 7, 2026  
**Status:** Complete  
**Track:** Options Pod — Iron Condor Test  
**Related:** [[TAA_PROJECT_STATUS]] | [[2026-04-07_pod2_redesign]] | [[2026-04-07_two_pod_comparison]]

## Result: Catastrophic Failure

| Metric | Iron Condor | Prior Put Spread | Prior -0.20d |
|--------|------------|-----------------|-------------|
| Annual return | **-11.5%** | 0.8% | 0.8% |
| Sharpe | **-0.920** | 0.314 | 0.141 |
| Max DD | **-84.6%** | -13.3% | -21.8% |
| Terminal ($100K) | **$17,465** | $110,760 | $108,846 |
| Win rate | 55% | 81% | 79% |
| Avg loss | **-$9,026** | -$3,193 | -$6,201 |

**The iron condor destroyed 83% of capital.** From $100K in 2010 to $17,465 by 2023. This is dramatically worse than either prior single-spread test.

## Why It Failed: The Call Spread Is the Killer

The call spread at +0.20 delta triggered 19 delta stops (33% of trades) with average loss of -$10,452 per event. These are catastrophic — happening during market rallies that Harvey predicted would be mild recoveries. But "recovery" doesn't mean "slow grind up" — it means V-shaped explosions that blast through +0.20 delta call strikes.

**The asymmetry hypothesis was backwards.** Harvey signals recovery, which should make the put side safe (correct — put side was fine). But recovery also means the call side is in extreme danger — short calls at +0.20 delta get steamrolled when the market rallies hard. The +0.35 delta stop on the call side is reached quickly during strong up-moves, locking in maximum losses on every rally.

### Leg P&L breakdown:
- Put leg profits: 20 put_leg_profit events, avg +$1,421
- Call leg stops: 19 call_delta_stop events, avg **-$10,452**
- Condor full profit: 7 events, avg +$3,001

The call leg losses ($198,588 total) overwhelm the put leg profits ($28,420) and condor profits ($21,007).

## Credit Economics: Also Wrong

The hypothesis was that the condor would collect ~$1.05 total credit ($0.65 put + $0.40 call). Actual: $1.12 total — close to prediction. But the split was $0.29 put + $0.82 call — the call side collected most of the premium. This is because at +0.20 delta, the call is much closer to ATM than the put at -0.10 delta, so it naturally has higher premium. But this higher premium comes with proportionally higher risk.

## Crisis Destruction

| Crisis | Condor Return | Condor DD |
|--------|--------------|-----------|
| 2018 Q4 | **-9.6%** (DD -10.0%) | Call stops during December relief rally |
| COVID | **-33.9%** (DD -26.9%) | Call stops during March crash bounce |
| 2022 bear | **-29.5%** (DD -31.9%) | Repeated call stops during bear market rallies |

The pattern: every crisis includes sharp counter-trend rallies. The call spread at +0.20 delta gets hit by these rallies — exactly the opposite of what the strategy intended.

## Correlation: Genuinely Negative (Doesn't Help)

Monthly correlation: -0.064 (essentially zero). Crisis correlations: 2011 -0.86, 2018 Q4 -0.72, COVID 0.04, 2022 -0.27. The condor IS negatively correlated with Faber — but it achieves this by losing money when Faber makes money (rallies) and making money when Faber loses (flat/down periods where condor decays profitably).

## Decision

**Iron condor Pod 2 REJECTED.** The call leg at +0.20 delta is fatally flawed during Harvey recovery signals. Adding call premium to a bullish macro thesis creates a structural contradiction: the strategy is long recovery via the put side and short recovery via the call side. The call side loses more than the put side wins.

**All three managed options Pod 2 designs have now been tested and failed:**
1. Put spread -0.10d/30DTE: 0.314 Sharpe, economically marginal
2. Put spread -0.20d/45DTE: 0.141 Sharpe, worse than prior
3. Iron condor -0.10d put / +0.20d call: -0.920 Sharpe, catastrophic

**The only viable options strategy was the unconstrained version** (held to expiry, no management): 1.151 Sharpe, 97% win rate. But this is not practical for live trading due to unacceptable tail risk on the occasional large loss.

**Pod 2 options research is concluded.** No managed options configuration produces risk-adjusted returns that justify inclusion in the production architecture.

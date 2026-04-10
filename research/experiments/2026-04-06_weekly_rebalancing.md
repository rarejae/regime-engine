# Weekly vs Monthly Rebalancing on Faber-Sweep-40-Daily-Daily

**Date:** April 6, 2026  
**Status:** Complete  
**Track:** Production Architecture  
**Related:** [[TAA_PROJECT_STATUS]] | [[2026-04-06_faber_daily_circuit_breaker]]

## Purpose

Test whether rebalancing the full Faber portfolio every Friday (instead of month-end only) improves performance at 100% SSO/QLD substitution. Both strategies use identical daily circuit breakers.

## Result: Weekly rebalancing significantly HURTS performance

| Strategy | Return | Vol | Sharpe | MaxDD | Terminal($1) | vs Monthly |
|----------|--------|-----|--------|-------|-------------|-----------|
| **MONTHLY-100** | **14.5%** | **15.8%** | **0.918** | **-18.1%** | **$24.65** | **baseline** |
| WEEKLY-100 | 13.2% | 15.4% | 0.859 | -21.0% | $18.43 | **-$6.22** |
| WEEKLY-100-NOCOST | 13.4% | 15.4% | 0.868 | -20.9% | $19.02 | -$5.63 |

**Weekly loses $6.22 terminal wealth** and 0.058 Sharpe vs monthly. Even without transaction costs (NOCOST), it still loses $5.63 and 0.050 Sharpe. The damage is NOT from transaction costs — it's from the signal itself.

## Why Weekly Rebalancing Hurts

### The leverage whipsaw problem

The weekly system changes the leverage condition (both IVV+QQQ at 3/3 vs not) **106 times over 24 years (4.4/year)** vs monthly's 48 changes (2.0/year). More than double the switching. Each leverage switch involves converting SSO↔IVV and QLD↔QQQ — and each switch catches a moment of uncertainty where the signal is transitioning.

The Faber SMA signal is designed for monthly timeframes. At weekly resolution, the signal catches mid-month noise — brief dips below one SMA that reverse within days. The monthly system ignores this noise because it only checks at month-end, by which time the transient has resolved.

### The 2022 paradox

Weekly significantly outperformed during the 2022 bear market: -8.3% vs -12.6% (monthly). It caught the trend break earlier and exited faster. But this benefit was overwhelmed by the damage during trending markets (2013, 2017, 2020, 2021) where weekly's frequent switching caused it to miss re-entry or exit prematurely during brief pullbacks in strong uptrends.

### GFC: weekly was worse

Monthly: +0.6%. Weekly: -1.5%. The weekly system caught some mid-month noise during the GFC recovery transition that the monthly system correctly ignored.

### Signal change frequency

```
Weekly score changes: 30.2/year (5.5/yr per equity ETF)
Monthly score changes: 11.8/year
Additional from weekly: 18.4/year — most of these are noise
```

VGLT is the worst offender: 7.5 score changes/year at weekly vs much fewer at monthly. Bond trend signals are noisy at weekly timeframes because VGLT's 126-day SMA creates frequent crossovers.

### Transaction costs are negligible

Additional TC from weekly: 0.075%/year — accounts for only $0.59 of the $6.22 terminal wealth loss. The remaining $5.63 is pure signal degradation.

## Crisis Detail

| Crisis | Monthly | Weekly | Diff |
|--------|---------|--------|------|
| GFC (2008-09) | +0.6% (DD -1.2%) | -1.5% (DD -2.0%) | Weekly **worse** |
| COVID (Feb-Mar 2020) | -16.7% (DD -18.1%) | -16.6% (DD -17.8%) | ~same |
| 2022 Bear | -12.6% (DD -13.8%) | -8.3% (DD -9.6%) | Weekly **better** |

Weekly is better in 2022 and worse in GFC. The net effect over 24 years strongly favors monthly — the damage from whipsaw during trending markets far outweighs the benefit from faster exits during bears.

## Age-65 Impact

```
MONTHLY-100:       $4,708,770 at age 65
WEEKLY-100:        $3,022,001 at age 65
Difference:       -$1,686,769  (36% less wealth)
```

Weekly rebalancing costs **$1.7 million** over a 40-year horizon.

## Key Insight

The Faber trend filter is a **low-frequency signal**. It was designed to capture multi-month trends, not weekly fluctuations. Checking it more frequently doesn't add information — it adds noise. The monthly cadence is not a compromise or limitation; it's the correct timeframe for the signal.

This is the same pattern seen throughout the project: the circuit breaker (which monitors a different condition — 3/3 SMA breach, a high-conviction exit signal) benefits from higher frequency because it's monitoring for an extreme event. The allocation signal (which SMA bin is each asset in) does NOT benefit from higher frequency because the bins change for noise reasons at weekly resolution.

**Slow allocation decisions, fast emergency exits** — the asymmetric design is confirmed correct.

## Decision

**Monthly rebalancing confirmed as optimal.** Weekly rebalancing rejected — loses $6.22 terminal and 0.058 Sharpe due to signal noise at weekly frequency. The current production architecture (monthly allocation + daily circuit breaker) is the correct asymmetric design.

# Weekly vs Monthly Rebalancing Test

**Date:** April 6, 2026
**Status:** Complete — Monthly confirmed optimal
**Track:** Production Architecture

## Result

Weekly rebalancing significantly HURTS performance.

| Strategy | Return | Sharpe | MaxDD | Terminal | vs Monthly |
|----------|--------|--------|-------|---------|-----------|
| MONTHLY-100 | 14.5% | 0.918 | -18.1% | $24.65 | baseline |
| WEEKLY-100 | 13.2% | 0.859 | -21.0% | $18.43 | **-$6.22** |
| WEEKLY-100-NOCOST | 13.4% | 0.868 | -20.9% | $19.02 | -$5.63 |

Age-65 impact: **-$1,686,769** (36% less wealth at age 65 from weekly rebalancing).

## Why Weekly Hurts

Leverage condition switches 106 times over 24yr at weekly (4.4/yr) vs 48 times monthly (2.0/yr) — more than double. Each switch catches mid-month noise that reverses within days. The monthly system correctly ignores these transients.

Transaction costs are negligible ($0.59 of $6.22 loss). The remaining $5.63 is pure signal degradation.

Signal changes per year: 30.2/yr (weekly) vs 11.8/yr (monthly). VGLT is worst offender at 7.5 score changes/year weekly — bond trend signals are too noisy at weekly timeframes.

## Crisis Performance

| Crisis | Monthly | Weekly |
|--------|---------|--------|
| GFC | +0.6% (DD -1.2%) | -1.5% (DD -2.0%) — weekly WORSE |
| COVID | -16.7% (DD -18.1%) | -16.6% (DD -17.8%) — similar |
| 2022 Bear | -12.6% (DD -13.8%) | -8.3% (DD -9.6%) — weekly better |

Weekly is better in 2022 but worse in GFC. Net 24-year effect strongly favors monthly.

## Key Insight

The Faber trend filter is a LOW-FREQUENCY signal. Checking it more frequently adds noise, not information. The circuit breaker benefits from daily monitoring because it watches for an extreme event (3/3 SMA breach). The allocation signal does NOT benefit — the score bins change for noise reasons at weekly resolution.

**Slow allocation decisions, fast emergency exits — asymmetric design confirmed correct.**

## Decision

Monthly rebalancing confirmed as optimal and final. Weekly rejected.

Final production architecture unchanged:
- Monthly allocation (month-end Faber scores)
- Daily circuit breaker (leverage exit only)

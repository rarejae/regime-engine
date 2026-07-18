---
date: 2026-04-13
experiment: V20 Directional State Transitions on V19d
status: FAILED — all variants. Directional hypothesis is inverted.
script: experiments/v20_directional/backtest.py
---

# V20: Directional Transitions — Results

## Verdict

**All V20 variants fail.** The directional hypothesis is empirically inverted: 3→2 mostly **recovers** (57% QQQ, 47% IVV next month), and 1→2 **never** recovers to score 3 (0% both assets). The directional information predicts the opposite of what happens. V19d's non-directional 70/30 at score 2 is correct.

---

## Core Metrics

| Strategy |   CAGR | Sharpe |  MaxDD | Term$1 |
|----------|-------:|-------:|-------:|-------:|
| V20-A    | 16.97% |  0.859 | -25.7% | $51.16 |
| V20-B    | 16.78% |  0.854 | -25.3% | $49.10 |
| V20-D    | 16.98% |  0.862 | -25.1% | $51.26 |
| **V19d** | 17.27% |  **0.866** | **-25.1%** | $54.60 |

V19d dominates every V20 variant on Sharpe. V20-D comes closest but still trails by -0.004 Sharpe.

---

## The Directional Hypothesis is Wrong (KEY FINDING)

### QQQ 3→2 (falling): 9 events
| Next month | Count | Pct |
|------------|------:|----:|
| →3 (recovered) | **4** | **57%** |
| →2 (stayed) | 0 | 0% |
| →1 (declined) | 1 | 14% |
| →0 (crashed) | 2 | 29% |
| **Further decline** | **3** | **43%** |

**Majority recovers.** Going defensive at 3→2 misses the 57% of cases where the score restores to 3 next month. The defensive treatment (V20-B: full cash at 3→2) is wrong ~60% of the time.

### QQQ 1→2 (rising): 4 events
| Next month | Count | Pct |
|------------|------:|----:|
| →3 (recovered) | **0** | **0%** |
| →2 (stayed) | 3 | 75% |
| →1 (declined) | 0 | 0% |
| →0 (crashed) | 1 | 25% |

**Zero recovery to full signal.** Being aggressive at 1→2 (V20-A: 100% equity) means adding equity exposure during a dead bounce. Not a single 1→2 event led to 3/3 the following month.

### IVV 3→2 (falling): 16 events — 47% recover, 27% decline
### IVV 1→2 (rising): 5 events — 0% recover, 60% crash to 0

The pattern is consistent across both assets: **falling is mostly false alarm, rising is a trap.**

---

## Why the Hypothesis Failed

The intuition "3→2 means trend is breaking, 1→2 means trend is restoring" assumes the fastest SMA (126-day) leads the others. In reality:

1. **3→2 often means a brief dip below the fastest SMA** during an intact uptrend. The 200-day and 252-day still hold. The market bounces back above the 126-day within 1-2 months in 57% of cases. The fastest SMA produces the most noise.

2. **1→2 often means a dead cat bounce** in a broken market. The price crosses above one SMA temporarily during a bear market rally, then resumes decline. The score goes 0→2→0 or 1→2→0, never reaching 3.

The multi-SMA system's directional information is the opposite of intuitive: short-term breaks are noise (3→2 recovers), short-term restores are traps (1→2 collapses).

---

## Sample Size Warning

QQQ: 9 falling events, 4 rising events. IVV: 16 falling, 5 rising. These are tiny samples. The 57% vs 43% QQQ falling split would not survive a statistical significance test. However, the consistent pattern across both assets (falling recovers, rising fails) and the consistent portfolio underperformance across all variants provides convergent evidence.

---

## Frontier Unchanged

V19d at 0.866 Sharpe, -25.1% MaxDD remains the balanced frontier point. Directional treatment adds complexity for worse performance.

---

## Cross-references

- [[experiments/V19C_FULL_UNLEVER_RESULTS]] — score-2 treatment is marginal (14% of months)
- [[experiments/V19D_GOLD_CB_RESULTS]] — V19d is the confirmed production spec

# CB Slippage Stress Report

**Generated from** packaged CB events + yfinance close cache.
**Proxy:** cost of selling one session later vs CB close (next-day return, sign-flipped for long exit).

## Events analyzed: 30

| Metric | Delay cost (bps) | 2× stress |
|--------|-----------------:|----------:|
| Mean | 24.0 | 48.1 |
| Median | -41.5 | -83.1 |
| P90 | 422.8 | 845.6 |
| Worst | 1170.9 | 2341.8 |

- CB events/year (approx): **1.13**
- Mean *adverse* delay (bps, positives only): **345.8**
- Rough portfolio drag if every CB delayed one session (weight-scaled): **~69.5 bps/year**

## Spec budget check

Live spec soft/hard CB limits: **40 / 100 bps** per event.

| Budget | Status |
|--------|--------|
| Median delay vs 40 bps soft | PASS |
| P90 vs 100 bps hard | FAIL — prefer same-day MOC / limit-into-close; next-open is not enough on crash gaps |

**Actionable finding:** P90 delay cost exceeds the 100 bps hard budget. Stage-3 live policy must prefer **sell into the CB close when possible**, not wait for the next open. Overnight gap risk is the dominant execution failure mode.

## Interpretation

- Negative delay_cost_bps means the next session *rose* — delayed sell helped (or hurt less).
- Positive means the market fell after the CB close signal — **this is the crash-morning risk**.
- Live policy (limit then market by 10:15) aims to keep realized slip near the median, not the worst.

See `slippage_stress.csv` for event-level detail.

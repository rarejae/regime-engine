# V19d intraday-MOC feasibility

**Question.** The V19d circuit breaker fires on a *close* below all 3 SMAs (126/200/252).
You can't fill at a close you don't yet know, so the honest
[execution-lag study](../v19d_execution_lag/) uses a T+1 fallback. But US market-on-close
(MOC) orders take entry until **15:50 ET**. If, at 15:50, "price below all 3 SMAs" already
predicts the 16:00 close-based signal, you can fire a MOC and **exit at the breach-day
close** — the execution-lag `t_close` mode — without look-ahead.

**Test.** Tiingo IEX intraday (2017+, the only free-tier intraday depth; IEX venue only).
Same 3 SMAs from daily *adjusted* closes; the raw 15:50 print is scaled by that day's
`adjClose/close` factor so the IAU 2021-05-24 2:1 split and all dividends cancel. For every
day we compare the close-based signal (what the backtest uses) to the 15:50 signal (what you
can execute).

## Result — 2017-01 → 2026-09, 2,437 days/ticker, 1,139 close-breach days

| Underlyer | breach days | agreement | caught | false-pos | missed |
|-----------|------------:|----------:|-------:|----------:|-------:|
| QQQ | 341 | 99.96% | 99.71% | 0 | 1 |
| IVV | 301 | 99.92% | 99.67% | 1 | 1 |
| IAU | 497 | 99.88% | 99.60% | 1 | 2 |
| **ALL** | **1,139** | **99.92%** | **99.65%** | **2** | **4** |

All 6 errors are **razor-edge crossings** (15:50→close move 0.006%–0.18%, price sitting on
the SMA). The 15:50→close move distribution: mean +0.001%, std 0.058%, 95%ile |move| 0.086%.
An SMA sits multiple percent from spot except on the exact crossing day, so a 10-minute
window essentially never flips a decisive breach.

**Conclusion.** Intraday-MOC is live-achievable. It promotes execution-lag `t_close` from
"reference" to the true **achievable base case**, and T+1 becomes the fallback for when you
miss the 15:50 cutoff (a razor-edge fraction of days).

## What it's worth (100% substitution, 2002-01 → 2026-03)

| Execution | CAGR | Sharpe | MaxDD | Aug-2015 DD |
|-----------|-----:|-------:|------:|------------:|
| prior (look-ahead, **impossible**) | 17.27% | 0.866 | −25.1% | −6.8% |
| **intraday-MOC (`t_close`, achievable)** | **14.98%** | **0.766** | **−32.7%** | **−13.3%** |
| T+1 close (missed-cutoff fallback) | 14.68% | 0.749 | −37.8% | −19.4% |
| T+1 open (worst realistic) | 14.23% | 0.730 | −40.2% | −23.7% |

- vs T+1 fallback: **+0.3pp CAGR, +5pp shallower MaxDD**, and it roughly **halves the
  flash-crash drawdown** (Aug 2015: −13.3% vs −19.4%) by exiting a day earlier.
- vs look-ahead `prior`: −2.3pp CAGR / −7.6pp deeper DD. **No execution recovers this** —
  it is the unavoidable breach-day decline you eat while holding to the close. The canonical
  17.27% / −25.1% was never achievable.

## Caveats
- IEX is one venue (~2–3% of volume); an IEX 15:50 print is not the consolidated close, but
  the below/above-SMA test only needs the side of a multi-percent gap, so this is immaterial.
- Intraday depth is 2017+ only — no GFC/2011/Aug-2015 *intraday*. Reliability is a
  microstructure fact (10 min vs a multi-percent SMA gap), so it generalizes, but pre-2017
  the honest model stays T+1.
- A real MOC fills at the official closing auction (= the backtest's close), so exit slippage
  is ~0 — the reason MOC, not a plain late-day market order, is the right instrument.

Run: `.venv/bin/python experiments/v19d_intraday_moc/backtest.py`
(free tier ~50 req/hour; 30 IEX + 3 EOD calls, cached to `_cache/`).

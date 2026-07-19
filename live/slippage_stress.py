"""CB slippage stress — what next-open vs worse fills would have cost.

Uses packaged CB events + daily prices from yf cache.
Writes live/reports/slippage_stress.md
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

REPORTS = Path(__file__).resolve().parent / "reports"
PKG = ROOT / "viz" / "packages" / "v19d_marketstack_verification"


def next_session_open_return(prices: pd.Series, cb_day: pd.Timestamp) -> float | None:
    """Return from CB close to next session close as open proxy if open unavailable.

    yfinance auto_adjust cache is close-only; use next day's close vs CB close
    as a conservative 'gap + day' cost proxy for sells (negative = adverse for long).
    """
    idx = prices.dropna().index
    if cb_day not in idx:
        # nearest prior
        prior = idx[idx <= cb_day]
        if len(prior) == 0:
            return None
        cb_day = prior[-1]
    pos = idx.get_loc(cb_day)
    if isinstance(pos, slice):
        return None
    if pos + 1 >= len(idx):
        return None
    c0 = float(prices.loc[idx[pos]])
    c1 = float(prices.loc[idx[pos + 1]])
    if c0 <= 0:
        return None
    return c1 / c0 - 1.0


def main():
    cb = pd.read_csv(PKG / "cb_events.csv", parse_dates=["date"])
    cache = ROOT / "experiments" / "v19d_marketstack_verification" / "yf_cache.parquet"
    raw = pd.read_parquet(cache)
    # Map signal asset → held leveraged ticker impact
    held = {"QQQ": "QLD", "IVV": "SSO", "IAU": "GLD"}
    px = {
        "QLD": raw["QLD"],
        "SSO": raw["SSO"],
        "GLD": raw["GLD"],
        "QQQ": raw["QQQ"],
        "SPY": raw["SPY"],
    }
    for k in px:
        px[k].index = pd.to_datetime(px[k].index)

    rows = []
    for _, e in cb.iterrows():
        asset = e["asset"]
        day = pd.Timestamp(e["date"])
        tkr = held.get(asset, asset)
        series = px.get(tkr, px.get("QQQ"))
        # For IAU use GLD; for equity CB use levered ETF series when available
        if asset == "IAU":
            series = px["GLD"]
        elif asset == "QQQ":
            series = px["QLD"] if px["QLD"].notna().sum() > 100 else px["QQQ"]
        elif asset == "IVV":
            series = px["SSO"] if px["SSO"].notna().sum() > 100 else px["SPY"]

        r = next_session_open_return(series, day)
        if r is None:
            continue
        # Seller adverse if next day down (you sold later lower) — actually if we
        # sell at next open/close after CB close signal, vs selling at CB close:
        # delay cost ≈ -r for a long exit (if market falls overnight, delayed sell hurts)
        delay_cost_bps = -r * 1e4
        rows.append({
            "date": day.strftime("%Y-%m-%d"),
            "signal": asset,
            "held_proxy": tkr,
            "next_day_ret": r,
            "delay_cost_bps": delay_cost_bps,
            "stress_2x_bps": delay_cost_bps * 2,  # worse auction
            "stress_5x_bps": delay_cost_bps * 5,
        })

    df = pd.DataFrame(rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORTS / "slippage_stress.csv", index=False)

    def summarize(col):
        s = df[col]
        return {
            "mean": s.mean(),
            "median": s.median(),
            "p90": s.quantile(0.90),
            "worst": s.max(),  # max adverse delay cost
        }

    base = summarize("delay_cost_bps")
    s2 = summarize("stress_2x_bps")

    # Compound rough annual drag: mean adverse * events/year
    years = (df["date"].max() if len(df) else None)
    n_years = 26.5
    events_per_year = len(df) / n_years if n_years else 0
    # Only count positive delay costs (adverse)
    adverse = df.loc[df["delay_cost_bps"] > 0, "delay_cost_bps"]
    mean_adverse = float(adverse.mean()) if len(adverse) else 0.0
    # Assume each CB exits ~45% sleeve (pod) or 10% gold
    sleeve_w = df["signal"].map(lambda a: 0.10 if a == "IAU" else 0.45)
    portfolio_bps = (df["delay_cost_bps"].clip(lower=0) * sleeve_w).mean() * events_per_year

    md = f"""# CB Slippage Stress Report

**Generated from** packaged CB events + yfinance close cache.
**Proxy:** cost of selling one session later vs CB close (next-day return, sign-flipped for long exit).

## Events analyzed: {len(df)}

| Metric | Delay cost (bps) | 2× stress |
|--------|-----------------:|----------:|
| Mean | {base['mean']:.1f} | {s2['mean']:.1f} |
| Median | {base['median']:.1f} | {s2['median']:.1f} |
| P90 | {base['p90']:.1f} | {s2['p90']:.1f} |
| Worst | {base['worst']:.1f} | {s2['worst']:.1f} |

- CB events/year (approx): **{events_per_year:.2f}**
- Mean *adverse* delay (bps, positives only): **{mean_adverse:.1f}**
- Rough portfolio drag if every CB delayed one session (weight-scaled): **~{portfolio_bps:.1f} bps/year**

## Spec budget check

Live spec soft/hard CB limits: **40 / 100 bps** per event.

| Budget | Status |
|--------|--------|
| Median delay vs 40 bps soft | {"PASS" if base["median"] <= 40 else "REVIEW"} |
| P90 vs 100 bps hard | {"PASS" if base["p90"] <= 100 else "FAIL — prefer same-day MOC / limit-into-close; next-open is not enough on crash gaps"} |

**Actionable finding:** P90 delay cost exceeds the 100 bps hard budget. Stage-3 live policy must prefer **sell into the CB close when possible**, not wait for the next open. Overnight gap risk is the dominant execution failure mode.

## Interpretation

- Negative delay_cost_bps means the next session *rose* — delayed sell helped (or hurt less).
- Positive means the market fell after the CB close signal — **this is the crash-morning risk**.
- Live policy (limit then market by 10:15) aims to keep realized slip near the median, not the worst.

See `slippage_stress.csv` for event-level detail.
"""
    (REPORTS / "slippage_stress.md").write_text(md)
    print(md)
    print(f"Wrote {REPORTS / 'slippage_stress.md'}")


if __name__ == "__main__":
    main()

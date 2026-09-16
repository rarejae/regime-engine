"""V19d leverage (substitution) sweep under REALISTIC T+1-close execution.

Only the substitution level changes (100/80/60/40%); every other rule is identical.
sub = fraction of a full-signal sleeve held in the 2x ETF (rest in the 1x underlying),
so sleeve leverage = 1 + sub. All runs use the T+1-close circuit-breaker fill.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv; load_dotenv()

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

from experiments.v11_beta_scaled.backtest import load_data, cagr, max_dd
from experiments.v19d_execution_lag.backtest import run_v19d_lag, metrics, START_DATE, END_DATE

CRISES = [("GFC 07-09", "2007-11-01", "2009-03-31"), ("COVID 2020", "2020-02-01", "2020-04-30"),
          ("2015 Aug", "2015-08-01", "2015-09-30"), ("2022 bear", "2022-01-01", "2022-12-31")]


def main():
    print("=" * 104)
    print("  V19d LEVERAGE SWEEP — realistic T+1-close execution (2002-01 → 2026-03)")
    print("=" * 104)
    print("  Loading data…")
    daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, *_ = load_data()

    subs = [1.0, 0.8, 0.6, 0.4]
    runs = {}
    for sub in subs:
        s, cb = run_v19d_lag(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                             START_DATE, exit_mode="t1_close", sub=sub)
        runs[sub] = (metrics(s), s)

    # benchmarks (buy-and-hold, no CB → execution-neutral)
    qqq = daily_ret["QQQ"].loc[START_DATE:END_DATE].dropna()
    ivv = daily_ret["IVV"].loc[START_DATE:END_DATE].dropna()

    print(f"\n  {'Substitution':<16}{'SleeveLev':>10}{'CAGR':>8}{'Vol':>8}{'Sharpe':>8}{'Sortino':>8}"
          f"{'MaxDD':>8}{'Calmar':>8}{'Term$1':>9}{'DCA':>9}")
    print("  " + "-" * 98)
    for sub in subs:
        m = runs[sub][0]
        print(f"  {int(sub*100)}% sub{'':<9}{1+sub:>9.1f}x{m['CAGR']:>8.2%}{m['Vol']:>8.2%}{m['Sharpe']:>8.3f}"
              f"{m['Sortino']:>8.3f}{m['MaxDD']:>8.1%}{m['Calmar']:>8.2f}${m['Terminal']:>8.2f}${m['DCA']/1e6:>7.2f}M")
    print("  " + "-" * 98)
    for nm, s in [("QQQ B&H", qqq), ("IVV B&H", ivv)]:
        m = metrics(s)
        print(f"  {nm:<16}{'1.0x':>10}{m['CAGR']:>8.2%}{m['Vol']:>8.2%}{m['Sharpe']:>8.3f}"
              f"{m['Sortino']:>8.3f}{m['MaxDD']:>8.1%}{m['Calmar']:>8.2f}${m['Terminal']:>8.2f}${m['DCA']/1e6:>7.2f}M")

    print(f"\n  Marginal trade-off stepping DOWN in leverage (realistic):")
    for i in range(len(subs) - 1):
        a, b = runs[subs[i]][0], runs[subs[i+1]][0]
        print(f"    {int(subs[i]*100)}%→{int(subs[i+1]*100)}%:  CAGR {a['CAGR']-b['CAGR']:+.2%}"
              f"   MaxDD {b['MaxDD']-a['MaxDD']:+.1f}pp better   Sharpe {b['Sharpe']-a['Sharpe']:+.3f}"
              .replace("pp better", "pp"))

    print(f"\n  CRISIS MAX DRAWDOWN by substitution:")
    print(f"    {'Crisis':<14}" + "".join(f"{int(s*100)}%sub".rjust(9) for s in subs))
    for name, c0, c1 in CRISES:
        row = f"    {name:<14}"
        for sub in subs:
            s = runs[sub][1]
            w = s[(s.index >= c0) & (s.index <= c1)]
            row += f"{max_dd(w):>9.1%}" if len(w) > 5 else f"{'n/a':>9}"
        print(row)
    print("\n  Done.")


if __name__ == "__main__":
    main()

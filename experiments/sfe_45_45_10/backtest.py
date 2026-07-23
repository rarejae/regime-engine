"""SFE-45/45/10 — same a priori Faber rules as SFE, gold sleeve capped at 10%.

Only change vs equal-weight SFE: fixed sleeves 45% QLD / 45% SSO / 10% GLD.
Signal machinery identical (classic 10-mo Faber, binary on/off, cash when off,
monthly only, no CB / guards / score tiers).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from experiments.sfe_simple_faber_equal.backtest import (
    W_45_45_10,
    W_EQUAL,
    YF_CACHE,
    assert_alignment,
    bh,
    crisis_dd,
    faber_on,
    load_prices,
    load_rfr,
    metrics,
    month_ends,
    prow,
    run_sfe,
    sixty_forty,
)


def main():
    print("=" * 120)
    print("  SFE 45/45/10 — GOLD CAPPED AT 10% (A PRIORI WEIGHT VARIANT)")
    print("=" * 120)

    px = load_prices()
    rfr = load_rfr(px.index)
    print(f"\n  Data: {px.index.min().date()} → {px.index.max().date()}")

    me = pd.DatetimeIndex(month_ends(px.index))
    qqq_m = px["QQQ"].reindex(me).dropna()
    on = faber_on(qqq_m)
    ok = True
    for i in range(11, min(30, len(qqq_m))):
        sma = qqq_m.iloc[i - 10 : i].mean()
        expected = bool(qqq_m.iloc[i - 1] > sma)
        if bool(on.iloc[i - 1]) != expected:
            ok = False
            break

    sfe_eq, log_eq = run_sfe(px, rfr, weights=W_EQUAL)
    sfe_10, log_10 = run_sfe(px, rfr, weights=W_45_45_10)
    assert_alignment(log_eq, ok)
    assert_alignment(log_10, ok)
    print("  Signal alignment: PASS")
    print(f"  Window: {sfe_10.index.min().date()} → {sfe_10.index.max().date()}")

    start = sfe_10.index.min()
    series = {
        "SFE 45/45/10": sfe_10,
        "SFE 1/3 equal": sfe_eq,
        "QQQ B&H": bh(px["QQQ"], start).reindex(sfe_10.index).fillna(0.0),
        "SPY B&H": bh(px["SPY"], start).reindex(sfe_10.index).fillna(0.0),
        "60/40": sixty_forty(px["SPY"], rfr, start).reindex(sfe_10.index).fillna(0.0),
    }

    print("\n" + "=" * 120)
    print("  CORE PERFORMANCE")
    print("=" * 120)
    print(
        f"  {'Strategy':<22} {'CAGR':>7} {'Vol':>7} {'Sharpe':>7} {'Sortino':>8} "
        f"{'MaxDD':>7} {'Calmar':>7} {'Term$1':>9} {'DCA':>9}"
    )
    print("  " + "-" * 100)
    mets = {k: metrics(v) for k, v in series.items()}
    for k, m in mets.items():
        print(prow(k, m))

    print("\n" + "=" * 120)
    print("  START-DATE SENSITIVITY (SFE variants)")
    print("=" * 120)
    print(f"  {'Window':<22} {'45/45/10 CAGR':>14} {'Sharpe':>7} {'MaxDD':>7} | {'1/3 CAGR':>10} {'Sharpe':>7} {'MaxDD':>7}")
    for name, st in [
        ("Full", start),
        ("2002-01", "2002-01-01"),
        ("2006-07", "2006-07-01"),
        ("2013-01", "2013-01-01"),
    ]:
        a = metrics(sfe_10.loc[st:])
        b = metrics(sfe_eq.loc[st:])
        print(
            f"  {name:<22} {a['CAGR']:>13.2%} {a['Sharpe']:>7.3f} {a['MaxDD']:>7.1%} | "
            f"{b['CAGR']:>9.2%} {b['Sharpe']:>7.3f} {b['MaxDD']:>7.1%}"
        )

    print("\n" + "=" * 120)
    print("  CRISIS DRAWDOWNS")
    print("=" * 120)
    crises = [
        ("Dot-com 00-02", "2000-03-01", "2002-10-31"),
        ("GFC 07-09", "2007-10-01", "2009-03-31"),
        ("COVID 2020", "2020-02-01", "2020-04-30"),
        ("2022 Bear", "2022-01-01", "2022-12-31"),
    ]
    print(f"  {'Crisis':<16} " + " ".join(f"{k:>14}" for k in series))
    for name, a, b in crises:
        row = f"  {name:<16}"
        for k, s in series.items():
            d = crisis_dd(s, a, b)
            row += f" {d:>13.1%}" if pd.notna(d) else f" {'n/a':>13}"
        print(row)

    print("\n" + "=" * 120)
    print("  EXPOSURE")
    print("=" * 120)
    for label, log in [("45/45/10", log_10), ("1/3 equal", log_eq)]:
        eff = log["w_qld"] * 2 + log["w_sso"] * 2
        print(
            f"  {label}: mean cash {log['w_cash'].mean():.1%} | "
            f"mean eff equity {eff.mean():.1%} | "
            f"mean gold {log['w_gld'].mean():.1%} | "
            f"max gold {log['w_gld'].max():.0%}"
        )

    # 2026 YTD
    print("\n  2026 YTD:")
    for k, s in series.items():
        y = s.loc["2026-01-01":]
        if len(y):
            print(f"    {k:<18} {(1 + y).prod() - 1:>7.2%}")

    out = ROOT / "research/data"
    out.mkdir(parents=True, exist_ok=True)
    sfe_10.to_csv(out / "sfe_454510_daily_returns.csv", header=["ret"])
    log_10.to_csv(out / "sfe_454510_monthly_allocations.csv", index=False)
    print(f"\n  Saved: research/data/sfe_454510_daily_returns.csv")
    print(f"  Saved: research/data/sfe_454510_monthly_allocations.csv")
    print("\n" + "=" * 120)
    print("  DONE")
    print("=" * 120)
    return mets, sfe_10, log_10


if __name__ == "__main__":
    main()

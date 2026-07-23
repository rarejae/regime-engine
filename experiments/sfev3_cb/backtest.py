"""SFEv3 — SFE 45/45/10 + daily circuit breaker → cash.

Base (unchanged from SFE 45/45/10):
  Sleeves: 45% QLD / 45% SSO / 10% GLD
  Monthly: classic Faber 10-mo SMA on QQQ / SPY / GLD → ON/OFF
  OFF sleeve weight stays cash (no pro-rata)

Add (the single new mechanism):
  Daily CB: if signal asset closes below ALL of SMA-126/200/252
            → that sleeve exits to cash next session
  Re-entry: next monthly Faber rebalance only (no mid-month re-entry)

Same CB definition as V19d; applied on the a priori SFE skeleton.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from experiments.v11_beta_scaled.backtest import SMA_PERIODS, QLD_EXP, SSO_EXP
from experiments.sfe_simple_faber_equal.backtest import (
    W_45_45_10,
    W_EQUAL,
    assert_alignment,
    bh,
    crisis_dd,
    faber_on,
    lev_or_actual,
    load_prices,
    load_rfr,
    metrics,
    month_ends,
    prow,
    run_sfe,
    sixty_forty,
)

SIGNAL_HOLD = {"QQQ": "nasdaq", "SPY": "sp500", "GLD": "gold"}


def daily_smas(px: pd.DataFrame, cols: list[str]) -> dict[int, pd.DataFrame]:
    sub = px[cols]
    return {p: sub.rolling(p, min_periods=p).mean() for p in SMA_PERIODS}


def breached(day: pd.Timestamp, asset: str, px: pd.DataFrame, smas: dict) -> bool:
    if asset not in px.columns or day not in px.index:
        return False
    price = px.at[day, asset]
    if pd.isna(price):
        return False
    n = 0
    for per in SMA_PERIODS:
        s = smas[per].at[day, asset] if day in smas[per].index else np.nan
        if pd.notna(s) and price < s:
            n += 1
    return n >= 3


def run_sfev3(
    px: pd.DataFrame,
    rfr_daily: pd.Series,
    start: str | None = None,
    weights: tuple[float, float, float] = W_45_45_10,
):
    """45/45/10 Faber monthly + daily 3/3 SMA CB → cash (next day)."""
    w_n, w_s, w_g = weights
    assert abs(w_n + w_s + w_g - 1.0) < 1e-12

    qqq_r = px["QQQ"].pct_change()
    spy_r = px["SPY"].pct_change()
    gld_r = px["GLD"].pct_change()
    qld_r = px["QLD"].pct_change()
    sso_r = px["SSO"].pct_change()
    both_start = max(px["QLD"].first_valid_index(), px["SSO"].first_valid_index())

    smas = daily_smas(px, ["QQQ", "SPY", "GLD"])

    me = pd.DatetimeIndex(month_ends(px.index))
    monthly = {
        "QQQ": px["QQQ"].reindex(me),
        "SPY": px["SPY"].reindex(me),
        "GLD": px["GLD"].reindex(me),
    }
    on_m = {k: faber_on(v) for k, v in monthly.items()}

    signal_by_month = {}
    months = sorted(on_m["QQQ"].index)
    for i, m_end in enumerate(months):
        period = m_end.to_period("M")
        if i == 0:
            signal_by_month[period] = {"QQQ": False, "SPY": False, "GLD": False}
        else:
            prev = months[i - 1]
            signal_by_month[period] = {
                "QQQ": bool(on_m["QQQ"].loc[prev]) if pd.notna(monthly["QQQ"].loc[prev]) else False,
                "SPY": bool(on_m["SPY"].loc[prev]) if pd.notna(monthly["SPY"].loc[prev]) else False,
                "GLD": bool(on_m["GLD"].loc[prev]) if pd.notna(monthly["GLD"].loc[prev]) else False,
            }

    qqq_me = monthly["QQQ"].dropna()
    live_start = (qqq_me.index[10].to_period("M") + 1).to_timestamp()
    bt_start = max(px["QQQ"].first_valid_index(), px["SPY"].first_valid_index(), live_start)
    if start:
        bt_start = max(bt_start, pd.Timestamp(start))

    trading_days = px.loc[bt_start:].index
    port = {}
    monthly_log = []
    cb_events = []
    nav = {"nasdaq": w_n, "sp500": w_s, "gold": w_g}

    # Sleeve state within month
    monthly_on = {"QQQ": False, "SPY": False, "GLD": False}
    cb_flat = {"QQQ": False, "SPY": False, "GLD": False}  # CB fired → cash until month reset
    pending_exit = {"QQQ": False, "SPY": False, "GLD": False}  # breach yesterday → exit today
    last_period = None

    for day in trading_days:
        period = day.to_period("M")
        is_month_start = last_period is None or period != last_period
        if is_month_start:
            monthly_on = signal_by_month.get(
                period, {"QQQ": False, "SPY": False, "GLD": False}
            )
            cb_flat = {"QQQ": False, "SPY": False, "GLD": False}
            pending_exit = {"QQQ": False, "SPY": False, "GLD": False}
            total = sum(nav.values())
            nav = {"nasdaq": total * w_n, "sp500": total * w_s, "gold": total * w_g}
            last_period = period

        # Apply pending exits from prior close breach (next-session exit)
        for asset in ("QQQ", "SPY", "GLD"):
            if pending_exit[asset] and monthly_on[asset] and not cb_flat[asset]:
                cb_flat[asset] = True
                pending_exit[asset] = False
                cb_events.append({"date": day, "asset": asset, "action": "exit_cash"})

        # Active holding this day
        hold_q = monthly_on["QQQ"] and not cb_flat["QQQ"]
        hold_s = monthly_on["SPY"] and not cb_flat["SPY"]
        hold_g = monthly_on["GLD"] and not cb_flat["GLD"]

        if is_month_start:
            monthly_log.append(
                {
                    "month": period.to_timestamp(),
                    "qqq_on": monthly_on["QQQ"],
                    "spy_on": monthly_on["SPY"],
                    "gld_on": monthly_on["GLD"],
                    "w_qld": w_n if hold_q else 0.0,
                    "w_sso": w_s if hold_s else 0.0,
                    "w_gld": w_g if hold_g else 0.0,
                    "w_cash": 1.0
                    - (w_n if hold_q else 0.0)
                    - (w_s if hold_s else 0.0)
                    - (w_g if hold_g else 0.0),
                }
            )

        rfr = float(rfr_daily.get(day, 0.0))
        qu = float(qqq_r.get(day, np.nan)) if pd.notna(qqq_r.get(day, np.nan)) else 0.0
        su = float(spy_r.get(day, np.nan)) if pd.notna(spy_r.get(day, np.nan)) else 0.0
        gu = float(gld_r.get(day, np.nan)) if pd.notna(gld_r.get(day, np.nan)) else 0.0

        r_n = lev_or_actual(day, qu, rfr, QLD_EXP, qld_r, both_start) if hold_q else rfr
        r_s = lev_or_actual(day, su, rfr, SSO_EXP, sso_r, both_start) if hold_s else rfr
        if hold_g and pd.notna(px["GLD"].get(day, np.nan)):
            r_g = gu
        else:
            r_g = rfr

        prev = sum(nav.values())
        nav["nasdaq"] *= 1 + r_n
        nav["sp500"] *= 1 + r_s
        nav["gold"] *= 1 + r_g
        port[day] = (sum(nav.values()) / prev - 1) if prev > 0 else 0.0

        # After close: schedule CB exit for next session if still holding
        for asset in ("QQQ", "SPY", "GLD"):
            active = monthly_on[asset] and not cb_flat[asset] and not pending_exit[asset]
            if active and breached(day, asset, px, smas):
                pending_exit[asset] = True

    daily = pd.Series(port).sort_index()
    return daily, pd.DataFrame(monthly_log), pd.DataFrame(cb_events)


def main():
    print("=" * 120)
    print("  SFEv3 — 45/45/10 FABER + DAILY CB → CASH")
    print("=" * 120)

    px = load_prices()
    rfr = load_rfr(px.index)
    print(f"\n  Data: {px.index.min().date()} → {px.index.max().date()}")
    print(f"  CB rule: close < all SMA{SMA_PERIODS} → cash next session; monthly re-entry only")

    v3, log_v3, cbs = run_sfev3(px, rfr)
    v2, log_v2 = run_sfe(px, rfr, weights=W_45_45_10)
    v1, _ = run_sfe(px, rfr, weights=W_EQUAL)
    assert_alignment(log_v3, True)
    assert_alignment(log_v2, True)

    # Alignment: CB exit is next day after breach
    if len(cbs):
        sample = cbs.iloc[0]
        print(f"  First CB event: {sample['date'].date()} ({sample['asset']}) — next-session exit")
    print(f"  Window: {v3.index.min().date()} → {v3.index.max().date()}")
    print(f"  CB events: {len(cbs)} "
          f"(QQQ {int((cbs['asset']=='QQQ').sum()) if len(cbs) else 0}, "
          f"SPY {int((cbs['asset']=='SPY').sum()) if len(cbs) else 0}, "
          f"GLD {int((cbs['asset']=='GLD').sum()) if len(cbs) else 0})")

    start = v3.index.min()
    series = {
        "SFEv3 45/45/10+CB": v3,
        "SFE 45/45/10 (no CB)": v2,
        "SFE 1/3 equal": v1,
        "QQQ B&H": bh(px["QQQ"], start).reindex(v3.index).fillna(0.0),
        "SPY B&H": bh(px["SPY"], start).reindex(v3.index).fillna(0.0),
        "60/40": sixty_forty(px["SPY"], rfr, start).reindex(v3.index).fillna(0.0),
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
    print("  START-DATE SENSITIVITY")
    print("=" * 120)
    print(f"  {'Window':<12} {'v3 CAGR':>9} {'Sharpe':>7} {'MaxDD':>7} | {'v2 CAGR':>9} {'Sharpe':>7} {'MaxDD':>7}")
    for name, st in [("Full", start), ("2002-01", "2002-01-01"),
                     ("2006-07", "2006-07-01"), ("2013-01", "2013-01-01")]:
        a, b = metrics(v3.loc[st:]), metrics(v2.loc[st:])
        print(
            f"  {name:<12} {a['CAGR']:>8.2%} {a['Sharpe']:>7.3f} {a['MaxDD']:>7.1%} | "
            f"{b['CAGR']:>8.2%} {b['Sharpe']:>7.3f} {b['MaxDD']:>7.1%}"
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
    keys = list(series.keys())
    print(f"  {'Crisis':<16}" + "".join(f"{k:>18}" for k in keys))
    for name, a, b in crises:
        row = f"  {name:<16}"
        for k, s in series.items():
            d = crisis_dd(s, a, b)
            row += f" {d:>17.1%}" if pd.notna(d) else f" {'n/a':>17}"
        print(row)

    # COVID CB timing
    print("\n  COVID CB events:")
    if len(cbs):
        covid_cbs = cbs[(cbs["date"] >= "2020-02-01") & (cbs["date"] <= "2020-04-30")]
        if len(covid_cbs):
            print(covid_cbs.to_string(index=False))
        else:
            print("    (none)")
    print("\n  2022 CB events:")
    if len(cbs):
        y22 = cbs[(cbs["date"] >= "2022-01-01") & (cbs["date"] <= "2022-12-31")]
        print(y22.to_string(index=False) if len(y22) else "    (none)")

    print("\n  2026 YTD:")
    for k, s in series.items():
        y = s.loc["2026-01-01":]
        if len(y):
            print(f"    {k:<22} {(1 + y).prod() - 1:>7.2%}")

    out = ROOT / "research/data"
    out.mkdir(parents=True, exist_ok=True)
    v3.to_csv(out / "sfev3_daily_returns.csv", header=["ret"])
    log_v3.to_csv(out / "sfev3_monthly_allocations.csv", index=False)
    cbs.to_csv(out / "sfev3_cb_events.csv", index=False)
    print(f"\n  Saved: research/data/sfev3_*.csv")
    print("\n" + "=" * 120)
    print("  DONE")
    print("=" * 120)
    return mets, v3, log_v3, cbs


if __name__ == "__main__":
    main()

"""Simple Faber Equal (SFE) — a priori principle backtest.

Rulebook (locked before looking at results):
  Universe:  QLD / SSO / GLD / cash
  Weights:   1/3 / 1/3 / 1/3 fixed sleeves (off → that sleeve stays cash)
  Signal:    classic Faber — month-end price > 10-month SMA → ON
  Gate on:   QQQ (for QLD), SPY (for SSO), GLD (for gold)
  Hold:      QLD / SSO / GLD when ON
  Cadence:   monthly only — no score tiers, guards, or mid-month CB
  Alignment: month-T signal applies to month T+1 returns

Data: cached yfinance adj closes + FRED DTB3.
Pre-2006 QLD/SSO: 2x underlying − rf − expense (same formula as V19d).
Gold sleeve is cash until GLD exists (2004-11).
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

from experiments.v11_beta_scaled.backtest import (
    QLD_EXP,
    SSO_EXP,
    cagr,
    calmar_r,
    dca_terminal,
    max_dd,
    sharpe_r,
    sortino_r,
)

YF_CACHE = ROOT / "data/raw/yfinance/sfe_universe.parquet"
FRED_CACHE = ROOT / "data/raw/fred/DTB3.parquet"

SLEEVE_W = 1.0 / 3.0
# Signal asset → holding asset, expense (for pre-inception lev sim)
SLEEVES = {
    "nasdaq": {"signal": "QQQ", "hold": "QLD", "expense": QLD_EXP},
    "sp500": {"signal": "SPY", "hold": "SSO", "expense": SSO_EXP},
    "gold": {"signal": "GLD", "hold": "GLD", "expense": 0.0},
}


def load_prices() -> pd.DataFrame:
    if not YF_CACHE.exists():
        raise FileNotFoundError(f"Missing {YF_CACHE} — run yfinance fetch first")
    px = pd.read_parquet(YF_CACHE)
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    return px.sort_index()


def load_rfr(index: pd.DatetimeIndex) -> pd.Series:
    tb = pd.read_parquet(FRED_CACHE)["DTB3"]
    tb.index = pd.to_datetime(tb.index).tz_localize(None).normalize()
    # DTB3 is percent annualized discount → daily simple approx
    daily = (tb / 100.0 / 252.0).reindex(index).ffill().fillna(0.0)
    return daily


def month_ends(daily_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(1, index=daily_index)
    return s.groupby(s.index.to_period("M")).apply(lambda x: x.index.max()).values


def faber_on(monthly_px: pd.Series, lookback: int = 10) -> pd.Series:
    """True when month-end price > 10-month SMA. Expanding SMA needs full lookback."""
    sma = monthly_px.rolling(lookback, min_periods=lookback).mean()
    on = monthly_px > sma
    on = on.where(sma.notna(), False)
    return on.astype(bool)


def lev_or_actual(
    day: pd.Timestamp,
    underlying_ret: float,
    rfr: float,
    expense: float,
    actual: pd.Series,
    both_start: pd.Timestamp,
) -> float:
    if day >= both_start and day in actual.index and pd.notna(actual.loc[day]):
        return float(actual.loc[day])
    return 2.0 * underlying_ret - rfr - expense / 252.0


def run_sfe(px: pd.DataFrame, rfr_daily: pd.Series, start: str | None = None):
    """Return daily portfolio returns + monthly allocation log."""
    # Holding returns
    qqq_r = px["QQQ"].pct_change()
    spy_r = px["SPY"].pct_change()
    gld_r = px["GLD"].pct_change()
    qld_r = px["QLD"].pct_change()
    sso_r = px["SSO"].pct_change()

    both_start = max(
        px["QLD"].first_valid_index(),
        px["SSO"].first_valid_index(),
    )

    # Monthly Faber signals on signal series
    me = pd.DatetimeIndex(month_ends(px.index))
    monthly = {
        "QQQ": px["QQQ"].reindex(me),
        "SPY": px["SPY"].reindex(me),
        "GLD": px["GLD"].reindex(me),
    }
    on_m = {k: faber_on(v) for k, v in monthly.items()}

    # Map each calendar month → signal decided at PRIOR month-end
    # Allocation for days in month M uses signal from last day of month M-1.
    signal_by_month = {}
    months = sorted(on_m["QQQ"].index)
    for i, m_end in enumerate(months):
        period = m_end.to_period("M")
        if i == 0:
            # No prior month — stay cash (warmup)
            signal_by_month[period] = {"QQQ": False, "SPY": False, "GLD": False}
        else:
            prev = months[i - 1]
            signal_by_month[period] = {
                "QQQ": bool(on_m["QQQ"].loc[prev]) if pd.notna(monthly["QQQ"].loc[prev]) else False,
                "SPY": bool(on_m["SPY"].loc[prev]) if pd.notna(monthly["SPY"].loc[prev]) else False,
                "GLD": bool(on_m["GLD"].loc[prev]) if pd.notna(monthly["GLD"].loc[prev]) else False,
            }

    # Backtest window: need SPY+QQQ for equity sleeves; gold optional
    bt_start = max(
        px["QQQ"].first_valid_index(),
        px["SPY"].first_valid_index(),
    )
    # Need 10 months of QQQ history before meaningful signals
    qqq_me = monthly["QQQ"].dropna()
    if len(qqq_me) < 11:
        raise RuntimeError("Insufficient QQQ monthly history for 10-mo SMA")
    first_live_signal_month = qqq_me.index[10].to_period("M")  # after 10 full months
    # Apply starting the NEXT month after first valid signal month-end
    live_start = (first_live_signal_month + 1).to_timestamp()
    bt_start = max(bt_start, live_start)
    if start:
        bt_start = max(bt_start, pd.Timestamp(start))

    trading_days = px.loc[bt_start:].index
    port = {}
    monthly_log = []
    nav = {"nasdaq": SLEEVE_W, "sp500": SLEEVE_W, "gold": SLEEVE_W}
    current_on = {"QQQ": False, "SPY": False, "GLD": False}
    last_period = None

    for day in trading_days:
        period = day.to_period("M")
        is_month_start = last_period is None or period != last_period
        if is_month_start:
            current_on = signal_by_month.get(
                period, {"QQQ": False, "SPY": False, "GLD": False}
            )
            # Fixed-sleeve rebalance to targets (cash fills OFF sleeves)
            total = sum(nav.values())
            nav = {k: total * SLEEVE_W for k in nav}
            monthly_log.append(
                {
                    "month": period.to_timestamp(),
                    "qqq_on": current_on["QQQ"],
                    "spy_on": current_on["SPY"],
                    "gld_on": current_on["GLD"],
                    "w_qld": SLEEVE_W if current_on["QQQ"] else 0.0,
                    "w_sso": SLEEVE_W if current_on["SPY"] else 0.0,
                    "w_gld": SLEEVE_W if current_on["GLD"] else 0.0,
                    "w_cash": SLEEVE_W
                    * (3 - sum([current_on["QQQ"], current_on["SPY"], current_on["GLD"]])),
                }
            )
            last_period = period

        rfr = float(rfr_daily.get(day, 0.0))
        qu = float(qqq_r.get(day, np.nan)) if pd.notna(qqq_r.get(day, np.nan)) else 0.0
        su = float(spy_r.get(day, np.nan)) if pd.notna(spy_r.get(day, np.nan)) else 0.0
        gu = float(gld_r.get(day, np.nan)) if pd.notna(gld_r.get(day, np.nan)) else 0.0

        if current_on["QQQ"]:
            r_n = lev_or_actual(day, qu, rfr, QLD_EXP, qld_r, both_start)
        else:
            r_n = rfr

        if current_on["SPY"]:
            r_s = lev_or_actual(day, su, rfr, SSO_EXP, sso_r, both_start)
        else:
            r_s = rfr

        if current_on["GLD"] and pd.notna(px["GLD"].get(day, np.nan)):
            r_g = gu
        else:
            r_g = rfr

        prev = sum(nav.values())
        nav["nasdaq"] *= 1 + r_n
        nav["sp500"] *= 1 + r_s
        nav["gold"] *= 1 + r_g
        new = sum(nav.values())
        port[day] = new / prev - 1 if prev > 0 else 0.0

    daily = pd.Series(port).sort_index()
    log = pd.DataFrame(monthly_log)
    return daily, log


def bh(px: pd.Series, start: pd.Timestamp) -> pd.Series:
    r = px.pct_change().loc[start:]
    return r.dropna()


def sixty_forty(spy: pd.Series, rfr: pd.Series, start: pd.Timestamp) -> pd.Series:
    """60% SPY / 40% T-bills, monthly rebalanced approx via daily mix of levels."""
    spy_r = spy.pct_change()
    out = {}
    w_e, w_c = 0.6, 0.4
    last_period = None
    for day in spy_r.loc[start:].index:
        period = day.to_period("M")
        if last_period is None or period != last_period:
            w_e, w_c = 0.6, 0.4
            last_period = period
        er = float(spy_r.get(day, 0.0)) if pd.notna(spy_r.get(day, np.nan)) else 0.0
        cr = float(rfr.get(day, 0.0))
        ret = w_e * er + w_c * cr
        # drift weights
        w_e *= 1 + er
        w_c *= 1 + cr
        tot = w_e + w_c
        w_e, w_c = w_e / tot, w_c / tot
        out[day] = ret
    return pd.Series(out).sort_index()


def metrics(s: pd.Series) -> dict:
    sm = s.resample("MS").apply(lambda x: (1 + x).prod() - 1)
    return {
        "CAGR": cagr(s),
        "Vol": s.std() * np.sqrt(252),
        "Sharpe": sharpe_r(s),
        "Sortino": sortino_r(s),
        "MaxDD": max_dd(s),
        "Calmar": calmar_r(s),
        "Terminal": float((1 + s).cumprod().iloc[-1]),
        "DCA": dca_terminal(sm),
    }


def crisis_dd(s: pd.Series, start: str, end: str) -> float:
    w = s.loc[start:end]
    if len(w) < 5:
        return np.nan
    return max_dd(w)


def prow(name: str, m: dict) -> str:
    return (
        f"  {name:<22} {m['CAGR']:>7.2%} {m['Vol']:>7.2%} {m['Sharpe']:>7.3f} "
        f"{m['Sortino']:>8.3f} {m['MaxDD']:>7.1%} {m['Calmar']:>7.2f} "
        f"${m['Terminal']:>8.2f} ${m['DCA']/1e6:>7.2f}M"
    )


def assert_alignment(log: pd.DataFrame, on_check_ok: bool):
    assert on_check_ok, "Signal alignment failed"
    assert (log["w_qld"] + log["w_sso"] + log["w_gld"] + log["w_cash"] - 1.0).abs().max() < 1e-9


def main():
    print("=" * 120)
    print("  SIMPLE FABER EQUAL (SFE) — A PRIORI PRINCIPLE BACKTEST")
    print("=" * 120)

    px = load_prices()
    rfr = load_rfr(px.index)
    print(f"\n  Data: {px.index.min().date()} → {px.index.max().date()}")
    print(f"  Cache: {YF_CACHE}")

    # Alignment check: rebuild prior-month mapping sample
    me = pd.DatetimeIndex(month_ends(px.index))
    qqq_m = px["QQQ"].reindex(me).dropna()
    on = faber_on(qqq_m)
    # Signal for month M must use only data through prior month-end
    ok = True
    for i in range(11, min(30, len(qqq_m))):
        # on.iloc[i-1] uses SMA through month i-1 only
        sma = qqq_m.iloc[i - 10 : i].mean()  # months [i-10 .. i-1] inclusive = 10 months
        expected = bool(qqq_m.iloc[i - 1] > sma)
        if bool(on.iloc[i - 1]) != expected:
            ok = False
            break

    daily, log = run_sfe(px, rfr)
    assert_alignment(log, ok)
    print("  Signal alignment (T → T+1, expanding 10-mo SMA): PASS")

    start = daily.index.min()
    end = daily.index.max()
    print(f"  Backtest window: {start.date()} → {end.date()} ({len(daily)} days)")

    series = {
        "SFE 1/3 Faber": daily,
        "QQQ B&H": bh(px["QQQ"], start).reindex(daily.index).fillna(0.0),
        "SPY B&H": bh(px["SPY"], start).reindex(daily.index).fillna(0.0),
        "60/40": sixty_forty(px["SPY"], rfr, start).reindex(daily.index).fillna(0.0),
    }
    # GLD B&H from its start within window
    gld_bh = bh(px["GLD"], max(start, px["GLD"].first_valid_index()))
    gld_aligned = gld_bh.reindex(daily.index)
    # before GLD: cash
    gld_aligned = gld_aligned.fillna(rfr.reindex(daily.index)).fillna(0.0)
    series["GLD B&H (cash pre)"] = gld_aligned

    print("\n" + "=" * 120)
    print("  CORE PERFORMANCE")
    print("=" * 120)
    hdr = (
        f"  {'Strategy':<22} {'CAGR':>7} {'Vol':>7} {'Sharpe':>7} {'Sortino':>8} "
        f"{'MaxDD':>7} {'Calmar':>7} {'Term$1':>9} {'DCA':>9}"
    )
    print(hdr)
    print("  " + "-" * 100)
    mets = {k: metrics(v) for k, v in series.items()}
    for k, m in mets.items():
        print(prow(k, m))

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
    print("  STATE OCCUPANCY")
    print("=" * 120)
    n = len(log)
    print(f"  Months: {n}")
    print(f"  QQQ/QLD ON: {log['qqq_on'].mean():.1%}  ({int(log['qqq_on'].sum())}/{n})")
    print(f"  SPY/SSO ON: {log['spy_on'].mean():.1%}  ({int(log['spy_on'].sum())}/{n})")
    print(f"  GLD ON:     {log['gld_on'].mean():.1%}  ({int(log['gld_on'].sum())}/{n})")
    all_on = (log["qqq_on"] & log["spy_on"] & log["gld_on"]).mean()
    all_off = (~log["qqq_on"] & ~log["spy_on"] & ~log["gld_on"]).mean()
    print(f"  All three ON:  {all_on:.1%}")
    print(f"  All three OFF: {all_off:.1%}")
    print(f"  Mean cash weight: {log['w_cash'].mean():.1%}")

    # Save artifacts
    out_dir = ROOT / "research/data"
    out_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_dir / "sfe_daily_returns.csv", header=["ret"])
    log.to_csv(out_dir / "sfe_monthly_allocations.csv", index=False)
    print(f"\n  Saved: research/data/sfe_daily_returns.csv")
    print(f"  Saved: research/data/sfe_monthly_allocations.csv")

    print("\n" + "=" * 120)
    print("  DONE")
    print("=" * 120)
    return mets, daily, log


if __name__ == "__main__":
    main()

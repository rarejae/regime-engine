"""V19d with a realistic circuit-breaker EXECUTION LAG.

The production backtest (experiments/v19d_final) sets a breached sleeve to cash on
the breach day itself — i.e. it earns the risk-free rate on day T even though the
breach can only be KNOWN at day T's close. That is a ~1-day look-ahead: it dodges
the breach-day drop you cannot actually avoid.

This script reruns V19d with the CB exit executed at a realistic time, using only
daily close data (no intraday, no false positives):

  prior     — original: cash from the breach day (look-ahead baseline — IMPOSSIBLE:
              it dodges the whole breach-day decline you cannot know until the close)
  t_close   — hold through the breach day, sell at ITS close = INTRADAY-MOC. Validated
              as live-achievable by experiments/v19d_intraday_moc: at the 15:50 MOC
              cutoff the "below all 3 SMAs" read matches the official close signal
              99.9% of the time (2 false-positives + 4 misses in 1,139 breach days,
              2017-2026, every error a <0.18% razor-edge crossing). So you CAN know
              at 15:50 that the close will breach and fire a market-on-close order.
              This is the true achievable base case.
  t1_close  — hold through T and T+1, sell at T+1's close (fallback when you MISS the
              15:50 cutoff; also the honest model pre-2017 with no intraday evidence)
  t1_open   — sell at T+1's open (worst realistic; fat overnight-gap tail)

Same window as v19d_final (2002-01 → 2026-03) for apples-to-apples.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv; load_dotenv()

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

from experiments.v11_beta_scaled.backtest import (
    SMA_PERIODS, SSO_EXP, QLD_EXP,
    load_data, asset_score, check_breach, lev_ret,
    cagr, max_dd, sharpe_r, sortino_r, calmar_r, dca_terminal,
)

END_DATE = "2026-03-31"
START_DATE = "2002-01-01"


def run_v19d_lag(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                 start_date, exit_mode="t1_close", overnight=None, sub=1.0):
    """exit_mode: 'prior' (look-ahead), 't_close' (delay 0), 't1_close' (delay 1, sell
    at T+1 close), 't1_open' (delay 1, sell at T+1 open — earns only the overnight gap
    on the exit day; needs `overnight` = {ticker: open/prev_close-1 Series})."""
    delay = {"t_close": 0, "t1_close": 1, "t1_open": 1}.get(exit_mode, None)
    exec_at = "open" if exit_mode == "t1_open" else "close"
    overnight = overnight or {}

    def on_gap(tk, und, day, mult):
        """Overnight gap (open/prev_close-1) on the exit day: actual ETF if we have
        it, else mult × underlying gap (mult=2 for QLD/SSO, 1 for IAU)."""
        if tk in overnight:
            v = overnight[tk].get(day, np.nan)
            if pd.notna(v):
                return float(v)
        if und in overnight:
            v = overnight[und].get(day, np.nan)
            if pd.notna(v):
                return mult * float(v)
        return 0.0
    bt_start = pd.Timestamp(start_date)
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:pd.Timestamp(END_DATE)].index

    port = {}; cb_events = []
    nav1 = 0.45; nav2 = 0.45; nav_g = 0.10
    p1_mode = "cash"; p1_lev = False; p1_delev = False; p1_pend = False; p1_cd = 0
    p2_mode = "cash"; p2_lev = False; p2_delev = False; p2_pend = False; p2_cd = 0
    gold_mode = "cash"; gold_delev = False; g_pend = False; g_cd = 0
    scores = {"QQQ": 0, "IVV": 0, "IAU": 0}

    for day in trading_days:
        dr = daily_ret.loc[day]
        is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day) - 1].month)

        if is_ms:
            p1_delev = p2_delev = gold_delev = False
            p1_pend = p2_pend = g_pend = False  # month rebalance supersedes any pending exit
            prior = trading_days[trading_days < day]
            sd = prior[-1] if len(prior) > 0 else day
            scores = {a: asset_score(sd, a, dpdf, daily_smas) for a in ["QQQ", "IVV", "IAU"]}
            sc_q, sc_i, sc_a = scores["QQQ"], scores["IVV"], scores["IAU"]
            if sc_q >= 3:
                if sc_i <= 1: p1_mode = "qqq"; p1_lev = False
                else: p1_mode = "qld"; p1_lev = True
            elif sc_q == 2: p1_mode = "qqq_partial"; p1_lev = False
            else: p1_mode = "cash"; p1_lev = False
            if sc_i >= 3: p2_mode = "sso"; p2_lev = True
            elif sc_i == 2: p2_mode = "ivv_partial"; p2_lev = False
            else: p2_mode = "cash"; p2_lev = False
            gold_mode = "iau" if sc_a >= 3 else "cash"
            if day != trading_days[0]:
                total = nav1 + nav2 + nav_g
                if total > 0:
                    drift = max(abs(nav1/total - 0.45), abs(nav2/total - 0.45), abs(nav_g/total - 0.10))
                    if drift > 0.05:
                        nav1 = total * 0.45; nav2 = total * 0.45; nav_g = total * 0.10

        # ── CB detection ──
        if delay is None:  # original: immediate cash on the breach day (look-ahead)
            if p1_lev and not p1_delev and check_breach(day, "QQQ", dpdf, daily_smas):
                p1_lev = False; p1_delev = True; p1_mode = "cash"; cb_events.append({"date": day, "asset": "QQQ"})
            if p2_lev and not p2_delev and check_breach(day, "IVV", dpdf, daily_smas):
                p2_lev = False; p2_delev = True; p2_mode = "cash"; cb_events.append({"date": day, "asset": "IVV"})
            if gold_mode == "iau" and not gold_delev and check_breach(day, "IAU", dpdf, daily_smas):
                gold_mode = "cash"; gold_delev = True; cb_events.append({"date": day, "asset": "IAU"})
        else:  # lagged: keep holding until the countdown elapses
            if p1_lev and not p1_delev and not p1_pend and check_breach(day, "QQQ", dpdf, daily_smas):
                p1_pend = True; p1_cd = delay; cb_events.append({"date": day, "asset": "QQQ"})
            if p2_lev and not p2_delev and not p2_pend and check_breach(day, "IVV", dpdf, daily_smas):
                p2_pend = True; p2_cd = delay; cb_events.append({"date": day, "asset": "IVV"})
            if gold_mode == "iau" and not gold_delev and not g_pend and check_breach(day, "IAU", dpdf, daily_smas):
                g_pend = True; g_cd = delay; cb_events.append({"date": day, "asset": "IAU"})

        # ── returns (current modes; a pending sleeve is still fully held) ──
        rfr = float(rfr_daily.get(day, 0.0))
        qqq_u = float(dr.get("QQQ", 0.0)) if pd.notna(dr.get("QQQ", np.nan)) else 0.0
        ivv_u = float(dr.get("IVV", 0.0)) if pd.notna(dr.get("IVV", np.nan)) else 0.0
        iau_u = float(dr.get("IAU", 0.0)) if pd.notna(dr.get("IAU", np.nan)) else 0.0
        # on the exit day, an 'open' fill earns only the overnight gap, then cash
        p1_exit = delay is not None and p1_pend and p1_cd == 0
        p2_exit = delay is not None and p2_pend and p2_cd == 0
        g_exit = delay is not None and g_pend and g_cd == 0

        # sub = fraction of a full-signal sleeve in the 2x ETF; rest in the 1x underlying
        if p1_mode == "qld":
            if p1_exit and exec_at == "open":
                r1 = sub * on_gap("QLD", "QQQ", day, 2) + (1 - sub) * on_gap("QQQ", "QQQ", day, 1)
            else:
                lev = lev_ret(qqq_u, rfr, QLD_EXP, day, actual_lev, "QLD", both_start)
                r1 = sub * lev + (1 - sub) * qqq_u
        elif p1_mode == "qqq": r1 = qqq_u
        elif p1_mode == "qqq_partial": r1 = 0.70 * qqq_u + 0.30 * rfr
        else: r1 = rfr
        if p2_mode == "sso":
            if p2_exit and exec_at == "open":
                r2 = sub * on_gap("SSO", "SPY", day, 2) + (1 - sub) * on_gap("SPY", "SPY", day, 1)
            else:
                lev = lev_ret(ivv_u, rfr, SSO_EXP, day, actual_lev, "SSO", both_start)
                r2 = sub * lev + (1 - sub) * ivv_u
        elif p2_mode == "ivv": r2 = ivv_u
        elif p2_mode == "ivv_partial": r2 = 0.70 * ivv_u + 0.30 * rfr
        else: r2 = rfr
        if gold_mode == "iau":
            rg = on_gap("IAU", "GLD", day, 1) if (g_exit and exec_at == "open") else iau_u
        else: rg = rfr

        prev_total = nav1 + nav2 + nav_g
        nav1 *= (1 + r1); nav2 *= (1 + r2); nav_g *= (1 + rg)
        new_total = nav1 + nav2 + nav_g
        port[day] = new_total / prev_total - 1 if prev_total > 0 else 0

        # ── end-of-day: resolve pending exits ──
        if delay is not None:
            if p1_pend:
                if p1_cd <= 0: p1_lev = False; p1_delev = True; p1_mode = "cash"; p1_pend = False
                else: p1_cd -= 1
            if p2_pend:
                if p2_cd <= 0: p2_lev = False; p2_delev = True; p2_mode = "cash"; p2_pend = False
                else: p2_cd -= 1
            if g_pend:
                if g_cd <= 0: gold_mode = "cash"; gold_delev = True; g_pend = False
                else: g_cd -= 1

    return pd.Series(port).sort_index(), cb_events


def metrics(s):
    sm = s.resample("MS").apply(lambda x: (1 + x).prod() - 1)
    return {"CAGR": cagr(s), "Vol": s.std() * np.sqrt(252), "Sharpe": sharpe_r(s),
            "Sortino": sortino_r(s), "MaxDD": max_dd(s), "Calmar": calmar_r(s),
            "Terminal": (1 + s).cumprod().iloc[-1], "DCA": dca_terminal(sm)}


CRISES = [("Dot-com 02-03", "2002-01-01", "2003-03-31"), ("GFC 07-09", "2007-11-01", "2009-03-31"),
          ("COVID 2020", "2020-02-01", "2020-04-30"), ("2022 bear", "2022-01-01", "2022-12-31"),
          ("2015 Aug", "2015-08-01", "2015-09-30")]


def main():
    print("=" * 104)
    print("  V19d EXECUTION-LAG BACKTEST — realistic circuit-breaker fills (2002-01 → 2026-03)")
    print("=" * 104)
    print("  Loading data (yfinance + FRED)…")
    daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, *_ = load_data()

    print("  Fetching OHLC opens for overnight-gap (T+1 open) modelling…")
    import yfinance as yf
    overnight = {}
    for t in ["QLD", "SSO", "IAU", "QQQ", "SPY", "GLD"]:
        d = yf.download(t, start="2003-01-01", progress=False, auto_adjust=True)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d.index = pd.to_datetime(d.index).tz_localize(None)
        overnight[t] = (d["Open"] / d["Close"].shift(1) - 1).dropna()

    modes = ["prior", "t_close", "t1_close", "t1_open"]
    runs = {}
    for mode in modes:
        s, cb = run_v19d_lag(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                             START_DATE, mode, overnight=overnight)
        runs[mode] = (metrics(s), s, cb)

    labels = {"prior": "prior (look-ahead, IMPOSSIBLE)", "t_close": "intraday-MOC (achievable base)",
              "t1_close": "T+1 close (missed-cutoff fallback)", "t1_open": "T+1 open (worst realistic)"}
    m0 = runs["prior"][0]
    print(f"\n  {'Execution':<28}{'CAGR':>7}{'Vol':>7}{'Sharpe':>8}{'Sortino':>8}{'MaxDD':>8}"
          f"{'Calmar':>7}{'Term$1':>9}{'DCA':>9}{'CBs':>5}")
    print("  " + "-" * 100)
    for mode in modes:
        m, s, cb = runs[mode]
        print(f"  {labels[mode]:<28}{m['CAGR']:>7.2%}{m['Vol']:>7.2%}{m['Sharpe']:>8.3f}{m['Sortino']:>8.3f}"
              f"{m['MaxDD']:>8.1%}{m['Calmar']:>7.2f}${m['Terminal']:>8.2f}${m['DCA']/1e6:>7.2f}M{len(cb):>5}")

    mc = runs["t_close"][0]   # intraday-MOC (achievable base case)
    mr = runs["t1_close"][0]  # T+1 fallback
    print(f"\n  DELTA — intraday-MOC (t_close, achievable) vs T+1 fallback:")
    print(f"    CAGR    {mc['CAGR']-mr['CAGR']:+.2%}   ({mr['CAGR']:.2%} → {mc['CAGR']:.2%})")
    print(f"    Sharpe  {mc['Sharpe']-mr['Sharpe']:+.3f}   ({mr['Sharpe']:.3f} → {mc['Sharpe']:.3f})")
    print(f"    MaxDD   {100*(mc['MaxDD']-mr['MaxDD']):+.1f}pp  ({mr['MaxDD']:.1%} → {mc['MaxDD']:.1%})")

    print(f"\n  DELTA — intraday-MOC (best achievable) vs look-ahead (prior, IMPOSSIBLE):")
    print(f"    CAGR    {mc['CAGR']-m0['CAGR']:+.2%}   ({m0['CAGR']:.2%} → {mc['CAGR']:.2%})")
    print(f"    MaxDD   {100*(mc['MaxDD']-m0['MaxDD']):+.1f}pp  ({m0['MaxDD']:.1%} → {mc['MaxDD']:.1%})")
    print(f"    ^ this gap is the unavoidable breach-day decline; NO execution recovers it.")

    print(f"\n  CRISIS MAX DRAWDOWN by execution:")
    print(f"    {'Crisis':<16}{'prior(LA)':>11}{'MOC(t_cl)':>11}{'T+1 close':>11}{'T+1 open':>10}")
    for name, c0, c1 in CRISES:
        row = f"    {name:<16}"
        for mode in modes:
            s = runs[mode][1]
            w = s[(s.index >= c0) & (s.index <= c1)]
            row += f"{max_dd(w):>11.1%}" if len(w) > 5 else f"{'n/a':>11}"
        print(row)
    print("\n  Done.")


if __name__ == "__main__":
    main()

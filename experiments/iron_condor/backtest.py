"""Asymmetric iron condor Pod 2 with independent leg management.

Put spread: -0.10 delta short put, 5pt wide (Harvey bullish conviction)
Call spread: +0.20 delta short call, 5pt wide (VRP harvest)
Combined credit ~$1.05 on ~$395 margin.
Legs managed independently — closing threatened leg while keeping profitable leg open.

2010-2023, $100K dedicated capital.
"""

import sys, os, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv; load_dotenv()

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

from taa.data import (load_daily_etf_returns, load_monthly_prices,
                       load_monthly_macro, load_monthly_asset_returns)
from taa.faber import compute_trend_scores, apply_faber_filter
from taa.harvey import compute_zscore_variables, find_similar_months, compute_expected_returns

BASELINE = {"IVV": 0.45, "QQQ": 0.25, "VGLT": 0.05, "IAU": 0.10, "DBC": 0.05, "cash": 0.10}
ASSETS = list(BASELINE.keys())
DAILY_SMA_PERIODS = [126, 200, 252]
SSO_EXP = 0.0089; QLD_EXP = 0.0095

SPREAD_WIDTH = 5.0
COMMISSION_PER_LEG_PAIR = 1.30  # 2 legs × $0.65
SLIPPAGE = 0.10
MAX_CONTRACTS = 75
MARGIN_PCT = 0.30
STARTING_CAPITAL = 100_000
PUT_SHORT_DELTA = -0.10
CALL_SHORT_DELTA = 0.20
PUT_DELTA_STOP = -0.25
CALL_DELTA_STOP = 0.35
DTE_MIN = 21; DTE_MAX = 38; DTE_IDEAL = 30


def load_data():
    import yfinance as yf
    daily_ret = load_daily_etf_returns()
    daily_ret = daily_ret[[c for c in daily_ret.columns if c in ASSETS]]
    ticker_map = {"IVV": "SPY", "QQQ": "QQQ", "VGLT": "TLT", "IAU": "GLD", "DBC": "DBC"}
    dp = {}
    for our, ticker in ticker_map.items():
        d = yf.download(ticker, start="1998-01-01", progress=False)
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p, "columns"): p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None); dp[our] = p
    dpdf = pd.DataFrame(dp).sort_index()
    daily_smas = {p: dpdf.rolling(p, min_periods=p).mean() for p in DAILY_SMA_PERIODS}

    from fredapi import Fred
    rfr_daily = pd.Series(0.0, index=daily_ret.index)
    key = os.environ.get("FRED_API_KEY"); vix_monthly = None; tbill_monthly = None
    if key:
        f = Fred(api_key=key)
        try:
            tb = f.get_series("DTB3", observation_start="1998-01-01"); tb.index = pd.to_datetime(tb.index)
            rfr_daily = (tb/100/252).reindex(daily_ret.index, method="ffill").fillna(0)
            tbill_monthly = tb.resample("MS").last().dropna() / 100 / 12
        except: pass
        try:
            vix = f.get_series("VIXCLS", observation_start="2009-01-01"); vix.index = pd.to_datetime(vix.index)
            vix_monthly = vix.resample("MS").last().dropna()
        except: pass
    if vix_monthly is None:
        vd = yf.download("^VIX", start="2009-01-01", progress=False)
        if vd is not None and not vd.empty:
            p = vd["Close"];
            if hasattr(p, "columns"): p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None); vix_monthly = p.resample("MS").last().dropna()

    macro = load_monthly_macro()
    z_data = compute_zscore_variables(macro)
    z_clean = z_data[[c for c in z_data.columns if c.endswith("_z")]].dropna()
    asset_ret_fwd = load_monthly_asset_returns().shift(-1)

    actual_lev = {}
    for ticker in ["SSO", "QLD"]:
        d = yf.download(ticker, start="2006-01-01", progress=False, auto_adjust=True)
        if d is not None and not d.empty:
            p = d["Close"];
            if hasattr(p, "columns"): p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None); actual_lev[ticker] = p.pct_change().dropna()
    both_start = max(actual_lev.get("SSO", pd.Series()).index.min(),
                     actual_lev.get("QLD", pd.Series()).index.min()) \
        if "SSO" in actual_lev and "QLD" in actual_lev else pd.Timestamp("2099-01-01")

    options_by_year = {}
    for yr in range(2010, 2024):
        p = Path(f"data/processed/spy_options_{yr}.parquet")
        if p.exists(): options_by_year[yr] = pd.read_parquet(p)

    return (daily_ret, dpdf, daily_smas, rfr_daily, vix_monthly, tbill_monthly,
            z_clean, asset_ret_fwd, actual_lev, both_start, options_by_year)


def find_contract(chain, opt_type, target_delta, dte_min, dte_max, dte_ideal):
    filt = chain[(chain["option_type"] == opt_type) &
                 (chain["dte"] >= dte_min) & (chain["dte"] <= dte_max) &
                 (chain["bid"] >= 0.03) & (chain["delta"].notna())].copy()
    if len(filt) == 0: return None
    filt["dte_dist"] = abs(filt["dte"] - dte_ideal)
    best = filt["dte_dist"].min()
    filt = filt[filt["dte_dist"] <= best + 3]
    filt["delta_dist"] = abs(filt["delta"] - target_delta)
    return filt.loc[filt["delta_dist"].idxmin()]


def get_option_mid_delta(chain, trade_date, strike, expiry, opt_type):
    dc = chain[(chain["trade_date"] == trade_date) & (chain["option_type"] == opt_type) &
               (chain["strike"] == strike) & (chain["expiry"] == expiry)]
    if len(dc) == 0: return None, None
    return float(dc.iloc[0]["mid"]), float(dc.iloc[0]["delta"])


def sma_scores(day, dpdf, smas):
    scores = {}
    for a in ["IVV", "QQQ", "VGLT", "IAU", "DBC"]:
        if a not in dpdf.columns: scores[a] = 0; continue
        p = dpdf.loc[:day, a]
        if len(p) == 0 or pd.isna(p.iloc[-1]): scores[a] = 0; continue
        price = p.iloc[-1]; sc = 0
        for per in DAILY_SMA_PERIODS:
            s = smas[per].loc[:day, a]
            if len(s) > 0 and pd.notna(s.iloc[-1]) and price > s.iloc[-1]: sc += 1
        scores[a] = sc
    return scores

def check_breach(day, dpdf, smas):
    for etf in ["IVV", "QQQ"]:
        if etf not in dpdf.columns: continue
        p = dpdf.loc[:day, etf]
        if len(p) == 0: continue
        price = p.iloc[-1]; b = 0
        for per in DAILY_SMA_PERIODS:
            s = smas[per].loc[:day, etf]
            if len(s) > 0 and pd.notna(s.iloc[-1]) and price < s.iloc[-1]: b += 1
        if b >= 3: return True
    return False

def run_faber(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start):
    bt = pd.Timestamp("2010-01-01"); td = daily_ret.loc[bt:pd.Timestamp("2023-12-31")].index
    cur = {a: 3 for a in ASSETS if a != "cash"}; wf = dict(BASELINE); la = False; dlv = False; r = {}
    for day in td:
        if day not in daily_ret.index: continue
        dr = daily_ret.loc[day]; avail = [a for a in ASSETS if a in dr.index and pd.notna(dr[a])]
        if len(avail) < 3: continue
        actual = {a: float(dr[a]) for a in avail}; rfr = float(rfr_daily.get(day, 0))
        is_ms = (day == td[0] or day.month != td[td.get_loc(day)-1].month)
        if is_ms:
            dlv = False; prior = td[td < day]; sd = prior[-1] if len(prior) > 0 else day
            cur = sma_scores(sd, dpdf, daily_smas)
            w1, pool = apply_faber_filter(cur, BASELINE); wf = dict(w1); wf["cash"] = wf.get("cash",0)+pool
            fc = cur.get("IVV",0)>=3 and cur.get("QQQ",0)>=3; la = fc
        if la and not dlv:
            if check_breach(day, dpdf, daily_smas): la = False; dlv = True
        iw=wf.get("IVV",0); qw=wf.get("QQQ",0); ir=actual.get("IVV",0); qr=actual.get("QQQ",0)
        base = sum(wf.get(a,0)*actual.get(a,0) for a in avail if a not in ["IVV","QQQ"])
        if la:
            sso = float(actual_lev.get("SSO",pd.Series()).get(day, 2*ir-rfr-SSO_EXP/252)) if day>=both_start else 2*ir-rfr-SSO_EXP/252
            qld = float(actual_lev.get("QLD",pd.Series()).get(day, 2*qr-rfr-QLD_EXP/252)) if day>=both_start else 2*qr-rfr-QLD_EXP/252
            if np.isnan(sso): sso = 2*ir-rfr-SSO_EXP/252
            if np.isnan(qld): qld = 2*qr-rfr-QLD_EXP/252
            r[day] = iw*sso + qw*qld + base
        else: r[day] = iw*ir + qw*qr + base
    return pd.Series(r).sort_index()


# ── Iron Condor Backtest ─────────────────────────────────────────────────────

def run_condor(vix_monthly, tbill_monthly, z_clean, asset_ret_fwd, options_by_year, dpdf):
    bt_start = pd.Timestamp("2010-01-01"); bt_end = pd.Timestamp("2023-12-31")
    trading_days = dpdf.index[(dpdf.index >= bt_start) & (dpdf.index <= bt_end)]
    month_ends = [d for i, d in enumerate(trading_days)
                  if i == len(trading_days)-1 or d.month != trading_days[i+1].month]

    # Position state
    pos = None  # dict: put_open, call_open, details
    trades = []
    daily_pnl = {d: 0.0 for d in trading_days}

    for day in trading_days:
        yr = day.year; is_me = day in month_ends; chain = options_by_year.get(yr)

        # ── Daily position monitoring ────────────────────────────────────
        if pos is not None and chain is not None:
            dte_now = (pos["expiry"] - day).days

            # Get current values for open legs
            put_val, put_delta_now = None, None
            call_val, call_delta_now = None, None

            if pos["put_open"]:
                sm, sd = get_option_mid_delta(chain, day, pos["put_short_strike"], pos["expiry"], "P")
                lm, _ = get_option_mid_delta(chain, day, pos["put_long_strike"], pos["expiry"], "P")
                if sm is not None and lm is not None:
                    put_val = sm - lm; put_delta_now = sd

            if pos["call_open"]:
                sm, sd = get_option_mid_delta(chain, day, pos["call_short_strike"], pos["expiry"], "C")
                lm, _ = get_option_mid_delta(chain, day, pos["call_long_strike"], pos["expiry"], "C")
                if sm is not None and lm is not None:
                    call_val = sm - lm; call_delta_now = sd

            # Combined value
            combined_val = 0.0
            if pos["put_open"] and put_val is not None: combined_val += put_val
            if pos["call_open"] and call_val is not None: combined_val += call_val

            total_credit = 0.0
            if pos["put_open"]: total_credit += pos["put_credit"]
            if pos["call_open"]: total_credit += pos["call_credit"]

            action = None; pnl_today = 0.0

            # Rule 1: Full condor profit target (both legs open, combined ≤ 50% of total credit)
            if pos["put_open"] and pos["call_open"] and total_credit > 0:
                if combined_val <= total_credit * 0.50:
                    action = "condor_profit"
                    exit_cost = combined_val * (1 + SLIPPAGE)
                    pps = (pos["put_credit_actual"] + pos["call_credit_actual"]) - exit_cost
                    pnl_today = (pps * pos["contracts"] * 100) - (COMMISSION_PER_LEG_PAIR * 2 * pos["contracts"])
                    pos["put_open"] = False; pos["call_open"] = False

            # Rule 2: Put delta stop
            if action is None and pos["put_open"] and put_delta_now is not None:
                if put_delta_now <= PUT_DELTA_STOP:
                    action = "put_delta_stop"
                    exit_cost = put_val * (1 + SLIPPAGE) if put_val is not None else pos["put_credit"]
                    pnl_put = (pos["put_credit_actual"] - exit_cost) * pos["contracts"] * 100
                    pnl_put -= COMMISSION_PER_LEG_PAIR * pos["contracts"]
                    pnl_today += pnl_put
                    pos["put_open"] = False

            # Rule 3: Call delta stop
            if action is None and pos["call_open"] and call_delta_now is not None:
                if call_delta_now >= CALL_DELTA_STOP:
                    action = "call_delta_stop"
                    exit_cost = call_val * (1 + SLIPPAGE) if call_val is not None else pos["call_credit"]
                    pnl_call = (pos["call_credit_actual"] - exit_cost) * pos["contracts"] * 100
                    pnl_call -= COMMISSION_PER_LEG_PAIR * pos["contracts"]
                    pnl_today += pnl_call
                    pos["call_open"] = False

            # Rule 4: Individual leg profit (≤ 25% of its own credit)
            if action is None:
                if pos["put_open"] and put_val is not None and put_val <= pos["put_credit"] * 0.25:
                    exit_cost = put_val * (1 + SLIPPAGE)
                    pnl_today += (pos["put_credit_actual"] - exit_cost) * pos["contracts"] * 100
                    pnl_today -= COMMISSION_PER_LEG_PAIR * pos["contracts"]
                    pos["put_open"] = False
                    action = "put_leg_profit"

                if pos["call_open"] and call_val is not None and call_val <= pos["call_credit"] * 0.25:
                    exit_cost = call_val * (1 + SLIPPAGE)
                    pnl_today += (pos["call_credit_actual"] - exit_cost) * pos["contracts"] * 100
                    pnl_today -= COMMISSION_PER_LEG_PAIR * pos["contracts"]
                    pos["call_open"] = False
                    if action is None: action = "call_leg_profit"
                    else: action = "both_leg_profit"

            # Rule 5: Time exit (≤ 7 DTE)
            if action is None and dte_now <= 7 and (pos["put_open"] or pos["call_open"]):
                action = "time_exit"
                if pos["put_open"] and put_val is not None:
                    exit_cost = put_val * (1 + SLIPPAGE)
                    pnl_today += (pos["put_credit_actual"] - exit_cost) * pos["contracts"] * 100
                    pnl_today -= COMMISSION_PER_LEG_PAIR * pos["contracts"]
                    pos["put_open"] = False
                if pos["call_open"] and call_val is not None:
                    exit_cost = call_val * (1 + SLIPPAGE)
                    pnl_today += (pos["call_credit_actual"] - exit_cost) * pos["contracts"] * 100
                    pnl_today -= COMMISSION_PER_LEG_PAIR * pos["contracts"]
                    pos["call_open"] = False

            # Expiry fallback
            if day >= pos["expiry"] and (pos["put_open"] or pos["call_open"]):
                ep = dpdf.loc[:day, "IVV"].iloc[-1] if "IVV" in dpdf.columns else None
                if ep is not None:
                    if pos["put_open"]:
                        ks=pos["put_short_strike"]; kl=pos["put_long_strike"]; ac=pos["put_credit_actual"]
                        if ep >= ks: pps = ac
                        elif ep <= kl: pps = ac - SPREAD_WIDTH
                        else: pps = ac - (ks - ep)
                        pnl_today += pps * pos["contracts"] * 100 - COMMISSION_PER_LEG_PAIR * pos["contracts"]
                    if pos["call_open"]:
                        ks=pos["call_short_strike"]; kl=pos["call_long_strike"]; ac=pos["call_credit_actual"]
                        if ep <= ks: pps = ac
                        elif ep >= kl: pps = ac - SPREAD_WIDTH
                        else: pps = ac - (ep - ks)
                        pnl_today += pps * pos["contracts"] * 100 - COMMISSION_PER_LEG_PAIR * pos["contracts"]
                action = action or "expiry"
                pos["put_open"] = False; pos["call_open"] = False

            if pnl_today != 0:
                daily_pnl[day] += pnl_today

            if action:
                trades.append({
                    "entry_date": pos["entry_date"], "exit_date": day,
                    "hold_days": (day - pos["entry_date"]).days,
                    "exit_reason": action,
                    "put_credit": pos["put_credit"], "call_credit": pos["call_credit"],
                    "total_credit": pos["put_credit"] + pos["call_credit"],
                    "contracts": pos["contracts"], "net_pnl": pnl_today,
                    "vix": pos["vix"], "harvey_er": pos["harvey_er"],
                    "underlying_entry": pos["underlying"],
                    "put_delta_exit": put_delta_now, "call_delta_exit": call_delta_now,
                })

            # Clean up if both legs closed
            if not pos["put_open"] and not pos["call_open"]:
                pos = None

        # ── Month-end entry ──────────────────────────────────────────────
        if is_me and pos is None:
            vix_val = None
            if vix_monthly is not None:
                vd = vix_monthly.index[vix_monthly.index <= day]
                if len(vd) > 0: vix_val = float(vix_monthly.loc[vd[-1]])
            if vix_val is None or vix_val < 18: continue

            z_prior = z_clean.index[z_clean.index < day]; harvey_er = 0.0
            if len(z_prior) > 0:
                try:
                    sim, _ = find_similar_months(z_clean, z_prior[-1])
                    er = compute_expected_returns(sim, asset_ret_fwd, ["IVV"])
                    harvey_er = er.get("IVV", 0.0)
                except ValueError: pass
            if harvey_er <= 0.005: continue

            if chain is None: continue
            cd = chain["trade_date"].unique(); valid = cd[cd <= day]
            if len(valid) == 0: continue
            trade_date = valid[-1]; dc = chain[chain["trade_date"] == trade_date]

            # Find put spread
            put_short = find_contract(dc, "P", PUT_SHORT_DELTA, DTE_MIN, DTE_MAX, DTE_IDEAL)
            if put_short is None: continue
            put_long_strike = float(put_short["strike"]) - SPREAD_WIDTH
            put_longs = dc[(dc["option_type"]=="P") & (dc["expiry"]==put_short["expiry"]) &
                           (abs(dc["strike"]-put_long_strike)<1.0)]
            if len(put_longs) == 0: continue
            put_long = put_longs.iloc[0]
            put_credit = float(put_short["mid"]) - float(put_long["mid"])
            if put_credit <= 0: continue

            # Find call spread — same expiry
            call_short = find_contract(
                dc[dc["expiry"] == put_short["expiry"]], "C", CALL_SHORT_DELTA, DTE_MIN, DTE_MAX, DTE_IDEAL)
            if call_short is None: continue
            call_long_strike = float(call_short["strike"]) + SPREAD_WIDTH
            call_longs = dc[(dc["option_type"]=="C") & (dc["expiry"]==put_short["expiry"]) &
                            (abs(dc["strike"]-call_long_strike)<1.0)]
            if len(call_longs) == 0: continue
            call_long = call_longs.iloc[0]
            call_credit = float(call_short["mid"]) - float(call_long["mid"])
            if call_credit <= 0: continue

            total_credit = put_credit + call_credit
            margin_per = (SPREAD_WIDTH * 100) - (total_credit * 100)
            if margin_per <= 0: continue
            contracts = min(int(STARTING_CAPITAL * MARGIN_PCT / margin_per), MAX_CONTRACTS)
            contracts = max(contracts, 1)

            pos = {
                "entry_date": day, "expiry": pd.Timestamp(put_short["expiry"]),
                "put_open": True, "call_open": True,
                "put_short_strike": float(put_short["strike"]),
                "put_long_strike": float(put_long["strike"]),
                "put_credit": put_credit, "put_credit_actual": put_credit * (1-SLIPPAGE),
                "call_short_strike": float(call_short["strike"]),
                "call_long_strike": float(call_long["strike"]),
                "call_credit": call_credit, "call_credit_actual": call_credit * (1-SLIPPAGE),
                "contracts": contracts, "underlying": float(put_short["underlying_close"]),
                "vix": vix_val, "harvey_er": harvey_er,
            }

    # Monthly P&L
    pnl_s = pd.Series(daily_pnl).sort_index()
    monthly_pnl = {}
    for me in month_ends:
        ms = pd.Timestamp(f"{me.year}-{me.month:02d}-01")
        mp = pnl_s[(pnl_s.index >= ms) & (pnl_s.index <= me)].sum()
        if mp == 0 and tbill_monthly is not None:
            tbd = tbill_monthly.index[tbill_monthly.index <= me]
            if len(tbd) > 0: mp = float(tbill_monthly.loc[tbd[-1]]) * STARTING_CAPITAL
        monthly_pnl[ms] = mp

    return pd.Series(monthly_pnl).sort_index(), pd.DataFrame(trades) if trades else pd.DataFrame()


def metrics(rets):
    ar = rets.mean()*12; av = rets.std()*np.sqrt(12)
    sh = ar/av if av > 0 else 0
    neg = rets[rets < 0]; ds = neg.std()*np.sqrt(12) if len(neg) > 3 else av
    so = ar/ds if ds > 0 else 0
    cum = (1+rets).cumprod(); dd = ((cum-cum.expanding().max())/cum.expanding().max()).min()
    cal = ar/abs(dd) if dd != 0 else 0
    return {"ar": ar, "av": av, "sh": sh, "sortino": so, "dd": dd, "calmar": cal, "final": cum.iloc[-1]}


def report(faber_daily, condor_monthly, trades_df, daily_ret):
    f_m = faber_daily.resample("MS").apply(lambda x: (1+x).prod()-1)
    s_r = condor_monthly / STARTING_CAPITAL
    common = f_m.dropna().index.intersection(s_r.dropna().index).sort_values()
    f = f_m.reindex(common); s = s_r.reindex(common); c = 0.90*f + 0.10*s

    ivv_d = daily_ret["IVV"].loc["2010-01-01":"2023-12-31"].dropna()
    vglt_d = daily_ret.get("VGLT", pd.Series(dtype=float)).loc["2010-01-01":"2023-12-31"].fillna(0)
    ivv_m = ivv_d.resample("MS").apply(lambda x: (1+x).prod()-1).reindex(common, fill_value=0)

    pf = metrics(f); ps = metrics(s); pc = metrics(c); pi = metrics(ivv_m)

    # ── 1. Pod 2 standalone ──────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  1. POD 2 STANDALONE — IRON CONDOR")
    print(f"{'='*120}")
    print(f"\n  Annualised return:  {ps['ar']:.1%}")
    print(f"  Volatility:         {ps['av']:.1%}")
    print(f"  Sharpe ratio:       {ps['sh']:.3f}")
    print(f"  Sortino ratio:      {ps['sortino']:.3f}")
    print(f"  Max drawdown:       {ps['dd']:.1%}")
    print(f"  Calmar ratio:       {ps['calmar']:.2f}")
    print(f"  Terminal ($100K):   ${STARTING_CAPITAL * ps['final']:,.0f}")

    if len(trades_df) > 0:
        wins = trades_df[trades_df["net_pnl"] > 0]; losses = trades_df[trades_df["net_pnl"] <= 0]
        print(f"  Total trades:       {len(trades_df)} ({len(trades_df)/14:.1f}/year)")
        print(f"  Win rate:           {len(wins)/len(trades_df)*100:.0f}%")
        if len(wins) > 0: print(f"  Avg winning trade:  ${wins['net_pnl'].mean():,.0f}")
        if len(losses) > 0: print(f"  Avg losing trade:   ${losses['net_pnl'].mean():,.0f}")
        gw = wins["net_pnl"].sum() if len(wins) > 0 else 0
        gl = abs(losses["net_pnl"].sum()) if len(losses) > 0 else 1
        print(f"  Profit factor:      {gw/gl:.2f}")
        print(f"  Avg total credit:   ${trades_df['total_credit'].mean():.2f}")
        print(f"  Avg put credit:     ${trades_df['put_credit'].mean():.2f}")
        print(f"  Avg call credit:    ${trades_df['call_credit'].mean():.2f}")

    # ── 2. Three-portfolio comparison ────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  2. THREE-PORTFOLIO COMPARISON")
    print(f"{'='*120}")
    print(f"\n  {'Portfolio':<22} {'Return':>8} {'Vol':>7} {'Sharpe':>8} {'Sortino':>8} "
          f"{'MaxDD':>8} {'Calmar':>8} {'Terminal':>12}")
    print(f"  {'-'*22} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*12}")
    for p, label in [(pf, "A: Faber Only"), (ps, "B: Condor"), (pc, "C: Combined 90/10"),
                      (pi, "IVV B&H")]:
        t = STARTING_CAPITAL * p["final"]
        print(f"  {label:<22} {p['ar']:>7.1%} {p['av']:>6.1%} {p['sh']:>8.3f} {p['sortino']:>8.3f} "
              f"{p['dd']:>7.1%} {p['calmar']:>8.2f} ${t:>11,.0f}")

    # ── 3. Annual NAV ────────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  3. ANNUAL NAV TABLE")
    print(f"{'='*120}")
    cf = (1+f).cumprod()*STARTING_CAPITAL; cs_ = (1+s).cumprod()*STARTING_CAPITAL
    cc = (1+c).cumprod()*STARTING_CAPITAL; ci = (1+ivv_m).cumprod()*STARTING_CAPITAL
    print(f"\n  {'Year':>6} {'A: Faber':>14} {'B: Condor':>14} {'C: Combined':>14} {'IVV B&H':>14}")
    for yr in range(2010, 2024):
        yf = cf[cf.index.year == yr]; ys = cs_[cs_.index.year == yr]
        yc = cc[cc.index.year == yr]; yi = ci[ci.index.year == yr]
        if len(yf) > 0:
            print(f"  {yr:>6} ${yf.iloc[-1]:>13,.0f} ${ys.iloc[-1]:>13,.0f} "
                  f"${yc.iloc[-1]:>13,.0f} ${yi.iloc[-1]:>13,.0f}")

    # ── 4. Crisis analysis ───────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  4. CRISIS ANALYSIS")
    print(f"{'='*120}")
    for cname, cs_d, ce in [("2011 correction", "2011-07", "2011-10"),
                             ("2018 Q4", "2018-10", "2018-12"),
                             ("COVID Feb-Mar 2020", "2020-02", "2020-03"),
                             ("2022 bear", "2022-01", "2022-10")]:
        print(f"\n  {cname}:")
        print(f"  {'Portfolio':<22} {'Return':>10} {'MaxDD':>10}")
        print(f"  {'-'*22} {'-'*10} {'-'*10}")
        for sr, label in [(f, "A: Faber"), (s, "B: Condor"), (c, "C: Combined")]:
            cr = sr[(sr.index >= pd.Timestamp(cs_d)) & (sr.index <= pd.Timestamp(ce))]
            if len(cr) > 0:
                cum = (1+cr).cumprod(); mdd = ((cum-cum.expanding().max())/cum.expanding().max()).min()
                print(f"  {label:<22} {(1+cr).prod()-1:>+9.1%} {mdd:>9.1%}")

    # ── 5. Correlation ───────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  5. CORRELATION")
    print(f"{'='*120}")
    corr = float(f.corr(s))
    print(f"\n  Monthly correlation (Faber vs Condor): {corr:.3f}")
    print(f"  Prior single spread correlation: 0.222")
    rc = f.rolling(12).corr(s).dropna()
    if len(rc) > 0: print(f"  Rolling 12-month mean: {rc.mean():.3f}")
    for cname, cs_d, ce in [("2011", "2011-07", "2011-11"), ("2018 Q4", "2018-10", "2018-12"),
                             ("COVID", "2020-02", "2020-04"), ("2022", "2022-01", "2022-12")]:
        cf_c = f[(f.index >= pd.Timestamp(cs_d)) & (f.index <= pd.Timestamp(ce))]
        cs_c = s[(s.index >= pd.Timestamp(cs_d)) & (s.index <= pd.Timestamp(ce))]
        ci_c = cf_c.index.intersection(cs_c.index)
        if len(ci_c) >= 3: print(f"    {cname}: {cf_c.reindex(ci_c).corr(cs_c.reindex(ci_c)):.2f}")
    both_loss = common[(f.reindex(common) < -0.01) & (s.reindex(common) < -0.01)]
    print(f"\n  Months where both lost >1%: {len(both_loss)}")
    for dt in both_loss:
        print(f"    {dt.strftime('%Y-%m')}: Faber {f.loc[dt]*100:+.1f}%, Condor {s.loc[dt]*100:+.1f}%")

    # ── 6. Leg management ────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  6. LEG MANAGEMENT BREAKDOWN")
    print(f"{'='*120}")
    if len(trades_df) > 0:
        print(f"\n  {'Exit Rule':<22} {'Trades':>8} {'%':>6} {'Avg Hold':>10} {'Avg P&L':>12}")
        print(f"  {'-'*22} {'-'*8} {'-'*6} {'-'*10} {'-'*12}")
        for reason in trades_df["exit_reason"].unique():
            rt = trades_df[trades_df["exit_reason"] == reason]
            print(f"  {reason:<22} {len(rt):>8} {len(rt)/len(trades_df)*100:>5.0f}% "
                  f"{rt['hold_days'].mean():>9.0f}d ${rt['net_pnl'].mean():>+11,.0f}")

        # Put stop analysis: what was call leg doing?
        put_stops = trades_df[trades_df["exit_reason"] == "put_delta_stop"]
        if len(put_stops) > 0:
            print(f"\n  When put leg was stopped ({len(put_stops)} times):")
            print(f"    Avg net trade P&L: ${put_stops['net_pnl'].mean():,.0f}")
            print(f"    (includes any call leg profit banked at same time or later)")

        print(f"\n  Largest loss: ${trades_df['net_pnl'].min():,.0f} ({trades_df.loc[trades_df['net_pnl'].idxmin(), 'exit_date'].strftime('%Y-%m-%d')})")
        print(f"  Largest win:  ${trades_df['net_pnl'].max():,.0f}")

    # ── 7. Value-add ─────────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  7. COMBINED VALUE-ADD")
    print(f"{'='*120}")
    sh_i = pc["sh"]-pf["sh"]; dd_i = (pc["dd"]-pf["dd"])*100
    tc = (pc["final"]-pf["final"])*STARTING_CAPITAL
    print(f"\n  Sharpe improvement (C vs A):   {sh_i:+.3f} ({pf['sh']:.3f} → {pc['sh']:.3f})")
    print(f"  Max DD improvement (C vs A):   {dd_i:+.1f}% ({pf['dd']:.1%} → {pc['dd']:.1%})")
    print(f"  Terminal cost (C vs A):         ${tc:+,.0f}")
    print(f"  Correlation:                    {corr:.3f}")

    worth = pc["sh"] > pf["sh"] and pc["dd"] > pf["dd"]
    print(f"\n  Is iron condor Pod 2 viable? {'YES' if worth else 'NO'}")
    print(f"  Does it outperform single put spread? ", end="")
    if ps["sh"] > 0.314:
        print(f"YES — Sharpe {ps['sh']:.3f} vs 0.314 (prior)")
    else:
        print(f"NO — Sharpe {ps['sh']:.3f} vs 0.314 (prior)")

    # ── 8. Economics comparison ──────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  8. ECONOMICS COMPARISON VS PRIOR TESTS")
    print(f"{'='*120}")
    avg_tc = trades_df["total_credit"].mean() if len(trades_df) > 0 else 0
    avg_pc = trades_df["put_credit"].mean() if len(trades_df) > 0 else 0
    avg_cc = trades_df["call_credit"].mean() if len(trades_df) > 0 else 0
    wins = trades_df[trades_df["net_pnl"]>0] if len(trades_df)>0 else pd.DataFrame()
    losses = trades_df[trades_df["net_pnl"]<=0] if len(trades_df)>0 else pd.DataFrame()
    wr = len(wins)/len(trades_df)*100 if len(trades_df) > 0 else 0

    print(f"\n  {'':>25} {'Put spread':>14} {'Put -0.20d':>14} {'Iron Condor':>14}")
    print(f"  {'':>25} {'(-0.10d 30DTE)':>14} {'(45DTE)':>14} {'(this test)':>14}")
    print(f"  {'-'*25} {'-'*14} {'-'*14} {'-'*14}")
    print(f"  {'Avg total credit':>25} {'$0.70':>14} {'$0.66':>14} ${avg_tc:>13.2f}")
    print(f"  {'  Put credit':>25} {'$0.70':>14} {'$0.66':>14} ${avg_pc:>13.2f}")
    print(f"  {'  Call credit':>25} {'—':>14} {'—':>14} ${avg_cc:>13.2f}")
    print(f"  {'Win rate':>25} {'81%':>14} {'79%':>14} {wr:>13.0f}%")
    print(f"  {'Standalone Sharpe':>25} {'0.314':>14} {'0.141':>14} {ps['sh']:>14.3f}")
    print(f"  {'Combined Sharpe':>25} {'1.072':>14} {'1.067':>14} {pc['sh']:>14.3f}")

    print()
    return pf, ps, pc


if __name__ == "__main__":
    print("=" * 120)
    print("  ASYMMETRIC IRON CONDOR POD 2 — Independent Leg Management")
    print("=" * 120)
    print(f"  Put: {PUT_SHORT_DELTA}d short, {SPREAD_WIDTH}pt | Call: +{CALL_SHORT_DELTA}d short, {SPREAD_WIDTH}pt")
    print(f"  Put stop: {PUT_DELTA_STOP} | Call stop: +{CALL_DELTA_STOP}")

    print(f"\n  Loading data...")
    (daily_ret, dpdf, daily_smas, rfr_daily, vix_monthly, tbill_monthly,
     z_clean, asset_ret_fwd, actual_lev, both_start, options_by_year) = load_data()

    print(f"  Running Pod 1 (Faber)...")
    faber_daily = run_faber(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start)

    print(f"  Running Pod 2 (Iron Condor)...")
    condor_monthly, trades_df = run_condor(
        vix_monthly, tbill_monthly, z_clean, asset_ret_fwd, options_by_year, dpdf)

    print(f"  Pod 1: {faber_daily.index.min().date()} to {faber_daily.index.max().date()}")
    print(f"  Pod 2: {condor_monthly.index.min().date()} to {condor_monthly.index.max().date()}, {len(trades_df)} trades")

    pf, ps, pc = report(faber_daily, condor_monthly, trades_df, daily_ret)

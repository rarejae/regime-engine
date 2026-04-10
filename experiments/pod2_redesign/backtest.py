"""Pod 2 redesign: -0.20 delta, 45 DTE, delta stop -0.35.

Fix for the prior test where -0.10 delta / 30 DTE spreads only collected
$0.70 credit, making the 50% profit target bank only $0.35 per contract.

New spec: -0.20 delta at 45 DTE collects ~$1.80 credit.
50% profit target now banks ~$0.90 — more than the prior strategy's max profit.

Reuses Faber-Sweep-40 Pod 1 results from the prior comparison.
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

# REDESIGNED PARAMETERS
SHORT_DELTA = -0.20        # was -0.10
TARGET_DTE = 45            # was 30
DTE_MIN = 35               # was 21
DTE_MAX = 55               # was 40
DELTA_STOP = -0.35         # was -0.25
SPREAD_WIDTH = 5.0
COMMISSION = 2.60
SLIPPAGE = 0.10
MAX_CONTRACTS = 60
MARGIN_PCT = 0.30
STARTING_CAPITAL = 100_000


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
            p.index = pd.to_datetime(p.index).tz_localize(None)
            dp[our] = p
    dpdf = pd.DataFrame(dp).sort_index()
    daily_smas = {p: dpdf.rolling(p, min_periods=p).mean() for p in DAILY_SMA_PERIODS}

    from fredapi import Fred
    rfr_daily = pd.Series(0.0, index=daily_ret.index)
    key = os.environ.get("FRED_API_KEY")
    vix_monthly = None; tbill_monthly = None
    if key:
        f = Fred(api_key=key)
        try:
            tb = f.get_series("DTB3", observation_start="1998-01-01")
            tb.index = pd.to_datetime(tb.index); rfr_daily = (tb / 100 / 252).reindex(daily_ret.index, method="ffill").fillna(0)
            tbill_monthly = tb.resample("MS").last().dropna() / 100 / 12
        except: pass
        try:
            vix = f.get_series("VIXCLS", observation_start="2009-01-01")
            vix.index = pd.to_datetime(vix.index); vix_monthly = vix.resample("MS").last().dropna()
        except: pass
    if vix_monthly is None:
        vd = yf.download("^VIX", start="2009-01-01", progress=False)
        if vd is not None and not vd.empty:
            p = vd["Close"]
            if hasattr(p, "columns"): p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None)
            vix_monthly = p.resample("MS").last().dropna()

    macro = load_monthly_macro()
    z_data = compute_zscore_variables(macro)
    z_clean = z_data[[c for c in z_data.columns if c.endswith("_z")]].dropna()
    asset_ret_fwd = load_monthly_asset_returns().shift(-1)

    actual_lev = {}
    for ticker in ["SSO", "QLD"]:
        d = yf.download(ticker, start="2006-01-01", progress=False, auto_adjust=True)
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p, "columns"): p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None)
            actual_lev[ticker] = p.pct_change().dropna()
    both_start = max(actual_lev.get("SSO", pd.Series()).index.min(),
                     actual_lev.get("QLD", pd.Series()).index.min()) \
        if "SSO" in actual_lev and "QLD" in actual_lev else pd.Timestamp("2099-01-01")

    options_by_year = {}
    for yr in range(2010, 2024):
        p = Path(f"data/processed/spy_options_{yr}.parquet")
        if p.exists(): options_by_year[yr] = pd.read_parquet(p)

    return (daily_ret, dpdf, daily_smas, rfr_daily, vix_monthly, tbill_monthly,
            z_clean, asset_ret_fwd, actual_lev, both_start, options_by_year)


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


def find_contract(chain, opt_type, target_delta, dte_min, dte_max, dte_ideal):
    filt = chain[(chain["option_type"] == opt_type) &
                 (chain["dte"] >= dte_min) & (chain["dte"] <= dte_max) &
                 (chain["bid"] >= 0.05) & (chain["delta"].notna())].copy()
    if len(filt) == 0: return None
    filt["dte_dist"] = abs(filt["dte"] - dte_ideal)
    best_dte = filt["dte_dist"].min()
    filt = filt[filt["dte_dist"] <= best_dte + 5]
    filt["delta_dist"] = abs(filt["delta"] - target_delta)
    return filt.loc[filt["delta_dist"].idxmin()]


def get_spread_value(chain, trade_date, short_strike, long_strike, expiry):
    dc = chain[chain["trade_date"] == trade_date]
    if len(dc) == 0: return None, None
    so = dc[(dc["option_type"] == "P") & (dc["strike"] == short_strike) & (dc["expiry"] == expiry)]
    lo = dc[(dc["option_type"] == "P") & (dc["strike"] == long_strike) & (dc["expiry"] == expiry)]
    if len(so) == 0 or len(lo) == 0: return None, None
    return float(so.iloc[0]["mid"]) - float(lo.iloc[0]["mid"]), float(so.iloc[0]["delta"])


# ── Pod 1: Faber-Sweep-40 ───────────────────────────────────────────────────

def run_faber(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start):
    bt_start = pd.Timestamp("2010-01-01")
    trading_days = daily_ret.loc[bt_start:pd.Timestamp("2023-12-31")].index
    cur = {a: 3 for a in ASSETS if a != "cash"}
    wf = dict(BASELINE); fc = False; la = False; dlv = False; results = {}

    for day in trading_days:
        if day not in daily_ret.index: continue
        dr = daily_ret.loc[day]
        avail = [a for a in ASSETS if a in dr.index and pd.notna(dr[a])]
        if len(avail) < 3: continue
        actual = {a: float(dr[a]) for a in avail}
        rfr = float(rfr_daily.get(day, 0))
        is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day)-1].month)

        if is_ms:
            dlv = False
            prior = trading_days[trading_days < day]
            sd = prior[-1] if len(prior) > 0 else day
            cur = sma_scores(sd, dpdf, daily_smas)
            w1, pool = apply_faber_filter(cur, BASELINE)
            wf = dict(w1); wf["cash"] = wf.get("cash", 0) + pool
            fc = cur.get("IVV", 0) >= 3 and cur.get("QQQ", 0) >= 3; la = fc

        if la and not dlv:
            if check_breach(day, dpdf, daily_smas): la = False; dlv = True

        iw = wf.get("IVV", 0); qw = wf.get("QQQ", 0)
        ir = actual.get("IVV", 0); qr = actual.get("QQQ", 0)
        base = sum(wf.get(a, 0) * actual.get(a, 0) for a in avail if a not in ["IVV", "QQQ"])
        if la:
            sso = float(actual_lev.get("SSO", pd.Series()).get(day, 2*ir-rfr-SSO_EXP/252)) if day >= both_start else 2*ir-rfr-SSO_EXP/252
            qld = float(actual_lev.get("QLD", pd.Series()).get(day, 2*qr-rfr-QLD_EXP/252)) if day >= both_start else 2*qr-rfr-QLD_EXP/252
            if np.isnan(sso): sso = 2*ir-rfr-SSO_EXP/252
            if np.isnan(qld): qld = 2*qr-rfr-QLD_EXP/252
            results[day] = iw*sso + qw*qld + base
        else:
            results[day] = iw*ir + qw*qr + base
    return pd.Series(results).sort_index()


# ── Pod 2: Redesigned Spreads ────────────────────────────────────────────────

def run_spreads(vix_monthly, tbill_monthly, z_clean, asset_ret_fwd, options_by_year, dpdf):
    bt_start = pd.Timestamp("2010-01-01"); bt_end = pd.Timestamp("2023-12-31")
    trading_days = dpdf.index[(dpdf.index >= bt_start) & (dpdf.index <= bt_end)]

    month_ends = []
    for i, d in enumerate(trading_days):
        if i == len(trading_days)-1 or d.month != trading_days[i+1].month:
            month_ends.append(d)

    open_pos = None; trades = []; daily_pnl = {d: 0.0 for d in trading_days}

    for day in trading_days:
        yr = day.year; is_me = day in month_ends

        # ── Daily position monitoring ────────────────────────────────────
        if open_pos is not None:
            chain = options_by_year.get(yr)
            closed = False

            if chain is not None:
                sv, sd = get_spread_value(chain, day, open_pos["short_strike"],
                                           open_pos["long_strike"], open_pos["expiry"])
                if sv is not None and sd is not None:
                    ec = open_pos["entry_credit"]
                    dte_now = (open_pos["expiry"] - day).days
                    reason = None

                    if sv <= ec * 0.50: reason = "profit_target"
                    elif sd <= DELTA_STOP: reason = "delta_stop"
                    elif sv >= ec * 2.0: reason = "premium_stop"
                    elif dte_now <= 7: reason = "time_exit"

                    if reason:
                        exit_val = sv * (1 + SLIPPAGE)
                        pps = open_pos["actual_credit"] - exit_val
                        net = (pps * open_pos["contracts"] * 100) - (COMMISSION * open_pos["contracts"])
                        trades.append({
                            "entry_date": open_pos["entry_date"], "exit_date": day,
                            "hold_days": (day - open_pos["entry_date"]).days,
                            "exit_reason": reason,
                            "short_strike": open_pos["short_strike"],
                            "long_strike": open_pos["long_strike"],
                            "entry_credit": ec, "exit_value": sv, "exit_delta": sd,
                            "pnl_per_share": pps, "contracts": open_pos["contracts"],
                            "net_pnl": net, "vix": open_pos["vix"], "harvey_er": open_pos["harvey_er"],
                            "underlying_entry": open_pos["underlying"],
                        })
                        daily_pnl[day] = net; open_pos = None; closed = True

            # Expiry fallback
            if not closed and open_pos is not None and day >= open_pos["expiry"]:
                ep = dpdf.loc[:day, "IVV"].iloc[-1] if "IVV" in dpdf.columns else None
                if ep is not None:
                    ks = open_pos["short_strike"]; kl = open_pos["long_strike"]
                    ac = open_pos["actual_credit"]
                    if ep >= ks: pps = ac
                    elif ep <= kl: pps = ac - SPREAD_WIDTH
                    else: pps = ac - (ks - ep)
                    net = (pps * open_pos["contracts"] * 100) - (COMMISSION * open_pos["contracts"])
                    trades.append({
                        "entry_date": open_pos["entry_date"], "exit_date": day,
                        "hold_days": (day - open_pos["entry_date"]).days,
                        "exit_reason": "expiry",
                        "short_strike": ks, "long_strike": kl,
                        "entry_credit": open_pos["entry_credit"], "exit_value": 0, "exit_delta": 0,
                        "pnl_per_share": pps, "contracts": open_pos["contracts"],
                        "net_pnl": net, "vix": open_pos["vix"], "harvey_er": open_pos["harvey_er"],
                        "underlying_entry": open_pos["underlying"],
                    })
                    daily_pnl[day] = net
                open_pos = None

        # ── Month-end entry ──────────────────────────────────────────────
        if is_me and open_pos is None:
            vix_val = None
            if vix_monthly is not None:
                vd = vix_monthly.index[vix_monthly.index <= day]
                if len(vd) > 0: vix_val = float(vix_monthly.loc[vd[-1]])
            if vix_val is None or vix_val < 18: continue

            z_prior = z_clean.index[z_clean.index < day]
            harvey_er = 0.0
            if len(z_prior) > 0:
                try:
                    sim, _ = find_similar_months(z_clean, z_prior[-1])
                    er = compute_expected_returns(sim, asset_ret_fwd, ["IVV"])
                    harvey_er = er.get("IVV", 0.0)
                except ValueError: pass
            if harvey_er <= 0.005: continue

            chain = options_by_year.get(yr)
            if chain is None: continue
            cd = chain["trade_date"].unique(); valid = cd[cd <= day]
            if len(valid) == 0: continue
            trade_date = valid[-1]

            dc = chain[chain["trade_date"] == trade_date]
            short = find_contract(dc, "P", SHORT_DELTA, DTE_MIN, DTE_MAX, TARGET_DTE)
            if short is None: continue

            long_strike = float(short["strike"]) - SPREAD_WIDTH
            longs = dc[(dc["option_type"] == "P") & (dc["expiry"] == short["expiry"]) &
                       (abs(dc["strike"] - long_strike) < 1.0)]
            if len(longs) == 0: continue
            lng = longs.iloc[0]

            nc = float(short["mid"]) - float(lng["mid"])
            if nc <= 0: continue
            actual_credit = nc * (1 - SLIPPAGE)
            margin_per = (SPREAD_WIDTH - nc) * 100
            if margin_per <= 0: continue
            contracts = min(int(STARTING_CAPITAL * MARGIN_PCT / margin_per), MAX_CONTRACTS)
            contracts = max(contracts, 1)

            open_pos = {
                "entry_date": day, "short_strike": float(short["strike"]),
                "long_strike": float(lng["strike"]),
                "entry_credit": nc, "actual_credit": actual_credit,
                "expiry": pd.Timestamp(short["expiry"]),
                "contracts": contracts, "underlying": float(short["underlying_close"]),
                "vix": vix_val, "harvey_er": harvey_er,
            }

    # Monthly P&L
    pnl_s = pd.Series(daily_pnl).sort_index()
    monthly_pnl = {}
    for me in month_ends:
        ms = pd.Timestamp(f"{me.year}-{me.month:02d}-01")
        mp = pnl_s[(pnl_s.index >= ms) & (pnl_s.index <= me)].sum()
        if mp == 0:
            tb_ret = 0.0
            if tbill_monthly is not None:
                tbd = tbill_monthly.index[tbill_monthly.index <= me]
                if len(tbd) > 0: tb_ret = float(tbill_monthly.loc[tbd[-1]])
            monthly_pnl[ms] = tb_ret * STARTING_CAPITAL
        else:
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


def report(faber_daily, spread_monthly, trades_df, daily_ret):
    faber_m = faber_daily.resample("MS").apply(lambda x: (1+x).prod()-1)
    spread_r = spread_monthly / STARTING_CAPITAL
    common = faber_m.dropna().index.intersection(spread_r.dropna().index).sort_values()
    f = faber_m.reindex(common); s = spread_r.reindex(common)
    c = 0.90 * f + 0.10 * s

    ivv_d = daily_ret["IVV"].loc["2010-01-01":"2023-12-31"].dropna()
    vglt_d = daily_ret.get("VGLT", pd.Series(dtype=float)).loc["2010-01-01":"2023-12-31"].fillna(0)
    ivv_m = ivv_d.resample("MS").apply(lambda x: (1+x).prod()-1).reindex(common, fill_value=0)
    b60_m = (0.6*ivv_d + 0.4*vglt_d.reindex(ivv_d.index, fill_value=0)).resample("MS").apply(
        lambda x: (1+x).prod()-1).reindex(common, fill_value=0)

    pf = metrics(f); ps = metrics(s); pc = metrics(c); pi = metrics(ivv_m); p6 = metrics(b60_m)

    # ── 1. Standalone Pod 2 ──────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  1. POD 2 STANDALONE (Redesigned: -0.20 delta, 45 DTE)")
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
        print(f"  Avg credit collected: ${trades_df['entry_credit'].mean():.2f}")

    # ── 2. Three-portfolio comparison ────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  2. FULL THREE-PORTFOLIO COMPARISON")
    print(f"{'='*120}")
    print(f"\n  {'Portfolio':<22} {'Return':>8} {'Vol':>7} {'Sharpe':>8} {'Sortino':>8} "
          f"{'MaxDD':>8} {'Calmar':>8} {'Terminal':>12}")
    print(f"  {'-'*22} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*12}")
    for p, label in [(pf, "A: Faber Only"), (ps, "B: Spreads (new)"), (pc, "C: Combined 90/10"),
                      (pi, "IVV B&H"), (p6, "60/40")]:
        t = STARTING_CAPITAL * p["final"]
        print(f"  {label:<22} {p['ar']:>7.1%} {p['av']:>6.1%} {p['sh']:>8.3f} {p['sortino']:>8.3f} "
              f"{p['dd']:>7.1%} {p['calmar']:>8.2f} ${t:>11,.0f}")

    # ── 3. Annual NAV ────────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  3. ANNUAL NAV TABLE")
    print(f"{'='*120}")
    cum_f = (1+f).cumprod()*STARTING_CAPITAL
    cum_s = (1+s).cumprod()*STARTING_CAPITAL
    cum_c = (1+c).cumprod()*STARTING_CAPITAL
    cum_i = (1+ivv_m).cumprod()*STARTING_CAPITAL
    print(f"\n  {'Year':>6} {'A: Faber':>14} {'B: Spreads':>14} {'C: Combined':>14} {'IVV B&H':>14}")
    for yr in range(2010, 2024):
        for cum, label in [(cum_f, ""), (cum_s, ""), (cum_c, ""), (cum_i, "")]:
            pass
        yf = cum_f[cum_f.index.year == yr]; ys = cum_s[cum_s.index.year == yr]
        yc = cum_c[cum_c.index.year == yr]; yi = cum_i[cum_i.index.year == yr]
        if len(yf) > 0:
            print(f"  {yr:>6} ${yf.iloc[-1]:>13,.0f} ${ys.iloc[-1]:>13,.0f} "
                  f"${yc.iloc[-1]:>13,.0f} ${yi.iloc[-1]:>13,.0f}")

    # ── 4. Crisis analysis ───────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  4. CRISIS ANALYSIS")
    print(f"{'='*120}")
    for cname, cs, ce in [("2011 correction", "2011-07", "2011-10"),
                           ("2018 Q4", "2018-10", "2018-12"),
                           ("COVID Feb-Mar 2020", "2020-02", "2020-03"),
                           ("2022 bear", "2022-01", "2022-10")]:
        print(f"\n  {cname}:")
        print(f"  {'Portfolio':<22} {'Return':>10} {'MaxDD':>10}")
        print(f"  {'-'*22} {'-'*10} {'-'*10}")
        for sr, label in [(f, "A: Faber"), (s, "B: Spreads"), (c, "C: Combined")]:
            cr = sr[(sr.index >= pd.Timestamp(cs)) & (sr.index <= pd.Timestamp(ce))]
            if len(cr) > 0:
                cum = (1+cr).cumprod(); mdd = ((cum-cum.expanding().max())/cum.expanding().max()).min()
                print(f"  {label:<22} {(1+cr).prod()-1:>+9.1%} {mdd:>9.1%}")

    # ── 5. Correlation ───────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  5. CORRELATION ANALYSIS")
    print(f"{'='*120}")
    corr = float(f.corr(s))
    print(f"\n  Monthly correlation (Faber vs Spreads): {corr:.3f}")
    rc = f.rolling(12).corr(s).dropna()
    print(f"  Rolling 12-month by year:")
    for yr in range(2011, 2024):
        yrc = rc[rc.index.year == yr]
        if len(yrc) > 0: print(f"    {yr}: {yrc.mean():.2f}")
    print(f"\n  Crisis correlations:")
    for cname, cs, ce in [("2011", "2011-07", "2011-11"), ("2018 Q4", "2018-10", "2018-12"),
                           ("COVID", "2020-02", "2020-04"), ("2022", "2022-01", "2022-12")]:
        cf = f[(f.index >= pd.Timestamp(cs)) & (f.index <= pd.Timestamp(ce))]
        cs_r = s[(s.index >= pd.Timestamp(cs)) & (s.index <= pd.Timestamp(ce))]
        ci = cf.index.intersection(cs_r.index)
        if len(ci) >= 3: print(f"    {cname}: {cf.reindex(ci).corr(cs_r.reindex(ci)):.2f}")
    both_loss = common[(f.reindex(common) < -0.01) & (s.reindex(common) < -0.01)]
    print(f"\n  Months where both lost >1%: {len(both_loss)}")
    for dt in both_loss:
        print(f"    {dt.strftime('%Y-%m')}: Faber {f.loc[dt]*100:+.1f}%, Spreads {s.loc[dt]*100:+.1f}%")

    # ── 6. Position management ───────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  6. POSITION MANAGEMENT BREAKDOWN")
    print(f"{'='*120}")
    if len(trades_df) > 0:
        print(f"\n  {'Exit Rule':<20} {'Trades':>8} {'%':>6} {'Avg Hold':>10} {'Avg P&L':>12}")
        print(f"  {'-'*20} {'-'*8} {'-'*6} {'-'*10} {'-'*12}")
        for reason in ["profit_target", "delta_stop", "premium_stop", "time_exit", "expiry"]:
            rt = trades_df[trades_df["exit_reason"] == reason]
            if len(rt) > 0:
                print(f"  {reason:<20} {len(rt):>8} {len(rt)/len(trades_df)*100:>5.0f}% "
                      f"{rt['hold_days'].mean():>9.0f}d ${rt['net_pnl'].mean():>+11,.0f}")
        print(f"\n  Largest loss: ${trades_df['net_pnl'].min():,.0f} ({trades_df.loc[trades_df['net_pnl'].idxmin(), 'exit_date'].strftime('%Y-%m-%d')})")
        print(f"  Largest win:  ${trades_df['net_pnl'].max():,.0f}")

    # ── 7. Value-add ─────────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  7. COMBINED PORTFOLIO VALUE-ADD")
    print(f"{'='*120}")
    sh_imp = pc["sh"] - pf["sh"]; dd_imp = (pc["dd"] - pf["dd"])*100
    term_cost = (pc["final"] - pf["final"]) * STARTING_CAPITAL
    print(f"\n  Sharpe improvement (C vs A):   {sh_imp:+.3f} ({pf['sh']:.3f} → {pc['sh']:.3f})")
    print(f"  Max DD improvement (C vs A):   {dd_imp:+.1f}% ({pf['dd']:.1%} → {pc['dd']:.1%})")
    print(f"  Terminal wealth cost (C vs A):  ${term_cost:+,.0f}")
    print(f"  Correlation:                    {corr:.3f}")

    worth = pc["sh"] > pf["sh"] and pc["dd"] > pf["dd"]
    print(f"\n  Is Pod 2 worth adding? {'YES' if worth else 'NO'} — ", end="")
    if worth:
        print(f"+{sh_imp:.3f} Sharpe, {dd_imp:+.1f}% DD, ${term_cost:+,.0f} terminal")
    else:
        print(f"Sharpe {sh_imp:+.3f}, DD {dd_imp:+.1f}%, terminal cost ${term_cost:+,.0f}")

    # ── 8. Comparison vs prior ───────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  8. PARAMETER CHANGE IMPACT (vs prior -0.10d/30DTE test)")
    print(f"{'='*120}")
    avg_credit = trades_df["entry_credit"].mean() if len(trades_df) > 0 else 0
    wins = trades_df[trades_df["net_pnl"] > 0] if len(trades_df) > 0 else pd.DataFrame()
    losses = trades_df[trades_df["net_pnl"] <= 0] if len(trades_df) > 0 else pd.DataFrame()
    avg_win = wins["net_pnl"].mean() if len(wins) > 0 else 0
    avg_loss = losses["net_pnl"].mean() if len(losses) > 0 else 0
    wr = len(wins)/len(trades_df)*100 if len(trades_df) > 0 else 0

    print(f"\n  {'':>25} {'Prior (-0.10d,30DTE)':>22} {'New (-0.20d,45DTE)':>22}")
    print(f"  {'-'*25} {'-'*22} {'-'*22}")
    print(f"  {'Avg credit collected':>25} {'$0.70':>22} ${avg_credit:>21.2f}")
    print(f"  {'50% profit target':>25} {'$0.35':>22} ${avg_credit*0.5:>21.2f}")
    print(f"  {'Avg winning trade':>25} {'$709':>22} ${avg_win:>21,.0f}")
    print(f"  {'Avg losing trade':>25} {'-$3,193':>22} ${avg_loss:>21,.0f}")
    print(f"  {'Win rate':>25} {'81%':>22} {wr:>21.0f}%")
    print(f"  {'Total trades':>25} {'32':>22} {len(trades_df):>22}")
    print(f"  {'Standalone Sharpe':>25} {'0.314':>22} {ps['sh']:>22.3f}")
    print(f"  {'Standalone return':>25} {'0.8%':>22} {ps['ar']:>21.1%}")
    print(f"  {'Combined Sharpe':>25} {'1.072':>22} {pc['sh']:>22.3f}")
    print(f"  {'Combined terminal':>25} {'$646,768':>22} ${STARTING_CAPITAL*pc['final']:>21,.0f}")

    print(f"\n  Did -0.20 delta / 45 DTE fix the problem? ", end="")
    if ps["sh"] > 0.5 and pc["sh"] > pf["sh"]:
        print("YES — meaningful standalone return and combined improvement")
    elif ps["sh"] > 0.314:
        print("PARTIALLY — improved but still marginal")
    else:
        print("NO — still insufficient standalone return")

    print()
    return pf, ps, pc


if __name__ == "__main__":
    print("=" * 120)
    print("  POD 2 REDESIGN: -0.20 Delta, 45 DTE, Delta Stop -0.35")
    print("=" * 120)
    print(f"  Short delta: {SHORT_DELTA}, DTE: {DTE_MIN}-{DTE_MAX} (target {TARGET_DTE})")
    print(f"  Delta stop: {DELTA_STOP}, Spread width: {SPREAD_WIDTH}pt")

    print(f"\n  Loading data...")
    (daily_ret, dpdf, daily_smas, rfr_daily, vix_monthly, tbill_monthly,
     z_clean, asset_ret_fwd, actual_lev, both_start, options_by_year) = load_data()

    print(f"  Running Pod 1 (Faber-Sweep-40)...")
    faber_daily = run_faber(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start)

    print(f"  Running Pod 2 (Redesigned spreads: -0.20d, 45 DTE)...")
    spread_monthly, trades_df = run_spreads(
        vix_monthly, tbill_monthly, z_clean, asset_ret_fwd, options_by_year, dpdf)

    print(f"  Pod 1: {faber_daily.index.min().date()} to {faber_daily.index.max().date()}")
    print(f"  Pod 2: {spread_monthly.index.min().date()} to {spread_monthly.index.max().date()}, {len(trades_df)} trades")

    pf, ps, pc = report(faber_daily, spread_monthly, trades_df, daily_ret)

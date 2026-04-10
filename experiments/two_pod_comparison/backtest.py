"""Three-portfolio direct comparison: Faber-only, Spreads-only, Combined 90/10.

Pod 2 uses fully specified position management:
  Entry: VIX>18, Harvey ER>+0.005, sell -0.10 delta put spread (5pt wide), 30 DTE
  Exit Rule 1 (profit): spread value ≤ 50% of entry credit → close
  Exit Rule 2 (delta stop): short delta ≤ -0.25 → close
  Exit Rule 3 (premium stop): spread value ≥ 2× entry credit → close
  Exit Rule 4 (time): DTE ≤ 7 → close
  No re-entry until next month-end evaluation.

Period: 2010-01-01 to 2023-12-31, $100K starting capital.
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
            tb.index = pd.to_datetime(tb.index)
            rfr_daily = (tb / 100 / 252).reindex(daily_ret.index, method="ffill").fillna(0)
            tbill_monthly = tb.resample("MS").last().dropna() / 100 / 12
        except: pass
        try:
            vix = f.get_series("VIXCLS", observation_start="2009-01-01")
            vix.index = pd.to_datetime(vix.index)
            vix_monthly = vix.resample("MS").last().dropna()
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


# ── Helpers ──────────────────────────────────────────────────────────────────

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

def find_contract(chain, opt_type, target_delta, dte_min=21, dte_max=40):
    filt = chain[(chain["option_type"] == opt_type) &
                 (chain["dte"] >= dte_min) & (chain["dte"] <= dte_max) &
                 (chain["bid"] >= 0.05) & (chain["delta"].notna())].copy()
    if len(filt) == 0: return None
    filt["dte_dist"] = abs(filt["dte"] - 30)
    best_dte = filt["dte_dist"].min()
    filt = filt[filt["dte_dist"] <= best_dte + 5]
    filt["delta_dist"] = abs(filt["delta"] - target_delta)
    return filt.loc[filt["delta_dist"].idxmin()]

def get_spread_value(chain, trade_date, short_strike, long_strike, expiry):
    """Get current mid value of spread from chain data."""
    dc = chain[chain["trade_date"] == trade_date]
    if len(dc) == 0: return None, None
    short_opt = dc[(dc["option_type"] == "P") & (dc["strike"] == short_strike) & (dc["expiry"] == expiry)]
    long_opt = dc[(dc["option_type"] == "P") & (dc["strike"] == long_strike) & (dc["expiry"] == expiry)]
    if len(short_opt) == 0 or len(long_opt) == 0: return None, None
    spread_val = float(short_opt.iloc[0]["mid"]) - float(long_opt.iloc[0]["mid"])
    short_delta = float(short_opt.iloc[0]["delta"])
    return spread_val, short_delta


# ── Pod 1: Faber-Sweep-40 ───────────────────────────────────────────────────

def run_faber(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start):
    bt_start = pd.Timestamp("2010-01-01")
    trading_days = daily_ret.loc[bt_start:pd.Timestamp("2023-12-31")].index
    cur = {a: 3 for a in ASSETS if a != "cash"}
    wf = dict(BASELINE); fc = False; la = False; dlv = False
    results = {}

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

        if la and day >= both_start:
            sso = float(actual_lev.get("SSO", pd.Series()).get(day, 2*ir - rfr - SSO_EXP/252))
            qld = float(actual_lev.get("QLD", pd.Series()).get(day, 2*qr - rfr - QLD_EXP/252))
            if np.isnan(sso): sso = 2*ir - rfr - SSO_EXP/252
            if np.isnan(qld): qld = 2*qr - rfr - QLD_EXP/252
            results[day] = iw * sso + qw * qld + base
        elif la:
            results[day] = iw * (2*ir - rfr - SSO_EXP/252) + qw * (2*qr - rfr - QLD_EXP/252) + base
        else:
            results[day] = iw * ir + qw * qr + base

    return pd.Series(results).sort_index()


# ── Pod 2: Harvey Spreads with Position Management ──────────────────────────

def run_spreads(vix_monthly, tbill_monthly, z_clean, asset_ret_fwd, options_by_year, dpdf):
    bt_start = pd.Timestamp("2010-01-01")
    bt_end = pd.Timestamp("2023-12-31")
    trading_days = dpdf.index[(dpdf.index >= bt_start) & (dpdf.index <= bt_end)]

    month_ends = []
    for i, d in enumerate(trading_days):
        if i == len(trading_days)-1 or d.month != trading_days[i+1].month:
            month_ends.append(d)

    # State
    open_position = None  # dict with details if position open
    trades = []
    daily_pnl = {}  # day → pnl contribution

    # Set daily T-bill for idle cash
    for day in trading_days:
        daily_pnl[day] = 0.0  # default: no contribution

    for day in trading_days:
        yr = day.year
        is_me = day in month_ends

        # ── Daily position monitoring (if open) ─────────────────────────
        if open_position is not None:
            chain = options_by_year.get(yr)
            if chain is not None:
                sv, sd = get_spread_value(chain, day,
                    open_position["short_strike"], open_position["long_strike"],
                    open_position["expiry"])

                if sv is not None and sd is not None:
                    entry_credit = open_position["entry_credit"]
                    dte_now = (open_position["expiry"] - day).days
                    close_reason = None

                    # Rule 1: Profit target (spread value ≤ 50% of entry credit)
                    if sv <= entry_credit * 0.50:
                        close_reason = "profit_target"
                    # Rule 2: Delta stop (short delta ≤ -0.25)
                    elif sd <= -0.25:
                        close_reason = "delta_stop"
                    # Rule 3: Premium stop (spread value ≥ 2× entry credit)
                    elif sv >= entry_credit * 2.0:
                        close_reason = "premium_stop"
                    # Rule 4: Time exit (DTE ≤ 7)
                    elif dte_now <= 7:
                        close_reason = "time_exit"

                    if close_reason:
                        actual_exit_value = sv * (1 + SLIPPAGE)  # paying to close
                        pps = open_position["actual_credit"] - actual_exit_value
                        contracts = open_position["contracts"]
                        net = (pps * contracts * 100) - (COMMISSION * contracts)
                        hold_days = (day - open_position["entry_date"]).days

                        trades.append({
                            "entry_date": open_position["entry_date"],
                            "exit_date": day,
                            "hold_days": hold_days,
                            "exit_reason": close_reason,
                            "short_strike": open_position["short_strike"],
                            "long_strike": open_position["long_strike"],
                            "entry_credit": entry_credit,
                            "exit_value": sv,
                            "pnl_per_share": pps,
                            "contracts": contracts,
                            "net_pnl": net,
                            "entry_underlying": open_position["underlying"],
                            "exit_delta": sd,
                            "vix": open_position["vix"],
                            "harvey_er": open_position["harvey_er"],
                        })
                        daily_pnl[day] = net
                        open_position = None

                # If position expired (past expiry date)
                if open_position is not None and day >= open_position["expiry"]:
                    # Close at expiry
                    exp_price = dpdf.loc[:day, "IVV"].iloc[-1] if "IVV" in dpdf.columns else None
                    if exp_price is not None:
                        ks = open_position["short_strike"]
                        kl = open_position["long_strike"]
                        ac = open_position["actual_credit"]
                        if exp_price >= ks: pps = ac
                        elif exp_price <= kl: pps = ac - SPREAD_WIDTH
                        else: pps = ac - (ks - exp_price)
                        contracts = open_position["contracts"]
                        net = (pps * contracts * 100) - (COMMISSION * contracts)

                        trades.append({
                            "entry_date": open_position["entry_date"],
                            "exit_date": day, "hold_days": (day - open_position["entry_date"]).days,
                            "exit_reason": "expiry",
                            "short_strike": ks, "long_strike": kl,
                            "entry_credit": open_position["entry_credit"],
                            "exit_value": 0, "pnl_per_share": pps,
                            "contracts": contracts, "net_pnl": net,
                            "entry_underlying": open_position["underlying"],
                            "exit_delta": 0, "vix": open_position["vix"],
                            "harvey_er": open_position["harvey_er"],
                        })
                        daily_pnl[day] = net
                    open_position = None

        # ── Month-end entry check ───────────────────────────────────────
        if is_me and open_position is None:
            # VIX
            vix_val = None
            if vix_monthly is not None:
                vd = vix_monthly.index[vix_monthly.index <= day]
                if len(vd) > 0: vix_val = float(vix_monthly.loc[vd[-1]])
            if vix_val is None or vix_val < 18: continue

            # Harvey
            z_prior = z_clean.index[z_clean.index < day]
            harvey_er = 0.0
            if len(z_prior) > 0:
                try:
                    sim, _ = find_similar_months(z_clean, z_prior[-1])
                    er = compute_expected_returns(sim, asset_ret_fwd, ["IVV"])
                    harvey_er = er.get("IVV", 0.0)
                except ValueError: pass
            if harvey_er <= 0.005: continue

            # Find contracts
            chain = options_by_year.get(yr)
            if chain is None: continue
            cd = chain["trade_date"].unique()
            valid = cd[cd <= day]
            if len(valid) == 0: continue
            trade_date = valid[-1]

            dc = chain[chain["trade_date"] == trade_date]
            short = find_contract(dc, "P", -0.10, 21, 40)
            if short is None: continue

            long_strike = float(short["strike"]) - SPREAD_WIDTH
            longs = dc[(dc["option_type"] == "P") &
                       (dc["expiry"] == short["expiry"]) &
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

            open_position = {
                "entry_date": day, "short_strike": float(short["strike"]),
                "long_strike": float(lng["strike"]),
                "entry_credit": nc, "actual_credit": actual_credit,
                "expiry": pd.Timestamp(short["expiry"]),
                "contracts": contracts, "underlying": float(short["underlying_close"]),
                "vix": vix_val, "harvey_er": harvey_er,
                "short_delta_entry": float(short["delta"]),
            }

    # Convert to monthly returns
    pnl_s = pd.Series(daily_pnl).sort_index()
    # For months with no trade: earn T-bill
    monthly_pnl = {}
    for me in month_ends:
        ms = pd.Timestamp(f"{me.year}-{me.month:02d}-01")
        month_days = pnl_s[(pnl_s.index >= ms) & (pnl_s.index <= me)]
        trade_pnl = month_days.sum()
        if trade_pnl == 0:
            # T-bill income
            tb_ret = 0.0
            if tbill_monthly is not None:
                tbd = tbill_monthly.index[tbill_monthly.index <= me]
                if len(tbd) > 0: tb_ret = float(tbill_monthly.loc[tbd[-1]])
            monthly_pnl[ms] = tb_ret * STARTING_CAPITAL
        else:
            monthly_pnl[ms] = trade_pnl

    return pd.Series(monthly_pnl).sort_index(), pd.DataFrame(trades) if trades else pd.DataFrame()


# ── Metrics ──────────────────────────────────────────────────────────────────

def metrics(rets):
    ar = rets.mean() * 12; av = rets.std() * np.sqrt(12)
    sh = ar / av if av > 0 else 0
    neg = rets[rets < 0]; ds = neg.std() * np.sqrt(12) if len(neg) > 3 else av
    so = ar / ds if ds > 0 else 0
    cum = (1 + rets).cumprod()
    dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
    cal = ar / abs(dd) if dd != 0 else 0
    return {"ar": ar, "av": av, "sh": sh, "sortino": so, "dd": dd, "calmar": cal, "final": cum.iloc[-1]}


# ── Report ───────────────────────────────────────────────────────────────────

def report(faber_daily, spread_monthly, trades_df, daily_ret):
    bt_start = pd.Timestamp("2010-01-01")

    # Monthly returns
    faber_m = faber_daily.resample("MS").apply(lambda x: (1 + x).prod() - 1)
    spread_r = spread_monthly / STARTING_CAPITAL

    # Align
    common = faber_m.dropna().index.intersection(spread_r.dropna().index).sort_values()
    f = faber_m.reindex(common); s = spread_r.reindex(common)

    # Combined 90/10
    c = 0.90 * f + 0.10 * s

    # Benchmarks
    ivv_daily = daily_ret["IVV"].loc[bt_start:"2023-12-31"].dropna()
    vglt_daily = daily_ret.get("VGLT", pd.Series(dtype=float)).loc[bt_start:"2023-12-31"].fillna(0)
    ivv_m = ivv_daily.resample("MS").apply(lambda x: (1 + x).prod() - 1).reindex(common, fill_value=0)
    b60_m = (0.6 * ivv_daily + 0.4 * vglt_daily.reindex(ivv_daily.index, fill_value=0)).resample("MS").apply(
        lambda x: (1 + x).prod() - 1).reindex(common, fill_value=0)

    pf = metrics(f); ps = metrics(s); pc = metrics(c)
    pi = metrics(ivv_m); p6 = metrics(b60_m)

    # ── 1. Performance table ─────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  1. FULL PERFORMANCE TABLE ($100,000 starting capital, 2010-2023)")
    print(f"{'='*120}")

    print(f"\n  {'Portfolio':<22} {'Return':>8} {'Vol':>7} {'Sharpe':>8} {'Sortino':>8} "
          f"{'MaxDD':>8} {'Calmar':>8} {'Terminal':>12}")
    print(f"  {'-'*22} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*12}")
    for p, label in [(pf, "A: Faber Only"), (ps, "B: Spreads Only"), (pc, "C: Combined 90/10"),
                      (pi, "IVV B&H"), (p6, "60/40")]:
        terminal = STARTING_CAPITAL * p["final"]
        print(f"  {label:<22} {p['ar']:>7.1%} {p['av']:>6.1%} {p['sh']:>8.3f} {p['sortino']:>8.3f} "
              f"{p['dd']:>7.1%} {p['calmar']:>8.2f} ${terminal:>11,.0f}")

    # ── 2. Annual NAV ────────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  2. ANNUAL NAV TABLE")
    print(f"{'='*120}")

    print(f"\n  {'Year':>6} {'A: Faber':>14} {'B: Spreads':>14} {'C: Combined':>14} {'IVV B&H':>14}")
    cum_f = (1 + f).cumprod() * STARTING_CAPITAL
    cum_s = (1 + s).cumprod() * STARTING_CAPITAL
    cum_c = (1 + c).cumprod() * STARTING_CAPITAL
    cum_i = (1 + ivv_m).cumprod() * STARTING_CAPITAL

    for yr in range(2010, 2024):
        yr_f = cum_f[cum_f.index.year == yr]
        yr_s = cum_s[cum_s.index.year == yr]
        yr_c = cum_c[cum_c.index.year == yr]
        yr_i = cum_i[cum_i.index.year == yr]
        if len(yr_f) > 0:
            print(f"  {yr:>6} ${yr_f.iloc[-1]:>13,.0f} ${yr_s.iloc[-1]:>13,.0f} "
                  f"${yr_c.iloc[-1]:>13,.0f} ${yr_i.iloc[-1]:>13,.0f}")

    # ── 3. Crisis analysis ───────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  3. CRISIS ANALYSIS")
    print(f"{'='*120}")

    for cname, cs, ce in [("2011 correction", "2011-07", "2011-10"),
                           ("2015-16 vol", "2015-08", "2016-02"),
                           ("2018 Q4", "2018-10", "2018-12"),
                           ("COVID Feb-Mar 2020", "2020-02", "2020-03"),
                           ("2022 bear", "2022-01", "2022-10")]:
        print(f"\n  {cname}:")
        print(f"  {'Portfolio':<22} {'Return':>10} {'MaxDD':>10}")
        print(f"  {'-'*22} {'-'*10} {'-'*10}")
        for sr, label in [(f, "A: Faber"), (s, "B: Spreads"), (c, "C: Combined"), (ivv_m, "IVV B&H")]:
            cr = sr[(sr.index >= pd.Timestamp(cs)) & (sr.index <= pd.Timestamp(ce))]
            if len(cr) > 0:
                cum = (1 + cr).cumprod()
                mdd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
                print(f"  {label:<22} {(1+cr).prod()-1:>+9.1%} {mdd:>9.1%}")

    # ── 4. Correlation ───────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  4. CORRELATION ANALYSIS")
    print(f"{'='*120}")

    corr = float(f.corr(s))
    print(f"\n  Monthly return correlation (Faber vs Spreads): {corr:.3f}")

    # Rolling
    rc = f.rolling(12).corr(s).dropna()
    print(f"\n  Rolling 12-month correlation by year:")
    for yr in range(2011, 2024):
        yrc = rc[rc.index.year == yr]
        if len(yrc) > 0:
            print(f"    {yr}: {yrc.mean():.2f}")

    # Crisis correlations
    print(f"\n  Crisis correlations:")
    for cname, cs, ce in [("2011", "2011-07", "2011-11"), ("2018 Q4", "2018-10", "2018-12"),
                           ("COVID", "2020-02", "2020-04"), ("2022", "2022-01", "2022-12")]:
        cf = f[(f.index >= pd.Timestamp(cs)) & (f.index <= pd.Timestamp(ce))]
        cs_r = s[(s.index >= pd.Timestamp(cs)) & (s.index <= pd.Timestamp(ce))]
        ci = cf.index.intersection(cs_r.index)
        if len(ci) >= 3:
            print(f"    {cname}: {cf.reindex(ci).corr(cs_r.reindex(ci)):.2f}")

    # Loss clustering
    both_loss = common[(f.reindex(common) < -0.01) & (s.reindex(common) < -0.01)]
    print(f"\n  Months where both pods lost >1%: {len(both_loss)}")
    for dt in both_loss:
        print(f"    {dt.strftime('%Y-%m')}: Faber {f.loc[dt]*100:+.1f}%, Spreads {s.loc[dt]*100:+.1f}%")

    # ── 5. Position management ───────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  5. POSITION MANAGEMENT EFFECTIVENESS")
    print(f"{'='*120}")

    if len(trades_df) > 0:
        print(f"\n  Total spreads opened: {len(trades_df)}")

        for reason in ["profit_target", "delta_stop", "premium_stop", "time_exit", "expiry"]:
            rt = trades_df[trades_df["exit_reason"] == reason]
            if len(rt) > 0:
                avg_hold = rt["hold_days"].mean()
                avg_pnl = rt["net_pnl"].mean()
                print(f"    {reason:<20}: {len(rt):>3} trades ({len(rt)/len(trades_df)*100:>4.0f}%) "
                      f"avg hold {avg_hold:>4.0f} days  avg P&L ${avg_pnl:>+8,.0f}")

        wins = trades_df[trades_df["net_pnl"] > 0]
        losses = trades_df[trades_df["net_pnl"] <= 0]
        print(f"\n  Win rate: {len(wins)/len(trades_df)*100:.0f}%")
        if len(wins) > 0: print(f"  Avg winning trade: ${wins['net_pnl'].mean():,.0f}")
        if len(losses) > 0: print(f"  Avg losing trade: ${losses['net_pnl'].mean():,.0f}")
        print(f"  Largest single loss: ${trades_df['net_pnl'].min():,.0f} ({trades_df.loc[trades_df['net_pnl'].idxmin(), 'exit_date'].strftime('%Y-%m-%d') if len(trades_df) > 0 else 'N/A'})")
        print(f"  Largest single win: ${trades_df['net_pnl'].max():,.0f}")
    else:
        print(f"\n  No trades executed.")

    # ── 6. Combined value-add ────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  6. COMBINED PORTFOLIO VALUE-ADD")
    print(f"{'='*120}")

    sh_imp = pc["sh"] - pf["sh"]
    dd_imp = (pc["dd"] - pf["dd"]) * 100
    term_cost = (pc["final"] - pf["final"]) * STARTING_CAPITAL

    print(f"\n  Sharpe improvement (C vs A): {sh_imp:+.3f} ({pf['sh']:.3f} → {pc['sh']:.3f})")
    print(f"  Max DD improvement (C vs A): {dd_imp:+.1f}% ({pf['dd']:.1%} → {pc['dd']:.1%})")
    print(f"  Terminal wealth cost (C vs A): ${term_cost:+,.0f}")
    print(f"  Correlation Faber/Spreads: {corr:.3f}")

    worth_it = pc["sh"] > pf["sh"] and pc["dd"] > pf["dd"]
    print(f"\n  Is Pod 2 worth adding? {'YES' if worth_it else 'CONDITIONAL'} — ", end="")
    if worth_it:
        print(f"+{sh_imp:.3f} Sharpe, +{dd_imp:.1f}% DD improvement, correlation {corr:.3f}")
    else:
        print(f"Sharpe {sh_imp:+.3f}, DD {dd_imp:+.1f}%, trade terminal cost ${term_cost:+,.0f}")

    # ── 7. Plain language summary ────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  7. PLAIN LANGUAGE SUMMARY")
    print(f"{'='*120}")

    print(f"\n  Starting with $100,000 in January 2010:")
    print(f"\n  Portfolio A (Faber Only) reached ${STARTING_CAPITAL * pf['final']:,.0f} by December 2023")
    print(f"  Portfolio B (Spreads Only) reached ${STARTING_CAPITAL * ps['final']:,.0f} by December 2023")
    print(f"  Portfolio C (Combined) reached ${STARTING_CAPITAL * pc['final']:,.0f} by December 2023")
    print(f"  IVV Buy & Hold reached ${STARTING_CAPITAL * pi['final']:,.0f} by December 2023")

    print(f"\n  Faber/Spreads monthly correlation: {corr:.3f}")
    print(f"  Combined Sharpe vs Faber-only: {pc['sh']:.3f} vs {pf['sh']:.3f} ({sh_imp:+.3f})")
    print(f"  Combined Max DD vs Faber-only: {pc['dd']:.1%} vs {pf['dd']:.1%} ({dd_imp:+.1f}%)")

    if len(trades_df) > 0:
        print(f"\n  The position management rules fired:")
        for reason in ["delta_stop", "profit_target", "time_exit", "premium_stop"]:
            rt = trades_df[trades_df["exit_reason"] == reason]
            if len(rt) > 0:
                print(f"    {reason}: {len(rt)} times, avg P&L ${rt['net_pnl'].mean():+,.0f}")

    print()


if __name__ == "__main__":
    print("=" * 120)
    print("  THREE-PORTFOLIO COMPARISON: Faber-Only vs Spreads-Only vs Combined 90/10")
    print("=" * 120)

    print(f"\n  Loading data...")
    (daily_ret, dpdf, daily_smas, rfr_daily, vix_monthly, tbill_monthly,
     z_clean, asset_ret_fwd, actual_lev, both_start, options_by_year) = load_data()

    print(f"  Running Pod 1 (Faber-Sweep-40)...")
    faber_daily = run_faber(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start)

    print(f"  Running Pod 2 (Harvey Spreads with position management)...")
    spread_monthly, trades_df = run_spreads(
        vix_monthly, tbill_monthly, z_clean, asset_ret_fwd, options_by_year, dpdf)

    print(f"  Faber: {faber_daily.index.min().date()} to {faber_daily.index.max().date()}")
    print(f"  Spreads: {spread_monthly.index.min().date()} to {spread_monthly.index.max().date()}, {len(trades_df)} trades")

    report(faber_daily, spread_monthly, trades_df, daily_ret)

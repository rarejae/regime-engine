"""Unconstrained Harvey-conditional vertical spread backtest.

Standalone strategy on $100K dedicated capital. No Faber cash gating.
Both put and call spreads. Measures standalone Sharpe and correlation with Pod 1.
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

DAILY_SMA_PERIODS = [126, 200, 252]
BASELINE = {"IVV": 0.45, "QQQ": 0.25, "VGLT": 0.05, "IAU": 0.10, "DBC": 0.05, "cash": 0.10}
ASSETS = list(BASELINE.keys())

CAPITAL = 100_000
VIX_THRESH_PUT = 18
VIX_THRESH_CALL = 15
HARVEY_THRESH = 0.005
SLIPPAGE = 0.10
COMMISSION = 2.60
SSO_EXP = 0.0089; QLD_EXP = 0.0095


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
        except Exception: pass
        try:
            vix = f.get_series("VIXCLS", observation_start="2009-01-01")
            vix.index = pd.to_datetime(vix.index)
            vix_monthly = vix.resample("MS").last().dropna()
        except Exception: pass

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
    asset_ret = load_monthly_asset_returns()
    asset_ret_fwd = asset_ret.shift(-1)

    actual_lev = {}
    for ticker in ["SSO", "QLD"]:
        d = yf.download(ticker, start="2006-01-01", progress=False, auto_adjust=True)
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p, "columns"): p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None)
            actual_lev[ticker] = p.pct_change().dropna()

    return (daily_ret, dpdf, daily_smas, rfr_daily, vix_monthly, tbill_monthly,
            z_clean, asset_ret_fwd, actual_lev)


def load_options(year):
    p = Path(f"data/processed/spy_options_{year}.parquet")
    return pd.read_parquet(p) if p.exists() else None


def find_contract(chain, opt_type, target_delta, dte_min=21, dte_max=35):
    filt = chain[
        (chain["option_type"] == opt_type) &
        (chain["dte"] >= dte_min) & (chain["dte"] <= dte_max) &
        (chain["bid_ask_pct"] <= 0.08) &
        (chain["bid"] >= 0.05) &
        (chain["delta"].notna())
    ].copy()
    if len(filt) == 0: return None
    filt["dte_dist"] = abs(filt["dte"] - 28)
    best_dte = filt["dte_dist"].min()
    filt = filt[filt["dte_dist"] <= best_dte + 5]
    filt["delta_dist"] = abs(filt["delta"] - target_delta)
    return filt.loc[filt["delta_dist"].idxmin()]


def select_spread(chain, direction, trade_date):
    dc = chain[chain["trade_date"] == trade_date]
    if len(dc) == 0: return None

    if direction == "PUT":
        short = find_contract(dc, "P", -0.10)
        if short is None: return None
        longs = dc[(dc["option_type"] == "P") & (dc["expiry"] == short["expiry"]) &
                    (dc["strike"] < short["strike"]) & (dc["delta"].notna())].copy()
        if len(longs) == 0: return None
        longs["dd"] = abs(longs["delta"] - (-0.05))
        lng = longs.loc[longs["dd"].idxmin()]
    else:
        short = find_contract(dc, "C", 0.10)
        if short is None: return None
        longs = dc[(dc["option_type"] == "C") & (dc["expiry"] == short["expiry"]) &
                    (dc["strike"] > short["strike"]) & (dc["delta"].notna())].copy()
        if len(longs) == 0: return None
        longs["dd"] = abs(longs["delta"] - 0.05)
        lng = longs.loc[longs["dd"].idxmin()]

    nc = float(short["mid"] - lng["mid"])
    sw = abs(float(short["strike"]) - float(lng["strike"]))
    if nc <= 0 or sw <= 0: return None

    return {"short_strike": float(short["strike"]), "long_strike": float(lng["strike"]),
            "net_credit": nc, "spread_width": sw,
            "expiry": pd.Timestamp(short["expiry"]), "dte": int(short["dte"]),
            "underlying": float(short["underlying_close"]),
            "short_delta": float(short["delta"]), "long_delta": float(lng["delta"])}


def spread_pnl(sp, direction, expiry_price):
    ac = sp["net_credit"] * (1 - SLIPPAGE)
    ks, kl, w = sp["short_strike"], sp["long_strike"], sp["spread_width"]
    if direction == "PUT":
        if expiry_price >= ks: p = ac
        elif expiry_price <= kl: p = ac - w
        else: p = ac - (ks - expiry_price)
    else:
        if expiry_price <= ks: p = ac
        elif expiry_price >= kl: p = ac - w
        else: p = ac - (expiry_price - ks)
    return p


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


def run_faber_s40_monthly(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev):
    """Generate Faber-Sweep-40 100% sub monthly returns for correlation analysis."""
    bt_start = pd.Timestamp("2010-01-01")
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:].index
    both_start = max(actual_lev["SSO"].index.min(), actual_lev["QLD"].index.min()) \
        if "SSO" in actual_lev and "QLD" in actual_lev else pd.Timestamp("2099-01-01")

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
            fc = cur.get("IVV", 0) >= 3 and cur.get("QQQ", 0) >= 3
            la = fc

        if la and not dlv:
            if check_breach(day, dpdf, daily_smas):
                la = False; dlv = True

        iw = wf.get("IVV", 0); qw = wf.get("QQQ", 0)
        ir = actual.get("IVV", 0); qr = actual.get("QQQ", 0)
        base = sum(wf.get(a, 0) * actual.get(a, 0) for a in avail if a not in ["IVV", "QQQ"])
        sub = 1.0 if la else 0.0
        if sub > 0 and day >= both_start:
            sso_r = float(actual_lev.get("SSO", pd.Series(dtype=float)).get(day, 2*ir - rfr - SSO_EXP/252))
            qld_r = float(actual_lev.get("QLD", pd.Series(dtype=float)).get(day, 2*qr - rfr - QLD_EXP/252))
            if np.isnan(sso_r): sso_r = 2*ir - rfr - SSO_EXP/252
            if np.isnan(qld_r): qld_r = 2*qr - rfr - QLD_EXP/252
        elif sub > 0:
            sso_r = 2*ir - rfr - SSO_EXP/252; qld_r = 2*qr - rfr - QLD_EXP/252
        else:
            sso_r = ir; qld_r = qr

        if sub > 0:
            results[day] = iw * sso_r + qw * qld_r + base
        else:
            results[day] = iw * ir + qw * qr + base

    return pd.Series(results).sort_index()


# ── Main backtest ────────────────────────────────────────────────────────────

def run_spread_backtest(vix_monthly, tbill_monthly, z_clean, asset_ret_fwd, dpdf):
    print(f"\n{'='*120}")
    print(f"  UNCONSTRAINED SPREAD BACKTEST (2010-2023)")
    print(f"{'='*120}")

    options_by_year = {}
    for yr in range(2010, 2024):
        df = load_options(yr)
        if df is not None: options_by_year[yr] = df

    # Build month-end dates
    trading_days = dpdf.index[(dpdf.index >= "2010-01-01") & (dpdf.index <= "2023-12-31")]
    month_ends = []
    for i, d in enumerate(trading_days):
        if i == len(trading_days)-1 or d.month != trading_days[i+1].month:
            month_ends.append(d)

    trades = []
    monthly_pnl = {}  # month → pnl for the spread strategy
    activation = {"total": 0, "vix_low": 0, "ambiguous": 0, "no_chain": 0,
                  "put_opened": 0, "call_opened": 0, "cash": 0}

    for me in month_ends:
        activation["total"] += 1
        yr = me.year

        # VIX
        vix_val = None
        if vix_monthly is not None:
            vd = vix_monthly.index[vix_monthly.index <= me]
            if len(vd) > 0: vix_val = float(vix_monthly.loc[vd[-1]])

        # Harvey ER
        z_prior = z_clean.index[z_clean.index < me]
        harvey_er = 0.0
        if len(z_prior) > 0:
            try:
                sim, _ = find_similar_months(z_clean, z_prior[-1])
                er = compute_expected_returns(sim, asset_ret_fwd, ["IVV"])
                harvey_er = er.get("IVV", 0.0)
            except ValueError: pass

        # Direction decision
        direction = None
        if harvey_er > HARVEY_THRESH and vix_val is not None and vix_val >= VIX_THRESH_PUT:
            direction = "PUT"
        elif harvey_er < -HARVEY_THRESH and vix_val is not None and vix_val >= VIX_THRESH_CALL:
            direction = "CALL"

        if direction is None:
            if vix_val is not None and vix_val < VIX_THRESH_CALL:
                activation["vix_low"] += 1
            elif abs(harvey_er) < HARVEY_THRESH:
                activation["ambiguous"] += 1
            else:
                activation["vix_low"] += 1

            # Cash month — T-bill return
            tb_ret = 0.0
            if tbill_monthly is not None:
                tb_dates = tbill_monthly.index[tbill_monthly.index <= me]
                if len(tb_dates) > 0: tb_ret = float(tbill_monthly.loc[tb_dates[-1]])
            ms = pd.Timestamp(f"{me.year}-{me.month:02d}-01")
            monthly_pnl[ms] = tb_ret * CAPITAL
            activation["cash"] += 1
            continue

        # Find chain
        chain = options_by_year.get(yr)
        if chain is None:
            activation["no_chain"] += 1
            ms = pd.Timestamp(f"{me.year}-{me.month:02d}-01")
            monthly_pnl[ms] = 0.0
            continue

        # Closest trade date
        cd = chain["trade_date"].unique()
        valid = cd[cd <= me]
        if len(valid) == 0:
            activation["no_chain"] += 1; continue
        trade_date = valid[-1]

        sp = select_spread(chain, direction, trade_date)
        if sp is None:
            activation["no_chain"] += 1; continue

        # Size
        max_loss_per = (sp["spread_width"] - sp["net_credit"]) * 100
        if max_loss_per <= 0: continue
        contracts = min(int(CAPITAL / max_loss_per), 50)
        contracts = max(contracts, 1)

        # Expiry price
        exp_dt = sp["expiry"]
        exp_price = None
        ec = options_by_year.get(exp_dt.year)
        if ec is not None:
            er = ec[ec["trade_date"] == exp_dt]
            if len(er) > 0: exp_price = float(er.iloc[0]["underlying_close"])
        if exp_price is None:
            ep = dpdf.loc[:exp_dt, "IVV"]
            if len(ep) > 0: exp_price = float(ep.iloc[-1])
        if exp_price is None: continue

        pps = spread_pnl(sp, direction, exp_price)
        net = (pps * contracts * 100) - (COMMISSION * contracts)

        ms = pd.Timestamp(f"{me.year}-{me.month:02d}-01")
        monthly_pnl[ms] = monthly_pnl.get(ms, 0) + net

        if direction == "PUT": activation["put_opened"] += 1
        else: activation["call_opened"] += 1

        trades.append({
            "date": me, "ms": ms, "direction": direction,
            "short_strike": sp["short_strike"], "long_strike": sp["long_strike"],
            "spread_width": sp["spread_width"], "net_credit": sp["net_credit"],
            "expiry": exp_dt, "dte": sp["dte"], "contracts": contracts,
            "underlying_entry": sp["underlying"], "underlying_expiry": exp_price,
            "pnl_per_share": pps, "net_pnl": net, "margin": max_loss_per * contracts,
            "vix": vix_val, "harvey_er": harvey_er,
            "market_moved": (exp_price / sp["underlying"]) - 1,
        })

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    pnl_series = pd.Series(monthly_pnl).sort_index()

    return trades_df, pnl_series, activation


def metrics_monthly(pnl_series, capital):
    """Compute strategy metrics from monthly P&L on fixed capital."""
    rets = pnl_series / capital
    ar = rets.mean() * 12; av = rets.std() * np.sqrt(12)
    sh = ar / av if av > 0 else 0
    neg = rets[rets < 0]
    ds = neg.std() * np.sqrt(12) if len(neg) > 3 else av
    so = ar / ds if ds > 0 else 0
    cum = (1 + rets).cumprod()
    dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
    cal = ar / abs(dd) if dd != 0 else 0
    return {"ar": ar, "av": av, "sh": sh, "sortino": so, "dd": dd, "calmar": cal, "final": cum.iloc[-1]}


def report(trades_df, pnl_series, activation, faber_daily, tbill_monthly):
    total = activation["total"]
    years = 14.0

    # ── Activation ───────────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  ACTIVATION FREQUENCY")
    print(f"{'='*120}")
    print(f"\n  Total months: {total}")
    print(f"  VIX too low:  {activation['vix_low']} ({activation['vix_low']/total*100:.0f}%)")
    print(f"  Harvey ambiguous: {activation['ambiguous']} ({activation['ambiguous']/total*100:.0f}%)")
    print(f"  No qualifying chain: {activation['no_chain']} ({activation['no_chain']/total*100:.0f}%)")
    print(f"  PUT spreads opened: {activation['put_opened']} ({activation['put_opened']/total*100:.0f}%)")
    print(f"  CALL spreads opened: {activation['call_opened']} ({activation['call_opened']/total*100:.0f}%)")
    print(f"  Cash months: {activation['cash']} ({activation['cash']/total*100:.0f}%)")

    # ── Standalone performance ───────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  STANDALONE PERFORMANCE ($100K dedicated capital)")
    print(f"{'='*120}")

    sm = metrics_monthly(pnl_series, CAPITAL)
    print(f"\n  Annualised return:  {sm['ar']:.2%}")
    print(f"  Volatility:         {sm['av']:.2%}")
    print(f"  Sharpe ratio:       {sm['sh']:.3f}")
    print(f"  Sortino ratio:      {sm['sortino']:.3f}")
    print(f"  Max drawdown:       {sm['dd']:.1%}")
    print(f"  Calmar ratio:       {sm['calmar']:.2f}")
    print(f"  Terminal $1:        ${sm['final']:.2f}")

    if len(trades_df) > 0:
        wins = trades_df[trades_df["net_pnl"] > 0]
        losses = trades_df[trades_df["net_pnl"] <= 0]
        print(f"\n  Total trades:       {len(trades_df)}")
        print(f"  Win rate:           {len(wins)/len(trades_df)*100:.0f}%")
        if len(wins) > 0: print(f"  Avg win:            ${wins['net_pnl'].mean():,.0f}")
        if len(losses) > 0: print(f"  Avg loss:           ${losses['net_pnl'].mean():,.0f}")
        gross_wins = wins["net_pnl"].sum() if len(wins) > 0 else 0
        gross_losses = abs(losses["net_pnl"].sum()) if len(losses) > 0 else 1
        print(f"  Profit factor:      {gross_wins/gross_losses:.2f}")
        print(f"  Total P&L:          ${trades_df['net_pnl'].sum():,.0f}")
        print(f"  Annual P&L:         ${trades_df['net_pnl'].sum()/years:,.0f}")

    # ── Direction breakdown ──────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  DIRECTION BREAKDOWN")
    print(f"{'='*120}")

    for dir_label in ["PUT", "CALL"]:
        dt = trades_df[trades_df["direction"] == dir_label] if len(trades_df) > 0 else pd.DataFrame()
        if len(dt) > 0:
            wr = (dt["net_pnl"] > 0).mean() * 100
            print(f"\n  {dir_label} spreads: {len(dt)} trades, {wr:.0f}% win rate, "
                  f"avg ${dt['net_pnl'].mean():,.0f}/trade, total ${dt['net_pnl'].sum():,.0f}")
        else:
            print(f"\n  {dir_label} spreads: 0 trades")

    # Cash months
    rets = pnl_series / CAPITAL
    cash_months = rets[abs(rets) < 0.0001]
    print(f"\n  Cash months: {len(cash_months)}")

    # ── Correlation with Pod 1 ───────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  CORRELATION WITH POD 1 (Faber-Sweep-40 100% sub)")
    print(f"{'='*120}")

    faber_m = faber_daily.resample("MS").apply(lambda x: (1 + x).prod() - 1)
    spread_m = pnl_series / CAPITAL

    common = faber_m.dropna().index.intersection(spread_m.dropna().index).sort_values()
    f = faber_m.reindex(common); s = spread_m.reindex(common)

    pearson = float(f.corr(s))
    spearman = float(f.rank().corr(s.rank()))

    print(f"\n  Pearson correlation (monthly):  {pearson:.3f}")
    print(f"  Spearman correlation (rank):   {spearman:.3f}")

    # Rolling 12-month
    rolling_corr = f.rolling(12).corr(s).dropna()
    print(f"\n  Rolling 12-month correlation:")
    print(f"    Min:  {rolling_corr.min():.3f}")
    print(f"    Max:  {rolling_corr.max():.3f}")
    print(f"    Mean: {rolling_corr.mean():.3f}")

    print(f"\n  Annual rolling correlation:")
    for yr in range(2011, 2024):
        rc = rolling_corr[rolling_corr.index.year == yr]
        if len(rc) > 0:
            print(f"    {yr}: {rc.mean():.3f}")

    # Crisis correlations
    print(f"\n  Crisis period correlations:")
    for cname, cs, ce in [("2011 correction", "2011-07", "2011-11"),
                           ("2018 Q4", "2018-10", "2018-12"),
                           ("COVID crash", "2020-02", "2020-04"),
                           ("2022 bear", "2022-01", "2022-12")]:
        cr_f = f[(f.index >= pd.Timestamp(cs)) & (f.index <= pd.Timestamp(ce))]
        cr_s = s[(s.index >= pd.Timestamp(cs)) & (s.index <= pd.Timestamp(ce))]
        common_c = cr_f.dropna().index.intersection(cr_s.dropna().index)
        if len(common_c) >= 3:
            c = cr_f.reindex(common_c).corr(cr_s.reindex(common_c))
            print(f"    {cname}: {c:.3f}")
        else:
            print(f"    {cname}: insufficient data")

    # Loss clustering
    print(f"\n  Months where BOTH pods lost >2%:")
    both_loss = common[(f.reindex(common) < -0.02) & (s.reindex(common) < -0.02)]
    if len(both_loss) > 0:
        for dt in both_loss:
            print(f"    {dt.strftime('%Y-%m')}: Pod1 {f.loc[dt]*100:+.1f}%, Pod2 {s.loc[dt]*100:+.1f}%")
    else:
        print(f"    None — losses do not cluster")

    # ── Combined portfolio ───────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  COMBINED PORTFOLIO (90/10 Faber + Spreads)")
    print(f"{'='*120}")

    combined = 0.90 * f + 0.10 * s
    fm = metrics_monthly(f * CAPITAL, CAPITAL)
    cm = metrics_monthly(combined * CAPITAL, CAPITAL)

    print(f"\n  {'Strategy':<22} {'Return':>8} {'Vol':>7} {'Sharpe':>8} {'MaxDD':>8} {'Terminal':>10}")
    print(f"  {'-'*22} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*10}")
    print(f"  {'Faber-only':<22} {fm['ar']:>7.1%} {fm['av']:>6.1%} {fm['sh']:>8.3f} {fm['dd']:>7.1%} ${fm['final']:>9.2f}")
    print(f"  {'Spreads-only':<22} {sm['ar']:>7.1%} {sm['av']:>6.1%} {sm['sh']:>8.3f} {sm['dd']:>7.1%} ${sm['final']:>9.2f}")
    print(f"  {'90/10 combined':<22} {cm['ar']:>7.1%} {cm['av']:>6.1%} {cm['sh']:>8.3f} {cm['dd']:>7.1%} ${cm['final']:>9.2f}")

    # ── VIX regime ───────────────────────────────────────────────────────
    if len(trades_df) > 0:
        print(f"\n{'='*120}")
        print(f"  VIX REGIME ANALYSIS")
        print(f"{'='*120}")

        for label, lo, hi in [("VIX 15-25", 15, 25), ("VIX 25-35", 25, 35), ("VIX > 35", 35, 200)]:
            bucket = trades_df[(trades_df["vix"] >= lo) & (trades_df["vix"] < hi)]
            if len(bucket) > 0:
                wr = (bucket["net_pnl"] > 0).mean() * 100
                print(f"\n  {label}: {len(bucket)} trades, {wr:.0f}% win, avg ${bucket['net_pnl'].mean():,.0f}")
            else:
                print(f"\n  {label}: 0 trades")

    # ── Harvey signal quality ────────────────────────────────────────────
    if len(trades_df) > 0:
        print(f"\n{'='*120}")
        print(f"  HARVEY SIGNAL QUALITY")
        print(f"{'='*120}")

        for dir_label in ["PUT", "CALL"]:
            dt = trades_df[trades_df["direction"] == dir_label]
            if len(dt) == 0: continue
            up = (dt["market_moved"] > 0.02).sum()
            flat = ((dt["market_moved"] >= -0.02) & (dt["market_moved"] <= 0.02)).sum()
            down = (dt["market_moved"] < -0.02).sum()
            won = (dt["net_pnl"] > 0).sum()
            print(f"\n  {dir_label} spreads ({len(dt)} trades):")
            print(f"    Market up >2%:    {up} ({up/len(dt)*100:.0f}%)")
            print(f"    Market flat ±2%:  {flat} ({flat/len(dt)*100:.0f}%)")
            print(f"    Market down >2%:  {down} ({down/len(dt)*100:.0f}%)")
            print(f"    Spread still won: {won} ({won/len(dt)*100:.0f}%)")

    # ── Key questions ────────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  KEY QUESTIONS")
    print(f"{'='*120}")
    print(f"\n  Q1. Standalone Sharpe: {sm['sh']:.3f}")
    print(f"  Q2. Correlation with Pod 1: {pearson:.3f} (Pearson), {spearman:.3f} (Spearman)")
    print(f"  Q3. Loss clustering: {'YES' if len(both_loss) > 0 else 'NO — losses independent'}")
    print(f"  Q4. Call spreads activated: {activation['call_opened']}")
    print(f"  Q5. 90/10 combined Sharpe: {cm['sh']:.3f} vs Faber-only {fm['sh']:.3f} ({cm['sh']-fm['sh']:+.3f})")

    print()
    return sm, fm, cm, pearson


if __name__ == "__main__":
    print("=" * 120)
    print("  UNCONSTRAINED HARVEY-CONDITIONAL VERTICAL SPREADS (2010-2023)")
    print("=" * 120)

    print(f"\n  Loading data...")
    (daily_ret, dpdf, daily_smas, rfr_daily, vix_monthly, tbill_monthly,
     z_clean, asset_ret_fwd, actual_lev) = load_data()

    print(f"  Running spread backtest...")
    trades_df, pnl_series, activation = run_spread_backtest(
        vix_monthly, tbill_monthly, z_clean, asset_ret_fwd, dpdf)

    print(f"  Generating Faber-Sweep-40 returns for correlation...")
    faber_daily = run_faber_s40_monthly(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev)
    faber_daily = faber_daily[faber_daily.index >= "2010-01-01"]

    sm, fm, cm, corr = report(trades_df, pnl_series, activation, faber_daily, tbill_monthly)

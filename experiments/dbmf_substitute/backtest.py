"""DBMF as signal-off cash substitute for Faber-Sweep-40.

Variant A: signal-off → freed equity weight to T-bills (baseline)
Variant B: signal-off → freed equity weight: 50% DBMF proxy / 50% T-bills

DBMF proxy: 5-asset equal-weight trend-following (SPY,TLT,GLD,USO,DBC).
Validated against actual DBMF from May 2019.

100% SSO/QLD substitution when both IVV+QQQ at 3/3. No lifecycle delevering.
Daily circuit breaker. Daily SMAs 126/200/252.
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv; load_dotenv()

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

from taa.data import load_daily_etf_returns, load_monthly_prices
from taa.faber import apply_faber_filter

BASELINE_W = {"IVV": 0.45, "QQQ": 0.25, "VGLT": 0.05, "IAU": 0.10, "DBC": 0.05, "cash": 0.10}
ASSETS = list(BASELINE_W.keys())
RISKY = ["IVV", "QQQ", "VGLT", "IAU", "DBC"]
SMA_PERIODS = [126, 200, 252]
SSO_EXP = 0.0089; QLD_EXP = 0.0095
DBMF_SPLIT = 0.50  # 50% DBMF / 50% T-bills


def load_data():
    import yfinance as yf

    daily_ret = load_daily_etf_returns()
    daily_ret = daily_ret[[c for c in daily_ret.columns if c in ASSETS]]

    from fredapi import Fred
    rfr_daily = pd.Series(0.0, index=daily_ret.index)
    key = os.environ.get("FRED_API_KEY")
    if key:
        tb = Fred(api_key=key).get_series("DTB3", observation_start="1998-01-01")
        tb.index = pd.to_datetime(tb.index)
        rfr_daily = (tb / 100 / 252).reindex(daily_ret.index, method="ffill").fillna(0)

    ticker_map = {"IVV": "SPY", "QQQ": "QQQ", "VGLT": "TLT", "IAU": "GLD", "DBC": "DBC"}
    dp = {}
    for our, ticker in ticker_map.items():
        d = yf.download(ticker, start="1998-01-01", progress=False)
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p, "columns"): p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None); dp[our] = p
    dpdf = pd.DataFrame(dp).sort_index()
    daily_smas = {p: dpdf.rolling(p, min_periods=p).mean() for p in SMA_PERIODS}

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

    # DBMF actual
    dbmf_actual = None
    d = yf.download("DBMF", start="2019-01-01", progress=False, auto_adjust=True)
    if d is not None and not d.empty:
        p = d["Close"]
        if hasattr(p, "columns"): p = p.iloc[:, 0]
        p.index = pd.to_datetime(p.index).tz_localize(None)
        dbmf_actual = p.pct_change().dropna()

    # DBMF proxy: 5-asset equal-weight trend-following
    proxy_tickers = {"SPY": "SPY", "TLT": "TLT", "GLD": "GLD", "USO": "USO", "DBC_P": "DBC"}
    proxy_prices = {}
    for name, ticker in proxy_tickers.items():
        d = yf.download(ticker, start="1998-01-01", progress=False)
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p, "columns"): p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None)
            proxy_prices[name] = p
    proxy_pdf = pd.DataFrame(proxy_prices).sort_index()

    # Build DBMF proxy: for each asset, +1 if above 200-day SMA, -1 if below
    proxy_sma200 = proxy_pdf.rolling(200, min_periods=200).mean()
    proxy_signals = (proxy_pdf > proxy_sma200).astype(int) * 2 - 1  # +1 or -1
    proxy_returns = proxy_pdf.pct_change()

    # Equal weight, signal × return, scaled to target ~10% vol
    raw_proxy = (proxy_signals * proxy_returns * 0.20).sum(axis=1).dropna()
    # Scale to 10% annualized vol
    raw_vol = raw_proxy.rolling(252, min_periods=126).std() * np.sqrt(252)
    target_vol = 0.10
    vol_scale = target_vol / raw_vol.replace(0, np.nan)
    vol_scale = vol_scale.clip(0.5, 3.0).fillna(1.0)
    dbmf_proxy = (raw_proxy * vol_scale).dropna()

    # Hybrid: actual DBMF where available, proxy before
    if dbmf_actual is not None:
        dbmf_start = dbmf_actual.index.min()
        pre = dbmf_proxy[dbmf_proxy.index < dbmf_start]
        dbmf_hybrid = pd.concat([pre, dbmf_actual]).sort_index()
        dbmf_hybrid = dbmf_hybrid[~dbmf_hybrid.index.duplicated(keep="last")]
    else:
        dbmf_hybrid = dbmf_proxy

    # T-bill daily return
    tbill_daily = rfr_daily

    return (daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
            dbmf_proxy, dbmf_actual, dbmf_hybrid, tbill_daily)


def validate_proxy(dbmf_proxy, dbmf_actual):
    print(f"\n{'='*100}")
    print(f"  DBMF PROXY VALIDATION (May 2019 – present)")
    print(f"{'='*100}")

    if dbmf_actual is None:
        print("  DBMF actual data not available — skipping validation")
        return

    common = dbmf_proxy.dropna().index.intersection(dbmf_actual.dropna().index).sort_values()
    p = dbmf_proxy.reindex(common).dropna()
    a = dbmf_actual.reindex(common).dropna()
    common2 = p.index.intersection(a.index)
    p = p.reindex(common2); a = a.reindex(common2)

    # Monthly
    p_m = p.resample("MS").apply(lambda x: (1+x).prod()-1)
    a_m = a.resample("MS").apply(lambda x: (1+x).prod()-1)
    m_corr = float(p_m.corr(a_m))
    ann_diff = abs(p.mean()*252 - a.mean()*252)

    print(f"\n  Overlap: {common2.min().date()} to {common2.max().date()} ({len(common2)} days)")
    print(f"  Monthly return correlation: {m_corr:.3f} (threshold > 0.75)")
    print(f"  Proxy ann return: {p.mean()*252:.1%}")
    print(f"  Actual DBMF ann return: {a.mean()*252:.1%}")
    print(f"  Annual return diff: {ann_diff:.1%} (threshold < 4%)")

    corr_ok = m_corr > 0.75
    diff_ok = ann_diff < 0.04
    print(f"\n  Correlation: {'PASS' if corr_ok else 'FAIL'}")
    print(f"  Return diff: {'PASS' if diff_ok else 'FAIL'}")

    # Year-by-year
    print(f"\n  Year-by-year:")
    for yr in range(2019, 2027):
        py = p[p.index.year == yr]; ay = a[a.index.year == yr]
        if len(py) > 50 and len(ay) > 50:
            pr = (1+py).prod()-1; ar = (1+ay).prod()-1
            print(f"    {yr}: proxy {pr:+.1%}, actual {ar:+.1%}, diff {pr-ar:+.1%}")


def sma_scores(day, dpdf, smas):
    scores = {}
    for a in RISKY:
        if a not in dpdf.columns: scores[a] = 0; continue
        p = dpdf.loc[:day, a]
        if len(p) == 0 or pd.isna(p.iloc[-1]): scores[a] = 0; continue
        price = p.iloc[-1]; sc = 0
        for per in SMA_PERIODS:
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
        for per in SMA_PERIODS:
            s = smas[per].loc[:day, etf]
            if len(s) > 0 and pd.notna(s.iloc[-1]) and price < s.iloc[-1]: b += 1
        if b >= 3: return True
    return False


def run_backtest(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                 dbmf_hybrid, tbill_daily):
    print(f"\n{'='*100}")
    print(f"  BACKTEST")
    print(f"{'='*100}")

    bt_start = pd.Timestamp("2002-01-01")
    bt_end = pd.Timestamp("2026-03-31")
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:bt_end].index
    print(f"  {len(trading_days)} trading days ({common_start.date()} to {trading_days[-1].date()})")

    # State
    cur_scores = {a: 3 for a in RISKY}
    w_faber = dict(BASELINE_W)
    leverage_active = False; delevered = False
    freed_equity = 0.0  # weight freed from IVV/QQQ when signal off

    results_a = {}; results_b = {}
    bench_ivv = {}; bench_6040 = {}
    signal_off_log = []
    monthly_detail_2022 = []

    for day in trading_days:
        if day not in daily_ret.index: continue
        dr = daily_ret.loc[day]
        avail = [a for a in ASSETS if a in dr.index and pd.notna(dr[a])]
        if len(avail) < 3: continue
        actual = {a: float(dr[a]) for a in avail}
        rfr = float(rfr_daily.get(day, 0))

        is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day)-1].month)

        if is_ms:
            delevered = False
            prior = trading_days[trading_days < day]
            sd = prior[-1] if len(prior) > 0 else day
            cur_scores = sma_scores(sd, dpdf, daily_smas)
            w1, pool = apply_faber_filter(cur_scores, BASELINE_W)
            w_faber = dict(w1); w_faber["cash"] = w_faber.get("cash", 0) + pool

            faber_conv = cur_scores.get("IVV", 0) >= 3 and cur_scores.get("QQQ", 0) >= 3
            leverage_active = faber_conv

            # Compute freed equity weight (IVV/QQQ that went to cash due to score ≤ 1)
            freed_ivv = BASELINE_W["IVV"] - w_faber.get("IVV", 0)
            freed_qqq = BASELINE_W["QQQ"] - w_faber.get("QQQ", 0)
            freed_equity = max(freed_ivv, 0) + max(freed_qqq, 0)

            signal_off = cur_scores.get("IVV", 0) <= 1 or cur_scores.get("QQQ", 0) <= 1

        # Daily circuit breaker
        if leverage_active and not delevered:
            if check_breach(day, dpdf, daily_smas):
                leverage_active = False; delevered = True

        iw = w_faber.get("IVV", 0); qw = w_faber.get("QQQ", 0)
        ir = actual.get("IVV", 0); qr = actual.get("QQQ", 0)
        base_ret = sum(w_faber.get(a, 0) * actual.get(a, 0) for a in avail if a not in ["IVV", "QQQ"])
        cash_w = w_faber.get("cash", 0)

        # Variant A — all freed weight earns T-bill (cash)
        if leverage_active:
            if day >= both_start:
                sso_r = float(actual_lev.get("SSO", pd.Series()).get(day, np.nan))
                qld_r = float(actual_lev.get("QLD", pd.Series()).get(day, np.nan))
                if np.isnan(sso_r): sso_r = 2*ir - rfr - SSO_EXP/252
                if np.isnan(qld_r): qld_r = 2*qr - rfr - QLD_EXP/252
            else:
                sso_r = 2*ir - rfr - SSO_EXP/252; qld_r = 2*qr - rfr - QLD_EXP/252
            eq_ret_a = iw * sso_r + qw * qld_r
        else:
            eq_ret_a = iw * ir + qw * qr

        ret_a = eq_ret_a + base_ret + cash_w * rfr
        results_a[day] = ret_a

        # Variant B — freed EQUITY weight split: 50% DBMF / 50% T-bills
        # Only freed IVV/QQQ weight goes to DBMF, not VGLT/IAU/DBC freed weight
        dbmf_r = float(dbmf_hybrid.get(day, 0)) if day in dbmf_hybrid.index else 0.0
        if np.isnan(dbmf_r): dbmf_r = 0.0

        # Cash in variant B: original cash weight + VGLT/IAU/DBC freed (same as A)
        # But freed equity goes 50% DBMF / 50% T-bills instead of all T-bills
        non_equity_cash = cash_w - freed_equity  # cash from non-equity sources + structural 10%
        # freed_equity portion: 50% DBMF, 50% T-bills
        dbmf_w = freed_equity * DBMF_SPLIT
        freed_tbill_w = freed_equity * (1 - DBMF_SPLIT)
        total_cash_b = non_equity_cash + freed_tbill_w

        ret_b = eq_ret_a + base_ret + total_cash_b * rfr + dbmf_w * dbmf_r
        results_b[day] = ret_b

        # Benchmarks
        bench_ivv[day] = actual.get("IVV", 0)
        bench_6040[day] = 0.60 * actual.get("IVV", 0) + 0.40 * actual.get("VGLT", 0)

    ps_a = pd.Series(results_a).sort_index()
    ps_b = pd.Series(results_b).sort_index()
    bi = pd.Series(bench_ivv).sort_index()
    b6 = pd.Series(bench_6040).sort_index()

    return ps_a, ps_b, bi, b6, dbmf_hybrid


def metrics_daily(s):
    ar = s.mean()*252; av = s.std()*np.sqrt(252)
    sh = ar/av if av > 0 else 0
    neg = s[s<0]; ds = neg.std()*np.sqrt(252) if len(neg)>10 else av
    so = ar/ds if ds>0 else 0
    cum = (1+s).cumprod(); dd = ((cum-cum.expanding().max())/cum.expanding().max()).min()
    cal = ar/abs(dd) if dd!=0 else 0
    return {"ar": ar, "av": av, "sh": sh, "sortino": so, "dd": dd, "calmar": cal, "final": cum.iloc[-1]}


def report(ps_a, ps_b, bi, b6, dbmf_hybrid, rfr_daily):
    ma = metrics_daily(ps_a); mb = metrics_daily(ps_b)
    mi = metrics_daily(bi); m6 = metrics_daily(b6)

    # DCA projection ($700/mo over 24 years)
    def dca_proj(ann_ret, monthly, years):
        r_m = (1 + ann_ret) ** (1/12) - 1
        return monthly * (((1 + r_m) ** (years * 12) - 1) / r_m) if r_m > 0 else monthly * years * 12
    dca_a = dca_proj(ma["ar"], 700, 24)
    dca_b = dca_proj(mb["ar"], 700, 24)

    print(f"\n{'='*100}")
    print(f"  RESULTS")
    print(f"{'='*100}")

    print(f"\n  {'Variant':<14} {'Return':>8} {'Vol':>7} {'Sharpe':>8} {'Sortino':>8} "
          f"{'MaxDD':>8} {'Calmar':>8} {'Terminal($1)':>13} {'DCA $700/mo':>12}")
    print(f"  {'-'*14} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*13} {'-'*12}")
    print(f"  {'A Baseline':<14} {ma['ar']:>7.2%} {ma['av']:>6.1%} {ma['sh']:>8.3f} {ma['sortino']:>8.3f} "
          f"{ma['dd']:>7.1%} {ma['calmar']:>8.2f} ${ma['final']:>12.2f} ${dca_a/1e6:>11.2f}M")
    print(f"  {'B DBMF50':<14} {mb['ar']:>7.2%} {mb['av']:>6.1%} {mb['sh']:>8.3f} {mb['sortino']:>8.3f} "
          f"{mb['dd']:>7.1%} {mb['calmar']:>8.2f} ${mb['final']:>12.2f} ${dca_b/1e6:>11.2f}M")

    # Crisis
    print(f"\n  Crisis comparison:")
    print(f"  {'Period':<24} {'A Baseline':>14} {'B DBMF50':>14}")
    print(f"  {'-'*24} {'-'*14} {'-'*14}")
    for cname, cs, ce in [("GFC 2008-09", "2008-09-01", "2009-03-31"),
                           ("COVID Feb-Mar 2020", "2020-02-19", "2020-03-23"),
                           ("2022 full year", "2022-01-03", "2022-12-30")]:
        for label, s in [("A", ps_a), ("B", ps_b)]:
            pass
        ca = ps_a[(ps_a.index >= pd.Timestamp(cs)) & (ps_a.index <= pd.Timestamp(ce))]
        cb = ps_b[(ps_b.index >= pd.Timestamp(cs)) & (ps_b.index <= pd.Timestamp(ce))]
        if len(ca) > 0 and len(cb) > 0:
            cum_a = (1+ca).cumprod(); dd_a = ((cum_a-cum_a.expanding().max())/cum_a.expanding().max()).min()
            cum_b = (1+cb).cumprod(); dd_b = ((cum_b-cum_b.expanding().max())/cum_b.expanding().max()).min()
            print(f"  {cname:<24} {(1+ca).prod()-1:>+6.1%} ({dd_a:>5.1%}) {(1+cb).prod()-1:>+6.1%} ({dd_b:>5.1%})")

    # 2022 month-by-month
    print(f"\n  2022 month-by-month:")
    print(f"  {'Month':>8} {'A ret':>8} {'B ret':>8} {'DBMF ret':>9} {'Signal':>8}")
    dbmf_m = dbmf_hybrid.resample("MS").apply(lambda x: (1+x).prod()-1) if len(dbmf_hybrid) > 0 else pd.Series(dtype=float)

    for mo in range(1, 13):
        ms = pd.Timestamp(f"2022-{mo:02d}-01")
        me = pd.Timestamp(f"2022-{mo:02d}-28") if mo != 12 else pd.Timestamp("2022-12-31")
        ca = ps_a[(ps_a.index >= ms) & (ps_a.index <= me)]
        cb = ps_b[(ps_b.index >= ms) & (ps_b.index <= me)]
        dm = dbmf_m.get(ms, np.nan)
        if len(ca) > 0:
            ra = (1+ca).prod()-1; rb = (1+cb).prod()-1
            sig = "ON" if rb != ra else "off"
            dm_str = f"{dm:>+8.1%}" if not np.isnan(dm) else f"{'N/A':>9}"
            print(f"  {ms.strftime('%Y-%m'):>8} {ra:>+7.1%} {rb:>+7.1%} {dm_str} {'  LEV' if sig == 'ON' else '  cash'}")

    # DBMF behavior during signal-off
    print(f"\n  DBMF behavior during signal-off periods:")
    # Compute monthly
    a_m = ps_a.resample("MS").apply(lambda x: (1+x).prod()-1)
    b_m = ps_b.resample("MS").apply(lambda x: (1+x).prod()-1)
    diff_m = b_m - a_m
    active_months = diff_m[abs(diff_m) > 0.0001]
    dbmf_better = (active_months > 0).sum()
    tbill_better = (active_months < 0).sum()

    print(f"  Months where DBMF allocation was active: {len(active_months)}")
    print(f"  Months DBMF beat T-bills: {dbmf_better} ({dbmf_better/max(len(active_months),1)*100:.0f}%)")
    print(f"  Months T-bills beat DBMF: {tbill_better} ({tbill_better/max(len(active_months),1)*100:.0f}%)")
    if len(active_months) > 0:
        print(f"  Mean monthly return diff (B-A): {active_months.mean()*100:+.2f}%")

    # Correlation
    ivv_daily = bi
    dbmf_d = dbmf_hybrid
    common = ivv_daily.dropna().index.intersection(dbmf_d.dropna().index)
    if len(common) > 252:
        full_corr = ivv_daily.reindex(common).corr(dbmf_d.reindex(common))
        print(f"\n  DBMF-IVV correlation (full period): {full_corr:.3f}")

    # Deltas
    print(f"\n  VARIANT B vs A DELTAS:")
    print(f"    Return: {(mb['ar']-ma['ar'])*100:+.2f}%")
    print(f"    Sharpe: {mb['sh']-ma['sh']:+.3f}")
    print(f"    MaxDD:  {(mb['dd']-ma['dd'])*100:+.1f}%")
    print(f"    Terminal: ${mb['final']-ma['final']:+.2f}")
    print(f"    DCA $700/mo: ${(dca_b-dca_a)/1e3:+.0f}K")

    # Verdict
    print(f"\n  VERDICT: ", end="")
    better_sharpe = mb["sh"] > ma["sh"]
    same_dd = abs(mb["dd"] - ma["dd"]) < 0.005
    worse_dd = mb["dd"] < ma["dd"] - 0.005
    if better_sharpe and not worse_dd:
        print(f"DBMF 50/50 IMPROVES Sharpe ({ma['sh']:.3f}→{mb['sh']:.3f}) without increasing MaxDD → ADOPT")
    elif better_sharpe and worse_dd:
        print(f"DBMF 50/50 improves Sharpe but worsens MaxDD ({ma['dd']:.1%}→{mb['dd']:.1%}) → CONDITIONAL")
    else:
        print(f"DBMF 50/50 does not improve Sharpe → REJECT")

    return ma, mb


if __name__ == "__main__":
    print("=" * 100)
    print("  DBMF CASH SUBSTITUTE RESEARCH")
    print("=" * 100)

    print(f"\n  Loading data...")
    (daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
     dbmf_proxy, dbmf_actual, dbmf_hybrid, tbill_daily) = load_data()

    print(f"  DBMF proxy: {dbmf_proxy.index.min().date()} to {dbmf_proxy.index.max().date()}")
    if dbmf_actual is not None:
        print(f"  DBMF actual: {dbmf_actual.index.min().date()} to {dbmf_actual.index.max().date()}")
    print(f"  Hybrid: {dbmf_hybrid.index.min().date()} to {dbmf_hybrid.index.max().date()}")

    validate_proxy(dbmf_proxy, dbmf_actual)

    ps_a, ps_b, bi, b6, _ = run_backtest(
        daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
        dbmf_hybrid, tbill_daily)

    ma, mb = report(ps_a, ps_b, bi, b6, dbmf_hybrid, rfr_daily)

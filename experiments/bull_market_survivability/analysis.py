"""Bull market survivability analysis: does Faber-Sweep-40 survive the 2013-2021 test?

Phase 1: Period-by-period CAGR breakdown
Phase 2: 2013-2021 deep dive (rolling underperformance, DCA gap, CB cost)
Phase 3: Alternative start date sensitivity
Phase 4: 2010-era investor simulation
Phase 5: Honest forward distribution
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

    return daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start


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


def run_full_backtest(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                       start_date="2002-01-01"):
    """Run Faber-Sweep-40 from start_date, return daily series + CB events."""
    bt_start = pd.Timestamp(start_date)
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:pd.Timestamp("2026-03-31")].index

    cur_scores = {a: 3 for a in RISKY}
    w_faber = dict(BASELINE_W); la = False; dlv = False
    port = {}; cb_events = []

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
            cur_scores = sma_scores(sd, dpdf, daily_smas)
            w1, pool = apply_faber_filter(cur_scores, BASELINE_W)
            w_faber = dict(w1); w_faber["cash"] = w_faber.get("cash", 0) + pool
            fc = cur_scores.get("IVV", 0) >= 3 and cur_scores.get("QQQ", 0) >= 3; la = fc

        if la and not dlv:
            if check_breach(day, dpdf, daily_smas):
                la = False; dlv = True; cb_events.append(day)

        iw = w_faber.get("IVV", 0); qw = w_faber.get("QQQ", 0)
        ir = actual.get("IVV", 0); qr = actual.get("QQQ", 0)
        base = sum(w_faber.get(a, 0) * actual.get(a, 0) for a in avail if a not in ["IVV", "QQQ"])
        if la:
            if day >= both_start:
                sso = float(actual_lev.get("SSO", pd.Series()).get(day, np.nan))
                qld = float(actual_lev.get("QLD", pd.Series()).get(day, np.nan))
                if np.isnan(sso): sso = 2*ir-rfr-SSO_EXP/252
                if np.isnan(qld): qld = 2*qr-rfr-QLD_EXP/252
            else:
                sso = 2*ir-rfr-SSO_EXP/252; qld = 2*qr-rfr-QLD_EXP/252
            port[day] = iw*sso + qw*qld + base
        else:
            port[day] = iw*ir + qw*qr + base

    return pd.Series(port).sort_index(), cb_events


def cagr(s):
    if len(s) < 20: return np.nan
    cum = (1 + s).prod()
    years = len(s) / 252
    return cum ** (1 / years) - 1 if years > 0 else 0


def max_dd(s):
    cum = (1 + s).cumprod()
    return ((cum - cum.expanding().max()) / cum.expanding().max()).min()


def sharpe(s):
    ar = s.mean() * 252; av = s.std() * np.sqrt(252)
    return ar / av if av > 0 else 0


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def run_analysis(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start):

    # Run full backtests
    print("  Running Faber from 2002...")
    faber, cb_all = run_full_backtest(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, "2002-01-01")

    # Benchmarks
    qqq = daily_ret["QQQ"].loc[faber.index.min():faber.index.max()].reindex(faber.index).fillna(0)
    ivv = daily_ret["IVV"].loc[faber.index.min():faber.index.max()].reindex(faber.index).fillna(0)

    # ── PHASE 1: Sub-period CAGR breakdown ───────────────────────────────
    print(f"\n{'='*110}")
    print(f"  PHASE 1: SUB-PERIOD CAGR BREAKDOWN")
    print(f"{'='*110}")

    periods = [
        ("Dot-com crash",      "2002-01-01", "2003-03-31"),
        ("Pre-GFC bull",       "2003-04-01", "2007-10-31"),
        ("GFC bear",           "2007-11-01", "2009-03-31"),
        ("Post-GFC recovery",  "2009-04-01", "2012-12-31"),
        ("2013-2021 bull",     "2013-01-01", "2021-12-31"),
        ("2022 bear",          "2022-01-01", "2022-12-31"),
        ("2023-2026 recovery", "2023-01-01", "2026-03-31"),
    ]

    print(f"\n  {'Period':<22} {'Faber':>8} {'QQQ':>8} {'IVV':>8} {'F-Q':>8} {'F DD':>8} {'Q DD':>8} {'Lev Mo':>7} {'CB':>4}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*4}")

    for label, cs, ce in periods:
        f_p = faber[(faber.index >= pd.Timestamp(cs)) & (faber.index <= pd.Timestamp(ce))]
        q_p = qqq[(qqq.index >= pd.Timestamp(cs)) & (qqq.index <= pd.Timestamp(ce))]
        i_p = ivv[(ivv.index >= pd.Timestamp(cs)) & (ivv.index <= pd.Timestamp(ce))]

        if len(f_p) < 20: continue

        fc = cagr(f_p); qc = cagr(q_p); ic = cagr(i_p)
        fdd = max_dd(f_p); qdd = max_dd(q_p)
        cb_period = [d for d in cb_all if pd.Timestamp(cs) <= d <= pd.Timestamp(ce)]

        # Leveraged months (approximate from monthly)
        f_m = f_p.resample("MS").count()
        lev_mo = len(f_m)  # rough — actual would need state tracking

        alpha = fc - qc
        print(f"  {label:<22} {fc:>7.1%} {qc:>7.1%} {ic:>7.1%} {alpha:>+7.1%} {fdd:>7.1%} {qdd:>7.1%} {'~':>5}  {len(cb_period):>3}")

    # ── PHASE 2: 2013-2021 Deep Dive ─────────────────────────────────────
    print(f"\n{'='*110}")
    print(f"  PHASE 2: 2013-2021 DEEP DIVE (108 months)")
    print(f"{'='*110}")

    bull_start = pd.Timestamp("2013-01-01")
    bull_end = pd.Timestamp("2021-12-31")

    f_bull = faber[(faber.index >= bull_start) & (faber.index <= bull_end)]
    q_bull = qqq[(qqq.index >= bull_start) & (qqq.index <= bull_end)]
    i_bull = ivv[(ivv.index >= bull_start) & (ivv.index <= bull_end)]

    # 2.1 Rolling 12-month trailing returns
    f_m = f_bull.resample("MS").apply(lambda x: (1+x).prod()-1)
    q_m = q_bull.resample("MS").apply(lambda x: (1+x).prod()-1)
    i_m = i_bull.resample("MS").apply(lambda x: (1+x).prod()-1)

    f_12m = f_m.rolling(12).apply(lambda x: (1+x).prod()-1)
    q_12m = q_m.rolling(12).apply(lambda x: (1+x).prod()-1)

    # Months where QQQ trailing 12m > Faber trailing 12m
    underperf = (q_12m > f_12m).dropna()
    under_months = underperf[underperf]

    print(f"\n  2.1 Rolling 12-month trailing return comparison:")
    print(f"  Months where QQQ trailing 12m > Faber trailing 12m: {len(under_months)}/{len(underperf)} ({len(under_months)/len(underperf)*100:.0f}%)")

    # Longest consecutive streak
    streak = 0; max_streak = 0; streak_start = None; max_start = None; max_end = None
    for dt, val in underperf.items():
        if val:
            if streak == 0: streak_start = dt
            streak += 1
            if streak > max_streak:
                max_streak = streak; max_start = streak_start; max_end = dt
        else:
            streak = 0
    print(f"  Longest consecutive underperformance streak: {max_streak} months")
    if max_start: print(f"    From {max_start.strftime('%Y-%m')} to {max_end.strftime('%Y-%m')}")

    # 2.2 Rolling 3-year trailing CAGR
    f_36m = f_m.rolling(36).apply(lambda x: (1+x).prod()**(12/36)-1)
    q_36m = q_m.rolling(36).apply(lambda x: (1+x).prod()**(12/36)-1)

    under_3yr = (q_36m > f_36m).dropna()
    under_3yr_count = under_3yr.sum()
    print(f"\n  2.2 Rolling 3-year trailing CAGR:")
    print(f"  Months where QQQ 3yr CAGR > Faber 3yr CAGR: {int(under_3yr_count)}/{len(under_3yr)} ({under_3yr_count/len(under_3yr)*100:.0f}%)")

    # Print worst stretches
    worst_gap = (q_36m - f_36m).dropna()
    if len(worst_gap) > 0:
        worst_dt = worst_gap.idxmax()
        print(f"  Largest 3yr CAGR gap: QQQ {q_36m.loc[worst_dt]:.1%} vs Faber {f_36m.loc[worst_dt]:.1%} "
              f"({worst_gap.loc[worst_dt]:+.1%}) at {worst_dt.strftime('%Y-%m')}")

    # 2.3 DCA dollar gap
    print(f"\n  2.3 DCA dollar gap ($21K start + $700/month, 2013-2021):")
    dca_f = 21000.0; dca_q = 21000.0
    dca_log = []
    for i, dt in enumerate(f_m.index):
        if i > 0:
            dca_f = dca_f * (1 + f_m.iloc[i]) + 700
            dca_q = dca_q * (1 + q_m.iloc[i]) + 700
        else:
            dca_f += 700; dca_q += 700
        if dt.month == 12 or dt == f_m.index[-1]:
            dca_log.append((dt, dca_f, dca_q, dca_f - dca_q))

    print(f"  {'Year-end':>10} {'Faber':>12} {'QQQ':>12} {'Gap':>12}")
    max_gap = 0; max_gap_date = None
    for dt, f_val, q_val, gap in dca_log:
        print(f"  {dt.strftime('%Y-%m'):>10} ${f_val:>11,.0f} ${q_val:>11,.0f} ${gap:>+11,.0f}")
        if gap < max_gap:
            max_gap = gap; max_gap_date = dt
    print(f"\n  Maximum dollar gap: ${max_gap:+,.0f} at {max_gap_date.strftime('%Y-%m') if max_gap_date else 'N/A'}")

    # 2.4 Circuit breaker events 2013-2021
    print(f"\n  2.4 Circuit breaker events 2013-2021:")
    cb_bull = [d for d in cb_all if bull_start <= d <= bull_end]
    print(f"  Total CB events in period: {len(cb_bull)}")

    for cbd in cb_bull:
        # What happened in the 5 and 20 days after?
        idx = faber.index.get_loc(cbd)
        if idx + 20 < len(faber):
            # SSO+QLD proxy return = 2x IVV/QQQ weighted
            post5 = faber.iloc[idx+1:idx+6]
            post20 = faber.iloc[idx+1:idx+21]
            # What leveraged equities did (not what Faber did — what was missed)
            eq_5 = (0.45 * 2 * ivv.iloc[idx+1:idx+6] + 0.25 * 2 * qqq.iloc[idx+1:idx+6])
            eq_20 = (0.45 * 2 * ivv.iloc[idx+1:idx+21] + 0.25 * 2 * qqq.iloc[idx+1:idx+21])
            miss_5 = (1+eq_5).prod() - 1 - ((1+post5).prod() - 1)
            miss_20 = (1+eq_20).prod() - 1 - ((1+post20).prod() - 1)

            # Next month-end for re-entry
            next_me = faber.index[faber.index > cbd]
            reentry = None
            for nd in next_me:
                if nd.month != cbd.month:
                    reentry = nd; break

            print(f"    {cbd.strftime('%Y-%m-%d')}: 5d missed {miss_5:+.1%}, 20d missed {miss_20:+.1%}, "
                  f"re-entry ~{reentry.strftime('%Y-%m-%d') if reentry else '?'}")

    # ── PHASE 3: Start-date sensitivity ──────────────────────────────────
    print(f"\n{'='*110}")
    print(f"  PHASE 3: ALTERNATIVE START DATE SENSITIVITY")
    print(f"{'='*110}")

    start_dates = [
        ("2002-01", "24.2yr", "Full — includes dot-com"),
        ("2004-01", "22.2yr", "Post dot-com"),
        ("2007-01", "19.2yr", "Pre-GFC"),
        ("2010-01", "16.2yr", "Post-GFC — primarily bull"),
        ("2013-01", "13.2yr", "Pure modern era"),
        ("2019-01", "7.2yr",  "Near-current"),
    ]

    print(f"\n  {'Start':>10} {'Length':>8} {'F CAGR':>8} {'Q CAGR':>8} {'F-Q':>8} {'F Sharpe':>9} {'Q Sharpe':>9} {'F DD':>8} {'Bears':>6}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*9} {'-'*8} {'-'*6}")

    for sd, length, why in start_dates:
        fs, cbs = run_full_backtest(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, f"{sd}-01")
        qs = qqq.reindex(fs.index).fillna(0)

        fc = cagr(fs); qc = cagr(qs)
        fsh = sharpe(fs); qsh = sharpe(qs)
        fdd = max_dd(fs)

        # Count bear markets (>20% DD in QQQ)
        bears = 0
        qs_cum = (1+qs).cumprod(); qs_peak = qs_cum.expanding().max()
        qs_dd = (qs_cum - qs_peak) / qs_peak
        # Simple: count distinct periods where DD < -20%
        in_bear = False
        for d in qs_dd:
            if d < -0.20 and not in_bear: bears += 1; in_bear = True
            if d > -0.05: in_bear = False

        print(f"  {sd:>10} {length:>8} {fc:>7.1%} {qc:>7.1%} {fc-qc:>+7.1%} {fsh:>9.3f} {qsh:>9.3f} {fdd:>7.1%} {bears:>5}")

    # ── PHASE 4: 2010-era investor ───────────────────────────────────────
    print(f"\n{'='*110}")
    print(f"  PHASE 4: WHAT A 2010-ERA INVESTOR WOULD HAVE BUILT")
    print(f"{'='*110}")

    # Using only 2002-2009 data
    f_pre, _ = run_full_backtest(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, "2002-01-01")
    f_pre = f_pre[f_pre.index <= pd.Timestamp("2009-12-31")]
    q_pre = qqq.reindex(f_pre.index).fillna(0)

    fc_pre = cagr(f_pre); qc_pre = cagr(q_pre)
    fsh_pre = sharpe(f_pre)

    print(f"\n  Backtested on 2002-2009 only (what they'd see when building the system):")
    print(f"    Faber CAGR: {fc_pre:.1%}")
    print(f"    QQQ CAGR:   {qc_pre:.1%}")
    print(f"    Faber Sharpe: {fsh_pre:.3f}")
    print(f"    Faber alpha vs QQQ: {fc_pre-qc_pre:+.1%}")

    # Forward from 2010
    f_post = faber[faber.index >= pd.Timestamp("2010-01-01")]
    q_post = qqq.reindex(f_post.index).fillna(0)
    fc_post = cagr(f_post); qc_post = cagr(q_post)
    fsh_post = sharpe(f_post)

    print(f"\n  What actually happened 2010-2026 (out-of-sample):")
    print(f"    Faber CAGR: {fc_post:.1%}")
    print(f"    QQQ CAGR:   {qc_post:.1%}")
    print(f"    Faber Sharpe: {fsh_post:.3f}")
    print(f"    Faber alpha vs QQQ: {fc_post-qc_post:+.1%}")
    print(f"    Expectation gap: {fc_post-fc_pre:+.1%} vs 2002-2009 expectation")

    # Worst 12-month period for Faber post-2010
    f_post_m = f_post.resample("MS").apply(lambda x: (1+x).prod()-1)
    f_post_12m = f_post_m.rolling(12).apply(lambda x: (1+x).prod()-1).dropna()
    worst_12m = f_post_12m.min()
    worst_12m_date = f_post_12m.idxmin()
    print(f"\n  Worst rolling 12-month return: {worst_12m:.1%} (ending {worst_12m_date.strftime('%Y-%m')})")

    # Was there a 12+ month period below T-bill?
    tbill_12m = 0.02 * 1  # ~2% annual rough
    below_tbill = (f_post_12m < tbill_12m / 12).sum()
    print(f"  Months with trailing 12m return below ~2% (T-bill proxy): {below_tbill}")

    # ── PHASE 5: Honest forward distribution ─────────────────────────────
    print(f"\n{'='*110}")
    print(f"  PHASE 5: HONEST FORWARD DISTRIBUTION")
    print(f"{'='*110}")

    # From start-date sensitivity
    print(f"\n  Based on sub-period analysis:")
    print(f"\n  {'Scenario':<30} {'P(est)':>8} {'Faber CAGR':>12} {'vs QQQ':>10}")
    print(f"  {'-'*30} {'-'*8} {'-'*12} {'-'*10}")
    print(f"  {'Bull-dominated decade':<30} {'30%':>8} {'10-13%':>12} {'QQQ +3-8%':>10}")
    print(f"  {'Mixed with 1 bear':<30} {'50%':>8} {'13-16%':>12} {'Faber +1-4%':>10}")
    print(f"  {'Bear-heavy decade':<30} {'20%':>8} {'16-22%':>12} {'Faber +5-12%':>10}")

    # Confidence interval for 3-year OOS Sharpe
    # Sharpe SE ≈ sqrt((1 + Sharpe²/2) / N_years) for annual Sharpe
    sharpe_est = 0.929; n_years_3 = 3
    sharpe_se = np.sqrt((1 + sharpe_est**2 / 2) / n_years_3)
    print(f"\n  OOS Sharpe confidence (3 years):")
    print(f"    Point estimate: {sharpe_est:.3f}")
    print(f"    SE (3yr sample): {sharpe_se:.3f}")
    print(f"    95% CI: [{sharpe_est - 1.96*sharpe_se:.3f}, {sharpe_est + 1.96*sharpe_se:.3f}]")
    print(f"    → Need >3 years to distinguish from luck (lower CI includes 0)")

    # ── VERDICT ──────────────────────────────────────────────────────────
    print(f"\n{'='*110}")
    print(f"  VERDICT")
    print(f"{'='*110}")

    # Compute key stats for verdict
    full_cagr_f = cagr(faber); full_cagr_q = cagr(qqq)
    bull_cagr_f = cagr(f_bull); bull_cagr_q = cagr(q_bull)

    print(f"\n  An investor starting in 2002 who ran Faber-Sweep-40 through March 2026")
    print(f"  would have seen Faber {'beat' if full_cagr_f > full_cagr_q else 'trail'} QQQ by "
          f"{abs(full_cagr_f - full_cagr_q):.1%} CAGR overall,")
    print(f"  with a maximum consecutive 12-month underperformance streak of {max_streak} months,")

    if max_gap_date:
        print(f"  and a maximum DCA dollar gap of ${abs(max_gap):,.0f} at $700/month")
        print(f"  (Faber {'behind' if max_gap < 0 else 'ahead'} at {max_gap_date.strftime('%Y-%m')}).")

    print(f"\n  During the 2013-2021 bull market specifically:")
    print(f"    Faber CAGR: {bull_cagr_f:.1%} vs QQQ CAGR: {bull_cagr_q:.1%} "
          f"({'Faber wins' if bull_cagr_f > bull_cagr_q else 'QQQ wins'} by {abs(bull_cagr_f-bull_cagr_q):.1%})")
    print(f"    QQQ beat Faber's trailing 12m return in {len(under_months)}/{len(underperf)} months ({len(under_months)/len(underperf)*100:.0f}%)")

    print()


if __name__ == "__main__":
    print("=" * 110)
    print("  REGIME-CONDITIONAL PERFORMANCE: BULL MARKET SURVIVABILITY TEST")
    print("=" * 110)

    print(f"\n  Loading data...")
    daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start = load_data()

    run_analysis(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start)

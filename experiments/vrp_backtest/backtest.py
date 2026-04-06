"""VRP (Volatility Risk Premium) Harvesting Backtest — Pod 2.

Uses CBOE PUT index (cash-secured put-write on S&P 500) as the VRP return series.
Tests standalone performance, correlation with Faber-Sweep-40, VRP filter,
combined portfolios, and extended history.
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv; load_dotenv()

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

from taa.data import load_daily_etf_returns, load_monthly_prices
from taa.faber import compute_trend_scores, apply_faber_filter

# ── Configuration ────────────────────────────────────────────────────────────

PUT_CSV = Path("data/raw/optionsdx/PUT_index_combined.csv")
BASELINE = {"IVV": 0.45, "QQQ": 0.25, "VGLT": 0.05, "IAU": 0.10, "DBC": 0.05, "cash": 0.10}
ASSETS = list(BASELINE.keys())
SSO_EXPENSE = 0.0089
QLD_EXPENSE = 0.0095
FABER_SUB = 0.40  # Faber-Sweep-40 substitution


# ── Data loading ─────────────────────────────────────────────────────────────

def load_put_index():
    """Load PUT index daily prices and returns."""
    df = pd.read_csv(PUT_CSV, parse_dates=["Date"], index_col="Date")
    df.index = pd.to_datetime(df.index)
    prices = df["Price"].sort_index().dropna()
    returns = prices.pct_change().dropna()
    return prices, returns


def load_faber_data():
    """Load data needed for Faber-Sweep-40."""
    daily_ret = load_daily_etf_returns()
    daily_ret = daily_ret[[c for c in daily_ret.columns if c in ASSETS]]
    monthly_prices = load_monthly_prices()
    trend_df = compute_trend_scores(monthly_prices)

    from fredapi import Fred
    rfr_daily = pd.Series(0.0, index=daily_ret.index)
    key = os.environ.get("FRED_API_KEY")
    if key:
        tb = Fred(api_key=key).get_series("DTB3", observation_start="1998-01-01")
        tb.index = pd.to_datetime(tb.index)
        rfr_daily = (tb / 100 / 252).reindex(daily_ret.index, method="ffill").fillna(0)

    return daily_ret, trend_df, rfr_daily


def run_faber_sweep40(daily_ret, trend_df, rfr_daily):
    """Generate Faber-Sweep-40 daily returns."""
    bt_start = pd.Timestamp("2002-01-01")
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:].index

    cur_trends = {a: 3 for a in ASSETS if a != "cash"}
    w_faber = dict(BASELINE)
    faber_conv = False
    results = {}

    for day in trading_days:
        if day not in daily_ret.index:
            continue
        dr = daily_ret.loc[day]
        avail = [a for a in ASSETS if a in dr.index and pd.notna(dr[a])]
        if len(avail) < 3:
            continue
        actual = {a: float(dr[a]) for a in avail}
        rfr = float(rfr_daily.get(day, 0))

        is_ms = (day == trading_days[0] or
                 day.month != trading_days[trading_days.get_loc(day) - 1].month)

        if is_ms:
            ts_c = trend_df.index[trend_df.index <= day]
            if len(ts_c) > 0:
                for a in ASSETS:
                    if a == "cash":
                        continue
                    if a in trend_df.columns:
                        v = trend_df.loc[ts_c[-1], a]
                        cur_trends[a] = int(v) if pd.notna(v) else 0

            w1, pool = apply_faber_filter(cur_trends, BASELINE)
            w_faber = dict(w1)
            w_faber["cash"] = w_faber.get("cash", 0) + pool
            faber_conv = (cur_trends.get("IVV", 0) >= 3 and
                          cur_trends.get("QQQ", 0) >= 3)

        iw = w_faber.get("IVV", 0)
        qw = w_faber.get("QQQ", 0)
        ir = actual.get("IVV", 0)
        qr = actual.get("QQQ", 0)
        base_ret = sum(w_faber.get(a, 0) * actual.get(a, 0)
                       for a in avail if a not in ["IVV", "QQQ"])

        sub = FABER_SUB if faber_conv else 0.0
        if sub <= 0:
            results[day] = iw * ir + qw * qr + base_ret
        else:
            sso = 2.0 * ir - rfr - SSO_EXPENSE / 252
            qld = 2.0 * qr - rfr - QLD_EXPENSE / 252
            results[day] = iw * ((1 - sub) * ir + sub * sso) + qw * ((1 - sub) * qr + sub * qld) + base_ret

    return pd.Series(results).sort_index()


def load_vix_and_rvol():
    """Load VIX daily close and compute 21-day realized vol of IVV."""
    import yfinance as yf

    vix = yf.download("^VIX", start="2000-01-01", progress=False)
    vix_close = None
    if vix is not None and not vix.empty:
        p = vix["Close"]
        if hasattr(p, "columns"):
            p = p.iloc[:, 0]
        p.index = pd.to_datetime(p.index).tz_localize(None)
        vix_close = p

    # IVV daily returns for realized vol
    ivv = yf.download("SPY", start="2000-01-01", progress=False)
    rvol_21d = None
    if ivv is not None and not ivv.empty:
        p = ivv["Close"]
        if hasattr(p, "columns"):
            p = p.iloc[:, 0]
        p.index = pd.to_datetime(p.index).tz_localize(None)
        daily_ret = p.pct_change().dropna()
        rvol_21d = daily_ret.rolling(21, min_periods=15).std() * np.sqrt(252) * 100  # in vol points

    return vix_close, rvol_21d


def load_tbill_monthly():
    """Load 3-month T-bill monthly return for VRP filter cash alternative."""
    from fredapi import Fred
    key = os.environ.get("FRED_API_KEY")
    if not key:
        return None
    tb = Fred(api_key=key).get_series("DTB3", observation_start="2000-01-01")
    tb.index = pd.to_datetime(tb.index)
    # Convert annualized rate to monthly return
    monthly = tb.resample("MS").last().dropna()
    return monthly / 100 / 12  # monthly return


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(s):
    s = s.dropna()
    if len(s) < 50:
        return None
    ar = s.mean() * 252
    av = s.std() * np.sqrt(252)
    sh = ar / av if av > 0 else 0
    neg = s[s < 0]
    ds = neg.std() * np.sqrt(252) if len(neg) > 10 else av
    sortino = ar / ds if ds > 0 else 0
    cum = (1 + s).cumprod()
    dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
    calmar = ar / abs(dd) if dd != 0 else 0
    final = cum.iloc[-1]
    return {"ar": ar, "av": av, "sh": sh, "sortino": sortino,
            "dd": dd, "calmar": calmar, "final": final}


def compute_metrics_monthly(s):
    s = s.dropna()
    if len(s) < 12:
        return None
    ar = s.mean() * 12
    av = s.std() * np.sqrt(12)
    sh = ar / av if av > 0 else 0
    neg = s[s < 0]
    ds = neg.std() * np.sqrt(12) if len(neg) > 5 else av
    sortino = ar / ds if ds > 0 else 0
    cum = (1 + s).cumprod()
    dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
    calmar = ar / abs(dd) if dd != 0 else 0
    final = cum.iloc[-1]
    return {"ar": ar, "av": av, "sh": sh, "sortino": sortino,
            "dd": dd, "calmar": calmar, "final": final}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: STANDALONE PUT INDEX ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def step1_standalone(put_ret, faber40_ret, ivv_ret, b60_ret):
    print(f"\n{'='*120}")
    print(f"  STEP 1: STANDALONE PUT INDEX ANALYSIS (2002-2026)")
    print(f"{'='*120}")

    bt_start = pd.Timestamp("2002-01-01")
    put_bt = put_ret[put_ret.index >= bt_start]

    p_put = compute_metrics(put_bt)
    p_fab = compute_metrics(faber40_ret)
    p_ivv = compute_metrics(ivv_ret)
    p_60 = compute_metrics(b60_ret)

    # Performance table
    print(f"\n  {'Strategy':<20} {'Return':>8} {'Vol':>7} {'Sharpe':>8} {'Sortino':>8} "
          f"{'MaxDD':>8} {'Calmar':>8} {'Terminal($1)':>13} {'vs 60/40':>10} {'vs IVV':>10}")
    print(f"  {'-'*20} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*13} {'-'*10} {'-'*10}")

    for p, label in [(p_put, "PUT Index (VRP)"), (p_fab, "Faber-Sweep-40"),
                      (p_ivv, "IVV B&H"), (p_60, "60/40")]:
        if p is None:
            continue
        vs60 = p["final"] - p_60["final"]
        vsivv = p["final"] - p_ivv["final"]
        print(f"  {label:<20} {p['ar']:>7.1%} {p['av']:>6.1%} {p['sh']:>8.3f} {p['sortino']:>8.3f} "
              f"{p['dd']:>7.1%} {p['calmar']:>8.2f} ${p['final']:>12.2f} {vs60:>+9.2f} {vsivv:>+9.2f}")

    # Crisis analysis
    print(f"\n  CRISIS ANALYSIS:")
    for cname, cs, ce in [("GFC (2008-09)", "2008-09-01", "2009-03-31"),
                           ("COVID (Feb-Mar 2020)", "2020-02-19", "2020-03-23"),
                           ("2022 Bear", "2022-01-03", "2022-10-31")]:
        print(f"\n  {cname}:")
        print(f"  {'Strategy':<20} {'Return':>10} {'MaxDD':>10}")
        print(f"  {'-'*20} {'-'*10} {'-'*10}")
        for sr, label in [(put_bt, "PUT Index"), (faber40_ret, "Faber-Sweep-40"),
                           (ivv_ret, "IVV B&H"), (b60_ret, "60/40")]:
            c = sr[(sr.index >= pd.Timestamp(cs)) & (sr.index <= pd.Timestamp(ce))]
            if len(c) > 0:
                cum = (1 + c).cumprod()
                mdd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
                print(f"  {label:<20} {(1+c).prod()-1:>+9.1%} {mdd:>9.1%}")

    # Return distribution
    put_monthly = put_bt.resample("MS").apply(lambda x: (1 + x).prod() - 1)

    print(f"\n  RETURN DISTRIBUTION (PUT Index, monthly):")
    print(f"    Mean monthly return:         {put_monthly.mean()*100:>+.2f}%")
    print(f"    Median monthly return:       {put_monthly.median()*100:>+.2f}%")
    print(f"    Skewness:                    {put_monthly.skew():.2f}  (expected: negative)")
    print(f"    Kurtosis:                    {put_monthly.kurtosis():.2f}  (expected: elevated)")
    print(f"    % months positive:           {(put_monthly > 0).mean()*100:.0f}%")
    worst_mo = put_monthly.idxmin()
    best_mo = put_monthly.idxmax()
    print(f"    Worst single month:          {put_monthly.min()*100:>-.1f}% ({worst_mo.strftime('%Y-%m')})")
    print(f"    Best single month:           {put_monthly.max()*100:>+.1f}% ({best_mo.strftime('%Y-%m')})")
    print(f"    Months with loss > 5%:       {(put_monthly < -0.05).sum()}")
    print(f"    Months with loss > 10%:      {(put_monthly < -0.10).sum()}")
    avg_gain = put_monthly[put_monthly > 0].mean()
    avg_loss = put_monthly[put_monthly < 0].mean()
    steamroller = abs(avg_loss / avg_gain) if avg_gain > 0 else 0
    print(f"    Avg gain in positive months: {avg_gain*100:>+.2f}%")
    print(f"    Avg loss in negative months: {avg_loss*100:>+.2f}%")
    print(f"    Steamroller ratio (|avg loss| / avg gain): {steamroller:.1f}x")

    return p_put, p_fab, p_ivv, p_60


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: CORRELATION WITH FABER-SWEEP-40
# ══════════════════════════════════════════════════════════════════════════════

def step2_correlation(put_ret, faber40_ret, ivv_ret, b60_ret):
    print(f"\n{'='*120}")
    print(f"  STEP 2: CORRELATION WITH FABER-SWEEP-40")
    print(f"{'='*120}")

    bt_start = pd.Timestamp("2002-01-01")

    # Monthly returns for correlation
    series = {
        "PUT Index": put_ret[put_ret.index >= bt_start].resample("MS").apply(lambda x: (1 + x).prod() - 1),
        "Faber-40": faber40_ret.resample("MS").apply(lambda x: (1 + x).prod() - 1),
        "IVV B&H": ivv_ret.resample("MS").apply(lambda x: (1 + x).prod() - 1),
        "60/40": b60_ret.resample("MS").apply(lambda x: (1 + x).prod() - 1),
    }

    df = pd.DataFrame(series).dropna()

    # Full-period correlation
    corr = df.corr()
    print(f"\n  Full-period correlation (2002-2026, monthly returns):")
    labels = list(corr.columns)
    print(f"  {'':>14}", end="")
    for l in labels:
        print(f" {l:>12}", end="")
    print()
    for r in labels:
        print(f"  {r:>14}", end="")
        for c in labels:
            print(f" {corr.loc[r, c]:>12.3f}", end="")
        print()

    # Crisis-period correlation
    print(f"\n  Crisis-period correlation (monthly returns):")
    print(f"  {'Crisis':<22} {'PUT vs Faber-40':>16} {'PUT vs IVV B&H':>16}")
    print(f"  {'-'*22} {'-'*16} {'-'*16}")

    for cname, cs, ce in [("GFC 2008-2009", "2008-01", "2009-12"),
                           ("COVID Feb-Apr 2020", "2020-02", "2020-04"),
                           ("2022 Bear", "2022-01", "2022-10")]:
        crisis = df[(df.index >= pd.Timestamp(cs)) & (df.index <= pd.Timestamp(ce))]
        if len(crisis) >= 3:
            pf = crisis["PUT Index"].corr(crisis["Faber-40"])
            pi = crisis["PUT Index"].corr(crisis["IVV B&H"])
            print(f"  {cname:<22} {pf:>16.3f} {pi:>16.3f}")
        else:
            print(f"  {cname:<22} {'insuff data':>16} {'insuff data':>16}")

    # Rolling 12-month correlation
    rolling = df["PUT Index"].rolling(12).corr(df["Faber-40"]).dropna()
    print(f"\n  Rolling 12-month correlation (PUT vs Faber-Sweep-40):")
    print(f"    Min:  {rolling.min():.3f}")
    print(f"    Max:  {rolling.max():.3f}")
    print(f"    Mean: {rolling.mean():.3f}")

    # Sustained high-correlation periods
    high_corr = rolling[rolling > 0.60]
    if len(high_corr) > 0:
        # Find consecutive runs
        runs = []
        current_run = [high_corr.index[0]]
        for i in range(1, len(high_corr)):
            diff = (high_corr.index[i] - high_corr.index[i-1]).days
            if diff < 45:  # within ~1 month
                current_run.append(high_corr.index[i])
            else:
                if len(current_run) >= 3:
                    runs.append((current_run[0], current_run[-1], len(current_run)))
                current_run = [high_corr.index[i]]
        if len(current_run) >= 3:
            runs.append((current_run[0], current_run[-1], len(current_run)))

        if runs:
            print(f"\n    Sustained periods (3+ months) with rolling corr > 0.60:")
            for start, end, n in runs:
                print(f"      {start.strftime('%Y-%m')} to {end.strftime('%Y-%m')} ({n} months)")
        else:
            print(f"    No sustained periods (3+ months) with rolling corr > 0.60")
    else:
        print(f"    Rolling correlation never exceeded 0.60")

    return corr


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: VRP FILTER TEST
# ══════════════════════════════════════════════════════════════════════════════

def step3_vrp_filter(put_ret, vix_close, rvol_21d, tbill_monthly):
    print(f"\n{'='*120}")
    print(f"  STEP 3: VRP FILTER TEST")
    print(f"{'='*120}")

    if vix_close is None or rvol_21d is None:
        print(f"  SKIP — VIX or realized vol data unavailable")
        return

    bt_start = pd.Timestamp("2002-01-01")

    # Monthly VRP spread: VIX close at month-end minus 21-day realized vol at month-end
    vix_monthly = vix_close.resample("MS").last().dropna()
    rvol_monthly = rvol_21d.resample("MS").last().dropna()

    vrp_spread = (vix_monthly - rvol_monthly).dropna()
    vrp_spread = vrp_spread[vrp_spread.index >= bt_start]

    # PUT monthly returns
    put_monthly = put_ret[put_ret.index >= bt_start].resample("MS").apply(lambda x: (1 + x).prod() - 1)

    # T-bill monthly returns for cash alternative
    if tbill_monthly is not None:
        tbill = tbill_monthly.reindex(put_monthly.index, method="ffill").fillna(0)
    else:
        tbill = pd.Series(0.0, index=put_monthly.index)

    print(f"\n  VRP spread (VIX - 21d realized vol) stats:")
    print(f"    Mean:   {vrp_spread.mean():.1f} vol points")
    print(f"    Median: {vrp_spread.median():.1f} vol points")
    print(f"    Min:    {vrp_spread.min():.1f} ({vrp_spread.idxmin().strftime('%Y-%m')})")
    print(f"    Max:    {vrp_spread.max():.1f} ({vrp_spread.idxmax().strftime('%Y-%m')})")
    print(f"    % months VRP >= 0: {(vrp_spread >= 0).mean()*100:.0f}%")

    thresholds = [("No filter", None), (">= 0", 0), (">= 2", 2), (">= 4", 4)]

    print(f"\n  {'Filter':<14} {'Return':>8} {'Vol':>7} {'Sharpe':>8} {'MaxDD':>8} {'Terminal':>10} {'% Active':>10}")
    print(f"  {'-'*14} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")

    for label, thresh in thresholds:
        # Align dates
        common = put_monthly.index.intersection(vrp_spread.index)
        pm = put_monthly.reindex(common)
        tb = tbill.reindex(common)
        vr = vrp_spread.reindex(common)

        if thresh is None:
            # No filter — always hold PUT
            filtered = pm
            pct_active = 1.0
        else:
            # Shift VRP spread by 1 month for PIT safety: use prior month's spread
            vr_shifted = vr.shift(1)
            active = vr_shifted >= thresh
            filtered = pm.where(active, tb)
            pct_active = active.dropna().mean()

        m = compute_metrics_monthly(filtered.dropna())
        if m:
            print(f"  {label:<14} {m['ar']:>7.1%} {m['av']:>6.1%} {m['sh']:>8.3f} {m['dd']:>7.1%} "
                  f"${m['final']:>9.2f} {pct_active:>9.0%}")

    print(f"\n  Note: VRP spread shifted by 1 month for PIT safety (use prior month's reading).")
    print(f"  Filter >= 0 is the natural boundary: only harvest when premium actually exists.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: COMBINED PORTFOLIO ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def step4_combined(put_ret, faber40_ret, ivv_ret, b60_ret):
    print(f"\n{'='*120}")
    print(f"  STEP 4: COMBINED PORTFOLIO ANALYSIS")
    print(f"{'='*120}")

    bt_start = pd.Timestamp("2002-01-01")

    # Monthly returns for monthly rebalancing
    put_m = put_ret[put_ret.index >= bt_start].resample("MS").apply(lambda x: (1 + x).prod() - 1)
    fab_m = faber40_ret.resample("MS").apply(lambda x: (1 + x).prod() - 1)
    ivv_m = ivv_ret.resample("MS").apply(lambda x: (1 + x).prod() - 1)
    b60_m = b60_ret.resample("MS").apply(lambda x: (1 + x).prod() - 1)

    common = put_m.index.intersection(fab_m.index)
    put_m = put_m.reindex(common)
    fab_m = fab_m.reindex(common)
    ivv_m = ivv_m.reindex(common)
    b60_m = b60_m.reindex(common)

    sleeves = [
        ("Faber-only",   1.00, 0.00),
        ("Combined-10",  0.90, 0.10),
        ("Combined-20",  0.80, 0.20),
        ("Combined-30",  0.70, 0.30),
        ("PUT-only",     0.00, 1.00),
    ]

    port_series = {}
    for name, fab_w, put_w in sleeves:
        port_series[name] = fab_w * fab_m + put_w * put_m

    p_ivv = compute_metrics_monthly(ivv_m)
    p_60 = compute_metrics_monthly(b60_m)

    # Performance table
    print(f"\n  {'Portfolio':<16} {'Return':>8} {'Vol':>7} {'Sharpe':>8} {'Sortino':>8} "
          f"{'MaxDD':>8} {'Calmar':>8} {'Terminal($1)':>13} {'vs Faber':>10} {'vs IVV':>10} {'vs 60/40':>10}")
    print(f"  {'-'*16} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*13} {'-'*10} {'-'*10} {'-'*10}")

    perfs = {}
    for name, _, _ in sleeves:
        m = compute_metrics_monthly(port_series[name])
        if m is None:
            continue
        perfs[name] = m
        faber_p = perfs.get("Faber-only", m)
        vs_fab = m["final"] - faber_p["final"]
        vs_ivv = m["final"] - p_ivv["final"] if p_ivv else 0
        vs_60 = m["final"] - p_60["final"] if p_60 else 0
        print(f"  {name:<16} {m['ar']:>7.1%} {m['av']:>6.1%} {m['sh']:>8.3f} {m['sortino']:>8.3f} "
              f"{m['dd']:>7.1%} {m['calmar']:>8.2f} ${m['final']:>12.2f} "
              f"{vs_fab:>+9.2f} {vs_ivv:>+9.2f} {vs_60:>+9.2f}")

    # Benchmarks
    for p, label in [(p_ivv, "IVV B&H"), (p_60, "60/40")]:
        if p is None:
            continue
        print(f"  {label:<16} {p['ar']:>7.1%} {p['av']:>6.1%} {p['sh']:>8.3f} {p['sortino']:>8.3f} "
              f"{p['dd']:>7.1%} {p['calmar']:>8.2f} ${p['final']:>12.2f}")

    # Crisis analysis
    print(f"\n  CRISIS ANALYSIS (combined portfolios):")
    for cname, cs, ce in [("GFC (2008-09)", "2008-09", "2009-03"),
                           ("COVID (Feb-Apr 2020)", "2020-02", "2020-04"),
                           ("2022 Bear", "2022-01", "2022-10")]:
        print(f"\n  {cname}:")
        print(f"  {'Portfolio':<16} {'Return':>10} {'MaxDD':>10}")
        print(f"  {'-'*16} {'-'*10} {'-'*10}")
        all_s = [(name, port_series[name]) for name, _, _ in sleeves]
        all_s += [("IVV B&H", ivv_m), ("60/40", b60_m)]
        for name, sr in all_s:
            c = sr[(sr.index >= pd.Timestamp(cs)) & (sr.index <= pd.Timestamp(ce))]
            if len(c) > 0:
                cum = (1 + c).cumprod()
                mdd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
                print(f"  {name:<16} {(1+c).prod()-1:>+9.1%} {mdd:>9.1%}")

    # Diversification benefit summary
    print(f"\n  DIVERSIFICATION BENEFIT SUMMARY:")
    print(f"  {'Sleeve':<12} {'Sharpe Δ':>10} {'MaxDD Δ':>10} {'Terminal Δ':>12} {'Vol Δ':>8}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*12} {'-'*8}")
    fab_p = perfs.get("Faber-only")
    if fab_p:
        for name in ["Combined-10", "Combined-20", "Combined-30"]:
            m = perfs.get(name)
            if m:
                print(f"  {name:<12} {m['sh']-fab_p['sh']:>+9.3f} {(m['dd']-fab_p['dd'])*100:>+9.1f}% "
                      f"${m['final']-fab_p['final']:>+11.2f} {(m['av']-fab_p['av'])*100:>+7.1f}%")

    return perfs, port_series


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: EXTENDED HISTORY (1988-2001)
# ══════════════════════════════════════════════════════════════════════════════

def step5_extended_history(put_ret):
    print(f"\n{'='*120}")
    print(f"  STEP 5: EXTENDED HISTORY CHECK (1988-2001)")
    print(f"{'='*120}")

    pre_bt = put_ret[(put_ret.index >= "1988-06-01") & (put_ret.index < "2002-01-01")]
    if len(pre_bt) < 252:
        print(f"  Insufficient data for pre-backtest period.")
        return

    m = compute_metrics(pre_bt)
    if m:
        print(f"\n  {'Period':<16} {'Return':>8} {'Vol':>7} {'Sharpe':>8} {'MaxDD':>8} {'Key Events'}")
        print(f"  {'-'*16} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*40}")
        print(f"  {'1988-2001':<16} {m['ar']:>7.1%} {m['av']:>6.1%} {m['sh']:>8.3f} {m['dd']:>7.1%} "
              f"1987 aftermath, LTCM 1998, dot-com")

    # Oct 1987 check (data starts Jun 1988, may not capture)
    oct87 = put_ret[(put_ret.index >= "1987-10-01") & (put_ret.index <= "1987-10-31")]
    if len(oct87) > 0:
        print(f"\n  Oct 1987 crash: PUT index return = {(1+oct87).prod()-1:+.1%}")
    else:
        print(f"\n  Oct 1987: Data starts Jun 1988 — crash period not captured.")

    # LTCM / Aug-Oct 1998
    ltcm = put_ret[(put_ret.index >= "1998-08-01") & (put_ret.index <= "1998-10-31")]
    if len(ltcm) > 0:
        cum = (1 + ltcm).cumprod()
        mdd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
        print(f"  LTCM 1998 (Aug-Oct): return = {(1+ltcm).prod()-1:+.1%}, DD = {mdd:.1%}")

    # Dot-com crash (Mar 2000 - Mar 2001)
    dotcom = put_ret[(put_ret.index >= "2000-03-01") & (put_ret.index <= "2001-03-31")]
    if len(dotcom) > 0:
        cum = (1 + dotcom).cumprod()
        mdd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
        print(f"  Dot-com crash (Mar 2000 - Mar 2001): return = {(1+dotcom).prod()-1:+.1%}, DD = {mdd:.1%}")

    print(f"\n  Note: Pre-2007 PUT index data is CBOE back-tested methodology (reliable but not live-calculated).")
    print(f"  Post-2007 is live. Full history covers 1988 crash aftermath, LTCM, dot-com, GFC, COVID, 2022.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: IMPLEMENTATION NOTES
# ══════════════════════════════════════════════════════════════════════════════

def step6_implementation():
    print(f"\n{'='*120}")
    print(f"  STEP 6: PRACTICAL IMPLEMENTATION NOTES (Schwab Roth IRA)")
    print(f"{'='*120}")

    print(f"""
  1. OPTIONS APPROVAL: Level 2 required at Schwab for cash-secured puts. Already approved.

  2. INSTRUMENT CHOICE:
     - IVV options: American-style, early assignment risk (manageable — assignment just means
       buying IVV at the strike, which is the desired outcome if ITM at expiry).
     - SPX options: European-style, cash-settled, no assignment risk. 100x notional per point.
       Potentially more capital-efficient but different margin treatment.

  3. MINIMUM POSITION SIZE:
     - 1 IVV contract = 100 shares. At ~$520/share, 1 contract requires ~$52,000 collateral.
     - For 10% VRP sleeve on $500K portfolio: $50,000 → 1 contract barely.
     - For 20% VRP sleeve on $500K portfolio: $100,000 → ~2 contracts.
     - Practical threshold: ~$250K portfolio to run even a 10% sleeve with 1 contract.

  4. CADENCE:
     - PUT index methodology: sell on third Friday of each month (standard monthly expiry).
     - Alternative: sell 30-45 DTE rolling, which provides slightly smoother premium collection
       but requires more frequent management.
     - Recommendation: match PUT index methodology (monthly, standard expiry) for clean tracking.

  5. TRANSACTION COSTS:
     - Bid-ask spread on IVV ATM options: ~$0.05-0.15 per share ($5-15 per contract).
     - At monthly frequency (12 trades/year): ~$60-180/year per contract.
     - As annual drag on $52K notional: ~0.12-0.35% of notional.
     - This is already implicitly included in the PUT index return (index methodology
       uses mid-market pricing; actual execution adds ~half the spread).
""")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 120)
    print("  VRP (VOLATILITY RISK PREMIUM) HARVESTING BACKTEST — POD 2")
    print("=" * 120)

    # Load PUT index
    print(f"\n  Loading PUT index from {PUT_CSV}...")
    put_prices, put_ret = load_put_index()
    print(f"  PUT index: {put_prices.index.min().date()} to {put_prices.index.max().date()} "
          f"({len(put_prices)} daily obs, {len(put_ret)} daily returns)")
    print(f"  Terminal value (from $100 in Jun 1988): ${put_prices.iloc[-1]:.2f}")

    # Generate Faber-Sweep-40 returns
    print(f"\n  Loading Faber data and generating Faber-Sweep-40 daily returns...")
    daily_ret, trend_df, rfr_daily = load_faber_data()
    faber40_ret = run_faber_sweep40(daily_ret, trend_df, rfr_daily)
    print(f"  Faber-Sweep-40: {faber40_ret.index.min().date()} to {faber40_ret.index.max().date()} "
          f"({len(faber40_ret)} daily returns)")

    # Benchmarks
    bt_start = pd.Timestamp("2002-01-01")
    ivv_ret = daily_ret["IVV"].loc[bt_start:].dropna()
    vglt_ret = daily_ret.get("VGLT", pd.Series(dtype=float)).loc[bt_start:].fillna(0)
    b60_ret = 0.60 * ivv_ret + 0.40 * vglt_ret.reindex(ivv_ret.index, fill_value=0)

    # VIX and realized vol for filter test
    print(f"\n  Loading VIX and realized vol for VRP filter...")
    vix_close, rvol_21d = load_vix_and_rvol()
    tbill_monthly = load_tbill_monthly()

    # Step 1: Standalone
    p_put, p_fab, p_ivv, p_60 = step1_standalone(put_ret, faber40_ret, ivv_ret, b60_ret)

    # Step 2: Correlation
    corr_matrix = step2_correlation(put_ret, faber40_ret, ivv_ret, b60_ret)

    # Step 3: VRP filter
    step3_vrp_filter(put_ret, vix_close, rvol_21d, tbill_monthly)

    # Step 4: Combined portfolios
    combined_perfs, combined_series = step4_combined(put_ret, faber40_ret, ivv_ret, b60_ret)

    # Step 5: Extended history
    step5_extended_history(put_ret)

    # Step 6: Implementation
    step6_implementation()

    print()

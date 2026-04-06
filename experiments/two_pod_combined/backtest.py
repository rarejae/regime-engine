"""Two-pod combined portfolio: Faber-Sweep-40 + VRP (PUT index) with Kritzman turbulence.

Pod 1: Faber-Sweep-40 (daily → monthly aggregation)
Pod 2: PUT index with VRP filter >= 0 (monthly)
Turbulence: Hybrid correlation-gated Kritzman (Option C from architecture doc)

Six strategies:
1. Faber-only (control)
2. VRP-only (filtered, control)
3. Two-Pod-10 (90/10, no turbulence)
4. Two-Pod-20 (80/20, no turbulence)
5. Two-Pod-10-Turb (90/10, with turbulence)
6. Two-Pod-20-Turb (80/20, with turbulence)
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
FABER_SUB = 0.40

BT_START = pd.Timestamp("2002-01-01")
TURB_ACTIVATION = pd.Timestamp("2005-01-01")  # 36 months after BT_START
TURB_WINDOW = 36
CORR_WINDOW = 3
CORR_THRESHOLD = 0.65
TURB_PERCENTILE = 75


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_all_data():
    """Load Faber data, PUT index, VIX, realized vol, T-bill rate."""
    # Faber data
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

    # T-bill monthly return for cash
    tbill_monthly = None
    if key:
        tb_ann = Fred(api_key=key).get_series("DTB3", observation_start="2000-01-01")
        tb_ann.index = pd.to_datetime(tb_ann.index)
        tbill_monthly = tb_ann.resample("MS").last().dropna() / 100 / 12

    # PUT index
    put_df = pd.read_csv(PUT_CSV, parse_dates=["Date"], index_col="Date")
    put_prices = put_df["Price"].sort_index().dropna()
    put_monthly_prices = put_prices.resample("MS").last().dropna()
    put_monthly_ret = put_monthly_prices.pct_change().dropna()

    # VIX and realized vol for VRP filter
    import yfinance as yf
    vix = yf.download("^VIX", start="2000-01-01", progress=False)
    vix_monthly = None
    if vix is not None and not vix.empty:
        p = vix["Close"]
        if hasattr(p, "columns"):
            p = p.iloc[:, 0]
        p.index = pd.to_datetime(p.index).tz_localize(None)
        vix_monthly = p.resample("MS").last().dropna()

    # IVV realized vol (21-day)
    ivv_daily = yf.download("SPY", start="2000-01-01", progress=False)
    rvol_monthly = None
    if ivv_daily is not None and not ivv_daily.empty:
        p = ivv_daily["Close"]
        if hasattr(p, "columns"):
            p = p.iloc[:, 0]
        p.index = pd.to_datetime(p.index).tz_localize(None)
        dr = p.pct_change().dropna()
        rvol_21d = dr.rolling(21, min_periods=15).std() * np.sqrt(252) * 100
        rvol_monthly = rvol_21d.resample("MS").last().dropna()

    # IVV monthly returns for benchmarks
    ivv_m = daily_ret["IVV"].resample("MS").apply(lambda x: (1 + x).prod() - 1)
    vglt_ret = daily_ret.get("VGLT", pd.Series(dtype=float))
    b60_m = (0.60 * daily_ret["IVV"] + 0.40 * vglt_ret.reindex(daily_ret["IVV"].index, fill_value=0)).resample("MS").apply(lambda x: (1 + x).prod() - 1)

    return (daily_ret, trend_df, rfr_daily, put_monthly_ret,
            vix_monthly, rvol_monthly, tbill_monthly, ivv_m, b60_m)


# ── Faber-Sweep-40 Monthly Returns ──────────────────────────────────────────

def compute_faber_sweep40_monthly(daily_ret, trend_df, rfr_daily):
    """Generate Faber-Sweep-40 monthly returns from daily backtest."""
    common_start = max(daily_ret.dropna(how="all").index.min(), BT_START)
    trading_days = daily_ret.loc[common_start:].index

    cur_trends = {a: 3 for a in ASSETS if a != "cash"}
    w_faber = dict(BASELINE)
    faber_conv = False
    daily_results = {}

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
            daily_results[day] = iw * ir + qw * qr + base_ret
        else:
            sso = 2.0 * ir - rfr - SSO_EXPENSE / 252
            qld = 2.0 * qr - rfr - QLD_EXPENSE / 252
            daily_results[day] = (iw * ((1 - sub) * ir + sub * sso) +
                                   qw * ((1 - sub) * qr + sub * qld) + base_ret)

    daily_s = pd.Series(daily_results).sort_index()
    monthly_s = daily_s.resample("MS").apply(lambda x: (1 + x).prod() - 1)
    return monthly_s


# ── VRP Filter ───────────────────────────────────────────────────────────────

def apply_vrp_filter(put_monthly_ret, vix_monthly, rvol_monthly, tbill_monthly):
    """Apply >= 0 VRP filter: only invest when VIX > 21d realized vol (PIT shifted)."""
    if vix_monthly is None or rvol_monthly is None:
        return put_monthly_ret  # no filter

    vrp_spread = (vix_monthly - rvol_monthly).dropna()
    vrp_shifted = vrp_spread.shift(1)  # PIT: use prior month's reading

    common = put_monthly_ret.index.intersection(vrp_shifted.dropna().index)
    filtered = put_monthly_ret.reindex(common).copy()

    for dt in common:
        if vrp_shifted.loc[dt] < 0:
            # VRP negative — hold T-bills instead
            if tbill_monthly is not None and dt in tbill_monthly.index:
                filtered.loc[dt] = tbill_monthly.loc[dt]
            else:
                filtered.loc[dt] = 0.0

    return filtered


# ── Turbulence Layer ─────────────────────────────────────────────────────────

def compute_turbulence_series(faber_m, vrp_m):
    """Compute hybrid turbulence signals for each month.

    Returns DataFrame with columns: turbulence, rolling_corr, turb_threshold,
    corr_flag, turb_flag, both_flag, scale_factor.
    """
    common = faber_m.dropna().index.intersection(vrp_m.dropna().index).sort_values()
    f = faber_m.reindex(common)
    v = vrp_m.reindex(common)

    results = []

    for i, dt in enumerate(common):
        row = {"date": dt, "faber_ret": f.iloc[i], "vrp_ret": v.iloc[i]}

        if i < TURB_WINDOW:
            row.update({"turbulence": np.nan, "rolling_corr": np.nan,
                         "turb_threshold": np.nan, "corr_flag": False,
                         "turb_flag": False, "both_flag": False, "scale": 1.0})
            results.append(row)
            continue

        # Trailing window
        f_window = f.iloc[i - TURB_WINDOW:i].values
        v_window = v.iloc[i - TURB_WINDOW:i].values
        joint = np.column_stack([f_window, v_window])

        # Mean and covariance (Ledoit-Wolf shrinkage)
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf()
            lw.fit(joint)
            sigma = lw.covariance_
            sigma_inv = np.linalg.inv(sigma)
        except Exception:
            sigma = np.cov(joint.T)
            sigma_inv = np.linalg.inv(sigma + np.eye(2) * 1e-8)

        mu = joint.mean(axis=0)

        # Current return vector
        r_t = np.array([f.iloc[i], v.iloc[i]])
        diff = r_t - mu
        turb_t = float(diff @ sigma_inv @ diff.T)

        # Turbulence threshold: 75th percentile of trailing window
        turb_history = []
        for j in range(max(0, i - TURB_WINDOW), i):
            if j >= TURB_WINDOW:
                r_j = np.array([f.iloc[j], v.iloc[j]])
                d_j = r_j - mu
                turb_history.append(float(d_j @ sigma_inv @ d_j.T))
        if turb_history:
            turb_threshold = np.percentile(turb_history, TURB_PERCENTILE)
        else:
            turb_threshold = turb_t

        # Rolling 3-month correlation
        if i >= CORR_WINDOW:
            f_recent = f.iloc[i - CORR_WINDOW:i]
            v_recent = v.iloc[i - CORR_WINDOW:i]
            if len(f_recent) >= CORR_WINDOW:
                rolling_corr = float(f_recent.corr(v_recent))
            else:
                rolling_corr = np.nan
        else:
            rolling_corr = np.nan

        # Flags
        corr_flag = (not np.isnan(rolling_corr)) and rolling_corr > CORR_THRESHOLD
        turb_flag = turb_t > turb_threshold
        both_flag = corr_flag and turb_flag

        # Only activate after TURB_ACTIVATION date
        if dt < TURB_ACTIVATION:
            scale = 1.0
        elif both_flag:
            scale = 0.50
        elif corr_flag or turb_flag:
            scale = 0.75
        else:
            scale = 1.0

        row.update({
            "turbulence": turb_t, "rolling_corr": rolling_corr,
            "turb_threshold": turb_threshold,
            "corr_flag": corr_flag, "turb_flag": turb_flag,
            "both_flag": both_flag, "scale": scale,
        })
        results.append(row)

    return pd.DataFrame(results).set_index("date")


# ── Combined Portfolio Returns ───────────────────────────────────────────────

def compute_combined_returns(faber_m, vrp_m, turb_df, tbill_monthly,
                              faber_weight, vrp_weight, use_turbulence):
    """Compute monthly combined portfolio returns."""
    common = faber_m.dropna().index.intersection(vrp_m.dropna().index).sort_values()
    f = faber_m.reindex(common)
    v = vrp_m.reindex(common)

    results = {}
    for dt in common:
        fw = faber_weight
        vw = vrp_weight

        if use_turbulence and dt in turb_df.index:
            scale = turb_df.loc[dt, "scale"]
            fw *= scale
            vw *= scale

        cash_w = 1.0 - fw - vw
        cash_ret = 0.0
        if tbill_monthly is not None and dt in tbill_monthly.index:
            cash_ret = float(tbill_monthly.loc[dt])

        results[dt] = fw * f.loc[dt] + vw * v.loc[dt] + cash_w * cash_ret

    return pd.Series(results).sort_index()


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(s):
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


# ── Report ───────────────────────────────────────────────────────────────────

def print_report(all_series, turb_df, faber_m, vrp_m, ivv_m, b60_m, strat_names):

    perfs = {}
    for name in strat_names:
        m = compute_metrics(all_series[name])
        if m:
            perfs[name] = m
    p_ivv = compute_metrics(ivv_m.loc[BT_START:])
    p_60 = compute_metrics(b60_m.loc[BT_START:])

    # ── Performance table ────────────────────────────────────────────────
    print(f"\n{'='*130}")
    print(f"  FULL PERFORMANCE TABLE")
    print(f"{'='*130}")
    header = (f"  {'Strategy':<22} {'Return':>8} {'Vol':>7} {'Sharpe':>8} {'Sortino':>8} "
              f"{'MaxDD':>8} {'Calmar':>8} {'Terminal($1)':>13} {'vs Faber':>10} {'vs IVV':>10} {'vs 60/40':>10}")
    print(header)
    print(f"  {'-'*22} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*13} {'-'*10} {'-'*10} {'-'*10}")

    fab_final = perfs.get("Faber-only", {}).get("final", 0)
    for name in strat_names:
        p = perfs.get(name)
        if p is None:
            continue
        vs_fab = p["final"] - fab_final
        vs_ivv = p["final"] - (p_ivv["final"] if p_ivv else 0)
        vs_60 = p["final"] - (p_60["final"] if p_60 else 0)
        print(f"  {name:<22} {p['ar']:>7.1%} {p['av']:>6.1%} {p['sh']:>8.3f} {p['sortino']:>8.3f} "
              f"{p['dd']:>7.1%} {p['calmar']:>8.2f} ${p['final']:>12.2f} "
              f"{vs_fab:>+9.2f} {vs_ivv:>+9.2f} {vs_60:>+9.2f}")

    for p, label in [(p_ivv, "IVV B&H"), (p_60, "60/40")]:
        if p:
            print(f"  {label:<22} {p['ar']:>7.1%} {p['av']:>6.1%} {p['sh']:>8.3f} {p['sortino']:>8.3f} "
                  f"{p['dd']:>7.1%} {p['calmar']:>8.2f} ${p['final']:>12.2f}")

    # ── Crisis analysis ──────────────────────────────────────────────────
    print(f"\n{'='*130}")
    print(f"  CRISIS ANALYSIS")
    print(f"{'='*130}")

    for cname, cs, ce in [("GFC (2008-09)", "2008-09", "2009-03"),
                           ("COVID (Feb-Apr 2020)", "2020-02", "2020-04"),
                           ("2022 Bear", "2022-01", "2022-10")]:
        print(f"\n  {cname}:")
        print(f"  {'Strategy':<22} {'Return':>10} {'MaxDD':>10}")
        print(f"  {'-'*22} {'-'*10} {'-'*10}")
        all_s = [(name, all_series[name]) for name in strat_names]
        all_s += [("IVV B&H", ivv_m), ("60/40", b60_m)]
        for name, sr in all_s:
            c = sr[(sr.index >= pd.Timestamp(cs)) & (sr.index <= pd.Timestamp(ce))]
            if len(c) > 0:
                cum = (1 + c).cumprod()
                mdd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
                print(f"  {name:<22} {(1+c).prod()-1:>+9.1%} {mdd:>9.1%}")

    # ── Turbulence diagnostics ───────────────────────────────────────────
    print(f"\n{'='*130}")
    print(f"  TURBULENCE LAYER DIAGNOSTICS")
    print(f"{'='*130}")

    active = turb_df[turb_df.index >= TURB_ACTIVATION].dropna(subset=["turbulence"])
    total_active = len(active)

    scale_dist = active["scale"].value_counts().sort_index()
    print(f"\n  Turbulence layer activation period: {TURB_ACTIVATION.date()} - {active.index.max().date()}")
    print(f"  Total months with turbulence active: {total_active}")

    print(f"\n  Scaling distribution:")
    for scale_val in [1.0, 0.75, 0.50]:
        n = int(scale_dist.get(scale_val, 0))
        pct = n / total_active * 100 if total_active > 0 else 0
        label = {1.0: "normal", 0.75: "one condition", 0.50: "both conditions"}[scale_val]
        print(f"    Scale = {scale_val:.2f} ({label}): {n:>4} months ({pct:.0f}%)")

    # Condition breakdown
    corr_only = ((active["corr_flag"]) & (~active["turb_flag"])).sum()
    turb_only = ((active["turb_flag"]) & (~active["corr_flag"])).sum()
    both = active["both_flag"].sum()
    print(f"\n  Condition breakdown:")
    print(f"    Corr > {CORR_THRESHOLD} only:         {corr_only} months")
    print(f"    Turb > 75th pct only:       {turb_only} months")
    print(f"    Both conditions:            {both} months")

    # Key trigger months
    both_months = active[active["both_flag"]].sort_values("turbulence", ascending=False)
    if len(both_months) > 0:
        print(f"\n  Top 10 months by turbulence (when both conditions met):")
        print(f"  {'Date':>12} {'Turbulence':>12} {'Roll Corr':>10} {'Faber Ret':>10} {'VRP Ret':>10}")
        for _, row in both_months.head(10).iterrows():
            print(f"  {row.name.strftime('%Y-%m'):>12} {row['turbulence']:>12.1f} {row['rolling_corr']:>10.3f} "
                  f"{row['faber_ret']*100:>+9.1f}% {row['vrp_ret']*100:>+9.1f}%")

    # Crisis trigger behavior
    for cname, cs, ce in [("GFC (2008-09)", "2008-09", "2009-03"),
                           ("COVID (Feb-Apr 2020)", "2020-02", "2020-04"),
                           ("2022 Bear (Jan-Oct)", "2022-01", "2022-10")]:
        crisis = active[(active.index >= pd.Timestamp(cs)) & (active.index <= pd.Timestamp(ce))]
        if len(crisis) == 0:
            print(f"\n  {cname}: no data in turbulence period")
            continue
        both_crisis = crisis[crisis["both_flag"]]
        any_crisis = crisis[(crisis["corr_flag"]) | (crisis["turb_flag"])]
        avg_scale = crisis["scale"].mean()
        print(f"\n  {cname}:")
        print(f"    Months in window: {len(crisis)}")
        print(f"    Both-condition triggers: {len(both_crisis)}")
        print(f"    Any-condition triggers: {len(any_crisis)}")
        print(f"    Average scale factor: {avg_scale:.2f}")
        if len(crisis) > 0:
            avg_faber = crisis["faber_ret"].mean() * 100
            avg_vrp = crisis["vrp_ret"].mean() * 100
            print(f"    Avg Faber return: {avg_faber:+.1f}%/mo")
            print(f"    Avg VRP return: {avg_vrp:+.1f}%/mo")

    # ── Turbulence value-add ─────────────────────────────────────────────
    print(f"\n{'='*130}")
    print(f"  TURBULENCE VALUE-ADD ANALYSIS")
    print(f"{'='*130}")

    for sleeve in [10, 20]:
        base_name = f"Two-Pod-{sleeve}"
        turb_name = f"Two-Pod-{sleeve}-Turb"
        pb = perfs.get(base_name)
        pt = perfs.get(turb_name)
        if pb and pt:
            print(f"\n  {turb_name} vs {base_name}:")
            print(f"    Sharpe:    {pb['sh']:.3f} → {pt['sh']:.3f} (delta: {pt['sh']-pb['sh']:+.3f})")
            print(f"    MaxDD:     {pb['dd']:.1%} → {pt['dd']:.1%} (delta: {(pt['dd']-pb['dd'])*100:+.1f}%)")
            print(f"    Return:    {pb['ar']:.1%} → {pt['ar']:.1%} (drag: {(pt['ar']-pb['ar'])*100:+.1f}%)")
            print(f"    Terminal:  ${pb['final']:.2f} → ${pt['final']:.2f} (delta: ${pt['final']-pb['final']:+.2f})")

    # ── Annual summary ───────────────────────────────────────────────────
    print(f"\n{'='*130}")
    print(f"  ANNUAL SUMMARY (Rolling Corr, Turbulence, Scale Factor)")
    print(f"{'='*130}")
    print(f"  {'Year':>6} {'Avg Corr':>10} {'Avg Turb':>10} {'Avg Scale':>10} {'Faber Ret':>10} {'VRP Ret':>10}")

    for yr in range(2005, 2027):
        yr_data = active[active.index.year == yr]
        if len(yr_data) == 0:
            continue
        avg_corr = yr_data["rolling_corr"].mean()
        avg_turb = yr_data["turbulence"].mean()
        avg_scale = yr_data["scale"].mean()
        avg_faber = yr_data["faber_ret"].mean() * 12
        avg_vrp = yr_data["vrp_ret"].mean() * 12
        corr_str = f"{avg_corr:.2f}" if not np.isnan(avg_corr) else "N/A"
        print(f"  {yr:>6} {corr_str:>10} {avg_turb:>10.1f} {avg_scale:>10.2f} {avg_faber:>+9.1%} {avg_vrp:>+9.1%}")

    # ── Key questions ────────────────────────────────────────────────────
    print(f"\n{'='*130}")
    print(f"  KEY QUESTIONS ANSWERED")
    print(f"{'='*130}")

    fab_p = perfs.get("Faber-only")
    for sleeve in [10, 20]:
        base = perfs.get(f"Two-Pod-{sleeve}")
        turb = perfs.get(f"Two-Pod-{sleeve}-Turb")
        if fab_p and base:
            print(f"\n  Q1. Two-Pod-{sleeve} vs Faber-only (no turbulence):")
            print(f"      Sharpe: {fab_p['sh']:.3f} → {base['sh']:.3f} ({base['sh']-fab_p['sh']:+.3f})")
            print(f"      Terminal: ${fab_p['final']:.2f} → ${base['final']:.2f} (${base['final']-fab_p['final']:+.2f})")
        if base and turb:
            print(f"  Q2. Turbulence layer on Two-Pod-{sleeve}:")
            print(f"      Sharpe: {base['sh']:.3f} → {turb['sh']:.3f} ({turb['sh']-base['sh']:+.3f})")
            print(f"      MaxDD: {base['dd']:.1%} → {turb['dd']:.1%}")

    # GFC check
    gfc_data = active[(active.index >= "2008-09") & (active.index <= "2009-03")]
    if len(gfc_data) > 0:
        gfc_both = gfc_data["both_flag"].sum()
        gfc_faber_avg = gfc_data["faber_ret"].mean()
        print(f"\n  Q3. GFC trigger behavior:")
        print(f"      Both-condition triggers during GFC: {gfc_both}")
        print(f"      Avg Faber return during GFC months: {gfc_faber_avg*100:+.1f}%/mo")
        if gfc_both == 0:
            print(f"      CORRECT: turbulence did NOT fire both conditions during GFC")
            print(f"      (Faber was positive, pods were diverging — hybrid gate worked)")
        else:
            print(f"      WARNING: turbulence fired during GFC — check if this hurt Faber returns")

    print()
    return perfs


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 130)
    print("  TWO-POD COMBINED PORTFOLIO: FABER-SWEEP-40 + VRP (PUT INDEX) + KRITZMAN TURBULENCE")
    print("=" * 130)

    print(f"\n  Loading data...")
    (daily_ret, trend_df, rfr_daily, put_monthly_ret,
     vix_monthly, rvol_monthly, tbill_monthly, ivv_m, b60_m) = load_all_data()

    print(f"  Generating Faber-Sweep-40 monthly returns...")
    faber_m = compute_faber_sweep40_monthly(daily_ret, trend_df, rfr_daily)
    print(f"  Faber-40: {faber_m.index.min().date()} to {faber_m.index.max().date()} ({len(faber_m)} months)")

    print(f"  Applying VRP filter (>= 0)...")
    vrp_filtered = apply_vrp_filter(put_monthly_ret, vix_monthly, rvol_monthly, tbill_monthly)
    vrp_m = vrp_filtered.loc[BT_START:]
    print(f"  VRP filtered: {vrp_m.index.min().date()} to {vrp_m.index.max().date()} ({len(vrp_m)} months)")

    print(f"  Computing turbulence series...")
    turb_df = compute_turbulence_series(faber_m, vrp_m)
    active_turb = turb_df[turb_df.index >= TURB_ACTIVATION].dropna(subset=["turbulence"])
    n_both = active_turb["both_flag"].sum()
    n_any = ((active_turb["corr_flag"]) | (active_turb["turb_flag"])).sum()
    print(f"  Turbulence computed: {len(turb_df)} months, {n_both} both-condition triggers, {n_any} any-condition triggers")

    # Build all strategies
    strat_names = [
        "Faber-only", "VRP-only",
        "Two-Pod-10", "Two-Pod-20",
        "Two-Pod-10-Turb", "Two-Pod-20-Turb",
    ]

    all_series = {}
    all_series["Faber-only"] = faber_m
    all_series["VRP-only"] = vrp_m

    for sleeve in [10, 20]:
        fab_w = 1.0 - sleeve / 100
        vrp_w = sleeve / 100
        all_series[f"Two-Pod-{sleeve}"] = compute_combined_returns(
            faber_m, vrp_m, turb_df, tbill_monthly, fab_w, vrp_w, use_turbulence=False)
        all_series[f"Two-Pod-{sleeve}-Turb"] = compute_combined_returns(
            faber_m, vrp_m, turb_df, tbill_monthly, fab_w, vrp_w, use_turbulence=True)

    perfs = print_report(all_series, turb_df, faber_m, vrp_m, ivv_m, b60_m, strat_names)

"""Two-pod S40 rerun: Faber-Sweep-40 + VRP, turbulence threshold raised to 0.80.

Changes from prior run (two_pod_combined):
1. Explicitly confirms Faber-Sweep-40 (40% SSO/QLD leverage) as Pod 1
2. Correlation threshold raised from 0.65 → 0.80
3. Adds Faber-1x control, 30% sleeve, and comparison table vs prior run

Eight strategies:
1. Faber-Sweep-40-only     (Pod 1 control)
2. Faber-1x-only           (unleveraged control)
3. VRP-only                (Pod 2 control, filtered)
4. Two-Pod-S40-10          (90/10, no turbulence)
5. Two-Pod-S40-20          (80/20, no turbulence)
6. Two-Pod-S40-30          (70/30, no turbulence)
7. Two-Pod-S40-10-Turb     (90/10, turbulence threshold 0.80)
8. Two-Pod-S40-20-Turb     (80/20, turbulence threshold 0.80)
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv; load_dotenv()

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

from taa.data import load_daily_etf_returns, load_monthly_prices
from taa.faber import compute_trend_scores, apply_faber_filter

PUT_CSV = Path("data/raw/optionsdx/PUT_index_combined.csv")
BASELINE = {"IVV": 0.45, "QQQ": 0.25, "VGLT": 0.05, "IAU": 0.10, "DBC": 0.05, "cash": 0.10}
ASSETS = list(BASELINE.keys())
SSO_EXPENSE = 0.0089
QLD_EXPENSE = 0.0095
FABER_SUB = 0.40

BT_START = pd.Timestamp("2002-01-01")
TURB_ACTIVATION = pd.Timestamp("2005-01-01")
TURB_WINDOW = 36
CORR_WINDOW = 3
CORR_THRESHOLD = 0.80  # CHANGED from 0.65
TURB_PERCENTILE = 75


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_all_data():
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

    tbill_monthly = None
    if key:
        tb_ann = Fred(api_key=key).get_series("DTB3", observation_start="2000-01-01")
        tb_ann.index = pd.to_datetime(tb_ann.index)
        tbill_monthly = tb_ann.resample("MS").last().dropna() / 100 / 12

    put_df = pd.read_csv(PUT_CSV, parse_dates=["Date"], index_col="Date")
    put_prices = put_df["Price"].sort_index().dropna()
    put_monthly_ret = put_prices.resample("MS").last().dropna().pct_change().dropna()

    import yfinance as yf
    vix_monthly = None
    vix = yf.download("^VIX", start="2000-01-01", progress=False)
    if vix is not None and not vix.empty:
        p = vix["Close"]
        if hasattr(p, "columns"): p = p.iloc[:, 0]
        p.index = pd.to_datetime(p.index).tz_localize(None)
        vix_monthly = p.resample("MS").last().dropna()

    rvol_monthly = None
    ivv_d = yf.download("SPY", start="2000-01-01", progress=False)
    if ivv_d is not None and not ivv_d.empty:
        p = ivv_d["Close"]
        if hasattr(p, "columns"): p = p.iloc[:, 0]
        p.index = pd.to_datetime(p.index).tz_localize(None)
        dr = p.pct_change().dropna()
        rvol_monthly = (dr.rolling(21, min_periods=15).std() * np.sqrt(252) * 100).resample("MS").last().dropna()

    ivv_m = daily_ret["IVV"].resample("MS").apply(lambda x: (1 + x).prod() - 1)
    vglt_ret = daily_ret.get("VGLT", pd.Series(dtype=float))
    b60_m = (0.60 * daily_ret["IVV"] + 0.40 * vglt_ret.reindex(daily_ret["IVV"].index, fill_value=0)
             ).resample("MS").apply(lambda x: (1 + x).prod() - 1)

    return (daily_ret, trend_df, rfr_daily, put_monthly_ret,
            vix_monthly, rvol_monthly, tbill_monthly, ivv_m, b60_m)


# ── Faber monthly returns (both 1x and Sweep-40) ────────────────────────────

def compute_faber_monthly(daily_ret, trend_df, rfr_daily, sub_pct):
    """Generate Faber monthly returns with given substitution percentage."""
    common_start = max(daily_ret.dropna(how="all").index.min(), BT_START)
    trading_days = daily_ret.loc[common_start:].index
    cur_trends = {a: 3 for a in ASSETS if a != "cash"}
    w_faber = dict(BASELINE)
    faber_conv = False
    daily_results = {}

    for day in trading_days:
        if day not in daily_ret.index: continue
        dr = daily_ret.loc[day]
        avail = [a for a in ASSETS if a in dr.index and pd.notna(dr[a])]
        if len(avail) < 3: continue
        actual = {a: float(dr[a]) for a in avail}
        rfr = float(rfr_daily.get(day, 0))

        is_ms = (day == trading_days[0] or
                 day.month != trading_days[trading_days.get_loc(day) - 1].month)
        if is_ms:
            ts_c = trend_df.index[trend_df.index <= day]
            if len(ts_c) > 0:
                for a in ASSETS:
                    if a == "cash": continue
                    if a in trend_df.columns:
                        v = trend_df.loc[ts_c[-1], a]
                        cur_trends[a] = int(v) if pd.notna(v) else 0
            w1, pool = apply_faber_filter(cur_trends, BASELINE)
            w_faber = dict(w1)
            w_faber["cash"] = w_faber.get("cash", 0) + pool
            faber_conv = (cur_trends.get("IVV", 0) >= 3 and cur_trends.get("QQQ", 0) >= 3)

        iw = w_faber.get("IVV", 0)
        qw = w_faber.get("QQQ", 0)
        ir = actual.get("IVV", 0)
        qr = actual.get("QQQ", 0)
        base_ret = sum(w_faber.get(a, 0) * actual.get(a, 0) for a in avail if a not in ["IVV", "QQQ"])

        sub = sub_pct if faber_conv else 0.0
        if sub <= 0:
            daily_results[day] = iw * ir + qw * qr + base_ret
        else:
            sso = 2.0 * ir - rfr - SSO_EXPENSE / 252
            qld = 2.0 * qr - rfr - QLD_EXPENSE / 252
            daily_results[day] = iw * ((1 - sub) * ir + sub * sso) + qw * ((1 - sub) * qr + sub * qld) + base_ret

    return pd.Series(daily_results).sort_index().resample("MS").apply(lambda x: (1 + x).prod() - 1)


# ── VRP Filter ───────────────────────────────────────────────────────────────

def apply_vrp_filter(put_monthly_ret, vix_monthly, rvol_monthly, tbill_monthly):
    if vix_monthly is None or rvol_monthly is None:
        return put_monthly_ret
    vrp_spread = (vix_monthly - rvol_monthly).dropna()
    vrp_shifted = vrp_spread.shift(1)
    common = put_monthly_ret.index.intersection(vrp_shifted.dropna().index)
    filtered = put_monthly_ret.reindex(common).copy()
    for dt in common:
        if vrp_shifted.loc[dt] < 0:
            filtered.loc[dt] = float(tbill_monthly.loc[dt]) if (tbill_monthly is not None and dt in tbill_monthly.index) else 0.0
    return filtered


# ── Turbulence ───────────────────────────────────────────────────────────────

def compute_turbulence(faber_m, vrp_m):
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

        f_w = f.iloc[i - TURB_WINDOW:i].values
        v_w = v.iloc[i - TURB_WINDOW:i].values
        joint = np.column_stack([f_w, v_w])

        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf(); lw.fit(joint)
            sigma = lw.covariance_; sigma_inv = np.linalg.inv(sigma)
        except Exception:
            sigma = np.cov(joint.T); sigma_inv = np.linalg.inv(sigma + np.eye(2) * 1e-8)

        mu = joint.mean(axis=0)
        r_t = np.array([f.iloc[i], v.iloc[i]])
        diff = r_t - mu
        turb_t = float(diff @ sigma_inv @ diff.T)

        turb_history = []
        for j in range(max(0, i - TURB_WINDOW), i):
            if j >= TURB_WINDOW:
                d_j = np.array([f.iloc[j], v.iloc[j]]) - mu
                turb_history.append(float(d_j @ sigma_inv @ d_j.T))
        turb_threshold = np.percentile(turb_history, TURB_PERCENTILE) if turb_history else turb_t

        rolling_corr = float(f.iloc[i - CORR_WINDOW:i].corr(v.iloc[i - CORR_WINDOW:i])) if i >= CORR_WINDOW else np.nan

        corr_flag = (not np.isnan(rolling_corr)) and rolling_corr > CORR_THRESHOLD
        turb_flag = turb_t > turb_threshold
        both_flag = corr_flag and turb_flag

        if dt < TURB_ACTIVATION:
            scale = 1.0
        elif both_flag:
            scale = 0.50
        elif corr_flag or turb_flag:
            scale = 0.75
        else:
            scale = 1.0

        row.update({"turbulence": turb_t, "rolling_corr": rolling_corr,
                     "turb_threshold": turb_threshold, "corr_flag": corr_flag,
                     "turb_flag": turb_flag, "both_flag": both_flag, "scale": scale})
        results.append(row)

    return pd.DataFrame(results).set_index("date")


# ── Combined returns ─────────────────────────────────────────────────────────

def combined_returns(faber_m, vrp_m, turb_df, tbill_monthly, fw, vw, use_turb):
    common = faber_m.dropna().index.intersection(vrp_m.dropna().index).sort_values()
    f = faber_m.reindex(common); v = vrp_m.reindex(common)
    out = {}
    for dt in common:
        f_w, v_w = fw, vw
        if use_turb and dt in turb_df.index:
            s = turb_df.loc[dt, "scale"]
            f_w *= s; v_w *= s
        cash_w = 1.0 - f_w - v_w
        cash_r = float(tbill_monthly.loc[dt]) if (tbill_monthly is not None and dt in tbill_monthly.index) else 0.0
        out[dt] = f_w * f.loc[dt] + v_w * v.loc[dt] + cash_w * cash_r
    return pd.Series(out).sort_index()


# ── Metrics ──────────────────────────────────────────────────────────────────

def metrics(s):
    s = s.dropna()
    if len(s) < 12: return None
    ar = s.mean() * 12; av = s.std() * np.sqrt(12)
    sh = ar / av if av > 0 else 0
    neg = s[s < 0]; ds = neg.std() * np.sqrt(12) if len(neg) > 5 else av
    so = ar / ds if ds > 0 else 0
    cum = (1 + s).cumprod()
    dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
    cal = ar / abs(dd) if dd != 0 else 0
    return {"ar": ar, "av": av, "sh": sh, "sortino": so, "dd": dd, "calmar": cal, "final": cum.iloc[-1]}


# ── Report ───────────────────────────────────────────────────────────────────

def report(all_s, turb_df, strat_names, ivv_m, b60_m):
    perfs = {n: metrics(all_s[n]) for n in strat_names}
    p_ivv = metrics(ivv_m.loc[BT_START:]); p_60 = metrics(b60_m.loc[BT_START:])
    s40_final = perfs.get("Faber-S40-only", {}).get("final", 0)

    # ── Performance ──────────────────────────────────────────────────────
    print(f"\n{'='*140}")
    print(f"  FULL PERFORMANCE TABLE")
    print(f"{'='*140}")
    print(f"  {'Strategy':<24} {'Return':>8} {'Vol':>7} {'Sharpe':>8} {'Sortino':>8} "
          f"{'MaxDD':>8} {'Calmar':>8} {'Terminal($1)':>13} {'vs S40':>9} {'vs IVV':>9} {'vs 60/40':>9}")
    print(f"  {'-'*24} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*13} {'-'*9} {'-'*9} {'-'*9}")

    for n in strat_names:
        p = perfs.get(n)
        if not p: continue
        vs_s40 = p["final"] - s40_final
        vs_ivv = p["final"] - (p_ivv["final"] if p_ivv else 0)
        vs_60 = p["final"] - (p_60["final"] if p_60 else 0)
        print(f"  {n:<24} {p['ar']:>7.1%} {p['av']:>6.1%} {p['sh']:>8.3f} {p['sortino']:>8.3f} "
              f"{p['dd']:>7.1%} {p['calmar']:>8.2f} ${p['final']:>12.2f} {vs_s40:>+8.2f} {vs_ivv:>+8.2f} {vs_60:>+8.2f}")
    for p, label in [(p_ivv, "IVV B&H"), (p_60, "60/40")]:
        if p:
            print(f"  {label:<24} {p['ar']:>7.1%} {p['av']:>6.1%} {p['sh']:>8.3f} {p['sortino']:>8.3f} "
                  f"{p['dd']:>7.1%} {p['calmar']:>8.2f} ${p['final']:>12.2f}")

    # ── Crisis ───────────────────────────────────────────────────────────
    print(f"\n{'='*140}")
    print(f"  CRISIS ANALYSIS")
    print(f"{'='*140}")
    for cname, cs, ce in [("GFC (2008-09)", "2008-09", "2009-03"),
                           ("COVID (Feb-Apr 2020)", "2020-02", "2020-04"),
                           ("2022 Bear", "2022-01", "2022-10")]:
        print(f"\n  {cname}:")
        print(f"  {'Strategy':<24} {'Return':>10} {'MaxDD':>10}")
        print(f"  {'-'*24} {'-'*10} {'-'*10}")
        all_items = [(n, all_s[n]) for n in strat_names] + [("IVV B&H", ivv_m), ("60/40", b60_m)]
        for name, sr in all_items:
            c = sr[(sr.index >= pd.Timestamp(cs)) & (sr.index <= pd.Timestamp(ce))]
            if len(c) > 0:
                cum = (1 + c).cumprod()
                mdd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
                print(f"  {name:<24} {(1+c).prod()-1:>+9.1%} {mdd:>9.1%}")

    # ── Turbulence diagnostics ───────────────────────────────────────────
    print(f"\n{'='*140}")
    print(f"  TURBULENCE LAYER DIAGNOSTICS (threshold = {CORR_THRESHOLD})")
    print(f"{'='*140}")

    active = turb_df[turb_df.index >= TURB_ACTIVATION].dropna(subset=["turbulence"])
    total = len(active)
    sd = active["scale"].value_counts().sort_index()

    print(f"\n  Activation period: {TURB_ACTIVATION.date()} - {active.index.max().date()} ({total} months)")
    print(f"\n  Scaling distribution:")
    for sv in [1.0, 0.75, 0.50]:
        n = int(sd.get(sv, 0)); pct = n / total * 100 if total > 0 else 0
        label = {1.0: "normal", 0.75: "one condition", 0.50: "both conditions"}[sv]
        print(f"    Scale {sv:.2f} ({label:>16}): {n:>4} months ({pct:.0f}%)")

    corr_only = ((active["corr_flag"]) & (~active["turb_flag"])).sum()
    turb_only = ((active["turb_flag"]) & (~active["corr_flag"])).sum()
    both = active["both_flag"].sum()
    print(f"\n  Condition breakdown:")
    print(f"    Corr > {CORR_THRESHOLD} only:         {corr_only} months")
    print(f"    Turb > 75th pct only:       {turb_only} months")
    print(f"    Both conditions:            {both} months")

    # Key triggers
    both_months = active[active["both_flag"]].sort_values("turbulence", ascending=False)
    if len(both_months) > 0:
        print(f"\n  All months with BOTH conditions (corr > {CORR_THRESHOLD} AND turb > 75th pct):")
        print(f"  {'Date':>12} {'Turbulence':>12} {'Roll Corr':>10} {'Faber Ret':>10} {'VRP Ret':>10}")
        for _, row in both_months.iterrows():
            print(f"  {row.name.strftime('%Y-%m'):>12} {row['turbulence']:>12.1f} {row['rolling_corr']:>10.3f} "
                  f"{row['faber_ret']*100:>+9.1f}% {row['vrp_ret']*100:>+9.1f}%")

    # Crisis trigger detail
    for cname, cs, ce in [("GFC (2008-09)", "2008-09", "2009-03"),
                           ("COVID (Feb-Apr 2020)", "2020-02", "2020-04"),
                           ("2022 Bear (Jan-Oct)", "2022-01", "2022-10")]:
        crisis = active[(active.index >= pd.Timestamp(cs)) & (active.index <= pd.Timestamp(ce))]
        if len(crisis) == 0: continue
        both_c = crisis["both_flag"].sum()
        any_c = ((crisis["corr_flag"]) | (crisis["turb_flag"])).sum()
        avg_sc = crisis["scale"].mean()
        print(f"\n  {cname}:")
        print(f"    Both-condition triggers: {both_c}  |  Any-condition: {any_c}  |  Avg scale: {avg_sc:.2f}")
        print(f"    Avg Faber ret: {crisis['faber_ret'].mean()*100:+.1f}%/mo  |  Avg VRP ret: {crisis['vrp_ret'].mean()*100:+.1f}%/mo")

    # ── Value-add vs prior run comparison ────────────────────────────────
    print(f"\n{'='*140}")
    print(f"  TURBULENCE VALUE-ADD & COMPARISON WITH PRIOR RUN")
    print(f"{'='*140}")

    # Prior run values (from 2026-04-05_two_pod_combined.md, threshold 0.65)
    prior = {
        "Two-Pod-10": {"sh": 1.061, "dd": -0.114, "final": 9.91},
        "Two-Pod-10-Turb": {"sh": 1.303, "dd": -0.082, "final": 7.67},
        "Two-Pod-20": {"sh": 1.078, "dd": -0.104, "final": 9.38},
        "Two-Pod-20-Turb": {"sh": 1.333, "dd": -0.079, "final": 7.37},
        "pct_delevered": 73,
    }

    pct_delev_now = (1 - sd.get(1.0, 0) / total) * 100 if total > 0 else 0

    print(f"\n  {'':>30} {'Prior (thr=0.65)':>18} {'This run (thr=0.80)':>20}")
    print(f"  {'-'*30} {'-'*18} {'-'*20}")

    for sleeve in [10, 20]:
        base_n = f"Two-Pod-S40-{sleeve}"
        turb_n = f"Two-Pod-S40-{sleeve}-Turb"
        pb = perfs.get(base_n); pt = perfs.get(turb_n)
        if pb and pt:
            prior_base_key = f"Two-Pod-{sleeve}"
            prior_turb_key = f"Two-Pod-{sleeve}-Turb"
            pp_b = prior.get(prior_base_key, {})
            pp_t = prior.get(prior_turb_key, {})

            print(f"\n  {base_n} Sharpe:           {pp_b.get('sh','?'):>18.3f} {pb['sh']:>20.3f}")
            print(f"  {turb_n} Sharpe:      {pp_t.get('sh','?'):>18.3f} {pt['sh']:>20.3f}")
            print(f"  {base_n} Terminal:         ${pp_b.get('final','?'):>17.2f} ${pb['final']:>19.2f}")
            print(f"  {turb_n} Terminal:    ${pp_t.get('final','?'):>17.2f} ${pt['final']:>19.2f}")

    print(f"\n  Pct months de-levered:        {prior['pct_delevered']:>17}% {pct_delev_now:>19.0f}%")

    # ── Sleeve tradeoff ──────────────────────────────────────────────────
    print(f"\n{'='*140}")
    print(f"  SLEEVE SIZE TRADEOFF (Faber-Sweep-40 base)")
    print(f"{'='*140}")
    s40 = perfs.get("Faber-S40-only", {})
    print(f"\n  {'Sleeve':>8} {'Return':>8} {'Sharpe':>8} {'MaxDD':>8} {'Terminal':>10} {'Sharpe vs S40':>14} {'Terminal vs S40':>16}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*14} {'-'*16}")
    for n in ["Faber-S40-only", "Two-Pod-S40-10", "Two-Pod-S40-20", "Two-Pod-S40-30",
              "Two-Pod-S40-10-Turb", "Two-Pod-S40-20-Turb"]:
        p = perfs.get(n)
        if not p: continue
        label = n.replace("Faber-S40-only", "0%").replace("Two-Pod-S40-", "").replace("-Turb", "+T")
        sh_d = p["sh"] - s40.get("sh", 0)
        t_d = p["final"] - s40.get("final", 0)
        print(f"  {label:>8} {p['ar']:>7.1%} {p['sh']:>8.3f} {p['dd']:>7.1%} ${p['final']:>9.2f} {sh_d:>+13.3f} ${t_d:>+15.2f}")

    # ── Key questions ────────────────────────────────────────────────────
    print(f"\n{'='*140}")
    print(f"  KEY QUESTIONS")
    print(f"{'='*140}")

    print(f"\n  Q1. How often does turbulence (threshold 0.80) fire?")
    print(f"      De-levered {pct_delev_now:.0f}% of months (target was ~20-30%, prior was 73%)")

    print(f"\n  Q2. Does it fire during crises and avoid normal markets?")
    for cname, cs, ce in [("COVID", "2020-02", "2020-04"), ("2022 Bear", "2022-01", "2022-10")]:
        cr = active[(active.index >= pd.Timestamp(cs)) & (active.index <= pd.Timestamp(ce))]
        if len(cr) > 0:
            print(f"      {cname}: {cr['both_flag'].sum()} both-triggers, avg scale {cr['scale'].mean():.2f}")

    p_s40 = perfs.get("Faber-S40-only")
    for sleeve in [10, 20]:
        base = perfs.get(f"Two-Pod-S40-{sleeve}")
        turb = perfs.get(f"Two-Pod-S40-{sleeve}-Turb")
        if p_s40 and base:
            print(f"\n  Q3-Q4. Two-Pod-S40-{sleeve} vs S40-only:")
            print(f"      No turb: Sharpe {p_s40['sh']:.3f}→{base['sh']:.3f} ({base['sh']-p_s40['sh']:+.3f}), "
                  f"Terminal ${p_s40['final']:.2f}→${base['final']:.2f} (${base['final']-p_s40['final']:+.2f})")
        if base and turb:
            print(f"      + Turb:  Sharpe {base['sh']:.3f}→{turb['sh']:.3f} ({turb['sh']-base['sh']:+.3f}), "
                  f"Terminal ${base['final']:.2f}→${turb['final']:.2f} (${turb['final']-base['final']:+.2f})")

    print(f"\n  Q5. At what sleeve does the tradeoff become unacceptable?")
    for sleeve in [10, 20, 30]:
        p = perfs.get(f"Two-Pod-S40-{sleeve}")
        if p:
            t_cost = s40.get("final", 0) - p["final"]
            sh_gain = p["sh"] - s40.get("sh", 0)
            print(f"      {sleeve}%: gain {sh_gain:+.3f} Sharpe, cost ${t_cost:.2f} terminal")

    print(f"\n  Q6. Final production architecture verdict:")
    best_sh = max((n, p["sh"]) for n, p in perfs.items() if p)
    best_term = max((n, p["final"]) for n, p in perfs.items() if p)
    print(f"      Best Sharpe: {best_sh[0]} ({best_sh[1]:.3f})")
    print(f"      Best terminal: {best_term[0]} (${best_term[1]:.2f})")

    print()
    return perfs


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 140)
    print("  TWO-POD S40 RERUN: Faber-Sweep-40 + VRP, TURBULENCE THRESHOLD 0.80")
    print("=" * 140)
    print(f"  Key change: correlation threshold {CORR_THRESHOLD} (was 0.65 in prior run)")

    print(f"\n  Loading data...")
    (daily_ret, trend_df, rfr_daily, put_monthly_ret,
     vix_monthly, rvol_monthly, tbill_monthly, ivv_m, b60_m) = load_all_data()

    print(f"  Computing Faber-Sweep-40 monthly...")
    faber_s40_m = compute_faber_monthly(daily_ret, trend_df, rfr_daily, FABER_SUB)
    print(f"  Computing Faber-1x monthly...")
    faber_1x_m = compute_faber_monthly(daily_ret, trend_df, rfr_daily, 0.0)

    print(f"  Applying VRP filter...")
    vrp_m = apply_vrp_filter(put_monthly_ret, vix_monthly, rvol_monthly, tbill_monthly).loc[BT_START:]

    print(f"  Computing turbulence (threshold {CORR_THRESHOLD})...")
    turb_df = compute_turbulence(faber_s40_m, vrp_m)
    active = turb_df[turb_df.index >= TURB_ACTIVATION].dropna(subset=["turbulence"])
    print(f"  Both-condition triggers: {active['both_flag'].sum()}, "
          f"any-condition: {((active['corr_flag'])|(active['turb_flag'])).sum()}")

    strat_names = [
        "Faber-S40-only", "Faber-1x-only", "VRP-only",
        "Two-Pod-S40-10", "Two-Pod-S40-20", "Two-Pod-S40-30",
        "Two-Pod-S40-10-Turb", "Two-Pod-S40-20-Turb",
    ]

    all_s = {
        "Faber-S40-only": faber_s40_m,
        "Faber-1x-only": faber_1x_m,
        "VRP-only": vrp_m,
    }
    for sleeve in [10, 20, 30]:
        fw = 1.0 - sleeve / 100; vw = sleeve / 100
        all_s[f"Two-Pod-S40-{sleeve}"] = combined_returns(faber_s40_m, vrp_m, turb_df, tbill_monthly, fw, vw, False)
    for sleeve in [10, 20]:
        fw = 1.0 - sleeve / 100; vw = sleeve / 100
        all_s[f"Two-Pod-S40-{sleeve}-Turb"] = combined_returns(faber_s40_m, vrp_m, turb_df, tbill_monthly, fw, vw, True)

    perfs = report(all_s, turb_df, strat_names, ivv_m, b60_m)

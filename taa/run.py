"""Main entry point: clean Faber+Harvey TAA backtest."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv; load_dotenv()

import numpy as np, pandas as pd

from taa.faber import compute_trend_scores, apply_faber_filter
from taa.harvey import compute_zscore_variables, find_similar_months, compute_expected_returns
from taa.allocation import BASELINE, ASSETS, EQUITY, direct_capital, normalize
from taa.data import load_monthly_macro, load_monthly_asset_returns, load_daily_etf_returns, load_monthly_prices, load_realized_vols

OUTPUT = Path("taa/output"); OUTPUT.mkdir(parents=True, exist_ok=True)
TC = 0.0010

print("=" * 80)
print("  CLEAN FABER + HARVEY TAA SYSTEM")
print("=" * 80)

# Load data
print("\nLoading data...")
raw_macro = load_monthly_macro()
asset_ret = load_monthly_asset_returns()
asset_ret_fwd = asset_ret.shift(-1)
daily_ret = load_daily_etf_returns()
daily_ret = daily_ret[[c for c in daily_ret.columns if c in ASSETS]]
monthly_prices = load_monthly_prices()
rvol_monthly = load_realized_vols()

# Compute signals
print("Computing Faber trend scores...")
trend_scores_df = compute_trend_scores(monthly_prices)

print("Computing Harvey z-scores...")
z_data = compute_zscore_variables(raw_macro)
z_cols = [c for c in z_data.columns if c.endswith("_z")]
z_clean = z_data[z_cols].dropna()

print(f"  Trend scores: {trend_scores_df.shape}")
print(f"  Z-scored data: {z_clean.shape} ({z_clean.index.min().date()} to {z_clean.index.max().date()})")

# Signal alignment assertion
print("\n--- Signal Alignment Assertion ---")
dt_dec07 = pd.Timestamp("2007-12-01")
if dt_dec07 in trend_scores_df.index and dt_dec07 in monthly_prices.index:
    # Trend score at Dec 2007 index = November 2007 signal (due to shift)
    ts = trend_scores_df.loc[dt_dec07]
    ivv_score = ts.get("IVV", "?")
    # The Nov 2007 monthly close
    nov_close = monthly_prices.loc[pd.Timestamp("2007-11-01"), "IVV"] if pd.Timestamp("2007-11-01") in monthly_prices.index else "?"
    print(f"  IVV trend score at index 2007-12: {ivv_score} (computed from Nov 2007 data)")

    # Harvey z-score: find what date would be used for Dec 2007 allocation
    z_before_dec = z_clean.index[z_clean.index < dt_dec07]
    z_date = z_before_dec[-1] if len(z_before_dec) > 0 else "none"
    print(f"  Harvey z-score date for Dec 2007: {z_date} (must be before 2007-12-01)")

    print(f"  Signal date: Dec 2007, Z-score through: {z_date}, Returns start: Jan 2008")
    ok = (z_date != "none" and z_date < dt_dec07)
    print(f"  {'PASS' if ok else 'FAIL'}")
else:
    print("  Data not available for Dec 2007")

# Backtest
common_start = max(daily_ret.dropna(how="all").index.min(), pd.Timestamp("2002-01-01"))
trading_days = daily_ret.loc[common_start:].index
print(f"\nBacktest: {len(trading_days)} days ({common_start.date()} to {trading_days.max().date()})")

strats = {"FH": {}, "F_only": {}, "B_base": {}, "bench_6040": {}, "bench_ivv": {}}
current_w = {c: None for c in strats}
total_tc = {c: 0.0 for c in strats}

w_fh = dict(BASELINE); w_f = dict(BASELINE)
current_trends = {a: 3 for a in ASSETS if a != "cash"}
harvey_er = {a: 0.0 for a in ASSETS}

conviction_log = []
pool_log = []
trace_log = {}
total_months = 0

for day in trading_days:
    if day not in daily_ret.index: continue
    dr = daily_ret.loc[day]
    avail = [a for a in ASSETS if a in dr.index and pd.notna(dr[a])]
    if len(avail) < 3: continue
    actual = {a: float(dr[a]) for a in avail}

    is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day)-1].month)

    if is_ms:
        total_months += 1

        # Faber
        ts_cands = trend_scores_df.index[trend_scores_df.index <= day]
        if len(ts_cands) > 0:
            ts = ts_cands[-1]
            for a in ASSETS:
                if a == "cash": continue
                if a in trend_scores_df.columns:
                    v = trend_scores_df.loc[ts, a]
                    current_trends[a] = int(v) if pd.notna(v) else 0

        # Harvey
        z_cands = z_clean.index[z_clean.index < day]
        if len(z_cands) > 0:
            z_dt = z_cands[-1]
            try:
                similar, min_dist = find_similar_months(z_clean, z_dt)
                harvey_er = compute_expected_returns(similar, asset_ret_fwd, avail)
            except ValueError:
                harvey_er = {a: 0.0 for a in avail}

        # Realized vols
        rvols = {}
        rv_cands = rvol_monthly.index[rvol_monthly.index <= day]
        if len(rv_cands) > 0:
            rv = rv_cands[-1]
            for a in ASSETS:
                if a == "cash": continue
                if a in rvol_monthly.columns:
                    v = rvol_monthly.loc[rv, a]
                    rvols[a] = float(v) if pd.notna(v) and v > 0 else 0.15

        # FH: Faber + Harvey
        w_step1, pool = apply_faber_filter(current_trends, BASELINE)
        w_step2, detail = direct_capital(dict(w_step1), pool, harvey_er, rvols)
        w_fh = normalize(w_step2)

        # F only: Faber, pool to cash
        w_f1, pool_f = apply_faber_filter(current_trends, BASELINE)
        w_f1["cash"] = w_f1.get("cash", 0) + pool_f
        w_f = normalize(w_f1)

        pool_log.append(pool)

        # Max conviction
        faber_conv = current_trends.get("IVV", 0) >= 3 and current_trends.get("QQQ", 0) >= 3
        harvey_conv = harvey_er.get("IVV", 0) > 0 and harvey_er.get("QQQ", 0) > 0
        is_max_conv = faber_conv and harvey_conv

        if is_max_conv:
            eq = sum(w_fh.get(a, 0) for a in EQUITY)
            # Get forward IVV return
            fwd = asset_ret_fwd.loc[day, "IVV"] if day in asset_ret_fwd.index and "IVV" in asset_ret_fwd.columns else np.nan
            conviction_log.append({
                "date": day, "ivv_w": w_fh.get("IVV", 0), "qqq_w": w_fh.get("QQQ", 0),
                "total_eq": eq, "ivv_fwd_1m": fwd,
            })

        # Trace
        key = day.strftime("%Y-%m")
        if key in ("2008-01", "2013-06", "2020-03", "2022-06"):
            trace_log[key] = {
                "trends": dict(current_trends), "pool": pool,
                "harvey_er": dict(harvey_er), "detail": detail,
                "final_fh": dict(w_fh), "final_f": dict(w_f),
            }

    w_base = dict(BASELINE)
    w_6040 = {a: 0 for a in avail}
    if "IVV" in avail: w_6040["IVV"] = 0.6
    if "VGLT" in avail: w_6040["VGLT"] = 0.4
    w_ivv = {a: (1 if a == "IVV" else 0) for a in avail}

    all_w = {"FH": (w_fh, is_ms), "F_only": (w_f, is_ms), "B_base": (w_base, is_ms),
             "bench_6040": (w_6040, is_ms), "bench_ivv": (w_ivv, False)}

    for cn, (new_w, trade) in all_w.items():
        if current_w[cn] is not None and not trade:
            w_used = current_w[cn]
        else:
            w_used = new_w
            to = sum(abs(new_w.get(a, 0) - (current_w[cn] or {}).get(a, 0)) for a in avail) / 2
            total_tc[cn] += to * TC
            current_w[cn] = new_w
        strats[cn][day] = sum(w_used.get(a, 0) * actual.get(a, 0) for a in avail)

results = {c: pd.Series(d).sort_index() for c, d in strats.items() if d}
n_years = len(trading_days) / 252
ivv = results.get("bench_ivv")
labels = {"FH": "Faber+Harvey", "F_only": "Faber Only", "B_base": "Baseline",
          "bench_6040": "60/40", "bench_ivv": "IVV B&H"}

# ── Report ────────────────────────────────────────────────────────────────────

print(f"\n{'='*80}")
print(f"  PERFORMANCE")
print(f"{'='*80}")
print(f"\n  {'Config':>14} {'AnnRet':>7} {'AnnVol':>7} {'Sharpe':>7} {'MaxDD':>8} {'CorrIVV':>8} {'AnnTC':>6}")
print(f"  {'-'*58}")
for cn in ["FH", "F_only", "B_base", "bench_6040", "bench_ivv"]:
    s = results.get(cn)
    if s is None or len(s) < 252: continue
    ar = s.mean()*252; av = s.std()*np.sqrt(252); sh = ar/av if av > 0 else 0
    cum = (1+s).cumprod(); dd = ((cum-cum.expanding().max())/cum.expanding().max()).min()
    corr = s.corr(ivv) if ivv is not None else 0
    tc = total_tc.get(cn, 0) / n_years
    print(f"  {labels[cn]:>14} {ar:>6.1%} {av:>6.1%} {sh:>7.2f} {dd:>7.1%} {corr:>7.2f} {tc:>5.2%}")

# Crisis
print(f"\n{'='*80}")
print(f"  CRISIS DRAWDOWNS")
print(f"{'='*80}")
for cn2, cs, ce in [("GFC","2008-09-01","2009-03-31"),("COVID","2020-02-19","2020-03-23"),("2022","2022-01-03","2022-10-31")]:
    print(f"\n  {cn2}:")
    for cn in ["FH","F_only","B_base","bench_ivv"]:
        s = results.get(cn)
        if s is None: continue
        c = s[(s.index>=pd.Timestamp(cs))&(s.index<=pd.Timestamp(ce))]
        if len(c) > 0: print(f"    {labels[cn]:>14}: {((1+c).prod()-1):>+7.1%}")

# Bull capture
print(f"\n{'='*80}")
print(f"  BULL MARKET CAPTURE")
print(f"{'='*80}")
for yr, desc in [(2013,"Harvey worst"),(2019,"Strong bull"),(2023,"Tech momentum")]:
    print(f"\n  {yr} ({desc}):")
    for cn in ["FH","F_only","B_base","bench_ivv"]:
        s = results.get(cn)
        if s is None: continue
        y = s[s.index.year == yr]
        if len(y) > 20: print(f"    {labels[cn]:>14}: {(1+y).prod()-1:>+7.1%}")

# Max conviction
print(f"\n{'='*80}")
print(f"  MAX CONVICTION ANALYSIS")
print(f"{'='*80}")
conv_df = pd.DataFrame(conviction_log)
n_conv = len(conv_df)
print(f"  Max conviction months: {n_conv}/{total_months} ({n_conv/total_months:.0%})")

if n_conv > 0:
    avg_eq_conv = conv_df["total_eq"].mean()
    # Avg equity in non-conviction months (approximate)
    all_eq = []
    for _, r in pd.DataFrame(conviction_log + [{"total_eq": 0}]).iterrows():
        all_eq.append(r.get("total_eq", 0))
    print(f"  Avg equity during conviction: {avg_eq_conv:.0%}")

    fwd_conv = conv_df["ivv_fwd_1m"].dropna()
    fwd_all = asset_ret["IVV"].dropna()
    print(f"  Avg IVV 1m fwd return (conviction months): {fwd_conv.mean():+.2%} (n={len(fwd_conv)})")
    print(f"  Avg IVV 1m fwd return (all months):        {fwd_all.mean():+.2%} (n={len(fwd_all)})")
    print(f"  Conviction justified: {'YES' if fwd_conv.mean() > fwd_all.mean() else 'NO'}")

    print(f"\n  {'Date':>10} {'IVV':>5} {'QQQ':>5} {'TotEq':>6} {'IVV_fwd':>8}")
    for _, r in conv_df.head(20).iterrows():
        fwd_s = f"{r['ivv_fwd_1m']:+.1%}" if pd.notna(r["ivv_fwd_1m"]) else "N/A"
        print(f"  {r['date'].strftime('%Y-%m'):>10} {r['ivv_w']:>4.0%} {r['qqq_w']:>4.0%} {r['total_eq']:>5.0%} {fwd_s:>8}")
    if n_conv > 20:
        print(f"  ... ({n_conv - 20} more)")

# Harvey capital direction
print(f"\n{'='*80}")
print(f"  HARVEY CAPITAL DIRECTION")
print(f"{'='*80}")
print(f"  Average pool from Faber: {np.mean(pool_log):.0%}")
print(f"  Pool > 0 in {sum(1 for p in pool_log if p > 0.01)}/{len(pool_log)} months ({sum(1 for p in pool_log if p > 0.01)/len(pool_log):.0%})")

# Decision traces
print(f"\n{'='*80}")
print(f"  DECISION TRACES")
print(f"{'='*80}")
for key in ["2008-01", "2013-06", "2020-03", "2022-06"]:
    t = trace_log.get(key)
    if t is None: print(f"\n  {key}: not in range"); continue
    print(f"\n  {key}:")
    print(f"    Trends: {t['trends']}")
    print(f"    Pool: {t['pool']:.0%}")
    er_s = ", ".join(f"{a}={v:+.3f}" for a, v in t["harvey_er"].items() if a != "cash" and abs(v) > 0.0001)
    print(f"    Harvey ER: {er_s}")
    if t["detail"].get("scores"):
        sc_s = ", ".join(f"{a}={d['score']:.3f}" for a, d in t["detail"]["scores"].items())
        print(f"    Inv-vol scores: {sc_s}")
    if t["detail"].get("directed_to"):
        dir_s = ", ".join(f"{a}={v:.0%}" for a, v in t["detail"]["directed_to"].items())
        print(f"    Pool directed to: {dir_s}")
    fh_s = "  ".join(f"{a}={v:.0%}" for a, v in sorted(t["final_fh"].items()) if v > 0.005)
    f_s = "  ".join(f"{a}={v:.0%}" for a, v in sorted(t["final_f"].items()) if v > 0.005)
    print(f"    FH weights: {fh_s}")
    print(f"    F  weights: {f_s}")

# Calendar years
print(f"\n{'='*80}")
print(f"  CALENDAR YEAR RETURNS")
print(f"{'='*80}")
print(f"  {'Year':>6} {'FH':>8} {'Faber':>8} {'Base':>8} {'IVV':>8}")
for yr in range(2002, 2025):
    row = f"  {yr:>6}"
    for cn in ["FH", "F_only", "B_base", "bench_ivv"]:
        s = results.get(cn)
        if s is None: row += f" {'--':>8}"; continue
        y = s[s.index.year == yr]
        row += f" {(1+y).prod()-1:>+7.1%}" if len(y) > 20 else f" {'--':>8}"
    print(row)

# Final values
print(f"\n{'='*80}")
print(f"  FINAL VALUES ($1)")
print(f"{'='*80}")
for cn in ["FH", "F_only", "B_base", "bench_6040", "bench_ivv"]:
    s = results.get(cn)
    if s is None or len(s) < 252: continue
    print(f"  {labels[cn]:>14}: ${(1+s).cumprod().iloc[-1]:>8.2f}")

print()

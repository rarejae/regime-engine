"""Backtest: Faber+Harvey with 1.25x leverage during max conviction."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv; load_dotenv()

import numpy as np, pandas as pd

from taa.faber import compute_trend_scores, apply_faber_filter
from taa.harvey import compute_zscore_variables, find_similar_months, compute_expected_returns
from taa.allocation import BASELINE, ASSETS, EQUITY, direct_capital, normalize
from taa.data import load_monthly_macro, load_monthly_asset_returns, load_daily_etf_returns, load_monthly_prices, load_realized_vols
from taa.leverage import build_synthetic_etfs, validate_synthetic

OUTPUT = Path("taa/output"); OUTPUT.mkdir(parents=True, exist_ok=True)
TC = 0.0010

print("=" * 80)
print("  FABER + HARVEY WITH 1.25x LEVERAGE DURING MAX CONVICTION")
print("=" * 80)

# Load data
raw_macro = load_monthly_macro()
asset_ret = load_monthly_asset_returns()
asset_ret_fwd = asset_ret.shift(-1)
daily_ret = load_daily_etf_returns()
daily_ret = daily_ret[[c for c in daily_ret.columns if c in ASSETS]]
monthly_prices = load_monthly_prices()
rvol_monthly = load_realized_vols()

# Build synthetic leveraged ETFs
print("\nBuilding synthetic SSO (2x IVV) and QLD (2x QQQ)...")
daily_with_lev = build_synthetic_etfs(daily_ret)
print(f"  Daily data: {daily_with_lev.shape}")

# Validate
print("\n--- Synthetic ETF Validation ---")
for synth_col, actual_ticker, name in [("SSO", "SSO", "SSO (2x SPY)"), ("QLD", "QLD", "QLD (2x QQQ)")]:
    if synth_col not in daily_with_lev.columns:
        print(f"  {name}: synthetic not available"); continue
    v = validate_synthetic(daily_with_lev[synth_col].dropna(), actual_ticker, name)
    print(f"\n  {v['name']} ({v['n_days']} overlapping days):")
    print(f"    Synth ann ret: {v['synth_ann_ret']:+.2%}")
    print(f"    Actual ann ret: {v['actual_ann_ret']:+.2%}")
    print(f"    Return diff: {v['ret_diff']:.2%} ({'< 1%' if v['ret_diff'] < 0.01 else '> 1% — CHECK'})")
    print(f"    Correlation: {v['correlation']:.4f} ({'>0.99' if v['correlation'] > 0.99 else 'CHECK'})")
    print(f"    Synth max DD: {v['synth_max_dd']:.1%}")
    print(f"    Actual max DD: {v['actual_max_dd']:.1%}")
    print(f"    DD diff: {v['dd_diff']:.1%}")
    print(f"    Status: {v['status']}")
    if v.get("annual"):
        print(f"    Annual comparison:")
        for yr, d in sorted(v["annual"].items()):
            print(f"      {yr}: synth={d['synthetic']:+.1%}, actual={d['actual']:+.1%}, diff={abs(d['synthetic']-d['actual']):.1%}")

# Compute signals
trend_scores_df = compute_trend_scores(monthly_prices)
z_data = compute_zscore_variables(raw_macro)
z_cols = [c for c in z_data.columns if c.endswith("_z")]
z_clean = z_data[z_cols].dropna()

# Backtest
common_start = max(daily_with_lev.dropna(how="all").index.min(), pd.Timestamp("2002-01-01"))
trading_days = daily_with_lev.loc[common_start:].index

print(f"\nBacktest: {len(trading_days)} days ({common_start.date()} to {trading_days.max().date()})")

strats = {"L_lev": {}, "FH": {}, "B_base": {}, "bench_6040": {}, "bench_ivv": {}}
current_w = {c: None for c in strats}
total_tc = {c: 0.0 for c in strats}

current_trends = {a: 3 for a in ASSETS if a != "cash"}
harvey_er = {a: 0.0 for a in ASSETS}
w_fh = dict(BASELINE); w_lev = dict(BASELINE)
is_leveraged = False
lev_months = 0; unlev_months = 0
lev_month_rets = []; unlev_month_rets = []
total_months = 0

ALL_ASSETS = ASSETS + ["SSO", "QLD"]

for day in trading_days:
    if day not in daily_with_lev.index: continue
    dr = daily_with_lev.loc[day]
    avail_base = [a for a in ASSETS if a in dr.index and pd.notna(dr[a])]
    if len(avail_base) < 3: continue
    actual = {a: float(dr[a]) for a in ALL_ASSETS if a in dr.index and pd.notna(dr.get(a, np.nan))}

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
                similar, _ = find_similar_months(z_clean, z_dt)
                harvey_er = compute_expected_returns(similar, asset_ret_fwd, avail_base)
            except ValueError:
                harvey_er = {a: 0.0 for a in avail_base}

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

        # FH allocation (base)
        w_step1, pool = apply_faber_filter(current_trends, BASELINE)
        w_step2, _ = direct_capital(dict(w_step1), pool, harvey_er, rvols)
        w_fh = normalize(w_step2)

        # Max conviction check
        faber_conv = current_trends.get("IVV", 0) >= 3 and current_trends.get("QQQ", 0) >= 3
        harvey_conv = harvey_er.get("IVV", 0) > 0 and harvey_er.get("QQQ", 0) > 0
        new_leveraged = faber_conv and harvey_conv

        # Build leveraged weights
        w_lev = dict(w_fh)
        # Zero out SSO/QLD first
        w_lev["SSO"] = 0; w_lev["QLD"] = 0

        if new_leveraged:
            # Replace 25% of IVV with SSO, 25% of QQQ with QLD
            ivv_w = w_lev.get("IVV", 0)
            qqq_w = w_lev.get("QQQ", 0)
            sso_w = ivv_w * 0.25
            qld_w = qqq_w * 0.25
            w_lev["IVV"] = ivv_w - sso_w
            w_lev["QQQ"] = qqq_w - qld_w
            w_lev["SSO"] = sso_w
            w_lev["QLD"] = qld_w
            lev_months += 1
        else:
            unlev_months += 1

        is_leveraged = new_leveraged

    # Compute daily returns
    w_base = dict(BASELINE); w_base["SSO"] = 0; w_base["QLD"] = 0
    w_6040 = {a: 0 for a in ALL_ASSETS}
    if "IVV" in avail_base: w_6040["IVV"] = 0.6
    if "VGLT" in avail_base: w_6040["VGLT"] = 0.4
    w_ivv = {a: 0 for a in ALL_ASSETS}; w_ivv["IVV"] = 1.0
    w_fh_full = dict(w_fh); w_fh_full.setdefault("SSO", 0); w_fh_full.setdefault("QLD", 0)

    all_w = {"L_lev": (w_lev, is_ms), "FH": (w_fh_full, is_ms), "B_base": (w_base, is_ms),
             "bench_6040": (w_6040, is_ms), "bench_ivv": (w_ivv, False)}

    for cn, (new_w, trade) in all_w.items():
        if current_w[cn] is not None and not trade:
            w_used = current_w[cn]
        else:
            w_used = new_w
            to = sum(abs(new_w.get(a, 0) - (current_w[cn] or {}).get(a, 0)) for a in ALL_ASSETS) / 2
            total_tc[cn] += to * TC
            current_w[cn] = new_w
        ret = sum(w_used.get(a, 0) * actual.get(a, 0) for a in ALL_ASSETS if a in actual)
        strats[cn][day] = ret

results = {c: pd.Series(d).sort_index() for c, d in strats.items() if d}
n_years = len(trading_days) / 252
ivv = results.get("bench_ivv")
labels = {"L_lev": "1.25x Lev", "FH": "Faber+Harvey", "B_base": "Baseline",
          "bench_6040": "60/40", "bench_ivv": "IVV B&H"}

# Report
print(f"\n{'='*80}")
print(f"  PERFORMANCE")
print(f"{'='*80}")
print(f"\n  {'Config':>14} {'AnnRet':>7} {'AnnVol':>7} {'Sharpe':>7} {'MaxDD':>8} {'CorrIVV':>8} {'AnnTC':>6}")
print(f"  {'-'*58}")
for cn in ["L_lev", "FH", "B_base", "bench_6040", "bench_ivv"]:
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
    for cn in ["L_lev", "FH", "B_base", "bench_ivv"]:
        s = results.get(cn)
        if s is None: continue
        c = s[(s.index>=pd.Timestamp(cs))&(s.index<=pd.Timestamp(ce))]
        if len(c) > 0:
            cum = (1+c).cumprod()
            total = (1+c).prod()-1
            intra_dd = ((cum-cum.expanding().max())/cum.expanding().max()).min()
            print(f"    {labels[cn]:>14}: {total:>+7.1%} (intra-period DD: {intra_dd:.1%})")

# Bull capture
print(f"\n{'='*80}")
print(f"  BULL CAPTURE")
print(f"{'='*80}")
for yr, desc in [(2013,"Harvey worst"),(2019,"Strong bull"),(2023,"Tech momentum")]:
    print(f"\n  {yr} ({desc}):")
    for cn in ["L_lev", "FH", "B_base", "bench_ivv"]:
        s = results.get(cn)
        if s is None: continue
        y = s[s.index.year == yr]
        if len(y) > 20: print(f"    {labels[cn]:>14}: {(1+y).prod()-1:>+7.1%}")

# Leverage analysis
print(f"\n{'='*80}")
print(f"  LEVERAGE ANALYSIS")
print(f"{'='*80}")
print(f"  Leveraged months:   {lev_months}/{total_months} ({lev_months/total_months:.0%})")
print(f"  Unleveraged months: {unlev_months}/{total_months} ({unlev_months/total_months:.0%})")

# Monthly returns during leveraged vs unleveraged
s_l = results["L_lev"]; s_fh = results["FH"]
monthly_l = s_l.resample("MS").apply(lambda x: (1+x).prod()-1)
monthly_fh = s_fh.resample("MS").apply(lambda x: (1+x).prod()-1)

# Determine which months were leveraged
lev_month_flags = {}
for day in trading_days:
    if day.day <= 5:
        ts_cands = trend_scores_df.index[trend_scores_df.index <= day]
        if len(ts_cands) > 0:
            t = current_trends  # approximate
            f_c = t.get("IVV",0) >= 3 and t.get("QQQ",0) >= 3
            h_c = harvey_er.get("IVV",0) > 0 and harvey_er.get("QQQ",0) > 0
            lev_month_flags[day.to_period("M").to_timestamp()] = f_c and h_c

print(f"\n  Worst month while leveraged: {monthly_l.min():+.1%}")
print(f"  Best month while leveraged:  {monthly_l.max():+.1%}")

# Drawdown analysis
print(f"\n{'='*80}")
print(f"  DRAWDOWN ANALYSIS")
print(f"{'='*80}")
cum_l = (1 + s_l).cumprod()
peak_l = cum_l.expanding().max()
dd_l = (cum_l - peak_l) / peak_l
max_dd_date = dd_l.idxmin()
max_dd_val = dd_l.min()
# Find peak date
peak_date = cum_l.loc[:max_dd_date].idxmax()
print(f"  Max drawdown: {max_dd_val:.1%}")
print(f"  Peak: {peak_date.date()}, Trough: {max_dd_date.date()}")

# Calendar years
print(f"\n{'='*80}")
print(f"  CALENDAR YEAR RETURNS")
print(f"{'='*80}")
print(f"  {'Year':>6} {'1.25x':>8} {'FH':>8} {'Base':>8} {'IVV':>8}")
for yr in range(2002, 2025):
    row = f"  {yr:>6}"
    for cn in ["L_lev", "FH", "B_base", "bench_ivv"]:
        s = results.get(cn)
        if s is None: row += f" {'--':>8}"; continue
        y = s[s.index.year == yr]
        row += f" {(1+y).prod()-1:>+7.1%}" if len(y) > 20 else f" {'--':>8}"
    print(row)

# Final values
print(f"\n{'='*80}")
print(f"  FINAL VALUES ($1)")
print(f"{'='*80}")
for cn in ["L_lev", "FH", "B_base", "bench_6040", "bench_ivv"]:
    s = results.get(cn)
    if s is None or len(s) < 252: continue
    print(f"  {labels[cn]:>14}: ${(1+s).cumprod().iloc[-1]:>8.2f}")

print()

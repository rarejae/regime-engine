"""Hierarchical v2: Faber → Harvey → per-asset HMMs → Kritzman."""

import logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv; load_dotenv()

import numpy as np, pandas as pd

from regime.config import RegimeConfig
from regime.data import fetch_monthly_history
from regime.transform import transform_variables, get_valid_zscored
from regime.similarity import compute_distances
from regime.faber_filter import compute_trend_signals, apply_faber_to_baseline
from regime.kritzman import (fetch_daily_basket, compute_turbulence, compute_turbulence_pctl,
                              compute_absorption_ratio, compute_ar_zscore)
from regime.hierarchical import BASELINE, ASSETS, EQUITY, step1_faber, step2_harvey, step4_kritzman, step5_normalize
from regime.per_asset_hmm import fetch_daily_prices, fit_all_asset_hmms, get_asset_zones, get_hmm_diagnostics, ASSETS_TO_FIT
from regime.run_daily_backtest import fetch_daily_etf_returns
# For single-HMM comparison
from regime.hmm_trend import fetch_daily_data, compute_features, fit_and_predict_rolling, apply_persistence_filter

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
OUTPUT = Path("experiments/signal_development/output"); OUTPUT.mkdir(parents=True, exist_ok=True)
TC = 0.0010; UNIVERSE = list(BASELINE.keys())
config = RegimeConfig()

# Load data
raw_macro = fetch_monthly_history(config)
transformed = transform_variables(raw_macro, config)
z_data = get_valid_zscored(transformed, config)
z_data_lagged = z_data.shift(1).dropna()
asset_ret = pd.read_parquet("data/macro/roth_asset_returns.parquet")
for c in list(asset_ret.columns):
    if c not in UNIVERSE: asset_ret = asset_ret.drop(columns=[c])
asset_ret_fwd = asset_ret.shift(-1)
daily_ret = fetch_daily_etf_returns()
daily_ret = daily_ret[[c for c in daily_ret.columns if c in UNIVERSE]]

# Faber
import yfinance as yf
etf_prices_dict = {}
for our, ticker in [("IVV","SPY"),("QQQ","QQQ"),("VGLT","TLT"),("IAU","GLD"),("DBC","DBC")]:
    d = yf.download(ticker, start="1998-01-01", progress=False)
    if d is not None and not d.empty:
        p = d["Close"]
        if hasattr(p,"columns"): p=p.iloc[:,0]
        p.index = pd.to_datetime(p.index).tz_localize(None)
        etf_prices_dict[our] = p
prices_df = pd.DataFrame(etf_prices_dict).sort_index()
trend_df = compute_trend_signals(prices_df).shift(1)

# Per-asset HMMs
print("Fitting 5 per-asset HMMs...")
hmm_preds = fit_all_asset_hmms(prices_df)

# Single SPY HMM (for comparison)
print("Fitting single SPY HMM...")
spy_raw = fetch_daily_data()
hmm_feat = compute_features(spy_raw)
single_hmm = fit_and_predict_rolling(hmm_feat).set_index("date")
zone_raw = pd.Series("neutral", index=single_hmm.index)
zone_raw[single_hmm["bull_prob"]>0.7]="bull"
zone_raw[single_hmm["bull_prob"]<0.3]="bear"
single_hmm["zone"] = apply_persistence_filter(zone_raw)

# Kritzman
basket = fetch_daily_basket()
turb_smooth = compute_turbulence(basket)
turb_pctl = compute_turbulence_pctl(turb_smooth, window=252)
ar = compute_absorption_ratio(basket, n_components=2)
ar_z_series = compute_ar_zscore(ar)
turb_m = turb_pctl.resample("MS").last().shift(1)
ar_z_m = ar_z_series.resample("MS").last().shift(1)

# Backtest
common_start = max(daily_ret.dropna(how="all").index.min(), pd.Timestamp("2002-01-01"))
# Also need HMM predictions
for a, pred in hmm_preds.items():
    common_start = max(common_start, pred.index.min())
trading_days = daily_ret.loc[common_start:].index

print(f"\n{'='*80}")
print(f"  HIERARCHICAL v2: PER-ASSET HMMs")
print(f"{'='*80}")
print(f"Backtest: {len(trading_days)} days ({common_start.date()} to {trading_days.max().date()})")

# HMM diagnostics
diag = get_hmm_diagnostics(hmm_preds)
print(f"\nPer-asset HMM diagnostics:")
print(f"  {'Asset':>6} {'Bull%':>6} {'Bear%':>6} {'Trans/yr':>9} {'AvgRun':>7} {'Noisy':>6}")
for a in ASSETS_TO_FIT:
    d = diag.get(a)
    if d:
        print(f"  {a:>6} {d['bull_pct']:>5.0%} {d['bear_pct']:>5.0%} {d['transitions_per_year']:>9.1f} {d['avg_run_length']:>7.0f} {'YES' if d['noisy'] else 'no':>6}")

strats = {"SH_perhmm": {}, "S1_single": {}, "F_faber": {}, "B_base": {},
          "bench_6040": {}, "bench_ivv": {}}
current_w = {c: None for c in strats}
total_tc = {c: 0.0 for c in strats}

current_trends = {a: True for a in ASSETS if a != "cash"}
harvey_er = {a: 0.0 for a in ASSETS}
trace_log = {}

for day in trading_days:
    if day not in daily_ret.index: continue
    dr = daily_ret.loc[day]
    avail = [a for a in UNIVERSE if a in dr.index and pd.notna(dr[a])]
    if len(avail) < 3: continue
    actual = {a: float(dr[a]) for a in avail}

    is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day)-1].month)

    if is_ms:
        # Faber
        tm_cands = trend_df.index[trend_df.index <= day]
        if len(tm_cands) > 0:
            tm = tm_cands[-1]
            for a in ASSETS:
                if a == "cash": continue
                if a in trend_df.columns:
                    s = trend_df.loc[tm, a]
                    current_trends[a] = bool(s) if pd.notna(s) else True

        # Harvey
        z_cands = z_data_lagged.index[z_data_lagged.index < day]
        if len(z_cands) > 0:
            z_dt = z_cands[-1]
            try:
                sim = compute_distances(z_data_lagged, z_dt, config)
                for a in avail:
                    if a not in asset_ret_fwd.columns: harvey_er[a]=0; continue
                    rets = [asset_ret_fwd.loc[d,a] for d in sim.similar_dates
                            if d in asset_ret_fwd.index and pd.notna(asset_ret_fwd.loc[d,a])]
                    harvey_er[a] = np.mean(rets) if rets else 0
            except ValueError: pass

        # Kritzman
        tp = turb_m.get(day, 0.5) if day in turb_m.index else 0.5
        if pd.isna(tp): tp = 0.5
        arz = ar_z_m.get(day, 0) if day in ar_z_m.index else 0
        if pd.isna(arz): arz = 0

        # === Config SH: Per-asset HMMs ===
        w1, pool, inelig = step1_faber(current_trends)
        w1, _ = step2_harvey(w1, pool, harvey_er)

        # Step 3: per-asset HMM
        zones = get_asset_zones(hmm_preds, day)
        for a in ASSETS_TO_FIT:
            if w1.get(a, 0) <= 0: continue
            z = zones.get(a, "neutral")
            if z == "trending":
                boost = w1[a] * 0.15
                if w1.get("cash", 0) - boost >= 0.03:
                    w1[a] += boost
                    w1["cash"] -= boost
            elif z == "not_trending":
                trim = w1[a] * 0.15
                w1[a] -= trim
                w1["cash"] = w1.get("cash", 0) + trim

        w1, _ = step4_kritzman(w1, tp, arz)
        w_sh = step5_normalize(w1)

        # === Config S1: Single SPY HMM (from previous version) ===
        w2, pool2, _ = step1_faber(current_trends)
        w2, _ = step2_harvey(w2, pool2, harvey_er)
        # Single HMM
        prev_month = day - pd.DateOffset(months=1)
        h_month = single_hmm[(single_hmm.index >= prev_month) & (single_hmm.index < day)]
        bp = h_month["bull_prob"].mean() if len(h_month) > 0 else 0.5
        if pd.isna(bp): bp = 0.5
        from regime.hierarchical import step3_hmm
        w2, _ = step3_hmm(w2, bp)
        w2, _ = step4_kritzman(w2, tp, arz)
        w_s1 = step5_normalize(w2)

        # Trace for key months
        key = day.strftime("%Y-%m")
        if key in ("2013-06", "2020-03", "2022-06"):
            trace_log[key] = {"zones": dict(zones), "inelig": inelig, "pool": pool,
                              "single_bp": bp, "final_sh": dict(w_sh), "final_s1": dict(w_s1)}

    # Faber alone
    w_fab = apply_faber_to_baseline(BASELINE, current_trends)
    w_base = dict(BASELINE)
    w_6040 = {a:0 for a in avail}
    if "IVV" in avail: w_6040["IVV"]=0.6
    if "VGLT" in avail: w_6040["VGLT"]=0.4
    w_ivv = {a:(1 if a=="IVV" else 0) for a in avail}

    all_w = {"SH_perhmm":(w_sh,is_ms),"S1_single":(w_s1,is_ms),"F_faber":(w_fab,is_ms),
             "B_base":(w_base,is_ms),"bench_6040":(w_6040,is_ms),"bench_ivv":(w_ivv,False)}

    for cn,(new_w,trade) in all_w.items():
        if current_w[cn] is not None and not trade: w_used=current_w[cn]
        else:
            w_used=new_w
            to=sum(abs(new_w.get(a,0)-(current_w[cn] or {}).get(a,0)) for a in avail)/2
            total_tc[cn]+=to*TC; current_w[cn]=new_w
        strats[cn][day]=sum(w_used.get(a,0)*actual.get(a,0) for a in avail)

results = {c:pd.Series(d).sort_index() for c,d in strats.items() if d}
n_years = len(trading_days)/252
ivv = results.get("bench_ivv")
labels = {"SH_perhmm":"PerAsset HMM","S1_single":"Single HMM","F_faber":"Faber",
          "B_base":"Baseline","bench_6040":"60/40","bench_ivv":"IVV B&H"}

# Report
print(f"\n{'='*80}")
print(f"  PERFORMANCE")
print(f"{'='*80}")
print(f"\n  {'Config':>14} {'AnnRet':>7} {'AnnVol':>7} {'Sharpe':>7} {'MaxDD':>8} {'CorrIVV':>8} {'AnnTC':>6}")
print(f"  {'-'*58}")
for cn in ["SH_perhmm","S1_single","F_faber","B_base","bench_6040","bench_ivv"]:
    s=results.get(cn)
    if s is None or len(s)<252: continue
    ar=s.mean()*252;av=s.std()*np.sqrt(252);sh=ar/av if av>0 else 0
    cum=(1+s).cumprod();dd=((cum-cum.expanding().max())/cum.expanding().max()).min()
    corr=s.corr(ivv) if ivv is not None else 0
    tc=total_tc.get(cn,0)/n_years
    print(f"  {labels[cn]:>14} {ar:>6.1%} {av:>6.1%} {sh:>7.2f} {dd:>7.1%} {corr:>7.2f} {tc:>5.2%}")

# Crisis
print(f"\n{'='*80}")
print(f"  CRISIS DRAWDOWNS")
print(f"{'='*80}")
for cn2,cs,ce in [("GFC","2008-09-01","2009-03-31"),("COVID","2020-02-19","2020-03-23"),("2022","2022-01-03","2022-10-31")]:
    print(f"\n  {cn2}:")
    for cn in ["SH_perhmm","S1_single","F_faber","B_base","bench_ivv"]:
        s=results.get(cn)
        if s is None: continue
        c=s[(s.index>=pd.Timestamp(cs))&(s.index<=pd.Timestamp(ce))]
        if len(c)>0: print(f"    {labels[cn]:>14}: {((1+c).prod()-1):>+7.1%}")

# Bull capture
print(f"\n{'='*80}")
print(f"  BULL CAPTURE")
print(f"{'='*80}")
for yr,desc in [(2013,"Harvey worst"),(2019,"Strong bull"),(2023,"Tech momentum")]:
    print(f"\n  {yr} ({desc}):")
    for cn in ["SH_perhmm","S1_single","F_faber","B_base","bench_ivv"]:
        s=results.get(cn)
        if s is None: continue
        y=s[s.index.year==yr]
        if len(y)>20: print(f"    {labels[cn]:>14}: {(1+y).prod()-1:>+7.1%}")

# Decision traces
print(f"\n{'='*80}")
print(f"  DECISION TRACES (per-asset vs single HMM)")
print(f"{'='*80}")
for key in ["2013-06","2020-03","2022-06"]:
    t = trace_log.get(key)
    if t is None: print(f"\n  {key}: not in range"); continue
    print(f"\n  {key}:")
    print(f"    Faber ineligible: {t['inelig']}, pool={t['pool']:.0%}")
    print(f"    Per-asset HMM zones: {t['zones']}")
    print(f"    Single HMM bull_prob: {t['single_bp']:.2f}")
    sh_eq = sum(t['final_sh'].get(a,0) for a in EQUITY)
    s1_eq = sum(t['final_s1'].get(a,0) for a in EQUITY)
    print(f"    Final equity — PerAsset: {sh_eq:.0%}, Single: {s1_eq:.0%}")
    sh_w = "  ".join(f"{a}={v:.0%}" for a,v in sorted(t['final_sh'].items()) if v>0.005)
    s1_w = "  ".join(f"{a}={v:.0%}" for a,v in sorted(t['final_s1'].items()) if v>0.005)
    print(f"    PerAsset weights: {sh_w}")
    print(f"    Single   weights: {s1_w}")

# Asset zone divergence
print(f"\n{'='*80}")
print(f"  ASSET ZONE DIVERGENCE (did per-asset HMMs differ across assets?)")
print(f"{'='*80}")
divergent_months = 0
total_checked = 0
for day in trading_days:
    if day.day <= 5:  # check roughly monthly
        zones = get_asset_zones(hmm_preds, day)
        unique_zones = set(zones.values())
        if len(unique_zones) > 1:
            divergent_months += 1
        total_checked += 1
if total_checked > 0:
    print(f"  Months with different zones across assets: {divergent_months}/{total_checked} ({divergent_months/total_checked:.0%})")

# Calendar years
print(f"\n{'='*80}")
print(f"  CALENDAR YEAR RETURNS")
print(f"{'='*80}")
print(f"  {'Year':>6} {'PerHMM':>8} {'Single':>8} {'Faber':>8} {'Base':>8} {'IVV':>8}")
for yr in range(2007,2025):
    row=f"  {yr:>6}"
    for cn in ["SH_perhmm","S1_single","F_faber","B_base","bench_ivv"]:
        s=results.get(cn)
        if s is None: row+=f" {'--':>8}"; continue
        y=s[s.index.year==yr]
        row+=f" {(1+y).prod()-1:>+7.1%}" if len(y)>20 else f" {'--':>8}"
    print(row)

# Final values
print(f"\n{'='*80}")
print(f"  FINAL VALUES ($1)")
print(f"{'='*80}")
for cn in ["SH_perhmm","S1_single","F_faber","B_base","bench_6040","bench_ivv"]:
    s=results.get(cn)
    if s is None or len(s)<252: continue
    print(f"  {labels[cn]:>14}: ${(1+s).cumprod().iloc[-1]:>8.2f}")

print()

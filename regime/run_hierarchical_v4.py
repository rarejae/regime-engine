"""Hierarchical v4: multi-timeframe Faber + Harvey+carry+value + HMM + Kritzman."""

import logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv; load_dotenv()

import numpy as np, pandas as pd

from regime.config import RegimeConfig
from regime.data import fetch_monthly_history
from regime.transform import transform_variables, get_valid_zscored
from regime.similarity import compute_distances
from regime.faber_filter import compute_trend_signals
from regime.hmm_trend import fetch_daily_data, compute_features, fit_and_predict_rolling, apply_persistence_filter
from regime.kritzman import (fetch_daily_basket, compute_turbulence, compute_turbulence_pctl,
                              compute_absorption_ratio, compute_ar_zscore)
from regime.hierarchical_v3 import (BASELINE, ASSETS, EQUITY, compute_multi_sma_signals,
                                     step1_multi_faber, step3_hmm, step4_kritzman, step5_normalize)
from regime.carry_value import compute_carry, compute_value
from regime.run_daily_backtest import fetch_daily_etf_returns

logging.basicConfig(level=logging.WARNING)
OUTPUT = Path("experiments/signal_development/output"); OUTPUT.mkdir(parents=True, exist_ok=True)
TC = 0.0010; UNIVERSE = list(BASELINE.keys())
config = RegimeConfig()

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

import yfinance as yf
ep = {}
for our, ticker in [("IVV","SPY"),("QQQ","QQQ"),("VGLT","TLT"),("IAU","GLD"),("DBC","DBC")]:
    d = yf.download(ticker, start="1998-01-01", progress=False)
    if d is not None and not d.empty:
        p = d["Close"]
        if hasattr(p,"columns"): p=p.iloc[:,0]
        p.index = pd.to_datetime(p.index).tz_localize(None)
        ep[our] = p
prices_df = pd.DataFrame(ep).sort_index()
multi_sma_df = compute_multi_sma_signals(prices_df).shift(1)
daily_rets_raw = prices_df.pct_change()
rvol_63 = daily_rets_raw.rolling(63, min_periods=30).std() * np.sqrt(252)
rvol_monthly = rvol_63.resample("MS").last().shift(1)

print("Computing carry and value signals...")
carry_df = compute_carry(prices_df)
value_df = compute_value(prices_df)
print(f"  Carry: {carry_df.shape}, Value: {value_df.shape}")

print("Fitting HMM...")
spy_raw = fetch_daily_data()
hmm_feat = compute_features(spy_raw)
hmm_pred = fit_and_predict_rolling(hmm_feat).set_index("date")
zone_raw = pd.Series("neutral", index=hmm_pred.index)
zone_raw[hmm_pred["bull_prob"]>0.7]="bull"
zone_raw[hmm_pred["bull_prob"]<0.3]="bear"
hmm_pred["zone"] = apply_persistence_filter(zone_raw)

basket = fetch_daily_basket()
turb_smooth = compute_turbulence(basket)
turb_pctl = compute_turbulence_pctl(turb_smooth, window=252)
ar = compute_absorption_ratio(basket, n_components=2)
ar_z_s = compute_ar_zscore(ar)
turb_m = turb_pctl.resample("MS").last().shift(1)
ar_z_m = ar_z_s.resample("MS").last().shift(1)

common_start = max(daily_ret.dropna(how="all").index.min(), hmm_pred.index.min(), pd.Timestamp("2002-01-01"))
trading_days = daily_ret.loc[common_start:].index

print(f"\n{'='*80}")
print(f"  HIERARCHICAL v4: CARRY + VALUE")
print(f"{'='*80}")
print(f"Backtest: {len(trading_days)} days ({common_start.date()} to {trading_days.max().date()})")


def step2_harvey_cv(weights, pool, harvey_er, rvols, carry, value, mode="full"):
    """Step 2 with carry and value. mode: 'full', 'carry_only', 'value_only', 'none'."""
    detail = {"pool": pool, "scores": {}, "directed_to": {}}
    if pool <= 0.001:
        return weights, detail

    candidates = {}
    for a in ASSETS:
        if a == "cash" or weights.get(a, 0) <= 0:
            continue
        er = harvey_er.get(a, 0)
        vol = rvols.get(a, 0.15)
        c = carry.get(a, 0) if mode in ("full", "carry_only") else 0
        v = value.get(a, 0) if mode in ("full", "value_only") else 0

        adjusted_er = er + c
        if adjusted_er <= 0 and mode == "none":
            adjusted_er = er  # original mode
            if adjusted_er <= 0:
                continue

        if adjusted_er <= 0:
            continue

        score = adjusted_er / max(vol, 0.01) + 0.01 * v
        if score > 0:
            candidates[a] = score
            detail["scores"][a] = {"er": er, "carry": c, "value_z": v, "vol": vol, "score": score}

    if not candidates:
        weights["cash"] = weights.get("cash", 0) + pool
        detail["directed_to"]["cash"] = pool
        return weights, detail

    # Single-eligible cap
    if len(candidates) == 1:
        a = list(candidates.keys())[0]
        alloc = min(pool, 0.40 - weights.get(a, 0))
        alloc = max(alloc, 0)
        weights[a] = weights.get(a, 0) + alloc
        weights["cash"] = weights.get("cash", 0) + (pool - alloc)
        detail["directed_to"][a] = alloc
        if pool - alloc > 0.001:
            detail["directed_to"]["cash"] = pool - alloc
        return weights, detail

    total_score = sum(candidates.values())
    for a, score in candidates.items():
        share = pool * (score / total_score)
        weights[a] = weights.get(a, 0) + share
        detail["directed_to"][a] = share

    return weights, detail


strats = {"V4": {}, "V3": {}, "V4c": {}, "V4v": {}, "B_base": {}, "bench_6040": {}, "bench_ivv": {}}
current_w = {c: None for c in strats}
total_tc = {c: 0.0 for c in strats}

current_strengths = {a: 3 for a in ASSETS if a != "cash"}
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
        ms_cands = multi_sma_df.index[multi_sma_df.index <= day]
        if len(ms_cands) > 0:
            ms = ms_cands[-1]
            for a in ASSETS:
                if a == "cash": continue
                if a in multi_sma_df.columns:
                    v = multi_sma_df.loc[ms, a]
                    current_strengths[a] = int(v) if pd.notna(v) else 0

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

        rv_cands = rvol_monthly.index[rvol_monthly.index <= day]
        rvols = {}
        if len(rv_cands) > 0:
            rv = rv_cands[-1]
            for a in ASSETS:
                if a == "cash": continue
                if a in rvol_monthly.columns:
                    v = rvol_monthly.loc[rv, a]
                    rvols[a] = float(v) if pd.notna(v) and v > 0 else 0.15

        # Get carry and value for this month
        carry = {}; value = {}
        for a in ASSETS:
            if a in carry_df.columns:
                c_cands = carry_df.index[carry_df.index <= day]
                if len(c_cands) > 0:
                    cv = carry_df.loc[c_cands[-1], a]
                    carry[a] = float(cv) if pd.notna(cv) else 0
            if a in value_df.columns:
                v_cands = value_df.index[value_df.index <= day]
                if len(v_cands) > 0:
                    vv = value_df.loc[v_cands[-1], a]
                    value[a] = float(vv) if pd.notna(vv) else 0

        prev_m = day - pd.DateOffset(months=1)
        hm = hmm_pred[(hmm_pred.index >= prev_m) & (hmm_pred.index < day)]
        bp = hm["bull_prob"].mean() if len(hm) > 0 else 0.5
        if pd.isna(bp): bp = 0.5
        tp = turb_m.get(day, 0.5) if day in turb_m.index else 0.5
        if pd.isna(tp): tp = 0.5
        arz = ar_z_m.get(day, 0) if day in ar_z_m.index else 0
        if pd.isna(arz): arz = 0

        # Max conviction check
        cond_f = current_strengths.get("IVV",0)>=3 and current_strengths.get("QQQ",0)>=3
        cond_h = harvey_er.get("IVV",0)>0 and harvey_er.get("QQQ",0)>0
        cond_m = bp > 0.7
        cond_k = not (tp > 0.95 and arz > 2.0)
        max_conv = cond_f and cond_h and cond_m and cond_k

        for cfg_name, mode in [("V4","full"),("V3","none"),("V4c","carry_only"),("V4v","value_only")]:
            w, pool, _ = step1_multi_faber(current_strengths)
            w, det = step2_harvey_cv(w, pool, harvey_er, rvols, carry, value, mode=mode)
            w, _ = step3_hmm(w, bp)
            w, _ = step4_kritzman(w, tp, arz)
            w = step5_normalize(w)
            if cfg_name == "V4":
                exec(f"w_{cfg_name} = w")
            globals()[f"w_{cfg_name}"] = w

        # Trace key months
        key = day.strftime("%Y-%m")
        if key in ("2007-01","2009-01","2022-01","2022-06","2024-01"):
            trace_log[key] = {"carry": dict(carry), "value": dict(value),
                              "final_v4": dict(w_V4), "final_v3": dict(w_V3)}

    w_base = dict(BASELINE)
    w_6040 = {a:0 for a in avail}
    if "IVV" in avail: w_6040["IVV"]=0.6
    if "VGLT" in avail: w_6040["VGLT"]=0.4
    w_ivv = {a:(1 if a=="IVV" else 0) for a in avail}

    all_w = {"V4":(w_V4,is_ms),"V3":(w_V3,is_ms),"V4c":(w_V4c,is_ms),"V4v":(w_V4v,is_ms),
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
labels = {"V4":"V4 Full","V3":"V3 (no CV)","V4c":"Carry Only","V4v":"Value Only",
          "B_base":"Baseline","bench_6040":"60/40","bench_ivv":"IVV B&H"}

print(f"\n{'='*80}")
print(f"  PERFORMANCE")
print(f"{'='*80}")
print(f"\n  {'Config':>12} {'AnnRet':>7} {'AnnVol':>7} {'Sharpe':>7} {'MaxDD':>8} {'CorrIVV':>8}")
print(f"  {'-'*50}")
for cn in ["V4","V3","V4c","V4v","B_base","bench_6040","bench_ivv"]:
    s=results.get(cn)
    if s is None or len(s)<252: continue
    ar=s.mean()*252;av=s.std()*np.sqrt(252);sh=ar/av if av>0 else 0
    cum=(1+s).cumprod();dd=((cum-cum.expanding().max())/cum.expanding().max()).min()
    corr=s.corr(ivv) if ivv is not None else 0
    print(f"  {labels[cn]:>12} {ar:>6.1%} {av:>6.1%} {sh:>7.2f} {dd:>7.1%} {corr:>7.2f}")

# Crisis
print(f"\n{'='*80}")
print(f"  CRISIS DRAWDOWNS")
print(f"{'='*80}")
for cn2,cs,ce in [("GFC","2008-09-01","2009-03-31"),("COVID","2020-02-19","2020-03-23"),("2022","2022-01-03","2022-10-31")]:
    print(f"\n  {cn2}:")
    for cn in ["V4","V3","V4c","V4v","B_base","bench_ivv"]:
        s=results.get(cn)
        if s is None: continue
        c=s[(s.index>=pd.Timestamp(cs))&(s.index<=pd.Timestamp(ce))]
        if len(c)>0: print(f"    {labels[cn]:>12}: {((1+c).prod()-1):>+7.1%}")

# Bull capture
print(f"\n{'='*80}")
print(f"  BULL CAPTURE")
print(f"{'='*80}")
for yr,desc in [(2013,"Harvey worst"),(2019,"Strong bull"),(2023,"Tech momentum")]:
    print(f"\n  {yr} ({desc}):")
    for cn in ["V4","V3","B_base","bench_ivv"]:
        s=results.get(cn)
        if s is None: continue
        y=s[s.index.year==yr]
        if len(y)>20: print(f"    {labels[cn]:>12}: {(1+y).prod()-1:>+7.1%}")

# Carry diagnostics
print(f"\n{'='*80}")
print(f"  CARRY DIAGNOSTICS (monthly carry per asset)")
print(f"{'='*80}")
for dt_str in ["2007-01","2009-01","2022-01","2024-01"]:
    dt = pd.Timestamp(dt_str+"-01")
    cands = carry_df.index[carry_df.index <= dt]
    if len(cands) == 0: continue
    cd = cands[-1]
    print(f"\n  {dt_str}:")
    for a in ASSETS:
        if a in carry_df.columns:
            v = carry_df.loc[cd, a]
            print(f"    {a:>6}: {v:+.4f} ({v*1200:+.1f} bps ann)" if pd.notna(v) else f"    {a:>6}: N/A")

# Value diagnostics
print(f"\n{'='*80}")
print(f"  VALUE DIAGNOSTICS (z-scores per asset)")
print(f"{'='*80}")
for dt_str in ["2009-01","2021-01","2022-10"]:
    dt = pd.Timestamp(dt_str+"-01")
    cands = value_df.index[value_df.index <= dt]
    if len(cands) == 0: continue
    vd = cands[-1]
    print(f"\n  {dt_str}:")
    for a in ASSETS:
        if a in value_df.columns:
            v = value_df.loc[vd, a]
            label = "cheap" if pd.notna(v) and v > 0.5 else ("expensive" if pd.notna(v) and v < -0.5 else "neutral")
            print(f"    {a:>6}: {v:+.2f} ({label})" if pd.notna(v) else f"    {a:>6}: N/A")

# Calendar years
print(f"\n{'='*80}")
print(f"  CALENDAR YEAR RETURNS")
print(f"{'='*80}")
print(f"  {'Year':>6} {'V4':>8} {'V3':>8} {'Carry':>8} {'Value':>8} {'IVV':>8}")
for yr in range(2002,2025):
    row=f"  {yr:>6}"
    for cn in ["V4","V3","V4c","V4v","bench_ivv"]:
        s=results.get(cn)
        if s is None: row+=f" {'--':>8}"; continue
        y=s[s.index.year==yr]
        row+=f" {(1+y).prod()-1:>+7.1%}" if len(y)>20 else f" {'--':>8}"
    print(row)

# Final values
print(f"\n{'='*80}")
print(f"  FINAL VALUES ($1)")
print(f"{'='*80}")
for cn in ["V4","V3","V4c","V4v","B_base","bench_6040","bench_ivv"]:
    s=results.get(cn)
    if s is None or len(s)<252: continue
    print(f"  {labels[cn]:>12}: ${(1+s).cumprod().iloc[-1]:>8.2f}")

# Contribution
print(f"\n{'='*80}")
print(f"  CONTRIBUTION ANALYSIS")
print(f"{'='*80}")
for cn, label in [("V4","V4 Full"),("V4c","Carry Only"),("V4v","Value Only"),("V3","V3 (base)")]:
    s = results.get(cn)
    if s is None or len(s) < 252: continue
    sh = s.mean()*252 / (s.std()*np.sqrt(252))
    print(f"  {label:>12} Sharpe: {sh:.3f}")
v3_sh = results["V3"].mean()*252 / (results["V3"].std()*np.sqrt(252)) if "V3" in results else 0
for cn, label in [("V4c","Carry"),("V4v","Value"),("V4","Carry+Value")]:
    s = results.get(cn)
    if s is None or len(s) < 252: continue
    sh = s.mean()*252 / (s.std()*np.sqrt(252))
    print(f"  {label:>12} contribution: {sh - v3_sh:+.3f} Sharpe")

print()

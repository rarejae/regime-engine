"""Ensemble voting backtest: Harvey + Faber + HMM + Kritzman → aggregate weights."""

import logging, sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv; load_dotenv()

import numpy as np, pandas as pd

from regime.config import RegimeConfig
from regime.data import fetch_monthly_history
from regime.transform import transform_variables, get_valid_zscored
from regime.similarity import compute_distances
from regime.multi_asset import allocate_similarity_weighted
from regime.faber_filter import compute_trend_signals, apply_faber_to_baseline
from regime.hmm_trend import fetch_daily_data, compute_features, fit_and_predict_rolling, apply_persistence_filter
from regime.kritzman import (fetch_daily_basket, compute_turbulence, compute_turbulence_pctl,
                              compute_absorption_ratio, compute_ar_zscore)
from regime.ensemble import (BASELINE, ASSETS, EQUITY, harvey_votes, faber_votes, hmm_votes,
                              kritzman_votes, aggregate_scores, scores_to_weights)
from regime.run_daily_backtest import fetch_daily_etf_returns

logging.basicConfig(level=logging.WARNING)
OUTPUT = Path("regime/output"); OUTPUT.mkdir(parents=True, exist_ok=True)
TC = 0.0010; UNIVERSE = list(BASELINE.keys())
config = RegimeConfig()

# ── Load all data ─────────────────────────────────────────────────────────────
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

# Faber: SMA signals (shifted for PIT)
import yfinance as yf
etf_prices = {}
for our, ticker in [("IVV","SPY"),("QQQ","QQQ"),("VGLT","TLT"),("IAU","GLD"),("DBC","DBC")]:
    d = yf.download(ticker, start="1998-01-01", progress=False)
    if d is not None and not d.empty:
        p = d["Close"]
        if hasattr(p,"columns"): p=p.iloc[:,0]
        p.index = pd.to_datetime(p.index).tz_localize(None)
        etf_prices[our] = p
prices_df = pd.DataFrame(etf_prices).sort_index()
trend_signals_df = compute_trend_signals(prices_df).shift(1)  # PIT safe

# HMM
spy_raw = fetch_daily_data()
hmm_features = compute_features(spy_raw)
print("Fitting HMM...")
hmm_pred = fit_and_predict_rolling(hmm_features).set_index("date")
zone_raw = pd.Series("neutral", index=hmm_pred.index)
zone_raw[hmm_pred["bull_prob"] > 0.7] = "bull"
zone_raw[hmm_pred["bull_prob"] < 0.3] = "bear"
hmm_pred["zone"] = apply_persistence_filter(zone_raw)

# Kritzman
basket = fetch_daily_basket()
turb_smooth = compute_turbulence(basket)
turb_pctl = compute_turbulence_pctl(turb_smooth, window=252)
ar = compute_absorption_ratio(basket, n_components=2)
ar_z = compute_ar_zscore(ar)
turb_monthly = turb_pctl.resample("MS").last().shift(1)
ar_z_monthly = ar_z.resample("MS").last().shift(1)

# ── Backtest ──────────────────────────────────────────────────────────────────
common_start = max(daily_ret.dropna(how="all").index.min(), hmm_pred.index.min(), pd.Timestamp("2002-01-01"))
trading_days = daily_ret.loc[common_start:].index

print("=" * 80)
print("  ENSEMBLE VOTING BACKTEST")
print("=" * 80)
print(f"Backtest: {len(trading_days)} days ({common_start.date()} to {trading_days.max().date()})")

strats = {"E_ensemble": {}, "F_faber": {}, "H_harvey": {}, "B_baseline": {},
          "bench_6040": {}, "bench_ivv": {}}
# Leave-one-out configs
loo_strats = {"no_harvey": {}, "no_faber": {}, "no_hmm": {}, "no_kritzman": {}}

current_w = {c: None for c in {**strats, **loo_strats}}
total_tc = {c: 0.0 for c in strats}
vote_log = []

harvey_er = {a: 0.0 for a in ASSETS}
uncond_means = {a: 0.0 for a in ASSETS}
signal_stds = {a: 0.01 for a in ASSETS}
signal_hist = {a: [] for a in ASSETS}
current_trends = {a: True for a in ASSETS if a != "cash"}

for day in trading_days:
    if day not in daily_ret.index: continue
    dr = daily_ret.loc[day]
    avail = [a for a in UNIVERSE if a in dr.index and pd.notna(dr[a])]
    if len(avail) < 3: continue
    actual = {a: float(dr[a]) for a in avail}

    is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day)-1].month)

    if is_ms:
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
                    if a in asset_ret.columns:
                        h = asset_ret.loc[:z_dt,a].dropna()
                        uncond_means[a] = h.mean() if len(h)>12 else 0
                    sig = harvey_er[a] - uncond_means.get(a,0)
                    signal_hist.setdefault(a,[]).append(sig)
                    signal_stds[a] = max(np.std(signal_hist[a][-120:]),1e-6) if len(signal_hist[a])>=12 else 0.01
            except ValueError: pass

        # Faber
        tm_cands = trend_signals_df.index[trend_signals_df.index <= day]
        if len(tm_cands) > 0:
            tm = tm_cands[-1]
            for a in ASSETS:
                if a=="cash": continue
                if a in trend_signals_df.columns:
                    s = trend_signals_df.loc[tm,a]
                    current_trends[a] = bool(s) if pd.notna(s) else True

    # HMM
    bp = hmm_pred.loc[day,"bull_prob"] if day in hmm_pred.index else 0.5
    if pd.isna(bp): bp = 0.5

    # Kritzman (monthly)
    tp = turb_monthly.get(day,0.5) if day in turb_monthly.index else 0.5
    if pd.isna(tp): tp = 0.5
    arz = ar_z_monthly.get(day,0) if day in ar_z_monthly.index else 0
    if pd.isna(arz): arz = 0

    # Compute all 4 votes
    v_h = harvey_votes(harvey_er, uncond_means, signal_stds)
    v_f = faber_votes(current_trends)
    v_m = hmm_votes(bp)
    v_k = kritzman_votes(tp, arz)

    # Ensemble
    all_votes = [v_h, v_f, v_m, v_k]
    scores = aggregate_scores(all_votes)
    w_ens = scores_to_weights(scores, 4)

    # Leave-one-out
    for loo_name, exclude_idx in [("no_harvey",0),("no_faber",1),("no_hmm",2),("no_kritzman",3)]:
        subset = [v for i,v in enumerate(all_votes) if i != exclude_idx]
        s = aggregate_scores(subset)
        loo_strats[loo_name][day] = None  # placeholder
        w_loo = scores_to_weights(s, 3)
        if current_w[loo_name] is not None and not is_ms:
            w_used = current_w[loo_name]
        else:
            w_used = w_loo; current_w[loo_name] = w_loo
        loo_strats[loo_name][day] = sum(w_used.get(a,0)*actual.get(a,0) for a in avail)

    # Faber alone
    w_fab = apply_faber_to_baseline(BASELINE, current_trends)
    # Harvey alone
    w_har = allocate_similarity_weighted(harvey_er, max_single=0.50)

    w_base = dict(BASELINE)
    w_6040 = {a:0 for a in avail}
    if "IVV" in avail: w_6040["IVV"]=0.6
    if "VGLT" in avail: w_6040["VGLT"]=0.4
    w_ivv = {a:(1 if a=="IVV" else 0) for a in avail}

    all_w = {"E_ensemble":(w_ens,is_ms),"F_faber":(w_fab,is_ms),"H_harvey":(w_har,is_ms),
             "B_baseline":(w_base,is_ms),"bench_6040":(w_6040,is_ms),"bench_ivv":(w_ivv,False)}

    if is_ms:
        vote_log.append({"date":day,"h_ivv":v_h.get("IVV",0),"f_ivv":v_f.get("IVV",0),
                         "m_ivv":v_m.get("IVV",0),"k_ivv":v_k.get("IVV",0),
                         "score_ivv":scores.get("IVV",0),"w_ivv":w_ens.get("IVV",0),
                         "bull_prob":bp,"turb_pctl":tp,"ar_z":arz})

    for cn,(new_w,trade) in all_w.items():
        if current_w[cn] is not None and not trade: w_used=current_w[cn]
        else:
            w_used=new_w
            to=sum(abs(new_w.get(a,0)-(current_w[cn] or {}).get(a,0)) for a in avail)/2
            total_tc[cn]+=to*TC; current_w[cn]=new_w
        strats[cn][day]=sum(w_used.get(a,0)*actual.get(a,0) for a in avail)

results = {c:pd.Series(d).sort_index() for c,d in strats.items() if d}
loo_results = {c:pd.Series(d).sort_index() for c,d in loo_strats.items() if d}
vdf = pd.DataFrame(vote_log).set_index("date")
vdf.to_parquet(OUTPUT/"ensemble_votes.parquet")

n_years = len(trading_days)/252
ivv = results.get("bench_ivv")
labels = {"E_ensemble":"Ensemble","F_faber":"Faber","H_harvey":"Harvey",
          "B_baseline":"Baseline","bench_6040":"60/40","bench_ivv":"IVV B&H"}

# ── Report ────────────────────────────────────────────────────────────────────

print(f"\n{'='*80}")
print(f"  PERFORMANCE")
print(f"{'='*80}")
print(f"\n  {'Config':>12} {'AnnRet':>7} {'AnnVol':>7} {'Sharpe':>7} {'MaxDD':>8} {'CorrIVV':>8} {'AnnTC':>6}")
print(f"  {'-'*56}")
for cn in ["E_ensemble","F_faber","H_harvey","B_baseline","bench_6040","bench_ivv"]:
    s=results.get(cn)
    if s is None or len(s)<252: continue
    ar=s.mean()*252;av=s.std()*np.sqrt(252);sh=ar/av if av>0 else 0
    cum=(1+s).cumprod();dd=((cum-cum.expanding().max())/cum.expanding().max()).min()
    corr=s.corr(ivv) if ivv is not None else 0
    tc=total_tc.get(cn,0)/n_years
    print(f"  {labels[cn]:>12} {ar:>6.1%} {av:>6.1%} {sh:>7.2f} {dd:>7.1%} {corr:>7.2f} {tc:>5.2%}")

# Crisis
print(f"\n{'='*80}")
print(f"  CRISIS DRAWDOWNS")
print(f"{'='*80}")
for cn2,cs,ce in [("GFC","2008-09-01","2009-03-31"),("COVID","2020-02-19","2020-03-23"),("2022","2022-01-03","2022-10-31")]:
    print(f"\n  {cn2}:")
    for cn in ["E_ensemble","F_faber","H_harvey","B_baseline","bench_ivv"]:
        s=results.get(cn)
        if s is None: continue
        c=s[(s.index>=pd.Timestamp(cs))&(s.index<=pd.Timestamp(ce))]
        if len(c)>0: print(f"    {labels[cn]:>12}: {((1+c).prod()-1):>+7.1%}")

# Bull capture
print(f"\n{'='*80}")
print(f"  BULL MARKET CAPTURE")
print(f"{'='*80}")
for yr,desc in [(2013,"Harvey worst"),(2019,"Strong bull"),(2023,"Tech momentum")]:
    print(f"\n  {yr} ({desc}):")
    for cn in ["E_ensemble","F_faber","H_harvey","B_baseline","bench_ivv"]:
        s=results.get(cn)
        if s is None: continue
        y=s[s.index.year==yr]
        if len(y)>20: print(f"    {labels[cn]:>12}: {(1+y).prod()-1:>+7.1%}")

# Key months vote breakdown
print(f"\n{'='*80}")
print(f"  VOTE BREAKDOWN — KEY MONTHS (IVV)")
print(f"{'='*80}")
print(f"  {'Date':>10} {'Harvey':>7} {'Faber':>6} {'HMM':>5} {'Kritz':>6} {'Score':>6} {'IVV_w':>6}")
for target in ["2008-01","2009-01","2013-06","2020-03","2022-06"]:
    ts = pd.Timestamp(target+"-01")
    sub = vdf[vdf.index >= ts]
    if len(sub) == 0: continue
    r = sub.iloc[0]
    print(f"  {sub.index[0].date()!s:>10} {r['h_ivv']:>+7.0f} {r['f_ivv']:>+6.0f} {r['m_ivv']:>+5.0f} {r['k_ivv']:>+6.0f} {r['score_ivv']:>+6.0f} {r['w_ivv']:>5.0%}")

# Calendar years
print(f"\n{'='*80}")
print(f"  CALENDAR YEAR RETURNS")
print(f"{'='*80}")
print(f"  {'Year':>6} {'Ensemb':>8} {'Faber':>8} {'Harvey':>8} {'Base':>8} {'IVV':>8}")
for yr in range(2002,2025):
    row=f"  {yr:>6}"
    for cn in ["E_ensemble","F_faber","H_harvey","B_baseline","bench_ivv"]:
        s=results.get(cn)
        if s is None: row+=f" {'--':>8}"; continue
        y=s[s.index.year==yr]
        row+=f" {(1+y).prod()-1:>+7.1%}" if len(y)>20 else f" {'--':>8}"
    print(row)

# Final values
print(f"\n{'='*80}")
print(f"  FINAL VALUES ($1)")
print(f"{'='*80}")
for cn in ["E_ensemble","F_faber","H_harvey","B_baseline","bench_6040","bench_ivv"]:
    s=results.get(cn)
    if s is None or len(s)<252: continue
    print(f"  {labels[cn]:>12}: ${(1+s).cumprod().iloc[-1]:>8.2f}")

# Vote agreement
print(f"\n{'='*80}")
print(f"  VOTE AGREEMENT ANALYSIS (IVV)")
print(f"{'='*80}")
agree_4 = ((vdf["h_ivv"]==vdf["f_ivv"])&(vdf["f_ivv"]==vdf["m_ivv"])&(vdf["m_ivv"]==vdf["k_ivv"])).mean()
same_sign = ((vdf[["h_ivv","f_ivv","m_ivv","k_ivv"]]>0).sum(axis=1)>=3) | ((vdf[["h_ivv","f_ivv","m_ivv","k_ivv"]]<0).sum(axis=1)>=3)
print(f"  All 4 agree exactly: {agree_4:.0%}")
print(f"  3+ same direction:   {same_sign.mean():.0%}")
avg_bull = vdf[vdf.index.year.isin([2013,2017,2019,2021])]["score_ivv"].mean()
avg_bear = vdf[vdf.index.year.isin([2002,2008,2022])]["score_ivv"].mean()
print(f"  Avg IVV score in bull years: {avg_bull:+.1f}")
print(f"  Avg IVV score in bear years: {avg_bear:+.1f}")

# Leave-one-out
print(f"\n{'='*80}")
print(f"  LEAVE-ONE-OUT ANALYSIS")
print(f"{'='*80}")
ens_s = results["E_ensemble"]
ens_sh = ens_s.mean()*252/(ens_s.std()*np.sqrt(252))
print(f"  Full ensemble Sharpe: {ens_sh:.2f}")
for loo_name, voter in [("no_harvey","Harvey"),("no_faber","Faber"),("no_hmm","HMM"),("no_kritzman","Kritzman")]:
    s=loo_results.get(loo_name)
    if s is None or len(s)<252: continue
    sh=s.mean()*252/(s.std()*np.sqrt(252))
    delta = ens_sh - sh
    print(f"  Without {voter:>10}: Sharpe={sh:.2f} (ensemble contribution: {delta:+.2f})")

print()

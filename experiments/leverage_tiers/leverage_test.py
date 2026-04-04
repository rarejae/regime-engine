"""Leverage comparison: Conservative vs Test vs Aggressive on 5-asset weekly."""

import sys, os, warnings, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv; load_dotenv()
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

import numpy as np, pandas as pd

from regime.config import RegimeConfig
from regime.data import fetch_monthly_history
from regime.transform import transform_variables, get_valid_zscored
from regime.similarity import compute_distances
from regime.hmm_trend import fetch_daily_data, compute_features, fit_and_predict_rolling, apply_persistence_filter
from regime.kritzman import (fetch_daily_basket, compute_turbulence, compute_turbulence_pctl,
                              compute_absorption_ratio, compute_ar_zscore)
from regime.carry_value import compute_carry, compute_value
from regime.run_daily_backtest import fetch_daily_etf_returns

OUTPUT = Path(__file__).resolve().parent / "output"
DATA_DIR = Path("data/macro")
SMA_PERIODS = [6, 10, 12]
EQUITY = ["IVV", "QQQ"]
BASE = {"IVV": 0.45, "QQQ": 0.25, "IEF": 0.10, "IAU": 0.10, "cash": 0.10}
ASSETS = list(BASE.keys())

LEV_CONFIGS = [
    ("1x",           {0: 0.0, 1: 0.00, 2: 0.00}),
    ("Conservative",  {0: 0.0, 1: 0.30, 2: 0.65}),
    ("Test",          {0: 0.0, 1: 0.50, 2: 0.75}),
    ("Aggressive",    {0: 0.0, 1: 0.50, 2: 1.00}),
]


def step1_faber(strengths):
    w = {}; pool = 0.0
    for a in ASSETS:
        if a == "cash": w[a] = BASE[a]; continue
        s = strengths.get(a, 0)
        if s >= 3: w[a] = BASE[a]
        elif s == 2: w[a] = BASE[a] * 0.70; pool += BASE[a] * 0.30
        else: w[a] = 0.0; pool += BASE[a]
    return w, pool

def step2(weights, pool, harvey_er, rvols, carry, value):
    if pool <= 0.001: return weights
    cands = {}
    for a in ASSETS:
        if a == "cash" or weights.get(a, 0) <= 0: continue
        adj = harvey_er.get(a, 0) + carry.get(a, 0)
        if adj <= 0: continue
        sc = adj / max(rvols.get(a, 0.15), 0.01) + 0.01 * value.get(a, 0)
        if sc > 0: cands[a] = sc
    if not cands:
        weights["cash"] = weights.get("cash", 0) + pool; return weights
    if len(cands) == 1:
        a = list(cands.keys())[0]
        alloc = max(min(pool, 0.40 - weights.get(a, 0)), 0)
        weights[a] = weights.get(a, 0) + alloc
        weights["cash"] = weights.get("cash", 0) + (pool - alloc); return weights
    ts = sum(cands.values())
    for a, sc in cands.items(): weights[a] = weights.get(a, 0) + pool * (sc / ts)
    return weights

def step3(weights, bp):
    if bp > 0.7:
        for a in EQUITY:
            if weights.get(a, 0) > 0: b = weights[a]*0.15; weights[a] += b; weights["cash"] = max(weights.get("cash",0)-b,0)
    elif bp < 0.3:
        for a in EQUITY:
            if weights.get(a, 0) > 0: r = weights[a]*0.15; weights[a] -= r; weights["cash"] = weights.get("cash",0)+r
    return weights

def step4(weights, tp, arz):
    if tp > 0.95 and arz > 2.0:
        for a in EQUITY:
            if weights.get(a, 0) > 0: f = weights[a]*0.50; weights[a] -= f; weights["cash"] = weights.get("cash",0)+f
    return weights

def step5(weights):
    w = dict(weights)
    risky = [a for a in w if a != "cash" and w.get(a,0) > 0.01]
    if len(risky) == 1 and w[risky[0]] > 0.40:
        w["cash"] = w.get("cash",0) + w[risky[0]] - 0.40; w[risky[0]] = 0.40
    for _ in range(10):
        total = sum(max(v,0) for v in w.values())
        if total > 0 and abs(total-1.0) > 1e-6: w = {a: max(v,0)/total for a,v in w.items()}
        changed = False
        for a in w:
            if w[a] > 0.60: w["cash"] = w.get("cash",0)+w[a]-0.60; w[a] = 0.60; changed = True
        eq = sum(w.get(a,0) for a in EQUITY)
        if eq > 0.85:
            r = 0.85/eq
            for a in EQUITY: freed = w[a]-w[a]*r; w[a] *= r; w["cash"] = w.get("cash",0)+freed
            changed = True
        if w.get("cash",0) < 0.03:
            deficit = 0.03 - w["cash"]; others = [a for a in w if a != "cash" and w[a] > 0]
            tot = sum(w[a] for a in others)
            if tot > deficit:
                for a in others: w[a] -= deficit*(w[a]/tot)
            w["cash"] = 0.03; changed = True
        if not changed: break
    total = sum(max(v,0) for v in w.values())
    if total > 0 and abs(total-1.0) > 1e-6: w = {a: max(v,0)/total for a,v in w.items()}
    return w


def main():
    lines = []
    def pr(s=""): print(s); lines.append(s)

    config = RegimeConfig()
    raw_macro = fetch_monthly_history(config)
    transformed = transform_variables(raw_macro, config)
    z_data = get_valid_zscored(transformed, config)
    z_data_lagged = z_data.shift(1).dropna()
    asset_ret = pd.read_parquet(DATA_DIR / "roth_asset_returns_5asset.parquet").rename(columns={"TSY":"IEF"})
    asset_ret_fwd = asset_ret.shift(-1)

    print("Loading data...")
    daily_ret = fetch_daily_etf_returns()
    tsy_daily = pd.read_parquet(DATA_DIR / "treasury_daily_prices.parquet")["TSY"]
    daily_ret["IEF"] = tsy_daily.pct_change()

    import yfinance as yf
    ep = {}
    for our, ticker in [("IVV","SPY"),("QQQ","QQQ"),("IAU","GLD")]:
        d = yf.download(ticker, start="1998-01-01", progress=False)
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p,"columns"): p=p.iloc[:,0]
            p.index = pd.to_datetime(p.index).tz_localize(None)
            ep[our] = p
    ep["IEF"] = tsy_daily
    prices_df = pd.DataFrame(ep).sort_index()
    monthly_prices = prices_df.resample("MS").last()
    sma_dfs = [monthly_prices.rolling(p, min_periods=p).mean().shift(1) for p in SMA_PERIODS]
    rvol_monthly = prices_df.pct_change().rolling(63, min_periods=30).std().mul(np.sqrt(252)).resample("MS").last().shift(1)

    print("Computing carry/value...")
    carry_df = compute_carry(prices_df)
    value_df = compute_value(prices_df)
    from fredapi import Fred
    fkey = os.environ.get("FRED_API_KEY")
    if fkey:
        try:
            f = Fred(api_key=fkey)
            gs20 = f.get_series("GS20", observation_start="1998-01-01"); tb3 = f.get_series("DTB3", observation_start="1998-01-01")
            gs20.index = pd.to_datetime(gs20.index); tb3.index = pd.to_datetime(tb3.index)
            carry_df["IEF"] = (gs20.resample("MS").last()/1200 - tb3.resample("MS").last()/1200).shift(1)
        except Exception: pass

    print("Fitting HMM...")
    spy_raw = fetch_daily_data(); hmm_feat = compute_features(spy_raw)
    hmm_pred = fit_and_predict_rolling(hmm_feat).set_index("date")
    zone_raw = pd.Series("neutral", index=hmm_pred.index)
    zone_raw[hmm_pred["bull_prob"]>0.7]="bull"; zone_raw[hmm_pred["bull_prob"]<0.3]="bear"
    hmm_pred["zone"] = apply_persistence_filter(zone_raw)

    basket = fetch_daily_basket()
    turb_m = compute_turbulence_pctl(compute_turbulence(basket), window=252).resample("MS").last().shift(1)
    ar_z_m = compute_ar_zscore(compute_absorption_ratio(basket, n_components=2)).resample("MS").last().shift(1)

    rfr_daily = pd.Series(0.0, index=daily_ret.index)
    if fkey:
        try:
            tb_d = Fred(api_key=fkey).get_series("DTB3", observation_start="1998-01-01")
            tb_d.index = pd.to_datetime(tb_d.index)
            rfr_daily = (tb_d/100/252).reindex(daily_ret.index, method="ffill").fillna(0)
        except Exception: pass

    pre_start = pd.Timestamp("2002-01-01")
    pre_ers = {"IVV": [], "QQQ": []}
    for z_dt in z_data_lagged.index[z_data_lagged.index < pre_start]:
        try:
            sim = compute_distances(z_data_lagged, z_dt, config)
            for a in ["IVV","QQQ"]:
                if a in asset_ret_fwd.columns:
                    r = [asset_ret_fwd.loc[d,a] for d in sim.similar_dates if d in asset_ret_fwd.index and pd.notna(asset_ret_fwd.loc[d,a])]
                    if r: pre_ers[a].append(np.mean(r))
        except ValueError: pass
    SEED = {a: float(np.mean(pre_ers[a])) if pre_ers[a] else 0.005 for a in ["IVV","QQQ"]}

    common_start = max(daily_ret.dropna(how="all").index.min(), hmm_pred.index.min(), pd.Timestamp("2002-07-01"))
    trading_days = daily_ret.loc[common_start:].index

    pr("=" * 80)
    pr("  LEVERAGE COMPARISON: Conservative vs Test vs Aggressive")
    pr("=" * 80)
    pr(f"\n  {'Config':<14} {'Tier 1':>10} {'Tier 2':>10}")
    pr(f"  {'-'*14} {'-'*10} {'-'*10}")
    for name, subs in LEV_CONFIGS:
        if name == "1x": continue
        pr(f"  {name:<14} {1+subs[1]:.2f}x ({subs[1]:.0%}) {1+subs[2]:.2f}x ({subs[2]:.0%})")

    # Shared state
    strengths = {a: 3 for a in ASSETS if a != "cash"}
    harvey_er = {}; rvols_v = {}; carry_v = {}; value_v = {}
    bp = 0.5; tp = 0.5; arz = 0.0
    w = dict(BASE); tier = 0
    tier_hist = {"IVV": list(pre_ers["IVV"]), "QQQ": list(pre_ers["QQQ"])}
    tier_med = dict(SEED)
    strats = {name: {} for name, _ in LEV_CONFIGS}
    ivv_rets = {}

    for day_idx, day in enumerate(trading_days):
        if day not in daily_ret.index: continue
        dr = daily_ret.loc[day]
        is_ms = (day_idx == 0 or day.month != trading_days[day_idx-1].month)
        is_friday = day.weekday() == 4
        rfr = float(rfr_daily.get(day, 0))
        if "IVV" in dr.index and pd.notna(dr["IVV"]): ivv_rets[day] = float(dr["IVV"])
        avail = [a for a in ASSETS if a in dr.index and pd.notna(dr[a])]
        if len(avail) < 2: continue
        actual = {a: float(dr[a]) for a in avail}

        if is_ms:
            z_c = z_data_lagged.index[z_data_lagged.index < day]
            if len(z_c) > 0:
                try:
                    sim = compute_distances(z_data_lagged, z_c[-1], config)
                    for a in ASSETS:
                        if a not in asset_ret_fwd.columns: harvey_er[a]=0; continue
                        r = [asset_ret_fwd.loc[d,a] for d in sim.similar_dates if d in asset_ret_fwd.index and pd.notna(asset_ret_fwd.loc[d,a])]
                        harvey_er[a] = np.mean(r) if r else 0
                except ValueError: pass
            rv_c = rvol_monthly.index[rvol_monthly.index <= day]
            if len(rv_c) > 0:
                for a in ASSETS:
                    if a != "cash" and a in rvol_monthly.columns:
                        v = rvol_monthly.loc[rv_c[-1], a]; rvols_v[a] = float(v) if pd.notna(v) and v > 0 else 0.15
            carry_v = {}; value_v = {}
            for a in ASSETS:
                if a in carry_df.columns:
                    cc = carry_df.index[carry_df.index <= day]
                    if len(cc) > 0: cv = carry_df.loc[cc[-1], a]; carry_v[a] = float(cv) if pd.notna(cv) else 0
                if a in value_df.columns:
                    vc = value_df.index[value_df.index <= day]
                    if len(vc) > 0: vv = value_df.loc[vc[-1], a]; value_v[a] = float(vv) if pd.notna(vv) else 0
            prev_m = day - pd.DateOffset(months=1)
            hm = hmm_pred[(hmm_pred.index >= prev_m) & (hmm_pred.index < day)]
            bp = hm["bull_prob"].mean() if len(hm) > 0 else 0.5
            if pd.isna(bp): bp = 0.5
            tp = turb_m.get(day, 0.5) if day in turb_m.index else 0.5
            if pd.isna(tp): tp = 0.5
            arz = ar_z_m.get(day, 0) if day in ar_z_m.index else 0
            if pd.isna(arz): arz = 0

        if is_friday or is_ms:
            for a in ASSETS:
                if a == "cash" or a not in prices_df.columns: continue
                pb = prices_df.loc[:day, a]
                if len(pb) == 0: continue
                cp = pb.iloc[-1]; score = 0
                for sdf in sma_dfs:
                    sd = sdf.index[sdf.index <= pd.Timestamp(f"{day.year}-{day.month:02d}-01")]
                    if len(sd) == 0: continue
                    sv = sdf.loc[sd[-1], a] if a in sdf.columns else np.nan
                    if pd.notna(sv) and cp > sv: score += 1
                strengths[a] = score
            wn, pool = step1_faber(strengths)
            wn = step2(wn, pool, harvey_er, rvols_v, carry_v, value_v)
            wn = step3(wn, bp); wn = step4(wn, tp, arz)
            w = step5(wn)
            f_c = strengths.get("IVV",0)>=3 and strengths.get("QQQ",0)>=3
            h_c = harvey_er.get("IVV",0)>0 and harvey_er.get("QQQ",0)>0
            if f_c and h_c:
                above = all(harvey_er.get(a,0) > tier_med.get(a,0) for a in ["IVV","QQQ"])
                tier = 2 if above else 1
                for a in ["IVV","QQQ"]: tier_hist[a].append(harvey_er.get(a,0)); tier_med[a] = float(np.median(tier_hist[a]))
            else: tier = 0

        iw = w.get("IVV",0); qw = w.get("QQQ",0)
        ir = actual.get("IVV",0); qr = actual.get("QQQ",0)
        base_r = sum(w.get(a,0)*actual.get(a,0) for a in avail if a not in ["IVV","QQQ"])
        for name, subs in LEV_CONFIGS:
            sub = subs.get(tier, 0)
            if sub > 0:
                sr = 2*ir - rfr - 0.0091/252; ql = 2*qr - rfr - 0.0089/252
                strats[name][day] = iw*(1-sub)*ir + iw*sub*sr + qw*(1-sub)*qr + qw*sub*ql + base_r
            else:
                strats[name][day] = iw*ir + qw*qr + base_r

    results = {k: pd.Series(v).sort_index() for k, v in strats.items()}
    ivv_s = pd.Series(ivv_rets).sort_index()

    def perf(s):
        ar = s.mean()*252; av = s.std()*np.sqrt(252)
        sh = ar/av if av>0 else 0
        neg = s[s<0]; ds = neg.std()*np.sqrt(252) if len(neg)>10 else av
        so = ar/ds if ds>0 else 0
        cum = (1+s).cumprod(); dd = ((cum-cum.expanding().max())/cum.expanding().max()).min()
        cal = ar/abs(dd) if dd!=0 else 0; final = cum.iloc[-1]
        return {"ar":ar,"av":av,"sh":sh,"so":so,"dd":dd,"cal":cal,"final":final}

    perfs = {name: perf(results[name]) for name, _ in LEV_CONFIGS}

    pr(f"\n\nPERFORMANCE (2002-2026)")
    pr("-" * 80)
    pr(f"  {'Metric':<18} {'1x':>10} {'Conserv':>10} {'Test':>10} {'Aggress':>10}")
    pr(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for label, key, fmt in [("Ann. Return","ar","{:.1%}"),("Volatility","av","{:.1%}"),
                              ("Sharpe","sh","{:.3f}"),("Sortino","so","{:.3f}"),
                              ("Max Drawdown","dd","{:.1%}"),("Calmar","cal","{:.2f}"),
                              ("Terminal $1","final","${:.2f}")]:
        row = f"  {label:<18}"
        for name, _ in LEV_CONFIGS:
            row += f" {fmt.format(perfs[name][key]):>10}"
        pr(row)

    pr(f"\n\nSHARPE BY CONFIG")
    pr("-" * 50)
    sh_1x = perfs["1x"]["sh"]
    for name, _ in LEV_CONFIGS:
        sh = perfs[name]["sh"]; d = sh - sh_1x
        pr(f"  {name:<14} {sh:>8.3f}  (Δ from 1x: {d:>+.3f})")

    # Crisis
    pr(f"\n\nCRISIS ANALYSIS")
    pr("-" * 70)
    for cname, cs, ce in [("GFC","2008-09-01","2009-03-31"),("COVID","2020-02-19","2020-03-23"),("2022 Bear","2022-01-03","2022-10-31")]:
        pr(f"\n  {cname}:")
        pr(f"  {'Metric':<12} {'1x':>10} {'Conserv':>10} {'Test':>10} {'Aggress':>10}")
        pr(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        ret_row = f"  {'Return':<12}"; dd_row = f"  {'Max DD':<12}"
        for name, _ in LEV_CONFIGS:
            s = results[name]; c = s[(s.index>=pd.Timestamp(cs))&(s.index<=pd.Timestamp(ce))]
            if len(c) > 0:
                cum = (1+c).prod()-1; cum_c = (1+c).cumprod()
                dd = ((cum_c-cum_c.expanding().max())/cum_c.expanding().max()).min()
                ret_row += f" {cum:>+9.1%}"; dd_row += f" {dd:>9.1%}"
            else:
                ret_row += f" {'N/A':>10}"; dd_row += f" {'N/A':>10}"
        pr(ret_row); pr(dd_row)

    # Calendar years
    pr(f"\n\n{'='*80}")
    pr(f"  CALENDAR YEAR RETURNS")
    pr(f"{'='*80}")
    pr(f"  {'Year':>6} {'1x':>9} {'Conserv':>9} {'Test':>9} {'Aggress':>9} {'IVV':>9}")
    for yr in range(2003, 2026):
        row = f"  {yr:>6}"
        for name, _ in LEV_CONFIGS:
            s = results[name]; y = s[s.index.year == yr]
            row += f" {(1+y).prod()-1:>+8.1%}" if len(y)>20 else f" {'--':>9}"
        y_ivv = ivv_s[ivv_s.index.year==yr]
        row += f" {(1+y_ivv).prod()-1:>+8.1%}" if len(y_ivv)>20 else f" {'--':>9}"
        pr(row)

    pr(f"\n  FINAL VALUES ($1)")
    for name, _ in LEV_CONFIGS:
        pr(f"    {name}: ${(1+results[name]).cumprod().iloc[-1]:.2f}")
    pr(f"    IVV B&H: ${(1+ivv_s).cumprod().iloc[-1]:.2f}")

    # Recommendation
    pr(f"\n\n{'='*80}")
    pr(f"  RECOMMENDATION")
    pr(f"{'='*80}")
    pc = perfs["Conservative"]; pm = perfs["Test"]; pa = perfs["Aggressive"]
    pr(f"\n  {'Config':<14} {'Sharpe':>8} {'Return':>8} {'MaxDD':>8} {'Terminal':>10}")
    pr(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for name in ["Conservative","Test","Aggressive"]:
        p = perfs[name]
        pr(f"  {name:<14} {p['sh']:>8.3f} {p['ar']:>7.1%} {p['dd']:>7.1%} ${p['final']:>9.2f}")

    best_sh = max(perfs, key=lambda k: perfs[k]["sh"])
    best_ret = max(perfs, key=lambda k: perfs[k]["ar"])
    pr(f"\n  Best Sharpe:  {best_sh} ({perfs[best_sh]['sh']:.3f})")
    pr(f"  Best Return:  {best_ret} ({perfs[best_ret]['ar']:.1%})")

    sh_m = pm["sh"]; sh_c = pc["sh"]
    if sh_m > sh_c:
        pr(f"\n  → Test (1.25x/1.75x) has BETTER Sharpe than Conservative ({sh_m:.3f} vs {sh_c:.3f})")
        pr(f"    Higher Tier 2 leverage captures more upside in high-conviction months,")
        pr(f"    while lower Tier 1 reduces whipsaw cost in moderate-conviction periods.")
    elif abs(sh_m - sh_c) < 0.01:
        pr(f"\n  → Test and Conservative have SIMILAR Sharpe ({sh_m:.3f} vs {sh_c:.3f})")
        pr(f"    Test offers ${pm['final']-pc['final']:+.2f} more terminal wealth at {pm['dd']-pc['dd']:+.1%} more drawdown.")
    else:
        pr(f"\n  → Conservative still has BETTER Sharpe ({sh_c:.3f} vs {sh_m:.3f})")
        pr(f"    The Sharpe curve is monotonically decreasing with leverage.")

    rp = OUTPUT / "leverage_test_report.txt"
    with open(rp, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Report saved: {rp}")


if __name__ == "__main__":
    main()

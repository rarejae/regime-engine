"""5-asset weekly backtest: IVV, QQQ, IEF, IAU, Cash."""

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
TIER_SUBS = {0: 0.0, 1: 0.30, 2: 0.65}
EQUITY = ["IVV", "QQQ"]

BASE_4 = {"IVV": 0.45, "QQQ": 0.25, "IAU": 0.15, "cash": 0.15}
BASE_5 = {"IVV": 0.45, "QQQ": 0.25, "IEF": 0.10, "IAU": 0.10, "cash": 0.10}


# ── Allocation steps ──────────────────────────────────────────────────────────

def step1_faber(strengths, baseline, assets):
    w = {}; pool = 0.0
    for a in assets:
        if a == "cash": w[a] = baseline[a]; continue
        s = strengths.get(a, 0)
        if s >= 3: w[a] = baseline[a]
        elif s == 2: w[a] = baseline[a] * 0.70; pool += baseline[a] * 0.30
        else: w[a] = 0.0; pool += baseline[a]
    return w, pool

def step2_harvey_cv(weights, pool, harvey_er, rvols, carry, value, assets):
    if pool <= 0.001: return weights
    candidates = {}
    for a in assets:
        if a == "cash" or weights.get(a, 0) <= 0: continue
        adj = harvey_er.get(a, 0) + carry.get(a, 0)
        if adj <= 0: continue
        score = adj / max(rvols.get(a, 0.15), 0.01) + 0.01 * value.get(a, 0)
        if score > 0: candidates[a] = score
    if not candidates:
        weights["cash"] = weights.get("cash", 0) + pool; return weights
    if len(candidates) == 1:
        a = list(candidates.keys())[0]
        alloc = max(min(pool, 0.40 - weights.get(a, 0)), 0)
        weights[a] = weights.get(a, 0) + alloc
        weights["cash"] = weights.get("cash", 0) + (pool - alloc); return weights
    ts = sum(candidates.values())
    for a, sc in candidates.items(): weights[a] = weights.get(a, 0) + pool * (sc / ts)
    return weights

def step3_hmm(weights, bp):
    if bp > 0.7:
        for a in EQUITY:
            if weights.get(a, 0) > 0:
                b = weights[a] * 0.15; weights[a] += b; weights["cash"] = max(weights.get("cash", 0) - b, 0)
    elif bp < 0.3:
        for a in EQUITY:
            if weights.get(a, 0) > 0:
                r = weights[a] * 0.15; weights[a] -= r; weights["cash"] = weights.get("cash", 0) + r
    return weights

def step4_kritzman(weights, tp, arz):
    if tp > 0.95 and arz > 2.0:
        for a in EQUITY:
            if weights.get(a, 0) > 0:
                f = weights[a] * 0.50; weights[a] -= f; weights["cash"] = weights.get("cash", 0) + f
    return weights

def step5_normalize(weights):
    w = dict(weights)
    risky = [a for a in w if a != "cash" and w.get(a, 0) > 0.01]
    if len(risky) == 1 and w[risky[0]] > 0.40:
        w["cash"] = w.get("cash", 0) + w[risky[0]] - 0.40; w[risky[0]] = 0.40
    for _ in range(10):
        total = sum(max(v, 0) for v in w.values())
        if total > 0 and abs(total - 1.0) > 1e-6: w = {a: max(v, 0) / total for a, v in w.items()}
        changed = False
        for a in w:
            if w[a] > 0.60: w["cash"] = w.get("cash", 0) + w[a] - 0.60; w[a] = 0.60; changed = True
        eq = sum(w.get(a, 0) for a in EQUITY)
        if eq > 0.85:
            r = 0.85 / eq
            for a in EQUITY: freed = w[a] - w[a] * r; w[a] *= r; w["cash"] = w.get("cash", 0) + freed
            changed = True
        if w.get("cash", 0) < 0.03:
            deficit = 0.03 - w["cash"]; others = [a for a in w if a != "cash" and w[a] > 0]
            tot = sum(w[a] for a in others)
            if tot > deficit:
                for a in others: w[a] -= deficit * (w[a] / tot)
            w["cash"] = 0.03; changed = True
        if not changed: break
    total = sum(max(v, 0) for v in w.values())
    if total > 0 and abs(total - 1.0) > 1e-6: w = {a: max(v, 0) / total for a, v in w.items()}
    return w


def main():
    lines = []
    def pr(s=""): print(s); lines.append(s)

    config = RegimeConfig()
    raw_macro = fetch_monthly_history(config)
    transformed = transform_variables(raw_macro, config)
    z_data = get_valid_zscored(transformed, config)
    z_data_lagged = z_data.shift(1).dropna()

    # Load 5-asset returns (has TSY column from pipeline)
    asset_ret = pd.read_parquet(DATA_DIR / "roth_asset_returns_5asset.parquet")
    asset_ret = asset_ret.rename(columns={"TSY": "IEF"})
    asset_ret_fwd = asset_ret.shift(-1)

    print("Loading daily data...")
    daily_ret = fetch_daily_etf_returns()

    # Add IEF daily returns from treasury prices
    tsy_daily_prices = pd.read_parquet(DATA_DIR / "treasury_daily_prices.parquet")["TSY"]
    daily_ret["IEF"] = tsy_daily_prices.pct_change()

    import yfinance as yf
    # Prices for Faber SMAs
    ep = {}
    for our, ticker in [("IVV","SPY"),("QQQ","QQQ"),("IAU","GLD")]:
        d = yf.download(ticker, start="1998-01-01", progress=False)
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p,"columns"): p=p.iloc[:,0]
            p.index = pd.to_datetime(p.index).tz_localize(None)
            ep[our] = p
    ep["IEF"] = tsy_daily_prices
    prices_df = pd.DataFrame(ep).sort_index()

    # Monthly SMAs for weekly Faber checks
    monthly_prices = prices_df.resample("MS").last()
    sma_dfs = []
    for p in SMA_PERIODS:
        sma_dfs.append(monthly_prices.rolling(p, min_periods=p).mean().shift(1))

    rvol_63 = prices_df.pct_change().rolling(63, min_periods=30).std() * np.sqrt(252)
    rvol_monthly = rvol_63.resample("MS").last().shift(1)

    print("Computing carry/value...")
    carry_df = compute_carry(prices_df)
    value_df = compute_value(prices_df)

    # Add IEF carry: term premium (GS20 - TB3MS)
    from fredapi import Fred
    fkey = os.environ.get("FRED_API_KEY")
    if fkey:
        fred_api = Fred(api_key=fkey)
        try:
            gs20 = fred_api.get_series("GS20", observation_start="1998-01-01")
            tb3 = fred_api.get_series("DTB3", observation_start="1998-01-01")
            if gs20 is not None and tb3 is not None:
                gs20.index = pd.to_datetime(gs20.index)
                tb3.index = pd.to_datetime(tb3.index)
                carry_df["IEF"] = (gs20.resample("MS").last() / 1200 - tb3.resample("MS").last() / 1200).shift(1)
        except Exception:
            pass

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
    turb_pctl_s = compute_turbulence_pctl(turb_smooth, window=252)
    ar_obj = compute_absorption_ratio(basket, n_components=2)
    ar_z_s = compute_ar_zscore(ar_obj)
    turb_m = turb_pctl_s.resample("MS").last().shift(1)
    ar_z_m = ar_z_s.resample("MS").last().shift(1)

    rfr_daily = pd.Series(0.0, index=daily_ret.index)
    if fkey:
        try:
            tb_d = Fred(api_key=fkey).get_series("DTB3", observation_start="1998-01-01")
            tb_d.index = pd.to_datetime(tb_d.index)
            rfr_daily = (tb_d / 100 / 252).reindex(daily_ret.index, method="ffill").fillna(0)
        except Exception:
            pass

    # ER seed
    pre_start = pd.Timestamp("2002-01-01")
    pre_ers = {"IVV": [], "QQQ": []}
    for z_dt in z_data_lagged.index[z_data_lagged.index < pre_start]:
        try:
            sim = compute_distances(z_data_lagged, z_dt, config)
            for a in ["IVV","QQQ"]:
                if a in asset_ret_fwd.columns:
                    rets = [asset_ret_fwd.loc[d,a] for d in sim.similar_dates
                            if d in asset_ret_fwd.index and pd.notna(asset_ret_fwd.loc[d,a])]
                    if rets: pre_ers[a].append(np.mean(rets))
        except ValueError: pass
    SEED = {a: float(np.mean(pre_ers[a])) if pre_ers[a] else 0.005 for a in ["IVV","QQQ"]}

    common_start = max(daily_ret.dropna(how="all").index.min(), hmm_pred.index.min(), pd.Timestamp("2002-07-01"))
    trading_days = daily_ret.loc[common_start:].index

    pr("=" * 80)
    pr("  5-ASSET WEEKLY BACKTEST RESULTS")
    pr("=" * 80)
    pr(f"\nBacktest: {len(trading_days)} days ({common_start.date()} to {trading_days.max().date()})")
    pr(f"\n  4-Asset: IVV 45%, QQQ 25%, IAU 15%, Cash 15%")
    pr(f"  5-Asset: IVV 45%, QQQ 25%, IEF 10%, IAU 10%, Cash 10%")

    # ── Run both configs with weekly rebalancing ──────────────────────────────

    configs = {
        "4A": {"base": BASE_4, "assets": list(BASE_4.keys())},
        "5A": {"base": BASE_5, "assets": list(BASE_5.keys())},
    }

    strats = {k: {} for k in configs}
    ivv_rets = {}

    harvey_er = {}; rvols = {}; carry_vals = {}; value_vals = {}
    bp = 0.5; tp = 0.5; arz = 0.0

    state = {}
    for k, cfg in configs.items():
        state[k] = {
            "strengths": {a: 3 for a in cfg["assets"] if a != "cash"},
            "w": dict(cfg["base"]),
            "tier": 0,
            "tier_hist": {"IVV": list(pre_ers["IVV"]), "QQQ": list(pre_ers["QQQ"])},
            "tier_med": dict(SEED),
            "tc_total": 0.0, "prev_w": None,
            "weight_snaps": [],  # (date, weights_dict, tier)
        }

    for day_idx, day in enumerate(trading_days):
        if day not in daily_ret.index: continue
        dr = daily_ret.loc[day]
        is_ms = (day_idx == 0 or day.month != trading_days[day_idx - 1].month)
        is_friday = day.weekday() == 4
        rfr = float(rfr_daily.get(day, 0))

        if "IVV" in dr.index and pd.notna(dr["IVV"]):
            ivv_rets[day] = float(dr["IVV"])

        # Update monthly signals
        if is_ms:
            # Harvey
            z_cands = z_data_lagged.index[z_data_lagged.index < day]
            if len(z_cands) > 0:
                try:
                    sim = compute_distances(z_data_lagged, z_cands[-1], config)
                    all_a = set()
                    for c in configs.values(): all_a.update(c["assets"])
                    for a in all_a:
                        if a not in asset_ret_fwd.columns: harvey_er[a]=0; continue
                        r = [asset_ret_fwd.loc[d,a] for d in sim.similar_dates
                             if d in asset_ret_fwd.index and pd.notna(asset_ret_fwd.loc[d,a])]
                        harvey_er[a] = np.mean(r) if r else 0
                except ValueError: pass

            rv_c = rvol_monthly.index[rvol_monthly.index <= day]
            if len(rv_c) > 0:
                for a in set().union(*[set(c["assets"]) for c in configs.values()]):
                    if a != "cash" and a in rvol_monthly.columns:
                        v = rvol_monthly.loc[rv_c[-1], a]
                        rvols[a] = float(v) if pd.notna(v) and v > 0 else 0.15

            carry_vals = {}; value_vals = {}
            for a in set().union(*[set(c["assets"]) for c in configs.values()]):
                if a in carry_df.columns:
                    cc = carry_df.index[carry_df.index <= day]
                    if len(cc) > 0: cv = carry_df.loc[cc[-1], a]; carry_vals[a] = float(cv) if pd.notna(cv) else 0
                if a in value_df.columns:
                    vc = value_df.index[value_df.index <= day]
                    if len(vc) > 0: vv = value_df.loc[vc[-1], a]; value_vals[a] = float(vv) if pd.notna(vv) else 0

            prev_m = day - pd.DateOffset(months=1)
            hm = hmm_pred[(hmm_pred.index >= prev_m) & (hmm_pred.index < day)]
            bp = hm["bull_prob"].mean() if len(hm) > 0 else 0.5
            if pd.isna(bp): bp = 0.5
            tp = turb_m.get(day, 0.5) if day in turb_m.index else 0.5
            if pd.isna(tp): tp = 0.5
            arz = ar_z_m.get(day, 0) if day in ar_z_m.index else 0
            if pd.isna(arz): arz = 0

        # Weekly rebalance
        if is_friday or is_ms:
            for k, cfg in configs.items():
                s = state[k]; assets = cfg["assets"]; base = cfg["base"]

                # Weekly Faber
                for a in assets:
                    if a == "cash" or a not in prices_df.columns: continue
                    pb = prices_df.loc[:day, a]
                    if len(pb) == 0: continue
                    cp = pb.iloc[-1]; score = 0
                    for sdf in sma_dfs:
                        sd = sdf.index[sdf.index <= pd.Timestamp(f"{day.year}-{day.month:02d}-01")]
                        if len(sd) == 0: continue
                        sv = sdf.loc[sd[-1], a] if a in sdf.columns else np.nan
                        if pd.notna(sv) and cp > sv: score += 1
                    s["strengths"][a] = score

                w, pool = step1_faber(s["strengths"], base, assets)
                w = step2_harvey_cv(w, pool, harvey_er, rvols, carry_vals, value_vals, assets)
                w = step3_hmm(w, bp)
                w = step4_kritzman(w, tp, arz)
                s["w"] = step5_normalize(w)

                # Tier
                f_c = s["strengths"].get("IVV",0)>=3 and s["strengths"].get("QQQ",0)>=3
                h_c = harvey_er.get("IVV",0)>0 and harvey_er.get("QQQ",0)>0
                if f_c and h_c:
                    above = all(harvey_er.get(a,0) > s["tier_med"].get(a,0) for a in ["IVV","QQQ"])
                    s["tier"] = 2 if above else 1
                    for a in ["IVV","QQQ"]:
                        s["tier_hist"][a].append(harvey_er.get(a,0))
                        s["tier_med"][a] = float(np.median(s["tier_hist"][a]))
                else: s["tier"] = 0

                if s["prev_w"] is not None:
                    to = sum(abs(s["w"].get(a,0) - s["prev_w"].get(a,0)) for a in assets) / 2
                    s["tc_total"] += to * 0.001
                s["prev_w"] = dict(s["w"])
                s["weight_snaps"].append((day, dict(s["w"]), s["tier"]))

        # Daily returns
        for k, cfg in configs.items():
            s = state[k]; wp = s["w"]; assets = cfg["assets"]
            avail = [a for a in assets if a in dr.index and pd.notna(dr[a])]
            if len(avail) < 2: continue
            actual = {a: float(dr[a]) for a in avail}

            iw = wp.get("IVV",0); qw = wp.get("QQQ",0)
            ir = actual.get("IVV",0); qr = actual.get("QQQ",0)
            base_r = sum(wp.get(a,0)*actual.get(a,0) for a in avail if a not in ["IVV","QQQ"])
            sub = TIER_SUBS.get(s["tier"], 0)
            if sub > 0:
                sr = 2*ir - rfr - 0.0091/252; ql = 2*qr - rfr - 0.0089/252
                ret = iw*(1-sub)*ir + iw*sub*sr + qw*(1-sub)*qr + qw*sub*ql + base_r
            else:
                ret = iw*ir + qw*qr + base_r
            strats[k][day] = ret

    results = {k: pd.Series(v).sort_index() for k, v in strats.items()}
    ivv_s = pd.Series(ivv_rets).sort_index()
    n_years = len(trading_days) / 252

    # ── Report ────────────────────────────────────────────────────────────────

    def perf(s):
        ar = s.mean()*252; av = s.std()*np.sqrt(252)
        sh = ar/av if av>0 else 0
        neg = s[s<0]; ds = neg.std()*np.sqrt(252) if len(neg)>10 else av
        so = ar/ds if ds>0 else 0
        cum = (1+s).cumprod(); dd = ((cum-cum.expanding().max())/cum.expanding().max()).min()
        cal = ar/abs(dd) if dd!=0 else 0
        final = cum.iloc[-1]; corr = s.corr(ivv_s)
        return {"ar":ar,"av":av,"sh":sh,"so":so,"dd":dd,"cal":cal,"final":final,"corr":corr}

    p4 = perf(results["4A"]); p5 = perf(results["5A"])

    pr(f"\n\nPERFORMANCE (2002-2026)")
    pr("-" * 70)
    pr(f"  {'Metric':<22} {'4-Asset':>12} {'5-Asset':>12} {'Delta':>12}")
    pr(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*12}")
    for name, key, fmt in [("Ann. Return","ar","{:.1%}"),("Volatility","av","{:.1%}"),
                             ("Sharpe","sh","{:.3f}"),("Sortino","so","{:.3f}"),
                             ("Max Drawdown","dd","{:.1%}"),("Calmar","cal","{:.2f}"),
                             ("Terminal $1","final","${:.2f}"),("Corr w/ IVV","corr","{:.3f}")]:
        v4 = p4[key]; v5 = p5[key]; d = v5 - v4
        if key == "final":
            pr(f"  {name:<22} {fmt.format(v4):>12} {fmt.format(v5):>12} {'+' if d>0 else ''}{fmt.format(d):>11}")
        elif key in ("sh","so","cal","corr"):
            pr(f"  {name:<22} {fmt.format(v4):>12} {fmt.format(v5):>12} {d:>+12.3f}")
        else:
            pr(f"  {name:<22} {fmt.format(v4):>12} {fmt.format(v5):>12} {d:>+11.1%}")

    # TC
    pr(f"\n  TC (ann.): 4A={state['4A']['tc_total']/n_years:.2%}, 5A={state['5A']['tc_total']/n_years:.2%}")
    pr(f"  Net Sharpe: 4A={(p4['ar']-state['4A']['tc_total']/n_years)/p4['av']:.3f}, 5A={(p5['ar']-state['5A']['tc_total']/n_years)/p5['av']:.3f}")

    # Leverage tiers
    pr(f"\n\nLEVERAGE TIERS")
    pr("-" * 50)
    for k in ["4A","5A"]:
        snaps = state[k]["weight_snaps"]
        tiers = [t for _,_,t in snaps]
        total = len(tiers)
        pr(f"  {('4-Asset' if k=='4A' else '5-Asset')}:")
        for t in [0,1,2]:
            n = tiers.count(t)
            pr(f"    Tier {t} ({['1.0x','1.3x','1.65x'][t]}): {n:>5} ({n/total*100:.1f}%)")

    # Crisis analysis
    pr(f"\n\n{'='*80}")
    pr(f"  CRISIS ANALYSIS")
    pr(f"{'='*80}")

    crises = [("GFC","2008-09-01","2009-03-31","2008-01-01","2009-06-30"),
              ("COVID","2020-02-19","2020-03-23","2020-01-01","2020-06-30"),
              ("2022 Bear","2022-01-03","2022-10-31","2021-10-01","2023-03-31")]

    for cname, cs, ce, ws, we in crises:
        pr(f"\n  {cname}:")
        pr(f"  {'Metric':<30} {'4-Asset':>12} {'5-Asset':>12}")
        pr(f"  {'-'*30} {'-'*12} {'-'*12}")

        for k in ["4A","5A"]:
            s = results[k]; snaps = state[k]["weight_snaps"]
            c = s[(s.index >= pd.Timestamp(cs)) & (s.index <= pd.Timestamp(ce))]
            if len(c) == 0: continue
            cum_ret = (1+c).prod() - 1
            cum_c = (1+c).cumprod()
            dd = ((cum_c - cum_c.expanding().max()) / cum_c.expanding().max()).min()

            # Average weights during wider window
            crisis_snaps = [(d,w,t) for d,w,t in snaps if pd.Timestamp(ws) <= d <= pd.Timestamp(we)]
            if crisis_snaps:
                eq_avg = np.mean([w.get("IVV",0)+w.get("QQQ",0) for _,w,_ in crisis_snaps])
                ief_avg = np.mean([w.get("IEF",0) for _,w,_ in crisis_snaps]) if k == "5A" else 0
                iau_avg = np.mean([w.get("IAU",0) for _,w,_ in crisis_snaps])
                cash_avg = np.mean([w.get("cash",0) for _,w,_ in crisis_snaps])

            if k == "4A":
                vals4 = {"ret": cum_ret, "dd": dd,
                          "eq": eq_avg if crisis_snaps else 0,
                          "ief": 0,
                          "iau": iau_avg if crisis_snaps else 0,
                          "cash": cash_avg if crisis_snaps else 0}
            else:
                vals5 = {"ret": cum_ret, "dd": dd,
                          "eq": eq_avg if crisis_snaps else 0,
                          "ief": ief_avg if crisis_snaps else 0,
                          "iau": iau_avg if crisis_snaps else 0,
                          "cash": cash_avg if crisis_snaps else 0}

        if "vals4" in dir() and "vals5" in dir():
            for name, k4, k5 in [("Return","ret","ret"),("Max DD","dd","dd"),
                                   ("Avg equity","eq","eq"),("Avg IEF","ief","ief"),
                                   ("Avg gold","iau","iau"),("Avg cash","cash","cash")]:
                v4 = vals4[k4]; v5 = vals5[k5]
                if name in ("Return","Max DD"):
                    pr(f"  {name:<30} {v4:>+11.1%} {v5:>+11.1%}")
                else:
                    pr(f"  {name:<30} {v4:>11.0%} {v5:>11.0%}")

    # Defensive asset analysis
    pr(f"\n\n{'='*80}")
    pr(f"  DEFENSIVE ASSET ANALYSIS (5-Asset)")
    pr(f"{'='*80}")

    snaps5 = state["5A"]["weight_snaps"]
    ief_wts = [w.get("IEF",0) for _,w,_ in snaps5]
    iau_wts = [w.get("IAU",0) for _,w,_ in snaps5]
    cash_wts = [w.get("cash",0) for _,w,_ in snaps5]
    ief_fab3 = sum(1 for _,w,_ in snaps5 if state["5A"]["strengths"].get("IEF",0) >= 3)
    iau_fab3 = sum(1 for _,w,_ in snaps5 if state["5A"]["strengths"].get("IAU",0) >= 3)

    pr(f"\n  Average weights (baseline in parens):")
    pr(f"    IEF:   {np.mean(ief_wts):.0%} (10%)")
    pr(f"    IAU:   {np.mean(iau_wts):.0%} (10%)")
    pr(f"    Cash:  {np.mean(cash_wts):.0%} (10%)")

    pr(f"\n  Faber 3/3 frequency:")
    pr(f"    IEF:   {ief_fab3/len(snaps5)*100:.0f}%")
    pr(f"    IAU:   {iau_fab3/len(snaps5)*100:.0f}%")

    # IEF-IAU correlation
    ief_d = daily_ret.get("IEF")
    iau_d = daily_ret.get("IAU")
    if ief_d is not None and iau_d is not None:
        common = ief_d.dropna().index.intersection(iau_d.dropna().index)
        tg_corr = ief_d.reindex(common).corr(iau_d.reindex(common))
        pr(f"\n  IEF-IAU daily return correlation: {tg_corr:.3f}")

    # Calendar years
    pr(f"\n\n{'='*80}")
    pr(f"  CALENDAR YEAR RETURNS")
    pr(f"{'='*80}")
    pr(f"  {'Year':>6} {'4-Asset':>9} {'5-Asset':>9} {'IVV':>9}")
    for yr in range(2003, 2026):
        row = f"  {yr:>6}"
        for k in ["4A","5A"]:
            s = results[k]; y = s[s.index.year == yr]
            row += f" {(1+y).prod()-1:>+8.1%}" if len(y) > 20 else f" {'--':>9}"
        y_ivv = ivv_s[ivv_s.index.year == yr]
        row += f" {(1+y_ivv).prod()-1:>+8.1%}" if len(y_ivv) > 20 else f" {'--':>9}"
        pr(row)

    # Final values
    pr(f"\n  FINAL VALUES ($1)")
    for k in ["4A","5A"]:
        label = "4-Asset" if k == "4A" else "5-Asset"
        s = results[k]
        pr(f"    {label}: ${(1+s).cumprod().iloc[-1]:.2f}")
    pr(f"    IVV B&H: ${(1+ivv_s).cumprod().iloc[-1]:.2f}")

    # Recommendation
    pr(f"\n\n{'='*80}")
    pr(f"  RECOMMENDATION")
    pr(f"{'='*80}")

    sh_diff = p5["sh"] - p4["sh"]
    dd_diff = p5["dd"] - p4["dd"]

    if sh_diff > 0.02:
        pr(f"\n  → ADOPT 5-ASSET MODEL")
        pr(f"  Sharpe improvement: {sh_diff:+.3f}")
        pr(f"  IEF adds crisis diversification that gold alone doesn't provide.")
    elif sh_diff > -0.02:
        pr(f"\n  → 5-ASSET MARGINALLY BETTER/EQUAL")
        pr(f"  Sharpe delta: {sh_diff:+.3f}")
        if dd_diff > 0.01:
            pr(f"  Key benefit: shallower drawdowns ({p5['dd']:.1%} vs {p4['dd']:.1%})")
        pr(f"  Adding IEF provides structural diversification without hurting returns.")
    else:
        pr(f"\n  → KEEP 4-ASSET")
        pr(f"  IEF doesn't justify the added complexity (Sharpe delta: {sh_diff:+.3f})")

    pr(f"\n  KEY QUESTION: Does IEF add value over gold-only defensive allocation?")
    if p5["dd"] > p4["dd"]:
        pr(f"  YES — Max drawdown improved from {p4['dd']:.1%} to {p5['dd']:.1%}")
    else:
        pr(f"  NO — Drawdowns are similar or worse")
    if p5["corr"] < p4["corr"]:
        pr(f"  YES — IVV correlation dropped from {p4['corr']:.3f} to {p5['corr']:.3f} (better diversification)")
    else:
        pr(f"  NO — IVV correlation unchanged")

    # Save
    rp = OUTPUT / "backtest_5asset_report.txt"
    with open(rp, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Report saved: {rp}")


if __name__ == "__main__":
    main()

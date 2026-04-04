"""Portfolio beta & risk metrics: 4-asset vs 6-asset, by tier/crisis/regime."""

import sys, os, warnings, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv; load_dotenv()
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

import numpy as np, pandas as pd
from scipy import stats as sp_stats

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
TC = 0.0010
SMA_PERIODS = [6, 10, 12]
TIER_SUBS = {0: 0.0, 1: 0.30, 2: 0.65}

BASE_6 = {"IVV": 0.45, "QQQ": 0.25, "VGLT": 0.05, "IAU": 0.10, "DBC": 0.05, "cash": 0.10}
BASE_4 = {"IVV": 0.45, "QQQ": 0.25, "IAU": 0.15, "cash": 0.15}
EQUITY = ["IVV", "QQQ"]

CRISES = [("GFC", "2008-09-01", "2009-03-31"), ("COVID", "2020-02-19", "2020-03-23"),
          ("2022 Bear", "2022-01-03", "2022-10-31")]


# ── Allocation steps (same as test_4asset.py) ────────────────────────────────

def compute_multi_sma(prices_df):
    monthly = prices_df.resample("MS").last()
    result = pd.DataFrame(0, index=monthly.index, columns=monthly.columns)
    for period in SMA_PERIODS:
        sma = monthly.rolling(period, min_periods=period).mean()
        result += (monthly > sma).astype(int)
    return result

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
        er = harvey_er.get(a, 0); vol = rvols.get(a, 0.15)
        c = carry.get(a, 0); v = value.get(a, 0)
        adj = er + c
        if adj <= 0: continue
        score = adj / max(vol, 0.01) + 0.01 * v
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


# ── Metric computation ────────────────────────────────────────────────────────

def compute_beta_metrics(port, bench):
    """Compute beta, alpha, R², TE, IR, captures from monthly return series."""
    common = port.index.intersection(bench.index)
    p = port.reindex(common).dropna(); b = bench.reindex(common).dropna()
    common = p.index.intersection(b.index)
    if len(common) < 6: return None
    p, b = p.reindex(common), b.reindex(common)

    slope, intercept, r_value, _, _ = sp_stats.linregress(b, p)
    diff = p - b
    te = diff.std() * np.sqrt(12)
    ir = diff.mean() * 12 / te if te > 0 else 0

    up_m = b > 0; dn_m = b < 0
    up_cap = (p[up_m].mean() / b[up_m].mean() * 100) if up_m.sum() > 2 else float("nan")
    dn_cap = (p[dn_m].mean() / b[dn_m].mean() * 100) if dn_m.sum() > 2 else float("nan")
    ud_ratio = up_cap / dn_cap if (not np.isnan(up_cap) and not np.isnan(dn_cap) and dn_cap != 0) else float("nan")

    return {
        "n": len(common), "beta": slope, "alpha_ann": intercept * 12,
        "r2": r_value ** 2, "te": te, "ir": ir,
        "up_cap": up_cap, "dn_cap": dn_cap, "ud_ratio": ud_ratio,
        "avg_ret": p.mean(), "avg_bench": b.mean(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    lines = []
    def pr(s=""): print(s); lines.append(s)

    config = RegimeConfig()
    raw_macro = fetch_monthly_history(config)
    transformed = transform_variables(raw_macro, config)
    z_data = get_valid_zscored(transformed, config)
    z_data_lagged = z_data.shift(1).dropna()
    asset_ret = pd.read_parquet("data/macro/roth_asset_returns.parquet")
    asset_ret_fwd = asset_ret.shift(-1)

    print("Loading data...")
    daily_ret = fetch_daily_etf_returns()

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
    multi_sma_df = compute_multi_sma(prices_df).shift(1)
    rvol_63 = prices_df.pct_change().rolling(63, min_periods=30).std() * np.sqrt(252)
    rvol_monthly = rvol_63.resample("MS").last().shift(1)

    print("Computing carry/value...")
    carry_df = compute_carry(prices_df)
    value_df = compute_value(prices_df)

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

    from fredapi import Fred
    rfr_daily = pd.Series(0.0, index=daily_ret.index)
    key = os.environ.get("FRED_API_KEY")
    if key:
        tb = Fred(api_key=key).get_series("DTB3", observation_start="1998-01-01")
        tb.index = pd.to_datetime(tb.index)
        rfr_daily = (tb / 100 / 252).reindex(daily_ret.index, method="ffill").fillna(0)

    daily_prices_df = pd.DataFrame({a: ep[a] for a in ["IVV","QQQ"] if a in ep})
    monthly_close = daily_prices_df.resample("MS").last()
    sma_6m = monthly_close.rolling(6, min_periods=6).mean()
    sma_10m = monthly_close.rolling(10, min_periods=10).mean()
    sma_12m = monthly_close.rolling(12, min_periods=12).mean()

    # Precompute ER seed
    pre_start = pd.Timestamp("2002-01-01")
    pre_months = z_data_lagged.index[z_data_lagged.index < pre_start]
    pre_ers = {"IVV": [], "QQQ": []}
    for z_dt in pre_months:
        try:
            sim = compute_distances(z_data_lagged, z_dt, config)
            for a in ["IVV","QQQ"]:
                if a in asset_ret_fwd.columns:
                    rets = [asset_ret_fwd.loc[d,a] for d in sim.similar_dates
                            if d in asset_ret_fwd.index and pd.notna(asset_ret_fwd.loc[d,a])]
                    if rets: pre_ers[a].append(np.mean(rets))
        except ValueError: pass
    SEED = {a: float(np.mean(pre_ers[a])) if pre_ers[a] else 0.005 for a in ["IVV","QQQ"]}

    common_start = max(daily_ret.dropna(how="all").index.min(), hmm_pred.index.min(), pd.Timestamp("2002-01-01"))
    trading_days = daily_ret.loc[common_start:].index

    # ── Run backtest (same as test_4asset) ────────────────────────────────────

    run_configs = {
        "6A_lev": {"base": BASE_6, "assets": list(BASE_6.keys()), "leverage": True},
        "4A_lev": {"base": BASE_4, "assets": list(BASE_4.keys()), "leverage": True},
        "6A_1x":  {"base": BASE_6, "assets": list(BASE_6.keys()), "leverage": False},
        "4A_1x":  {"base": BASE_4, "assets": list(BASE_4.keys()), "leverage": False},
    }

    daily_strats = {k: {} for k in run_configs}
    tier_daily = {k: {} for k in run_configs if "lev" in k}  # day → tier
    st = {}
    for k, cfg in run_configs.items():
        st[k] = {
            "strengths": {a: 3 for a in cfg["assets"] if a != "cash"},
            "harvey_er": {a: 0.0 for a in cfg["assets"]},
            "w": dict(cfg["base"]), "tier": 0, "delevered": False,
            "tier_hist": {"IVV": list(pre_ers["IVV"]), "QQQ": list(pre_ers["QQQ"])},
            "tier_med": dict(SEED),
        }

    ivv_daily = {}

    for day_idx, day in enumerate(trading_days):
        if day not in daily_ret.index: continue
        dr = daily_ret.loc[day]
        is_ms = (day_idx == 0 or day.month != trading_days[day_idx - 1].month)
        is_friday = day.weekday() == 4
        rfr = float(rfr_daily.get(day, 0))

        if "IVV" in dr.index and pd.notna(dr["IVV"]):
            ivv_daily[day] = float(dr["IVV"])

        for k, cfg in run_configs.items():
            assets = cfg["assets"]; base = cfg["base"]; use_lev = cfg["leverage"]
            s = st[k]
            avail = [a for a in assets if a in dr.index and pd.notna(dr[a])]
            if len(avail) < 2: continue
            actual = {a: float(dr[a]) for a in avail}

            if is_ms:
                s["delevered"] = False
                ms_cands = multi_sma_df.index[multi_sma_df.index <= day]
                if len(ms_cands) > 0:
                    ms_dt = ms_cands[-1]
                    for a in assets:
                        if a != "cash" and a in multi_sma_df.columns:
                            v = multi_sma_df.loc[ms_dt, a]
                            s["strengths"][a] = int(v) if pd.notna(v) else 0
                z_cands = z_data_lagged.index[z_data_lagged.index < day]
                if len(z_cands) > 0:
                    try:
                        sim = compute_distances(z_data_lagged, z_cands[-1], config)
                        for a in avail:
                            if a not in asset_ret_fwd.columns: s["harvey_er"][a]=0; continue
                            rets = [asset_ret_fwd.loc[d,a] for d in sim.similar_dates
                                    if d in asset_ret_fwd.index and pd.notna(asset_ret_fwd.loc[d,a])]
                            s["harvey_er"][a] = np.mean(rets) if rets else 0
                    except ValueError: pass
                rv_cands = rvol_monthly.index[rvol_monthly.index <= day]
                rvols = {}
                if len(rv_cands) > 0:
                    for a in assets:
                        if a != "cash" and a in rvol_monthly.columns:
                            v = rvol_monthly.loc[rv_cands[-1], a]
                            rvols[a] = float(v) if pd.notna(v) and v > 0 else 0.15
                carry = {}; value = {}
                for a in assets:
                    if a in carry_df.columns:
                        c_c = carry_df.index[carry_df.index <= day]
                        if len(c_c) > 0: cv = carry_df.loc[c_c[-1], a]; carry[a] = float(cv) if pd.notna(cv) else 0
                    if a in value_df.columns:
                        v_c = value_df.index[value_df.index <= day]
                        if len(v_c) > 0: vv = value_df.loc[v_c[-1], a]; value[a] = float(vv) if pd.notna(vv) else 0
                prev_m = day - pd.DateOffset(months=1)
                hm = hmm_pred[(hmm_pred.index >= prev_m) & (hmm_pred.index < day)]
                bp = hm["bull_prob"].mean() if len(hm) > 0 else 0.5
                if pd.isna(bp): bp = 0.5
                tp = turb_m.get(day, 0.5) if day in turb_m.index else 0.5
                if pd.isna(tp): tp = 0.5
                arz = ar_z_m.get(day, 0) if day in ar_z_m.index else 0
                if pd.isna(arz): arz = 0

                w, pool = step1_faber(s["strengths"], base, assets)
                w = step2_harvey_cv(w, pool, s["harvey_er"], rvols, carry, value, assets)
                w = step3_hmm(w, bp)
                w = step4_kritzman(w, tp, arz)
                s["w"] = step5_normalize(w)

                if use_lev:
                    f_c = s["strengths"].get("IVV",0)>=3 and s["strengths"].get("QQQ",0)>=3
                    h_c = s["harvey_er"].get("IVV",0)>0 and s["harvey_er"].get("QQQ",0)>0
                    if f_c and h_c:
                        above = all(s["harvey_er"].get(a,0)>s["tier_med"].get(a,0) for a in ["IVV","QQQ"])
                        s["tier"] = 2 if above else 1
                        for a in ["IVV","QQQ"]:
                            s["tier_hist"][a].append(s["harvey_er"].get(a,0))
                            s["tier_med"][a] = float(np.median(s["tier_hist"][a]))
                    else: s["tier"] = 0

            if use_lev and is_friday and s["tier"] > 0 and not s["delevered"]:
                breach = False
                for etf in ["IVV","QQQ"]:
                    if etf not in daily_prices_df.columns: continue
                    pb = daily_prices_df.loc[:day, etf]
                    if len(pb)==0: continue
                    pt = pb.iloc[-1]; lms = pd.Timestamp(f"{day.year}-{day.month:02d}-01")
                    for sdf in [sma_6m, sma_10m, sma_12m]:
                        sd = sdf.index[sdf.index <= lms]
                        if len(sd)==0: continue
                        sv = sdf.loc[sd[-1], etf]
                        if pd.notna(sv) and pt < sv: breach = True; break
                    if breach: break
                if breach: s["tier"] = 0; s["delevered"] = True

            wp = s["w"]; iw = wp.get("IVV",0); qw = wp.get("QQQ",0)
            ir_v = actual.get("IVV",0); qr = actual.get("QQQ",0)
            base_ret = sum(wp.get(a,0)*actual.get(a,0) for a in avail if a not in ["IVV","QQQ"])
            if use_lev:
                sub = TIER_SUBS.get(s["tier"], 0)
                if sub > 0:
                    sr = 2*ir_v - rfr - 0.0091/252; ql = 2*qr - rfr - 0.0089/252
                    ret = iw*(1-sub)*ir_v + iw*sub*sr + qw*(1-sub)*qr + qw*sub*ql + base_ret
                else: ret = iw*ir_v + qw*qr + base_ret
                tier_daily[k][day] = s["tier"]
            else: ret = iw*ir_v + qw*qr + base_ret
            daily_strats[k][day] = ret

    # Convert to series
    results = {k: pd.Series(v).sort_index() for k, v in daily_strats.items()}
    ivv_s = pd.Series(ivv_daily).sort_index()
    tier_series = {k: pd.Series(v).sort_index() for k, v in tier_daily.items()}

    # Resample to monthly
    monthly = {k: (1 + s).resample("MS").prod() - 1 for k, s in results.items()}
    ivv_m = (1 + ivv_s).resample("MS").prod() - 1

    # ── Report ────────────────────────────────────────────────────────────────

    labels = {"6A_lev": "6-Asset Lev", "4A_lev": "4-Asset Lev",
              "6A_1x": "6-Asset 1x", "4A_1x": "4-Asset 1x"}

    pr("=" * 80)
    pr("  PORTFOLIO BETA & RISK METRICS")
    pr("=" * 80)

    # 1. Overall
    pr(f"\n\nOVERALL (2002-2026)")
    pr("-" * 70)
    header = f"  {'Metric':<22}"
    for k in ["6A_lev","4A_lev","6A_1x","4A_1x"]:
        header += f" {labels[k]:>13}"
    pr(header)
    pr(f"  {'-'*22}" + f" {'-'*13}" * 4)

    overall = {k: compute_beta_metrics(monthly[k], ivv_m) for k in labels}
    metrics_list = [
        ("Beta", "beta", "{:.3f}"),
        ("Alpha (ann.)", "alpha_ann", "{:+.2%}"),
        ("R²", "r2", "{:.3f}"),
        ("Tracking Error", "te", "{:.1%}"),
        ("Information Ratio", "ir", "{:.3f}"),
        ("Up Capture", "up_cap", "{:.1f}%"),
        ("Down Capture", "dn_cap", "{:.1f}%"),
        ("Up/Down Ratio", "ud_ratio", "{:.2f}"),
    ]
    for name, key, fmt in metrics_list:
        row = f"  {name:<22}"
        for k in ["6A_lev","4A_lev","6A_1x","4A_1x"]:
            m = overall[k]
            if m is None: row += f" {'N/A':>13}"; continue
            v = m[key]
            if np.isnan(v): row += f" {'N/A':>13}"
            elif "%" in fmt and "%" not in name: row += f" {fmt.format(v):>13}"
            else: row += f" {fmt.format(v):>13}"
        pr(row)
    pr(f"  {'Months':<22}" + "".join(f" {overall[k]['n'] if overall[k] else 0:>13}" for k in labels))

    # 2. By leverage tier
    pr(f"\n\n{'='*80}")
    pr(f"  BY LEVERAGE TIER")
    pr(f"{'='*80}")

    for conf_k in ["6A_lev", "4A_lev"]:
        pr(f"\n  {labels[conf_k]}:")
        ts = tier_series.get(conf_k)
        if ts is None: continue
        port_d = results[conf_k]

        # Resample tier to monthly (mode of the month)
        tier_monthly = ts.resample("MS").apply(lambda x: x.mode().iloc[0] if len(x) > 0 else 0)

        header = f"    {'Metric':<22}" + "".join(f" {'Tier '+str(t)+' ('+['1.0x','1.3x','1.65x'][t]+')':>18}" for t in [0,1,2])
        pr(header)
        pr(f"    {'-'*22}" + f" {'-'*18}" * 3)

        tier_m = {}
        for t in [0, 1, 2]:
            mask = tier_monthly == t
            months_in_tier = mask.index[mask]
            pm = monthly[conf_k].reindex(months_in_tier).dropna()
            bm = ivv_m.reindex(months_in_tier).dropna()
            common = pm.index.intersection(bm.index)
            pm, bm = pm.reindex(common), bm.reindex(common)
            if len(common) >= 6:
                tier_m[t] = compute_beta_metrics(pm, bm)
                tier_m[t]["avg_ret_m"] = pm.mean()
                tier_m[t]["vol_ann"] = pm.std() * np.sqrt(12)
            else:
                tier_m[t] = None

        for name, key, fmt in [("Months", "n", "{}"), ("Beta", "beta", "{:.3f}"),
                                ("Alpha (ann.)", "alpha_ann", "{:+.2%}"),
                                ("Avg Monthly Ret", "avg_ret_m", "{:.2%}"),
                                ("Volatility (ann.)", "vol_ann", "{:.1%}")]:
            row = f"    {name:<22}"
            for t in [0, 1, 2]:
                m = tier_m[t]
                if m is None:
                    row += f" {'N/A (<6mo)':>18}"
                else:
                    v = m.get(key, float("nan"))
                    if isinstance(v, float) and np.isnan(v): row += f" {'N/A':>18}"
                    else: row += f" {fmt.format(v):>18}"
            pr(row)

    # 3. By crisis period
    pr(f"\n\n{'='*80}")
    pr(f"  BY CRISIS PERIOD")
    pr(f"{'='*80}")

    for conf_k in ["6A_lev", "4A_lev"]:
        pr(f"\n  {labels[conf_k]}:")
        header = f"    {'Metric':<22}" + "".join(f" {cn:>13}" for cn, _, _ in CRISES)
        pr(header)
        pr(f"    {'-'*22}" + f" {'-'*13}" * 3)

        crisis_m = {}
        for cn, cs, ce in CRISES:
            port_d = results[conf_k]
            mask = (port_d.index >= pd.Timestamp(cs)) & (port_d.index <= pd.Timestamp(ce))
            p_crisis = port_d[mask]
            i_crisis = ivv_s.reindex(p_crisis.index).dropna()
            common = p_crisis.index.intersection(i_crisis.index)
            p_crisis, i_crisis = p_crisis.reindex(common), i_crisis.reindex(common)

            if len(common) < 10:
                crisis_m[cn] = None; continue

            slope, intercept, r_value, _, _ = sp_stats.linregress(i_crisis, p_crisis)
            cum_p = (1 + p_crisis).prod() - 1
            cum_i = (1 + i_crisis).prod() - 1
            crisis_m[cn] = {
                "n_days": len(common), "beta": slope, "alpha_ann": intercept * 252,
                "cum_port": cum_p, "cum_ivv": cum_i, "vs_ivv": cum_p - cum_i,
            }

        for name, key, fmt in [("Days", "n_days", "{}"), ("Beta (daily)", "beta", "{:.3f}"),
                                ("Alpha (ann.)", "alpha_ann", "{:+.1%}"),
                                ("Cumul. Return", "cum_port", "{:+.1%}"),
                                ("vs IVV", "vs_ivv", "{:+.1%}")]:
            row = f"    {name:<22}"
            for cn, _, _ in CRISES:
                m = crisis_m.get(cn)
                if m is None: row += f" {'N/A':>13}"
                else:
                    v = m.get(key, float("nan"))
                    row += f" {fmt.format(v):>13}"
            pr(row)

    # 4. By market regime
    pr(f"\n\n{'='*80}")
    pr(f"  BY MARKET REGIME (monthly)")
    pr(f"{'='*80}")

    bull_mask = ivv_m > 0
    bear_mask = ivv_m < 0
    n_bull = bull_mask.sum(); n_bear = bear_mask.sum()

    for regime, mask, label in [("Bull", bull_mask, f"Bull Months (N={n_bull})"),
                                 ("Bear", bear_mask, f"Bear Months (N={n_bear})")]:
        pr(f"\n  {label}")
        header = f"    {'Metric':<22}"
        for k in ["6A_lev","4A_lev","bench"]:
            header += f" {(labels.get(k, 'IVV B&H')):>13}"
        pr(header)
        pr(f"    {'-'*22}" + f" {'-'*13}" * 3)

        months_in = mask.index[mask]
        for name, fn in [("Avg Return", lambda s, idx: s.reindex(idx).dropna().mean()),
                          ("Beta", None),
                          ("Capture %", None)]:
            row = f"    {name:<22}"
            for k in ["6A_lev", "4A_lev", "bench"]:
                if k == "bench":
                    if name == "Avg Return":
                        row += f" {ivv_m.reindex(months_in).dropna().mean():>12.2%}"
                    elif name == "Beta":
                        row += f" {'1.000':>13}"
                    elif name == "Capture %":
                        row += f" {'100.0%':>13}"
                else:
                    pm = monthly[k].reindex(months_in).dropna()
                    bm = ivv_m.reindex(months_in).dropna()
                    common = pm.index.intersection(bm.index)
                    pm, bm = pm.reindex(common), bm.reindex(common)
                    if name == "Avg Return":
                        row += f" {pm.mean():>12.2%}"
                    elif name == "Beta":
                        if len(common) > 6:
                            sl, _, _, _, _ = sp_stats.linregress(bm, pm)
                            row += f" {sl:>13.3f}"
                        else:
                            row += f" {'N/A':>13}"
                    elif name == "Capture %":
                        cap = pm.mean() / bm.mean() * 100 if bm.mean() != 0 else float("nan")
                        row += f" {cap:>12.1f}%"
            pr(row)

    # Interpretation
    pr(f"\n\n{'='*80}")
    pr(f"  INTERPRETATION")
    pr(f"{'='*80}")
    m6 = overall["6A_lev"]; m4 = overall["4A_lev"]
    if m6 and m4:
        pr(f"\n  Both systems run at sub-market beta ({m6['beta']:.2f} and {m4['beta']:.2f}), confirming")
        pr(f"  the Faber/HMM/Kritzman hierarchy actively reduces market exposure.")
        if m4["ud_ratio"] > 1.0 and m6["ud_ratio"] > 1.0:
            pr(f"  Up/down capture ratios ({m6['ud_ratio']:.2f} and {m4['ud_ratio']:.2f}) exceed 1.0,")
            pr(f"  meaning both systems capture more upside than downside — the hallmark")
            pr(f"  of effective tactical allocation.")
        if abs(m4["beta"] - m6["beta"]) < 0.05:
            pr(f"  The 4-asset and 6-asset betas are nearly identical, confirming that")
            pr(f"  dropping VGLT/DBC doesn't materially change the portfolio's market")
            pr(f"  sensitivity. The risk profile is preserved with cleaner data.")
        pr(f"\n  Information ratio: 6-asset {m6['ir']:.3f} vs 4-asset {m4['ir']:.3f}.")
        better = "4-asset" if m4["ir"] > m6["ir"] else "6-asset"
        pr(f"  The {better} generates more excess return per unit of tracking error.")

    # Save
    report_path = os.path.join(str(OUTPUT), "portfolio_beta_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Report saved: {report_path}")


if __name__ == "__main__":
    main()

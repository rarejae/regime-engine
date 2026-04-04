"""Weekly vs monthly rebalancing — 4-asset system comparison."""

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
TC = 0.0010
SMA_PERIODS = [6, 10, 12]
TIER_SUBS = {0: 0.0, 1: 0.30, 2: 0.65}
BASE = {"IVV": 0.45, "QQQ": 0.25, "IAU": 0.15, "cash": 0.15}
ASSETS = list(BASE.keys())
EQUITY = ["IVV", "QQQ"]


# ── Allocation steps ──────────────────────────────────────────────────────────

def compute_multi_sma(prices_df):
    monthly = prices_df.resample("MS").last()
    result = pd.DataFrame(0, index=monthly.index, columns=monthly.columns)
    for p in SMA_PERIODS:
        sma = monthly.rolling(p, min_periods=p).mean()
        result += (monthly > sma).astype(int)
    return result


def compute_weekly_faber(daily_prices, monthly_sma_dfs):
    """Check daily prices against monthly SMAs — return Faber score (0-3) per asset per day."""
    scores = pd.DataFrame(0, index=daily_prices.index, columns=daily_prices.columns)
    for sma_df in monthly_sma_dfs:
        # Forward-fill monthly SMA to daily frequency
        sma_daily = sma_df.reindex(daily_prices.index, method="ffill")
        scores += (daily_prices > sma_daily).astype(int)
    return scores


def step1_faber(strengths):
    w = {}; pool = 0.0
    for a in ASSETS:
        if a == "cash": w[a] = BASE[a]; continue
        s = strengths.get(a, 0)
        if s >= 3: w[a] = BASE[a]
        elif s == 2: w[a] = BASE[a] * 0.70; pool += BASE[a] * 0.30
        else: w[a] = 0.0; pool += BASE[a]
    return w, pool

def step2_harvey_cv(weights, pool, harvey_er, rvols, carry, value):
    if pool <= 0.001: return weights
    candidates = {}
    for a in ASSETS:
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


def full_allocation(strengths, harvey_er, rvols, carry, value, bp, tp, arz):
    """Run all 5 steps, return final weights."""
    w, pool = step1_faber(strengths)
    w = step2_harvey_cv(w, pool, harvey_er, rvols, carry, value)
    w = step3_hmm(w, bp)
    w = step4_kritzman(w, tp, arz)
    return step5_normalize(w)


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

    # Monthly SMAs (for monthly Faber) — shifted for PIT
    multi_sma_monthly = compute_multi_sma(prices_df).shift(1)

    # Monthly SMA DataFrames for weekly price checks
    monthly_prices = prices_df.resample("MS").last()
    sma_dfs = []
    for p in SMA_PERIODS:
        sma_dfs.append(monthly_prices.rolling(p, min_periods=p).mean().shift(1))  # PIT shifted

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
    fkey = os.environ.get("FRED_API_KEY")
    if fkey:
        tb = Fred(api_key=fkey).get_series("DTB3", observation_start="1998-01-01")
        tb.index = pd.to_datetime(tb.index)
        rfr_daily = (tb / 100 / 252).reindex(daily_ret.index, method="ffill").fillna(0)

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

    common_start = max(daily_ret.dropna(how="all").index.min(), hmm_pred.index.min(), pd.Timestamp("2002-01-01"))
    trading_days = daily_ret.loc[common_start:].index

    # ── Run both monthly and weekly versions ──────────────────────────────────

    # Shared monthly-updated signals
    harvey_er = {a: 0.0 for a in ASSETS}
    rvols = {a: 0.15 for a in ASSETS if a != "cash"}
    carry = {}; value = {}
    bp = 0.5; tp = 0.5; arz = 0.0

    # Monthly state
    m_strengths = {a: 3 for a in ASSETS if a != "cash"}
    m_w = dict(BASE); m_tier = 0; m_delevered = False
    m_tier_hist = {"IVV": list(pre_ers["IVV"]), "QQQ": list(pre_ers["QQQ"])}
    m_tier_med = dict(SEED)
    m_rets = {}; m_tc_total = 0.0; m_prev_w = None
    m_weight_snaps = []  # (date, weights, tier)
    m_delever_events = []

    # Weekly state
    w_strengths = {a: 3 for a in ASSETS if a != "cash"}
    w_w = dict(BASE); w_tier = 0; w_delevered_this_month = False
    w_tier_hist = {"IVV": list(pre_ers["IVV"]), "QQQ": list(pre_ers["QQQ"])}
    w_tier_med = dict(SEED)
    w_rets = {}; w_tc_total = 0.0; w_prev_w = None
    w_weight_snaps = []
    w_rebal_count = 0
    w_delever_events = []

    # IVV benchmark
    ivv_rets = {}

    for day_idx, day in enumerate(trading_days):
        if day not in daily_ret.index: continue
        dr = daily_ret.loc[day]
        avail = [a for a in ASSETS if a in dr.index and pd.notna(dr[a])]
        if len(avail) < 2: continue
        actual = {a: float(dr[a]) for a in avail}
        rfr = float(rfr_daily.get(day, 0))

        if "IVV" in actual:
            ivv_rets[day] = actual["IVV"]

        is_ms = (day_idx == 0 or day.month != trading_days[day_idx - 1].month)
        is_friday = day.weekday() == 4

        # ── Update monthly signals (shared) ───────────────────────────────────
        if is_ms:
            m_delevered = False
            w_delevered_this_month = False

            # Monthly Faber
            ms_cands = multi_sma_monthly.index[multi_sma_monthly.index <= day]
            if len(ms_cands) > 0:
                ms_dt = ms_cands[-1]
                for a in ASSETS:
                    if a != "cash" and a in multi_sma_monthly.columns:
                        v = multi_sma_monthly.loc[ms_dt, a]
                        m_strengths[a] = int(v) if pd.notna(v) else 0

            # Harvey
            z_cands = z_data_lagged.index[z_data_lagged.index < day]
            if len(z_cands) > 0:
                try:
                    sim = compute_distances(z_data_lagged, z_cands[-1], config)
                    for a in avail:
                        if a not in asset_ret_fwd.columns: harvey_er[a]=0; continue
                        r = [asset_ret_fwd.loc[d,a] for d in sim.similar_dates
                             if d in asset_ret_fwd.index and pd.notna(asset_ret_fwd.loc[d,a])]
                        harvey_er[a] = np.mean(r) if r else 0
                except ValueError: pass

            # Rvols
            rv_c = rvol_monthly.index[rvol_monthly.index <= day]
            if len(rv_c) > 0:
                for a in ASSETS:
                    if a != "cash" and a in rvol_monthly.columns:
                        v = rvol_monthly.loc[rv_c[-1], a]
                        rvols[a] = float(v) if pd.notna(v) and v > 0 else 0.15

            # Carry/value
            carry = {}; value = {}
            for a in ASSETS:
                if a in carry_df.columns:
                    cc = carry_df.index[carry_df.index <= day]
                    if len(cc) > 0: cv = carry_df.loc[cc[-1], a]; carry[a] = float(cv) if pd.notna(cv) else 0
                if a in value_df.columns:
                    vc = value_df.index[value_df.index <= day]
                    if len(vc) > 0: vv = value_df.loc[vc[-1], a]; value[a] = float(vv) if pd.notna(vv) else 0

            # HMM
            prev_m = day - pd.DateOffset(months=1)
            hm = hmm_pred[(hmm_pred.index >= prev_m) & (hmm_pred.index < day)]
            bp = hm["bull_prob"].mean() if len(hm) > 0 else 0.5
            if pd.isna(bp): bp = 0.5
            tp = turb_m.get(day, 0.5) if day in turb_m.index else 0.5
            if pd.isna(tp): tp = 0.5
            arz = ar_z_m.get(day, 0) if day in ar_z_m.index else 0
            if pd.isna(arz): arz = 0

            # ── Monthly rebalance ─────────────────────────────────────────────
            m_w = full_allocation(m_strengths, harvey_er, rvols, carry, value, bp, tp, arz)

            # Monthly tier
            f_c = m_strengths.get("IVV",0)>=3 and m_strengths.get("QQQ",0)>=3
            h_c = harvey_er.get("IVV",0)>0 and harvey_er.get("QQQ",0)>0
            if f_c and h_c:
                above = all(harvey_er.get(a,0) > m_tier_med.get(a,0) for a in ["IVV","QQQ"])
                m_tier = 2 if above else 1
                for a in ["IVV","QQQ"]:
                    m_tier_hist[a].append(harvey_er.get(a,0))
                    m_tier_med[a] = float(np.median(m_tier_hist[a]))
            else: m_tier = 0

            # TC for monthly
            if m_prev_w is not None:
                to = sum(abs(m_w.get(a,0) - m_prev_w.get(a,0)) for a in ASSETS) / 2
                m_tc_total += to * TC
            m_prev_w = dict(m_w)
            m_weight_snaps.append((day, dict(m_w), m_tier))

        # ── Weekly rebalance (every Friday) ───────────────────────────────────
        if is_friday or is_ms:
            # Compute weekly Faber: check current daily price vs monthly SMAs
            for a in ASSETS:
                if a == "cash": continue
                if a not in prices_df.columns: continue
                p_before = prices_df.loc[:day, a]
                if len(p_before) == 0: continue
                current_price = p_before.iloc[-1]
                score = 0
                for sma_df in sma_dfs:
                    sma_dates = sma_df.index[sma_df.index <= pd.Timestamp(f"{day.year}-{day.month:02d}-01")]
                    if len(sma_dates) == 0: continue
                    sv = sma_df.loc[sma_dates[-1], a] if a in sma_df.columns else np.nan
                    if pd.notna(sv) and current_price > sv:
                        score += 1
                w_strengths[a] = score

            w_w_new = full_allocation(w_strengths, harvey_er, rvols, carry, value, bp, tp, arz)

            # Weekly tier
            f_c = w_strengths.get("IVV",0)>=3 and w_strengths.get("QQQ",0)>=3
            h_c = harvey_er.get("IVV",0)>0 and harvey_er.get("QQQ",0)>0
            if f_c and h_c:
                above = all(harvey_er.get(a,0) > w_tier_med.get(a,0) for a in ["IVV","QQQ"])
                w_tier = 2 if above else 1
                for a in ["IVV","QQQ"]:
                    w_tier_hist[a].append(harvey_er.get(a,0))
                    w_tier_med[a] = float(np.median(w_tier_hist[a]))
            else: w_tier = 0

            # TC for weekly
            if w_prev_w is not None:
                to = sum(abs(w_w_new.get(a,0) - w_prev_w.get(a,0)) for a in ASSETS) / 2
                w_tc_total += to * TC
            w_prev_w = dict(w_w_new)
            w_w = w_w_new
            w_rebal_count += 1
            w_weight_snaps.append((day, dict(w_w), w_tier))

        # ── Monthly circuit breaker ───────────────────────────────────────────
        if is_friday and m_tier > 0 and not m_delevered:
            breach = False
            for etf in ["IVV","QQQ"]:
                if etf not in prices_df.columns: continue
                pb = prices_df.loc[:day, etf]
                if len(pb)==0: continue
                pt = pb.iloc[-1]; lms = pd.Timestamp(f"{day.year}-{day.month:02d}-01")
                for sdf in sma_dfs:
                    sd = sdf.index[sdf.index <= lms]
                    if len(sd)==0: continue
                    sv = sdf.loc[sd[-1], etf] if etf in sdf.columns else np.nan
                    if pd.notna(sv) and pt < sv: breach = True; break
                if breach: break
            if breach:
                m_delever_events.append(day)
                m_tier = 0; m_delevered = True

        # ── Compute daily returns ─────────────────────────────────────────────
        for tag, wp, tier_val in [("m", m_w, m_tier), ("w", w_w, w_tier)]:
            iw = wp.get("IVV",0); qw = wp.get("QQQ",0)
            ir = actual.get("IVV",0); qr = actual.get("QQQ",0)
            base_r = sum(wp.get(a,0)*actual.get(a,0) for a in avail if a not in ["IVV","QQQ"])
            sub = TIER_SUBS.get(tier_val, 0)
            if sub > 0:
                sr = 2*ir - rfr - 0.0091/252; ql = 2*qr - rfr - 0.0089/252
                ret = iw*(1-sub)*ir + iw*sub*sr + qw*(1-sub)*qr + qw*sub*ql + base_r
            else:
                ret = iw*ir + qw*qr + base_r
            if tag == "m": m_rets[day] = ret
            else: w_rets[day] = ret

    # Convert
    ms = pd.Series(m_rets).sort_index()
    ws = pd.Series(w_rets).sort_index()
    ivv_s = pd.Series(ivv_rets).sort_index()
    n_years = len(trading_days) / 252

    m_snap_df = pd.DataFrame([(d, ww.get("IVV",0)+ww.get("QQQ",0), ww.get("IVV",0), ww.get("QQQ",0), ww.get("IAU",0), ww.get("cash",0), t) for d, ww, t in m_weight_snaps],
                              columns=["date","equity","IVV","QQQ","IAU","cash","tier"]).set_index("date")
    w_snap_df = pd.DataFrame([(d, ww.get("IVV",0)+ww.get("QQQ",0), ww.get("IVV",0), ww.get("QQQ",0), ww.get("IAU",0), ww.get("cash",0), t) for d, ww, t in w_weight_snaps],
                              columns=["date","equity","IVV","QQQ","IAU","cash","tier"]).set_index("date")

    # ── Report ────────────────────────────────────────────────────────────────

    def perf(s):
        ar = s.mean()*252; av = s.std()*np.sqrt(252)
        sh = ar/av if av>0 else 0
        neg = s[s<0]; ds = neg.std()*np.sqrt(252) if len(neg)>10 else av
        so = ar/ds if ds>0 else 0
        cum = (1+s).cumprod(); dd = ((cum-cum.expanding().max())/cum.expanding().max()).min()
        final = cum.iloc[-1]
        return ar, av, sh, so, dd, final

    pr("=" * 80)
    pr("  WEEKLY vs MONTHLY REBALANCING — 4-ASSET SYSTEM")
    pr("=" * 80)

    pm = perf(ms); pw = perf(ws)

    pr(f"\n\nPERFORMANCE COMPARISON (2002-2026)")
    pr("-" * 60)
    pr(f"  {'Metric':<22} {'Monthly':>12} {'Weekly':>12} {'Delta':>12}")
    pr(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*12}")
    for name, mi, wi in [("Ann. Return", pm[0], pw[0]), ("Volatility", pm[1], pw[1]),
                          ("Sharpe", pm[2], pw[2]), ("Sortino", pm[3], pw[3]),
                          ("Max Drawdown", pm[4], pw[4])]:
        if name in ("Sharpe", "Sortino"):
            pr(f"  {name:<22} {mi:>12.3f} {wi:>12.3f} {wi-mi:>+12.3f}")
        else:
            pr(f"  {name:<22} {mi:>11.1%} {wi:>11.1%} {wi-mi:>+11.1%}")
    pr(f"  {'Terminal $1':<22} ${pm[5]:>11.2f} ${pw[5]:>11.2f} ${pw[5]-pm[5]:>+11.2f}")

    # Turnover
    pr(f"\n\nTURNOVER ANALYSIS")
    pr("-" * 60)
    m_monthly_to = []; m_prev = None
    for _, ww, _ in m_weight_snaps:
        if m_prev is not None:
            m_monthly_to.append(sum(abs(ww.get(a,0) - m_prev.get(a,0)) for a in ASSETS) / 2)
        m_prev = ww
    w_weekly_to = []; w_prev = None
    for _, ww, _ in w_weight_snaps:
        if w_prev is not None:
            w_weekly_to.append(sum(abs(ww.get(a,0) - w_prev.get(a,0)) for a in ASSETS) / 2)
        w_prev = ww

    m_avg_to = np.mean(m_monthly_to) if m_monthly_to else 0
    w_avg_to_per_rebal = np.mean(w_weekly_to) if w_weekly_to else 0
    w_avg_to_monthly = w_avg_to_per_rebal * (w_rebal_count / len(m_weight_snaps)) if len(m_weight_snaps) > 0 else 0

    m_tc_ann = m_tc_total / n_years
    w_tc_ann = w_tc_total / n_years

    pr(f"  {'Metric':<28} {'Monthly':>12} {'Weekly':>12}")
    pr(f"  {'-'*28} {'-'*12} {'-'*12}")
    pr(f"  {'Total rebalances':<28} {len(m_weight_snaps):>12} {w_rebal_count:>12}")
    pr(f"  {'Avg turnover per rebalance':<28} {m_avg_to:>11.1%} {w_avg_to_per_rebal:>11.1%}")
    pr(f"  {'Avg turnover per month':<28} {m_avg_to:>11.1%} {w_avg_to_monthly:>11.1%}")
    pr(f"  {'Est. TC (ann.)':<28} {m_tc_ann:>11.2%} {w_tc_ann:>11.2%}")
    pr(f"  {'Net return (after TC)':<28} {pm[0]-m_tc_ann:>11.1%} {pw[0]-w_tc_ann:>11.1%}")

    # Allocation smoothness
    pr(f"\n\nALLOCATION SMOOTHNESS")
    pr("-" * 60)
    m_eq = m_snap_df["equity"]
    w_eq_monthly = w_snap_df["equity"].resample("MS").last()
    m_deq = m_eq.diff().dropna()
    w_deq_all = w_snap_df["equity"].diff().dropna()

    pr(f"  {'Metric':<32} {'Monthly':>12} {'Weekly':>12}")
    pr(f"  {'-'*32} {'-'*12} {'-'*12}")
    pr(f"  {'Avg equity weight':<32} {m_eq.mean():>11.0%} {w_snap_df['equity'].mean():>11.0%}")
    pr(f"  {'Std dev of Δequity (per rebal)':<32} {m_deq.std():>11.1%} {w_deq_all.std():>11.1%}")
    pr(f"  {'Max single Δequity':<32} {m_deq.abs().max():>11.0%} {w_deq_all.abs().max():>11.0%}")
    pr(f"  {'Swings >30% equity':<32} {(m_deq.abs()>0.30).sum():>12} {(w_deq_all.abs()>0.30).sum():>12}")
    pr(f"  {'Swings >50% equity':<32} {(m_deq.abs()>0.50).sum():>12} {(w_deq_all.abs()>0.50).sum():>12}")

    # Crisis response
    pr(f"\n\n{'='*80}")
    pr(f"  CRISIS RESPONSE")
    pr(f"{'='*80}")
    crises = [("GFC", "2008-09-01", "2009-03-31", "2008-01-01", "2009-12-31"),
              ("COVID", "2020-02-19", "2020-03-23", "2020-01-01", "2020-06-30"),
              ("2022 Bear", "2022-01-03", "2022-10-31", "2021-07-01", "2023-03-31")]

    for cname, cs, ce, ws_start, ws_end in crises:
        pr(f"\n  {cname}:")
        pr(f"  {'Metric':<28} {'Monthly':>14} {'Weekly':>14}")
        pr(f"  {'-'*28} {'-'*14} {'-'*14}")

        # Pre-crisis equity
        cs_ts = pd.Timestamp(cs)
        for tag, snap in [("m", m_snap_df), ("w", w_snap_df)]:
            pre = snap[snap.index < cs_ts]
            if len(pre) == 0: continue

        m_pre = m_snap_df[m_snap_df.index < cs_ts]["equity"].iloc[-1] if len(m_snap_df[m_snap_df.index < cs_ts]) > 0 else float("nan")
        w_pre = w_snap_df[w_snap_df.index < cs_ts]["equity"].iloc[-1] if len(w_snap_df[w_snap_df.index < cs_ts]) > 0 else float("nan")

        # During crisis
        m_during = m_snap_df[(m_snap_df.index >= pd.Timestamp(ws_start)) & (m_snap_df.index <= pd.Timestamp(ws_end))]
        w_during = w_snap_df[(w_snap_df.index >= pd.Timestamp(ws_start)) & (w_snap_df.index <= pd.Timestamp(ws_end))]

        m_min_eq = m_during["equity"].min() if len(m_during) > 0 else float("nan")
        w_min_eq = w_during["equity"].min() if len(w_during) > 0 else float("nan")
        m_min_dt = m_during["equity"].idxmin().strftime("%Y-%m-%d") if len(m_during) > 0 else "N/A"
        w_min_dt = w_during["equity"].idxmin().strftime("%Y-%m-%d") if len(w_during) > 0 else "N/A"

        # Drawdown
        m_crisis = ms[(ms.index >= pd.Timestamp(cs)) & (ms.index <= pd.Timestamp(ce))]
        w_crisis = ws[(ws.index >= pd.Timestamp(cs)) & (ws.index <= pd.Timestamp(ce))]
        m_dd = ((1+m_crisis).cumprod().pipe(lambda c: (c-c.expanding().max())/c.expanding().max())).min() if len(m_crisis) > 0 else float("nan")
        w_dd = ((1+w_crisis).cumprod().pipe(lambda c: (c-c.expanding().max())/c.expanding().max())).min() if len(w_crisis) > 0 else float("nan")

        pr(f"  {'Pre-crisis equity':<28} {m_pre:>13.0%} {w_pre:>13.0%}")
        pr(f"  {'Min equity during':<28} {m_min_eq:>13.0%} {w_min_eq:>13.0%}")
        pr(f"  {'Date of min equity':<28} {m_min_dt:>14} {w_min_dt:>14}")
        pr(f"  {'Drawdown':<28} {m_dd:>13.1%} {w_dd:>13.1%}")

    # COVID zoom
    pr(f"\n\n{'='*80}")
    pr(f"  ALLOCATION TIMELINE — COVID ZOOM (2020-01 to 2020-06)")
    pr(f"{'='*80}")
    pr(f"\n  {'Date':>12} {'M-Equity':>10} {'W-Equity':>10} {'Delta':>8} {'M-Cash':>8} {'W-Cash':>8}")
    pr(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")

    covid_start = pd.Timestamp("2020-01-01")
    covid_end = pd.Timestamp("2020-06-30")
    w_covid = w_snap_df[(w_snap_df.index >= covid_start) & (w_snap_df.index <= covid_end)]
    # For monthly, resample to show at same frequency
    m_covid = m_snap_df[(m_snap_df.index >= covid_start) & (m_snap_df.index <= covid_end)]

    # Merge on weekly dates
    for dt in w_covid.index:
        w_eq = w_covid.loc[dt, "equity"]
        w_ca = w_covid.loc[dt, "cash"]
        # Find most recent monthly snap
        m_before = m_snap_df[m_snap_df.index <= dt]
        if len(m_before) == 0: continue
        m_eq = m_before["equity"].iloc[-1]
        m_ca = m_before["cash"].iloc[-1]
        delta = w_eq - m_eq
        note = ""
        if dt.strftime("%Y-%m-%d") == "2020-03-20": note = "  ← SPY bottom"
        pr(f"  {dt.strftime('%Y-%m-%d'):>12} {m_eq:>9.0%} {w_eq:>9.0%} {delta:>+7.0%} {m_ca:>7.0%} {w_ca:>7.0%}{note}")

    # Leverage tiers
    pr(f"\n\n{'='*80}")
    pr(f"  LEVERAGE TIER COMPARISON")
    pr(f"{'='*80}")
    m_tiers = m_snap_df["tier"]
    w_tiers = w_snap_df["tier"]
    pr(f"\n  {'Metric':<22} {'Monthly':>12} {'Weekly':>12}")
    pr(f"  {'-'*22} {'-'*12} {'-'*12}")
    for t in [0, 1, 2]:
        mp = (m_tiers == t).mean() * 100
        wp = (w_tiers == t).mean() * 100
        lev = {0:"1.00x", 1:"1.30x", 2:"1.65x"}[t]
        pr(f"  {'Tier '+str(t)+' ('+lev+')':<22} {mp:>11.1f}% {wp:>11.1f}%")
    pr(f"  {'De-lever events':<22} {len(m_delever_events):>12} {'N/A (built-in)':>12}")

    # Recommendation
    pr(f"\n\n{'='*80}")
    pr(f"  RECOMMENDATION")
    pr(f"{'='*80}")
    sh_diff = pw[2] - pm[2]
    dd_diff = pw[4] - pm[4]
    tc_diff = w_tc_ann - m_tc_ann

    if sh_diff > 0.02 and dd_diff > 0:
        pr(f"\n  → SWITCH TO WEEKLY REBALANCING")
        pr(f"  Sharpe improvement: {sh_diff:+.3f}")
        pr(f"  Drawdown improvement: {dd_diff:+.1%}")
        pr(f"  Additional TC cost: {tc_diff:+.2%} ann. — offset by better risk-adjusted returns.")
    elif sh_diff > -0.02 and dd_diff > 0.01:
        pr(f"\n  → ADOPT WEEKLY REBALANCING (modest improvement)")
        pr(f"  Sharpe delta: {sh_diff:+.3f}, Drawdown delta: {dd_diff:+.1%}")
        pr(f"  Main benefit is smoother allocation transitions and faster crisis response,")
        pr(f"  not raw performance. TC cost increase of {tc_diff:+.2%} ann. is manageable.")
    elif sh_diff < -0.03:
        pr(f"\n  → KEEP MONTHLY REBALANCING")
        pr(f"  Weekly costs {-sh_diff:.3f} Sharpe from whipsaw/over-trading.")
        pr(f"  The monthly Faber filter with weekly circuit breaker is already a good hybrid.")
    else:
        pr(f"\n  → KEEP MONTHLY (no clear advantage to weekly)")
        pr(f"  Sharpe delta: {sh_diff:+.3f}, Drawdown delta: {dd_diff:+.1%}")
        pr(f"  Insufficient improvement to justify additional complexity and turnover.")
        pr(f"  The current monthly rebalancing + weekly circuit breaker is already a")
        pr(f"  hybrid approach that captures most of the benefit.")

    # Save
    rp = os.path.join(str(OUTPUT), "weekly_vs_monthly_report.txt")
    with open(rp, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Report saved: {rp}")


if __name__ == "__main__":
    main()

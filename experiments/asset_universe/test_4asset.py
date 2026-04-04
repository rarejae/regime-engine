"""4-asset universe test: IVV, QQQ, IAU, Cash (validated proxies only).

Runs both 6-asset (current) and 4-asset side by side, including leverage tiers
and weekly circuit breaker, to produce direct performance comparison.
"""

import logging, sys, os, warnings
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

OUTPUT = Path(__file__).resolve().parent / "output"; OUTPUT.mkdir(parents=True, exist_ok=True)
TC = 0.0010
SMA_PERIODS = [6, 10, 12]
TIER_SUBS = {0: 0.0, 1: 0.30, 2: 0.65}

# ── Universe definitions ──────────────────────────────────────────────────────

BASE_6 = {"IVV": 0.45, "QQQ": 0.25, "VGLT": 0.05, "IAU": 0.10, "DBC": 0.05, "cash": 0.10}
BASE_4 = {"IVV": 0.45, "QQQ": 0.25, "IAU": 0.15, "cash": 0.15}
EQUITY = ["IVV", "QQQ"]


# ── Allocation steps (universe-agnostic) ──────────────────────────────────────

def compute_multi_sma(prices_df):
    monthly = prices_df.resample("MS").last()
    result = pd.DataFrame(0, index=monthly.index, columns=monthly.columns)
    for period in SMA_PERIODS:
        sma = monthly.rolling(period, min_periods=period).mean()
        result += (monthly > sma).astype(int)
    return result


def step1_faber(strengths, baseline, assets):
    w = {}
    pool = 0.0
    for a in assets:
        if a == "cash":
            w[a] = baseline[a]
            continue
        s = strengths.get(a, 0)
        if s >= 3:
            w[a] = baseline[a]
        elif s == 2:
            w[a] = baseline[a] * 0.70
            pool += baseline[a] * 0.30
        else:
            w[a] = 0.0
            pool += baseline[a]
    return w, pool


def step2_harvey_cv(weights, pool, harvey_er, rvols, carry, value, assets, mode="full"):
    if pool <= 0.001:
        return weights
    candidates = {}
    for a in assets:
        if a == "cash" or weights.get(a, 0) <= 0:
            continue
        er = harvey_er.get(a, 0)
        vol = rvols.get(a, 0.15)
        c = carry.get(a, 0) if mode in ("full", "carry_only") else 0
        v = value.get(a, 0) if mode in ("full", "value_only") else 0
        adjusted_er = er + c
        if adjusted_er <= 0:
            continue
        score = adjusted_er / max(vol, 0.01) + 0.01 * v
        if score > 0:
            candidates[a] = score
    if not candidates:
        weights["cash"] = weights.get("cash", 0) + pool
        return weights
    if len(candidates) == 1:
        a = list(candidates.keys())[0]
        alloc = min(pool, 0.40 - weights.get(a, 0))
        alloc = max(alloc, 0)
        weights[a] = weights.get(a, 0) + alloc
        weights["cash"] = weights.get("cash", 0) + (pool - alloc)
        return weights
    total_score = sum(candidates.values())
    for a, score in candidates.items():
        weights[a] = weights.get(a, 0) + pool * (score / total_score)
    return weights


def step3_hmm(weights, bull_prob):
    if bull_prob > 0.7:
        for a in EQUITY:
            if weights.get(a, 0) > 0:
                boost = weights[a] * 0.15
                weights[a] += boost
                weights["cash"] = max(weights.get("cash", 0) - boost, 0)
    elif bull_prob < 0.3:
        for a in EQUITY:
            if weights.get(a, 0) > 0:
                red = weights[a] * 0.15
                weights[a] -= red
                weights["cash"] = weights.get("cash", 0) + red
    return weights


def step4_kritzman(weights, tp, arz):
    if tp > 0.95 and arz > 2.0:
        for a in EQUITY:
            if weights.get(a, 0) > 0:
                freed = weights[a] * 0.50
                weights[a] -= freed
                weights["cash"] = weights.get("cash", 0) + freed
    return weights


def step5_normalize(weights, assets):
    w = dict(weights)
    risky = [a for a in w if a != "cash" and w.get(a, 0) > 0.01]
    if len(risky) == 1 and w[risky[0]] > 0.40:
        excess = w[risky[0]] - 0.40
        w[risky[0]] = 0.40
        w["cash"] = w.get("cash", 0) + excess
    for _ in range(10):
        total = sum(max(v, 0) for v in w.values())
        if total > 0 and abs(total - 1.0) > 1e-6:
            w = {a: max(v, 0) / total for a, v in w.items()}
        changed = False
        for a in w:
            if w[a] > 0.60:
                w["cash"] = w.get("cash", 0) + w[a] - 0.60
                w[a] = 0.60
                changed = True
        eq = sum(w.get(a, 0) for a in EQUITY)
        if eq > 0.85:
            r = 0.85 / eq
            for a in EQUITY:
                freed = w[a] - w[a] * r
                w[a] *= r
                w["cash"] = w.get("cash", 0) + freed
            changed = True
        if w.get("cash", 0) < 0.03:
            deficit = 0.03 - w["cash"]
            others = [a for a in w if a != "cash" and w[a] > 0]
            tot = sum(w[a] for a in others)
            if tot > deficit:
                for a in others:
                    w[a] -= deficit * (w[a] / tot)
            w["cash"] = 0.03
            changed = True
        if not changed:
            break
    total = sum(max(v, 0) for v in w.values())
    if total > 0 and abs(total - 1.0) > 1e-6:
        w = {a: max(v, 0) / total for a, v in w.items()}
    return w


# ── Data loading ──────────────────────────────────────────────────────────────

def main():
    lines = []
    def pr(s=""):
        print(s); lines.append(s)

    config = RegimeConfig()
    raw_macro = fetch_monthly_history(config)
    transformed = transform_variables(raw_macro, config)
    z_data = get_valid_zscored(transformed, config)
    z_data_lagged = z_data.shift(1).dropna()

    asset_ret = pd.read_parquet("data/macro/roth_asset_returns.parquet")
    asset_ret_fwd = asset_ret.shift(-1)

    print("Loading daily ETF data...")
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
    daily_rets_raw = prices_df.pct_change()
    rvol_63 = daily_rets_raw.rolling(63, min_periods=30).std() * np.sqrt(252)
    rvol_monthly = rvol_63.resample("MS").last().shift(1)

    print("Computing carry and value signals...")
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
    ar = compute_absorption_ratio(basket, n_components=2)
    ar_z_s = compute_ar_zscore(ar)
    turb_m = turb_pctl_s.resample("MS").last().shift(1)
    ar_z_m = ar_z_s.resample("MS").last().shift(1)

    # Risk-free rate for leverage
    from fredapi import Fred
    rfr_daily = pd.Series(0.0, index=daily_ret.index)
    key = os.environ.get("FRED_API_KEY")
    if key:
        tb = Fred(api_key=key).get_series("DTB3", observation_start="1998-01-01")
        tb.index = pd.to_datetime(tb.index)
        rfr_daily = (tb / 100 / 252).reindex(daily_ret.index, method="ffill").fillna(0)

    # Daily prices for weekly SMA checks
    daily_prices_df = pd.DataFrame({a: ep[a] for a in ["IVV","QQQ"] if a in ep})
    monthly_close = daily_prices_df.resample("MS").last()
    sma_6m = monthly_close.rolling(6, min_periods=6).mean()
    sma_10m = monthly_close.rolling(10, min_periods=10).mean()
    sma_12m = monthly_close.rolling(12, min_periods=12).mean()

    # Precompute expanding medians for leverage tiers (both universes use same seed)
    # Seed from pre-backtest ER
    pre_start = pd.Timestamp("2002-01-01")
    pre_months = z_data_lagged.index[z_data_lagged.index < pre_start]
    pre_ers = {"IVV": [], "QQQ": []}
    for z_dt in pre_months:
        try:
            sim = compute_distances(z_data_lagged, z_dt, config)
            for a in ["IVV", "QQQ"]:
                if a in asset_ret_fwd.columns:
                    rets = [asset_ret_fwd.loc[d,a] for d in sim.similar_dates
                            if d in asset_ret_fwd.index and pd.notna(asset_ret_fwd.loc[d,a])]
                    if rets:
                        pre_ers[a].append(np.mean(rets))
        except ValueError:
            pass
    SEED = {a: float(np.mean(pre_ers[a])) if pre_ers[a] else 0.005 for a in ["IVV", "QQQ"]}

    common_start = max(daily_ret.dropna(how="all").index.min(), hmm_pred.index.min(), pd.Timestamp("2002-01-01"))
    trading_days = daily_ret.loc[common_start:].index

    pr("=" * 80)
    pr("  4-ASSET UNIVERSE TEST")
    pr("=" * 80)
    pr(f"\nBacktest: {len(trading_days)} days ({common_start.date()} to {trading_days.max().date()})")
    pr(f"\nBASELINE WEIGHTS")
    pr("-" * 40)
    pr(f"  6-Asset (current): " + ", ".join(f"{a} {w:.0%}" for a, w in BASE_6.items()))
    pr(f"  4-Asset (new):     " + ", ".join(f"{a} {w:.0%}" for a, w in BASE_4.items()))

    # ── Run both universes ────────────────────────────────────────────────────

    configs = {
        "6A_lev": {"base": BASE_6, "assets": list(BASE_6.keys()), "leverage": True},
        "4A_lev": {"base": BASE_4, "assets": list(BASE_4.keys()), "leverage": True},
        "6A_1x":  {"base": BASE_6, "assets": list(BASE_6.keys()), "leverage": False},
        "4A_1x":  {"base": BASE_4, "assets": list(BASE_4.keys()), "leverage": False},
        "bench_ivv": {"bench": True},
        "bench_6040": {"bench_6040": True},
    }

    strats = {k: {} for k in configs}
    current_weights = {k: None for k in configs}
    total_tc_map = {k: 0.0 for k in configs}

    # Per-config state
    state = {}
    for k, cfg in configs.items():
        if cfg.get("bench") or cfg.get("bench_6040"):
            continue
        state[k] = {
            "strengths": {a: 3 for a in cfg["assets"] if a != "cash"},
            "harvey_er": {a: 0.0 for a in cfg["assets"]},
            "w": dict(cfg["base"]),
            "tier": 0,
            "delevered": False,
            "tier_hist": {"IVV": list(pre_ers["IVV"]), "QQQ": list(pre_ers["QQQ"])},
            "tier_med": dict(SEED),
            "delever_events": [],
        }

    # Tracking for era attribution
    era_data = {"6A_lev": [], "4A_lev": []}
    etf_starts = {
        "IVV": pd.Timestamp("2000-05-01"), "QQQ": pd.Timestamp("1999-03-01"),
        "VXUS": pd.Timestamp("2001-08-01"), "VGLT": pd.Timestamp("2009-11-01"),
        "IAU": pd.Timestamp("2004-11-01"), "DBC": pd.Timestamp("2006-02-01"),
        "VNQ": pd.Timestamp("2004-09-01"),
    }

    # Tier frequency tracking
    tier_counts = {k: {0: 0, 1: 0, 2: 0} for k in configs if "lev" in k}
    total_lev_days = 0

    for day_idx, day in enumerate(trading_days):
        if day not in daily_ret.index:
            continue
        dr = daily_ret.loc[day]
        is_ms = (day_idx == 0 or day.month != trading_days[day_idx - 1].month)
        is_friday = day.weekday() == 4
        rfr = float(rfr_daily.get(day, 0))

        for k, cfg in configs.items():
            if cfg.get("bench"):
                avail = [a for a in ["IVV"] if a in dr.index and pd.notna(dr[a])]
                if avail:
                    strats[k][day] = float(dr["IVV"])
                continue
            if cfg.get("bench_6040"):
                avail = ["IVV", "VGLT"]
                r = 0.0
                for a, w in [("IVV", 0.6), ("VGLT", 0.4)]:
                    if a in dr.index and pd.notna(dr[a]):
                        r += w * float(dr[a])
                strats[k][day] = r
                continue

            assets = cfg["assets"]
            base = cfg["base"]
            use_leverage = cfg["leverage"]
            st = state[k]

            avail = [a for a in assets if a in dr.index and pd.notna(dr[a])]
            if len(avail) < 2:
                continue
            actual = {a: float(dr[a]) for a in avail}

            if is_ms:
                st["delevered"] = False

                # Update SMA strengths
                ms_cands = multi_sma_df.index[multi_sma_df.index <= day]
                if len(ms_cands) > 0:
                    ms_dt = ms_cands[-1]
                    for a in assets:
                        if a == "cash": continue
                        if a in multi_sma_df.columns:
                            v = multi_sma_df.loc[ms_dt, a]
                            st["strengths"][a] = int(v) if pd.notna(v) else 0

                # Harvey expected returns
                z_cands = z_data_lagged.index[z_data_lagged.index < day]
                sim_dates = []
                if len(z_cands) > 0:
                    z_dt = z_cands[-1]
                    try:
                        sim = compute_distances(z_data_lagged, z_dt, config)
                        sim_dates = sim.similar_dates
                        for a in avail:
                            if a not in asset_ret_fwd.columns:
                                st["harvey_er"][a] = 0
                                continue
                            rets = [asset_ret_fwd.loc[d,a] for d in sim.similar_dates
                                    if d in asset_ret_fwd.index and pd.notna(asset_ret_fwd.loc[d,a])]
                            st["harvey_er"][a] = np.mean(rets) if rets else 0
                    except ValueError:
                        pass

                # Realized vols
                rv_cands = rvol_monthly.index[rvol_monthly.index <= day]
                rvols = {}
                if len(rv_cands) > 0:
                    for a in assets:
                        if a == "cash": continue
                        if a in rvol_monthly.columns:
                            v = rvol_monthly.loc[rv_cands[-1], a]
                            rvols[a] = float(v) if pd.notna(v) and v > 0 else 0.15

                # Carry and value
                carry = {}; value = {}
                for a in assets:
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

                # HMM
                prev_m = day - pd.DateOffset(months=1)
                hm = hmm_pred[(hmm_pred.index >= prev_m) & (hmm_pred.index < day)]
                bp = hm["bull_prob"].mean() if len(hm) > 0 else 0.5
                if pd.isna(bp): bp = 0.5

                # Kritzman
                tp = turb_m.get(day, 0.5) if day in turb_m.index else 0.5
                if pd.isna(tp): tp = 0.5
                arz = ar_z_m.get(day, 0) if day in ar_z_m.index else 0
                if pd.isna(arz): arz = 0

                # Allocation
                w, pool = step1_faber(st["strengths"], base, assets)
                w = step2_harvey_cv(w, pool, st["harvey_er"], rvols, carry, value, assets)
                w = step3_hmm(w, bp)
                w = step4_kritzman(w, tp, arz)
                st["w"] = step5_normalize(w, assets)

                # Leverage tier
                if use_leverage:
                    f_c = st["strengths"].get("IVV",0) >= 3 and st["strengths"].get("QQQ",0) >= 3
                    h_c = st["harvey_er"].get("IVV",0) > 0 and st["harvey_er"].get("QQQ",0) > 0
                    if f_c and h_c:
                        above = all(st["harvey_er"].get(a,0) > st["tier_med"].get(a,0) for a in ["IVV","QQQ"])
                        st["tier"] = 2 if above else 1
                        for a in ["IVV","QQQ"]:
                            st["tier_hist"][a].append(st["harvey_er"].get(a,0))
                            st["tier_med"][a] = float(np.median(st["tier_hist"][a]))
                    else:
                        st["tier"] = 0

                # Era attribution (for levered configs)
                if k in era_data and sim_dates:
                    relevant = [a for a in avail if a in etf_starts]
                    latest_etf = max((etf_starts[a] for a in relevant), default=pd.Timestamp("1900-01-01"))
                    n_proxy = sum(1 for d in sim_dates if d < latest_etf)
                    n_etf = sum(1 for d in sim_dates if d >= latest_etf)
                    era_data[k].append({"date": day, "n_proxy": n_proxy, "n_etf": n_etf,
                                        "total": len(sim_dates)})

            # Weekly circuit breaker (leverage configs only)
            if use_leverage and is_friday and st["tier"] > 0 and not st["delevered"]:
                breach = False
                for etf in ["IVV", "QQQ"]:
                    if etf not in daily_prices_df.columns: continue
                    prices_before = daily_prices_df.loc[:day, etf]
                    if len(prices_before) == 0: continue
                    price_today = prices_before.iloc[-1]
                    last_ms = pd.Timestamp(f"{day.year}-{day.month:02d}-01")
                    for sma_df in [sma_6m, sma_10m, sma_12m]:
                        sma_dates = sma_df.index[sma_df.index <= last_ms]
                        if len(sma_dates) == 0: continue
                        sma_val = sma_df.loc[sma_dates[-1], etf]
                        if pd.notna(sma_val) and price_today < sma_val:
                            breach = True
                            break
                    if breach: break
                if breach:
                    st["delever_events"].append(day)
                    st["tier"] = 0
                    st["delevered"] = True

            # Compute daily return
            wp = st["w"]
            iw = wp.get("IVV", 0); qw = wp.get("QQQ", 0)
            ir = actual.get("IVV", 0); qr = actual.get("QQQ", 0)
            base_ret = sum(wp.get(a, 0) * actual.get(a, 0) for a in avail if a not in ["IVV", "QQQ"])

            if use_leverage:
                sub = TIER_SUBS.get(st["tier"], 0)
                if sub > 0:
                    sr = 2 * ir - rfr - 0.0091 / 252
                    ql = 2 * qr - rfr - 0.0089 / 252
                    ret = iw * (1 - sub) * ir + iw * sub * sr + qw * (1 - sub) * qr + qw * sub * ql + base_ret
                else:
                    ret = iw * ir + qw * qr + base_ret
                tier_counts[k][st["tier"]] += 1
            else:
                ret = iw * ir + qw * qr + base_ret

            # Transaction costs
            if current_weights[k] is not None and is_ms:
                to = sum(abs(wp.get(a, 0) - current_weights[k].get(a, 0)) for a in avail) / 2
                ret -= to * TC
                total_tc_map[k] += to * TC

            strats[k][day] = ret
            if is_ms:
                current_weights[k] = dict(wp)

        total_lev_days += 1

    results = {k: pd.Series(v).sort_index() for k, v in strats.items() if v}

    # ── Report ────────────────────────────────────────────────────────────────

    def perf(s):
        ar = s.mean() * 252; av = s.std() * np.sqrt(252)
        sh = ar / av if av > 0 else 0
        neg = s[s < 0]; ds = neg.std() * np.sqrt(252) if len(neg) > 10 else av
        sortino = ar / ds if ds > 0 else 0
        cum = (1 + s).cumprod()
        dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
        calmar = ar / abs(dd) if dd != 0 else 0
        final = cum.iloc[-1]
        return ar, av, sh, sortino, dd, calmar, final

    pr(f"\n\n{'='*80}")
    pr(f"  PERFORMANCE COMPARISON")
    pr(f"{'='*80}")

    labels = {"6A_lev": "6-Asset Lev", "4A_lev": "4-Asset Lev",
              "6A_1x": "6-Asset 1x", "4A_1x": "4-Asset 1x",
              "bench_ivv": "IVV B&H", "bench_6040": "60/40"}

    pr(f"\n  {'Config':>14} {'AnnRet':>7} {'AnnVol':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>8} {'Calmar':>7} {'Final$':>8}")
    pr(f"  {'-'*70}")
    for k in ["6A_lev", "4A_lev", "6A_1x", "4A_1x", "bench_6040", "bench_ivv"]:
        s = results.get(k)
        if s is None or len(s) < 252: continue
        ar, av, sh, so, dd, cal, final = perf(s)
        pr(f"  {labels[k]:>14} {ar:>6.1%} {av:>6.1%} {sh:>7.2f} {so:>8.2f} {dd:>7.1%} {cal:>7.2f} ${final:>7.2f}")

    # Delta
    pr(f"\n  DELTA (4-Asset minus 6-Asset)")
    for suffix in ["lev", "1x"]:
        s6 = results.get(f"6A_{suffix}")
        s4 = results.get(f"4A_{suffix}")
        if s6 is None or s4 is None: continue
        p6 = perf(s6); p4 = perf(s4)
        tag = "Levered" if suffix == "lev" else "Unlevered"
        pr(f"  {tag}:  ΔReturn={p4[0]-p6[0]:+.1%}  ΔVol={p4[1]-p6[1]:+.1%}  ΔSharpe={p4[2]-p6[2]:+.3f}  ΔMaxDD={p4[4]-p6[4]:+.1%}  ΔFinal=${p4[6]-p6[6]:+.2f}")

    # Crisis
    pr(f"\n\n{'='*80}")
    pr(f"  CRISIS PERIODS")
    pr(f"{'='*80}")
    for cn, cs, ce in [("GFC","2008-09-01","2009-03-31"),("COVID","2020-02-19","2020-03-23"),("2022 Bear","2022-01-03","2022-10-31")]:
        pr(f"\n  {cn}:")
        pr(f"  {'Config':>14} {'Return':>8} {'MaxDD':>8}")
        for k in ["6A_lev","4A_lev","6A_1x","4A_1x","bench_ivv"]:
            s = results.get(k)
            if s is None: continue
            c = s[(s.index >= pd.Timestamp(cs)) & (s.index <= pd.Timestamp(ce))]
            if len(c) == 0: continue
            cum = (1+c).cumprod()
            dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
            pr(f"  {labels[k]:>14} {(1+c).prod()-1:>+7.1%} {dd:>7.1%}")

    # Leverage tiers
    pr(f"\n\n{'='*80}")
    pr(f"  LEVERAGE TIER FREQUENCY")
    pr(f"{'='*80}")
    for k in ["6A_lev", "4A_lev"]:
        counts = tier_counts.get(k, {})
        total = sum(counts.values())
        if total == 0: continue
        pr(f"\n  {labels[k]}:")
        for t in [0, 1, 2]:
            pct = counts[t] / total * 100
            lev = {0: "1.00x", 1: "1.30x", 2: "1.65x"}[t]
            pr(f"    Tier {t} ({lev}): {counts[t]:>5} days ({pct:>5.1f}%)")
        # De-lever events
        st = state.get(k, {})
        n_delev = len(st.get("delever_events", []))
        pr(f"    Weekly de-lever events: {n_delev}")

    # Era attribution
    pr(f"\n\n{'='*80}")
    pr(f"  PROXY ERA ATTRIBUTION")
    pr(f"{'='*80}")
    for k in ["6A_lev", "4A_lev"]:
        ed = era_data.get(k, [])
        if not ed: continue
        edf = pd.DataFrame(ed)
        total_sim = edf["total"].sum()
        total_proxy = edf["n_proxy"].sum()
        total_etf = edf["n_etf"].sum()
        pct_proxy = total_proxy / total_sim * 100 if total_sim > 0 else 0
        pct_etf = total_etf / total_sim * 100 if total_sim > 0 else 0
        pr(f"\n  {labels[k]}:")
        pr(f"    Total analog months used:    {total_sim:>6}")
        pr(f"    From proxy era (pre-ETF):    {total_proxy:>6} ({pct_proxy:.1f}%)")
        pr(f"    From ETF era:                {total_etf:>6} ({pct_etf:.1f}%)")

    # Signal quality: ER correlation between universes
    pr(f"\n\n{'='*80}")
    pr(f"  SIGNAL QUALITY")
    pr(f"{'='*80}")
    pr(f"  (Proxy era attribution improvement)")
    for k in ["6A_lev", "4A_lev"]:
        ed = era_data.get(k, [])
        if ed:
            edf = pd.DataFrame(ed)
            total_sim = edf["total"].sum()
            total_proxy = edf["n_proxy"].sum()
            pr(f"    {labels[k]}: {total_proxy/total_sim*100:.1f}% proxy-era analogs")

    # Calendar years
    pr(f"\n\n{'='*80}")
    pr(f"  CALENDAR YEAR RETURNS")
    pr(f"{'='*80}")
    pr(f"  {'Year':>6} {'6A Lev':>8} {'4A Lev':>8} {'6A 1x':>8} {'4A 1x':>8} {'IVV':>8}")
    for yr in range(2002, 2026):
        row = f"  {yr:>6}"
        for k in ["6A_lev","4A_lev","6A_1x","4A_1x","bench_ivv"]:
            s = results.get(k)
            if s is None: row += f" {'--':>8}"; continue
            y = s[s.index.year == yr]
            row += f" {(1+y).prod()-1:>+7.1%}" if len(y) > 20 else f" {'--':>8}"
        pr(row)

    # Final values
    pr(f"\n{'='*80}")
    pr(f"  FINAL VALUES ($1)")
    pr(f"{'='*80}")
    for k in ["6A_lev","4A_lev","6A_1x","4A_1x","bench_6040","bench_ivv"]:
        s = results.get(k)
        if s is None or len(s) < 252: continue
        pr(f"  {labels[k]:>14}: ${(1+s).cumprod().iloc[-1]:>8.2f}")

    # Recommendation
    pr(f"\n\n{'='*80}")
    pr(f"  RECOMMENDATION")
    pr(f"{'='*80}")
    s6 = results.get("6A_lev"); s4 = results.get("4A_lev")
    if s6 is not None and s4 is not None:
        p6 = perf(s6); p4 = perf(s4)
        sh_diff = p4[2] - p6[2]
        ret_diff = p4[0] - p6[0]
        dd_diff = p4[4] - p6[4]  # more negative = worse

        if sh_diff >= -0.05 and p4[4] >= p6[4] - 0.02:
            pr(f"\n  → SWITCH TO 4-ASSET")
            pr(f"  The 4-asset universe maintains competitive risk-adjusted returns")
            pr(f"  (Sharpe delta: {sh_diff:+.3f}) while eliminating two assets with")
            pr(f"  broken proxy data. Data integrity improvement outweighs any")
            pr(f"  marginal performance difference.")
        elif sh_diff < -0.10:
            pr(f"\n  → KEEP 6-ASSET")
            pr(f"  Dropping VGLT and DBC costs {-sh_diff:.3f} Sharpe — too large to justify")
            pr(f"  on data quality grounds alone. Instead, shorten Harvey lookback")
            pr(f"  for VGLT/DBC to ETF-only era.")
        else:
            pr(f"\n  → CONDITIONAL: 4-asset preferred if data quality is priority")
            pr(f"  Sharpe delta: {sh_diff:+.3f}, Return delta: {ret_diff:+.1%}, MaxDD delta: {dd_diff:+.1%}")
            pr(f"  Modest performance trade-off. Consider 4-asset for production,")
            pr(f"  6-asset for research (with era-aware weighting).")

    # Save
    report_path = os.path.join(str(OUTPUT), "4asset_test_results.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Report saved: {report_path}")


if __name__ == "__main__":
    main()

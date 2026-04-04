"""5-Asset model with treasury futures: validate, build, and backtest."""

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
SMA_PERIODS = [6, 10, 12]
TIER_SUBS = {0: 0.0, 1: 0.30, 2: 0.65}
EQUITY = ["IVV", "QQQ"]

# Universe definitions
BASE_4 = {"IVV": 0.45, "QQQ": 0.25, "IAU": 0.15, "cash": 0.15}
BASE_5 = {"IVV": 0.45, "QQQ": 0.25, "TSY": 0.10, "IAU": 0.10, "cash": 0.10}
BASE_6 = {"IVV": 0.45, "QQQ": 0.25, "VGLT": 0.05, "IAU": 0.10, "DBC": 0.05, "cash": 0.10}


# ── Allocation steps (universe-agnostic) ──────────────────────────────────────

def compute_multi_sma(prices_df):
    monthly = prices_df.resample("MS").last()
    result = pd.DataFrame(0, index=monthly.index, columns=monthly.columns)
    for p in SMA_PERIODS:
        sma = monthly.rolling(p, min_periods=p).mean()
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

def full_allocation(strengths, harvey_er, rvols, carry, value, bp, tp, arz, baseline, assets):
    w, pool = step1_faber(strengths, baseline, assets)
    w = step2_harvey_cv(w, pool, harvey_er, rvols, carry, value, assets)
    w = step3_hmm(w, bp)
    w = step4_kritzman(w, tp, arz)
    return step5_normalize(w)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    lines = []
    def pr(s=""): print(s); lines.append(s)

    import yfinance as yf

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1: Source Treasury Futures Data
    # ══════════════════════════════════════════════════════════════════════════

    pr("=" * 80)
    pr("  5-ASSET MODEL WITH TREASURY FUTURES")
    pr("=" * 80)

    pr(f"\n\nPHASE 1: DATA AVAILABILITY")
    pr("-" * 50)

    futures_data = {}
    for ticker, name in [("ZN=F","10Y T-Note"), ("ZB=F","30Y T-Bond")]:
        d = yf.download(ticker, start="1970-01-01", progress=False)
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p, "columns"): p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None)
            futures_data[ticker] = p
            pr(f"  {ticker} ({name}): {p.index.min().date()} to {p.index.max().date()}, {len(p)} daily obs")
            # Monthly returns
            mr = p.resample("MS").last().pct_change().dropna()
            pr(f"    Monthly returns: {len(mr)} months, mean={mr.mean()*100:.2f}%/mo, vol={mr.std()*np.sqrt(12)*100:.1f}% ann")
        else:
            pr(f"  {ticker} ({name}): NOT AVAILABLE")

    etf_data = {}
    for ticker, name in [("TLT","20+Y Treasury ETF"), ("IEF","7-10Y Treasury ETF")]:
        d = yf.download(ticker, start="2000-01-01", progress=False)
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p, "columns"): p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None)
            etf_data[ticker] = p
            pr(f"  {ticker} ({name}): {p.index.min().date()} to {p.index.max().date()}, {len(p)} daily obs")

    pr(f"\n  Note: yfinance futures data starts Sep 2000, not 1977/1982.")
    pr(f"  Pre-2000 treasury proxy not available from free sources.")
    pr(f"  Extended backtest (1982-2002) not possible without paid data.")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2: Validate Against ETF Overlap
    # ══════════════════════════════════════════════════════════════════════════

    pr(f"\n\n{'='*80}")
    pr(f"  PHASE 2: TREASURY FUTURES vs ETF VALIDATION")
    pr(f"{'='*80}")

    best_ticker = None; best_etf = None; best_corr = 0

    for fut_ticker, etf_ticker, name in [("ZN=F","IEF","10-YEAR"), ("ZB=F","TLT","30-YEAR")]:
        pr(f"\n{name}: {fut_ticker} vs {etf_ticker}")
        pr("-" * 50)

        if fut_ticker not in futures_data or etf_ticker not in etf_data:
            pr("  Data not available — skipping"); continue

        fp = futures_data[fut_ticker]; ep = etf_data[etf_ticker]
        fr = fp.resample("MS").last().pct_change().dropna()
        er = ep.resample("MS").last().pct_change().dropna()
        fr.index = fr.index.to_period("M").to_timestamp()
        er.index = er.index.to_period("M").to_timestamp()

        common = fr.index.intersection(er.index).sort_values()
        fr_c = fr.reindex(common).dropna(); er_c = er.reindex(common).dropna()
        common = fr_c.index.intersection(er_c.index)
        fr_c, er_c = fr_c.reindex(common), er_c.reindex(common)

        if len(common) < 12:
            pr("  Insufficient overlap"); continue

        slope, intercept, r_value, _, _ = sp_stats.linregress(er_c, fr_c)
        diff = fr_c - er_c
        corr = fr_c.corr(er_c)

        pr(f"  Overlap:         {common.min().strftime('%Y-%m')} to {common.max().strftime('%Y-%m')} ({len(common)} months)")
        pr(f"  Correlation:     {corr:.4f}")
        pr(f"  Beta:            {slope:.4f}")
        pr(f"  R²:              {r_value**2:.4f}")
        pr(f"  Ann. Return Fut: {fr_c.mean()*12*100:.2f}%")
        pr(f"  Ann. Return ETF: {er_c.mean()*12*100:.2f}%")
        pr(f"  Return diff:     {diff.mean()*12*100:.2f}%")
        pr(f"  Tracking error:  {diff.std()*np.sqrt(12)*100:.2f}%")

        passed = corr > 0.90
        pr(f"  PASS/FAIL:       {'PASS' if passed else 'FAIL'} (corr {'>' if passed else '<='} 0.90)")

        if corr > best_corr:
            best_corr = corr; best_ticker = fut_ticker; best_etf = etf_ticker

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 3: Choose Instrument
    # ══════════════════════════════════════════════════════════════════════════

    pr(f"\n\n{'='*80}")
    pr(f"  PHASE 3: INSTRUMENT SELECTION")
    pr(f"{'='*80}")

    # Also compare correlation with equities
    spy_d = yf.download("SPY", start="2000-01-01", progress=False)
    spy_p = spy_d["Close"]
    if hasattr(spy_p, "columns"): spy_p = spy_p.iloc[:, 0]
    spy_p.index = pd.to_datetime(spy_p.index).tz_localize(None)
    spy_m = spy_p.resample("MS").last().pct_change().dropna()
    spy_m.index = spy_m.index.to_period("M").to_timestamp()

    pr(f"\n  Correlation with SPY (monthly returns):")
    for ticker in ["ZN=F", "ZB=F"]:
        if ticker not in futures_data: continue
        fr = futures_data[ticker].resample("MS").last().pct_change().dropna()
        fr.index = fr.index.to_period("M").to_timestamp()
        common = fr.index.intersection(spy_m.index)
        corr_eq = fr.reindex(common).corr(spy_m.reindex(common))
        pr(f"    {ticker}: {corr_eq:.3f}")

    # Gold correlation
    gld_d = yf.download("GLD", start="2004-01-01", progress=False)
    gld_p = gld_d["Close"]
    if hasattr(gld_p, "columns"): gld_p = gld_p.iloc[:, 0]
    gld_p.index = pd.to_datetime(gld_p.index).tz_localize(None)
    gld_m = gld_p.resample("MS").last().pct_change().dropna()
    gld_m.index = gld_m.index.to_period("M").to_timestamp()

    pr(f"\n  Correlation with GLD (monthly returns):")
    for ticker in ["ZN=F", "ZB=F"]:
        if ticker not in futures_data: continue
        fr = futures_data[ticker].resample("MS").last().pct_change().dropna()
        fr.index = fr.index.to_period("M").to_timestamp()
        common = fr.index.intersection(gld_m.index)
        corr_gld = fr.reindex(common).corr(gld_m.reindex(common))
        pr(f"    {ticker}: {corr_gld:.3f}")

    # Decision
    use_fut = best_ticker or "ZB=F"
    use_etf = best_etf or "TLT"
    pr(f"\n  RECOMMENDATION: Use {use_fut} (proxy for {use_etf})")
    pr(f"    Reason: Highest ETF correlation ({best_corr:.3f})")
    if use_fut == "ZB=F":
        pr(f"    ZB (30-year) has more duration → bigger crisis hedge potential")
    else:
        pr(f"    ZN (10-year) has tighter ETF tracking")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 4 & 5: Build and Backtest 5-Asset Model
    # ══════════════════════════════════════════════════════════════════════════

    pr(f"\n\n{'='*80}")
    pr(f"  PHASE 5: FULL BACKTEST COMPARISON")
    pr(f"{'='*80}")

    config = RegimeConfig()
    raw_macro = fetch_monthly_history(config)
    transformed = transform_variables(raw_macro, config)
    z_data = get_valid_zscored(transformed, config)
    z_data_lagged = z_data.shift(1).dropna()
    asset_ret = pd.read_parquet("data/macro/roth_asset_returns.parquet")
    asset_ret_fwd = asset_ret.shift(-1)

    print("Loading daily ETF data...")
    daily_ret = fetch_daily_etf_returns()

    # Build price + daily return datasets including treasury
    ep = {}
    ticker_map = {"IVV": "SPY", "QQQ": "QQQ", "VGLT": "TLT", "IAU": "GLD", "DBC": "DBC"}
    for our, ticker in ticker_map.items():
        d = yf.download(ticker, start="1998-01-01", progress=False)
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p, "columns"): p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None)
            ep[our] = p

    # Add treasury: use best futures for price/SMA, ETF for daily returns
    tsy_etf_d = yf.download(use_etf, start="1998-01-01", progress=False)
    if tsy_etf_d is not None and not tsy_etf_d.empty:
        tp = tsy_etf_d["Close"]
        if hasattr(tp, "columns"): tp = tp.iloc[:, 0]
        tp.index = pd.to_datetime(tp.index).tz_localize(None)
        ep["TSY"] = tp
        # Add TSY daily returns
        daily_ret["TSY"] = tp.pct_change()

    # Also add TSY to roth asset returns for Harvey lookups
    # Use futures monthly returns for history, splice with ETF
    tsy_fut = futures_data.get(use_fut)
    if tsy_fut is not None:
        fut_m = tsy_fut.resample("MS").last().pct_change().dropna()
        fut_m.index = fut_m.index.to_period("M").to_timestamp()
        etf_m_tsy = tp.resample("MS").last().pct_change().dropna() if tp is not None else None
        if etf_m_tsy is not None:
            etf_m_tsy.index = etf_m_tsy.index.to_period("M").to_timestamp()
            # Splice: futures where ETF not available, ETF after
            cutoff = etf_m_tsy.index.min()
            tsy_returns = pd.concat([fut_m[fut_m.index < cutoff], etf_m_tsy]).sort_index()
            tsy_returns = tsy_returns[~tsy_returns.index.duplicated(keep="last")]
        else:
            tsy_returns = fut_m
        asset_ret["TSY"] = tsy_returns
        asset_ret_fwd = asset_ret.shift(-1)

    prices_df = pd.DataFrame(ep).sort_index()

    # Monthly SMAs for weekly Faber
    monthly_prices = prices_df.resample("MS").last()
    sma_dfs = []
    for p in SMA_PERIODS:
        sma_dfs.append(monthly_prices.rolling(p, min_periods=p).mean().shift(1))

    rvol_63 = prices_df.pct_change().rolling(63, min_periods=30).std() * np.sqrt(252)
    rvol_monthly = rvol_63.resample("MS").last().shift(1)

    print("Computing carry/value...")
    carry_df = compute_carry(prices_df)
    value_df = compute_value(prices_df)

    # Add TSY carry: term premium (GS20 - TB3MS)
    from fredapi import Fred
    fkey = os.environ.get("FRED_API_KEY")
    if fkey:
        fred_api = Fred(api_key=fkey)
        gs20 = fred_api.get_series("GS20", observation_start="1998-01-01")
        tb3 = fred_api.get_series("DTB3", observation_start="1998-01-01")
        if gs20 is not None and tb3 is not None:
            gs20.index = pd.to_datetime(gs20.index)
            tb3.index = pd.to_datetime(tb3.index)
            gs20_m = gs20.resample("MS").last() / 1200
            tb3_m = tb3.resample("MS").last() / 1200
            carry_df["TSY"] = (gs20_m - tb3_m).shift(1)

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
        tb_d = Fred(api_key=fkey).get_series("DTB3", observation_start="1998-01-01")
        tb_d.index = pd.to_datetime(tb_d.index)
        rfr_daily = (tb_d / 100 / 252).reindex(daily_ret.index, method="ffill").fillna(0)

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

    pr(f"\nUsing {use_fut} for treasury, mapped to {use_etf} for live")
    pr(f"Backtest: {len(trading_days)} days ({common_start.date()} to {trading_days.max().date()})")

    # ── Run 3 configs with weekly rebalancing ─────────────────────────────────

    configs = {
        "4A": {"base": BASE_4, "assets": list(BASE_4.keys())},
        "5A": {"base": BASE_5, "assets": list(BASE_5.keys())},
        "6A": {"base": BASE_6, "assets": list(BASE_6.keys())},
    }

    strats = {k: {} for k in configs}
    ivv_rets = {}

    # Shared monthly signals
    harvey_er = {}; rvols = {}; carry = {}; value = {}
    bp = 0.5; tp = 0.5; arz = 0.0

    # Per-config state
    state = {}
    for k, cfg in configs.items():
        state[k] = {
            "strengths": {a: 3 for a in cfg["assets"] if a != "cash"},
            "w": dict(cfg["base"]),
            "tier": 0,
            "tier_hist": {"IVV": list(pre_ers["IVV"]), "QQQ": list(pre_ers["QQQ"])},
            "tier_med": dict(SEED),
            "tc_total": 0.0, "prev_w": None,
            "weight_snaps": [],
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
            z_cands = z_data_lagged.index[z_data_lagged.index < day]
            if len(z_cands) > 0:
                try:
                    sim = compute_distances(z_data_lagged, z_cands[-1], config)
                    all_assets = set()
                    for cfg in configs.values(): all_assets.update(cfg["assets"])
                    for a in all_assets:
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

            carry = {}; value = {}
            for a in set().union(*[set(c["assets"]) for c in configs.values()]):
                if a in carry_df.columns:
                    cc = carry_df.index[carry_df.index <= day]
                    if len(cc) > 0: cv = carry_df.loc[cc[-1], a]; carry[a] = float(cv) if pd.notna(cv) else 0
                if a in value_df.columns:
                    vc = value_df.index[value_df.index <= day]
                    if len(vc) > 0: vv = value_df.loc[vc[-1], a]; value[a] = float(vv) if pd.notna(vv) else 0

            prev_m = day - pd.DateOffset(months=1)
            hm = hmm_pred[(hmm_pred.index >= prev_m) & (hmm_pred.index < day)]
            bp = hm["bull_prob"].mean() if len(hm) > 0 else 0.5
            if pd.isna(bp): bp = 0.5
            tp = turb_m.get(day, 0.5) if day in turb_m.index else 0.5
            if pd.isna(tp): tp = 0.5
            arz = ar_z_m.get(day, 0) if day in ar_z_m.index else 0
            if pd.isna(arz): arz = 0

        # Weekly rebalance (Friday or month start)
        if is_friday or is_ms:
            for k, cfg in configs.items():
                s = state[k]; assets = cfg["assets"]; base = cfg["base"]

                # Weekly Faber: check daily price vs monthly SMAs
                for a in assets:
                    if a == "cash": continue
                    if a not in prices_df.columns: continue
                    pb = prices_df.loc[:day, a]
                    if len(pb) == 0: continue
                    cp = pb.iloc[-1]
                    score = 0
                    for sdf in sma_dfs:
                        sd = sdf.index[sdf.index <= pd.Timestamp(f"{day.year}-{day.month:02d}-01")]
                        if len(sd) == 0: continue
                        sv = sdf.loc[sd[-1], a] if a in sdf.columns else np.nan
                        if pd.notna(sv) and cp > sv: score += 1
                    s["strengths"][a] = score

                s["w"] = full_allocation(s["strengths"], harvey_er, rvols, carry, value, bp, tp, arz, base, assets)

                # Tier
                f_c = s["strengths"].get("IVV",0)>=3 and s["strengths"].get("QQQ",0)>=3
                h_c = harvey_er.get("IVV",0)>0 and harvey_er.get("QQQ",0)>0
                if f_c and h_c:
                    above = all(harvey_er.get(a,0)>s["tier_med"].get(a,0) for a in ["IVV","QQQ"])
                    s["tier"] = 2 if above else 1
                    for a in ["IVV","QQQ"]:
                        s["tier_hist"][a].append(harvey_er.get(a,0))
                        s["tier_med"][a] = float(np.median(s["tier_hist"][a]))
                else: s["tier"] = 0

                # TC
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
        final = cum.iloc[-1]; corr = s.corr(ivv_s)
        return ar, av, sh, so, dd, final, corr

    labels = {"4A":"4-Asset", "5A":"5-Asset", "6A":"6-Asset (ref)"}

    pr(f"\n\nPERFORMANCE COMPARISON")
    pr("-" * 70)
    pr(f"  {'Metric':<22} {'4-Asset':>12} {'5-Asset':>12} {'6-Asset(ref)':>14}")
    pr(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*14}")
    perfs = {k: perf(results[k]) for k in configs}
    for name, idx, fmt in [("Ann. Return",0,"{:.1%}"),("Volatility",1,"{:.1%}"),
                             ("Sharpe",2,"{:.3f}"),("Sortino",3,"{:.3f}"),
                             ("Max Drawdown",4,"{:.1%}"),("Terminal $1",5,"${:.2f}"),
                             ("Corr with IVV",6,"{:.3f}")]:
        row = f"  {name:<22}"
        for k in ["4A","5A","6A"]:
            v = perfs[k][idx]
            row += f" {fmt.format(v):>12}" if k != "6A" else f" {fmt.format(v):>14}"
        pr(row)

    # TC
    pr(f"\n  Transaction costs (ann.):")
    for k in ["4A","5A","6A"]:
        pr(f"    {labels[k]}: {state[k]['tc_total']/n_years:.2%}")

    # Crisis analysis
    pr(f"\n\n{'='*80}")
    pr(f"  CRISIS PERFORMANCE")
    pr(f"{'='*80}")

    crises = [("GFC","2008-09-01","2009-03-31"), ("COVID","2020-02-19","2020-03-23"), ("2022 Bear","2022-01-03","2022-10-31")]
    for cname, cs, ce in crises:
        pr(f"\n  {cname}:")
        pr(f"  {'Metric':<28} {'4-Asset':>10} {'5-Asset':>10} {'6-Asset':>10}")
        pr(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*10}")

        for k in ["4A","5A","6A"]:
            s = results[k]
            c = s[(s.index >= pd.Timestamp(cs)) & (s.index <= pd.Timestamp(ce))]
            if len(c) == 0: continue
            cum_ret = (1+c).prod() - 1
            cum_s = (1+c).cumprod()
            dd = ((cum_s - cum_s.expanding().max()) / cum_s.expanding().max()).min()

            # Get weights during this period
            snaps = state[k]["weight_snaps"]
            crisis_snaps = [(d, w, t) for d, w, t in snaps
                            if pd.Timestamp(cs) <= d <= pd.Timestamp(ce)]

            if k == "4A":
                tsy_w = "N/A"; tsy_r = "N/A"
            else:
                tsy_key = "TSY" if k == "5A" else "VGLT"
                if crisis_snaps:
                    tsy_w = f"{np.mean([w.get(tsy_key,0) for _,w,_ in crisis_snaps]):.0%}"
                else:
                    tsy_w = "N/A"
                # TSY return during crisis
                tsy_daily = daily_ret.get(tsy_key if k == "5A" else "VGLT")
                if tsy_daily is not None:
                    tc = tsy_daily[(tsy_daily.index >= pd.Timestamp(cs)) & (tsy_daily.index <= pd.Timestamp(ce))].dropna()
                    tsy_r = f"{(1+tc).prod()-1:+.1%}" if len(tc) > 0 else "N/A"
                else:
                    tsy_r = "N/A"

            iau_snaps = [w.get("IAU",0) for _,w,_ in crisis_snaps] if crisis_snaps else []
            iau_w = f"{np.mean(iau_snaps):.0%}" if iau_snaps else "N/A"

        # Print crisis rows
        for metric in ["Portfolio return", "Portfolio max DD"]:
            row = f"  {metric:<28}"
            for k in ["4A","5A","6A"]:
                s = results[k]
                c = s[(s.index >= pd.Timestamp(cs)) & (s.index <= pd.Timestamp(ce))]
                if len(c) == 0: row += f" {'N/A':>10}"; continue
                if metric == "Portfolio return":
                    row += f" {(1+c).prod()-1:>+9.1%}"
                else:
                    cum_s = (1+c).cumprod()
                    dd = ((cum_s - cum_s.expanding().max()) / cum_s.expanding().max()).min()
                    row += f" {dd:>9.1%}"
            pr(row)

    # Defensive asset analysis
    pr(f"\n\n{'='*80}")
    pr(f"  DEFENSIVE ASSET ANALYSIS (5-Asset)")
    pr(f"{'='*80}")

    snaps_5 = state["5A"]["weight_snaps"]
    if snaps_5:
        tsy_wts = [w.get("TSY",0) for _,w,_ in snaps_5]
        iau_wts = [w.get("IAU",0) for _,w,_ in snaps_5]
        cash_wts = [w.get("cash",0) for _,w,_ in snaps_5]
        tsy_fab3 = sum(1 for _,w,_ in snaps_5 if state["5A"]["strengths"].get("TSY",0) >= 3) if snaps_5 else 0
        iau_fab3 = sum(1 for _,w,_ in snaps_5 if state["5A"]["strengths"].get("IAU",0) >= 3) if snaps_5 else 0

        pr(f"\n  Avg treasury weight:   {np.mean(tsy_wts):.0%} (baseline 10%)")
        pr(f"  Avg gold weight:       {np.mean(iau_wts):.0%} (baseline 10%)")
        pr(f"  Avg cash weight:       {np.mean(cash_wts):.0%} (baseline 10%)")
        pr(f"\n  Treasury Faber 3/3:    {tsy_fab3/len(snaps_5)*100:.0f}% of rebalances")
        pr(f"  Gold Faber 3/3:        {iau_fab3/len(snaps_5)*100:.0f}% of rebalances")

        # Treasury-gold correlation
        tsy_m = daily_ret.get("TSY")
        gld_m2 = daily_ret.get("IAU")
        if tsy_m is not None and gld_m2 is not None:
            common = tsy_m.dropna().index.intersection(gld_m2.dropna().index)
            tg_corr = tsy_m.reindex(common).corr(gld_m2.reindex(common))
            pr(f"\n  Correlation: Treasury vs Gold (daily): {tg_corr:.3f}")
            pr(f"  {'Low correlation = good diversification' if abs(tg_corr) < 0.3 else 'Moderate correlation — some redundancy'}")

    # Recommendation
    pr(f"\n\n{'='*80}")
    pr(f"  RECOMMENDATION")
    pr(f"{'='*80}")

    p4 = perfs["4A"]; p5 = perfs["5A"]; p6 = perfs["6A"]
    sh_4v5 = p5[2] - p4[2]
    dd_4v5 = p5[4] - p4[4]
    corr_4v5 = p5[6] - p4[6]

    if p5[2] > p4[2] and p5[2] > p6[2]:
        pr(f"\n  → ADOPT 5-ASSET MODEL")
        pr(f"  Best Sharpe: {p5[2]:.3f} (vs 4-asset {p4[2]:.3f}, 6-asset {p6[2]:.3f})")
    elif p5[2] > p6[2] and abs(p5[2] - p4[2]) < 0.03:
        pr(f"\n  → 5-ASSET PREFERRED over 6-asset, competitive with 4-asset")
        pr(f"  Sharpe: 5A={p5[2]:.3f} vs 4A={p4[2]:.3f} vs 6A={p6[2]:.3f}")
    elif p4[2] > p5[2]:
        pr(f"\n  → KEEP 4-ASSET (treasury doesn't add enough value)")
        pr(f"  Sharpe: 4A={p4[2]:.3f} vs 5A={p5[2]:.3f}")
    else:
        pr(f"\n  → MIXED: 5-asset has Sharpe {p5[2]:.3f} vs 4-asset {p4[2]:.3f}")

    pr(f"\n  Treasury adds value if it provides crisis diversification that gold doesn't.")
    pr(f"  IVV correlation: 4A={p4[6]:.3f}, 5A={p5[6]:.3f} ({'lower=better' if p5[6]<p4[6] else 'no improvement'})")
    if p5[4] > p4[4]:
        pr(f"  Max DD: 5A={p5[4]:.1%} vs 4A={p4[4]:.1%} (5A shallower drawdowns)")
    else:
        pr(f"  Max DD: 5A={p5[4]:.1%} vs 4A={p4[4]:.1%} (4A has shallower drawdowns)")

    # Save
    rp = os.path.join(str(OUTPUT), "treasury_5asset_report.txt")
    with open(rp, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Report saved: {rp}")


if __name__ == "__main__":
    main()

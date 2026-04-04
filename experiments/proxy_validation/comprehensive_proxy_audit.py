"""Comprehensive proxy quality audit: beta, R², crisis analysis, era attribution."""

import os, sys, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dotenv import load_dotenv; load_dotenv()
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from regime.config import RegimeConfig
from regime.data import fetch_monthly_history
from regime.transform import transform_variables, get_valid_zscored
from regime.similarity import compute_distances
from regime.multi_asset import allocate_similarity_weighted

OUTDIR = os.path.join(os.path.dirname(__file__), "output")
ROTH_PARQUET = os.path.join(os.path.dirname(__file__), "..", "..", "data", "macro", "roth_asset_returns.parquet")

CRISES = [
    ("GFC",      "2008-09", "2009-03"),
    ("COVID",    "2020-02", "2020-04"),
    ("2022 Bear","2022-01", "2022-10"),
]

PASS_CRITERIA = {
    "corr":       lambda v: abs(v) > 0.90,
    "beta":       lambda v: 0.85 <= v <= 1.15,
    "ann_diff":   lambda v: abs(v) < 0.02,
    "te_ann":     lambda v: v < 0.08,
    "crisis_corr":lambda v: v > 0.85,
}


# ── Fetchers ──────────────────────────────────────────────────────────────────

def fred(sid, start="1955-01-01"):
    try:
        from fredapi import Fred
        key = os.environ.get("FRED_API_KEY")
        if key:
            s = Fred(api_key=key).get_series(sid, observation_start=start)
            s.index = pd.to_datetime(s.index)
            return s.dropna()
    except Exception:
        pass
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={start}"
        df = pd.read_csv(url, index_col=0, parse_dates=True)
        return pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
    except Exception:
        return None


def yf_monthly_ret(ticker, start="1985-01-01"):
    import yfinance as yf
    d = yf.download(ticker, start=start, interval="1mo", progress=False)
    if d is None or d.empty:
        return None
    p = d["Close"]
    if hasattr(p, "columns"):
        p = p.iloc[:, 0]
    p.index = pd.to_datetime(p.index).tz_localize(None).to_period("M").to_timestamp()
    return p.pct_change().dropna()


def yf_daily(ticker, start="1955-01-01"):
    import yfinance as yf
    d = yf.download(ticker, start=start, progress=False)
    if d is None or d.empty:
        return None
    p = d["Close"]
    if hasattr(p, "columns"):
        p = p.iloc[:, 0]
    p.index = pd.to_datetime(p.index).tz_localize(None)
    return p


# ── Build all proxy/ETF pairs ────────────────────────────────────────────────

def build_all_pairs():
    """Return list of (label, asset, proxy_series, etf_series) tuples."""
    pairs = []

    # IVV — ^GSPC
    print("  Fetching IVV + ^GSPC...")
    etf_ivv = yf_monthly_ret("IVV", "2000-05-01")
    gspc = yf_daily("^GSPC", "1927-01-01")
    if gspc is not None and etf_ivv is not None:
        proxy = gspc.resample("MS").last().pct_change().dropna()
        proxy.index = proxy.index.to_period("M").to_timestamp()
        pairs.append(("^GSPC", "IVV", proxy, etf_ivv))

        # ^GSPC + 1.8% div
        proxy_div = proxy + 0.018 / 12
        pairs.append(("^GSPC+1.8%div", "IVV", proxy_div, etf_ivv))

    # QQQ — ^NDX
    print("  Fetching QQQ + ^NDX...")
    etf_qqq = yf_monthly_ret("QQQ", "1999-03-01")
    ndx = yf_daily("^NDX", "1985-01-01")
    if ndx is not None and etf_qqq is not None:
        proxy = ndx.resample("MS").last().pct_change().dropna()
        proxy.index = proxy.index.to_period("M").to_timestamp()
        pairs.append(("^NDX", "QQQ", proxy, etf_qqq))

    # VGLT — GS10 dur=17
    print("  Fetching VGLT + GS10...")
    etf_vglt = yf_monthly_ret("VGLT", "2009-11-01")
    if etf_vglt is None or len(etf_vglt) < 10:
        etf_vglt = yf_monthly_ret("TLT", "2002-07-01")
    gs10 = fred("GS10", "1960-01-01")
    if gs10 is not None and etf_vglt is not None:
        gs10_m = gs10.resample("MS").last().dropna()
        dy = gs10_m.diff()
        coupon = gs10_m.shift(1) / 1200
        proxy = (-17.0 * dy / 100 + coupon).dropna()
        pairs.append(("GS10 dur=17", "VGLT", proxy, etf_vglt))

    # IAU — GC=F
    print("  Fetching IAU + GC=F...")
    etf_gld = yf_monthly_ret("GLD", "2004-11-01")
    if etf_gld is None or len(etf_gld) < 10:
        etf_gld = yf_monthly_ret("IAU", "2005-01-01")
    gc = yf_daily("GC=F", "2000-01-01")
    if gc is not None and etf_gld is not None:
        proxy = gc.resample("MS").last().pct_change().dropna()
        proxy.index = proxy.index.to_period("M").to_timestamp()
        pairs.append(("GC=F", "IAU", proxy, etf_gld))

    # IAU — WPU1017 (known broken, for comparison)
    print("  Fetching IAU + WPU1017...")
    wpu = fred("WPU1017", "1960-01-01")
    if wpu is not None and etf_gld is not None:
        proxy = wpu.resample("MS").last().pct_change().dropna()
        pairs.append(("WPU1017", "IAU", proxy, etf_gld))

    # DBC — PPIACO
    print("  Fetching DBC + PPIACO...")
    etf_dbc = yf_monthly_ret("DBC", "2006-02-01")
    ppiaco = fred("PPIACO", "1960-01-01")
    if ppiaco is not None and etf_dbc is not None:
        proxy = ppiaco.resample("MS").last().pct_change().dropna()
        pairs.append(("PPIACO", "DBC", proxy, etf_dbc))

    # DBC — DCOILWTICO
    print("  Fetching DBC + DCOILWTICO...")
    doil = fred("DCOILWTICO", "1986-01-01")
    if doil is not None and etf_dbc is not None:
        proxy = doil.resample("MS").last().pct_change().dropna()
        pairs.append(("DCOILWTICO", "DBC", proxy, etf_dbc))

    return pairs


# ── Core metrics ──────────────────────────────────────────────────────────────

def compute_core(proxy, etf):
    common = proxy.index.intersection(etf.index).sort_values()
    p = proxy.reindex(common).dropna()
    e = etf.reindex(common).dropna()
    common = p.index.intersection(e.index)
    if len(common) < 12:
        return None
    p, e = p.reindex(common), e.reindex(common)

    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(e, p)
    diff = p - e

    return {
        "n": len(common),
        "start": common.min(),
        "end": common.max(),
        "corr": p.corr(e),
        "beta": slope,
        "alpha_ann": intercept * 12,
        "r2": r_value ** 2,
        "mean_etf": e.mean(),
        "mean_proxy": p.mean(),
        "ann_diff": diff.mean() * 12,
        "te_monthly": diff.std(),
        "te_ann": diff.std() * np.sqrt(12),
        "max_div": diff.abs().max(),
        "max_div_date": diff.abs().idxmax(),
        "p_series": p,
        "e_series": e,
    }


def compute_crisis(proxy, etf, crisis_start, crisis_end):
    cs = pd.Timestamp(crisis_start)
    ce = pd.Timestamp(crisis_end)
    common = proxy.index.intersection(etf.index)
    mask = (common >= cs) & (common <= ce)
    crisis_idx = common[mask]
    if len(crisis_idx) < 2:
        return None
    p = proxy.reindex(crisis_idx).dropna()
    e = etf.reindex(crisis_idx).dropna()
    crisis_idx = p.index.intersection(e.index)
    if len(crisis_idx) < 2:
        return None
    p, e = p.reindex(crisis_idx), e.reindex(crisis_idx)

    corr = p.corr(e) if len(crisis_idx) >= 3 else float("nan")
    if len(crisis_idx) >= 3:
        slope, *_ = sp_stats.linregress(e, p)
    else:
        slope = float("nan")

    cum_etf = (1 + e).prod() - 1
    cum_proxy = (1 + p).prod() - 1
    diff = p - e
    max_div = diff.abs().max()

    return {
        "n": len(crisis_idx),
        "corr": corr,
        "beta": slope,
        "cum_etf": cum_etf,
        "cum_proxy": cum_proxy,
        "max_div": max_div,
    }


# ── Phase 2: Era attribution ─────────────────────────────────────────────────

def run_era_attribution(pr_fn):
    """Run full backtest and attribute performance to proxy-era vs ETF-era analogs."""
    pr_fn("\n\n" + "=" * 80)
    pr_fn("  PHASE 2: BACKTEST WITH PROXY/ETF ERA ATTRIBUTION")
    pr_fn("=" * 80)

    config = RegimeConfig()
    raw_macro = fetch_monthly_history(config)
    transformed = transform_variables(raw_macro, config)
    z_data = get_valid_zscored(transformed, config)

    asset_ret = pd.read_parquet(ROTH_PARQUET)

    etf_starts = {
        "IVV": pd.Timestamp("2000-05-01"),
        "QQQ": pd.Timestamp("1999-03-01"),
        "VXUS": pd.Timestamp("2001-08-01"),
        "VGLT": pd.Timestamp("2009-11-01"),
        "IAU": pd.Timestamp("2004-11-01"),
        "DBC": pd.Timestamp("2006-02-01"),
        "VNQ": pd.Timestamp("2004-09-01"),
    }
    # Earliest date where ALL major ETFs have data
    all_etf_date = max(etf_starts.values())  # 2009-11

    start = pd.Timestamp(config.backtest_start)
    end = pd.Timestamp(config.backtest_end)
    bt_dates = z_data.index[(z_data.index >= start) & (z_data.index <= end)]
    bt_dates = bt_dates[bt_dates.isin(asset_ret.index)]

    # Track per-month: analog composition, performance
    records = []
    total_similar = 0
    total_proxy_era = 0
    total_etf_era = 0
    total_mixed = 0
    proxy_era_ers = []
    etf_era_ers = []

    for dt in bt_dates:
        avail = [a for a in asset_ret.columns if pd.notna(asset_ret.loc[dt, a])]
        avail_etfs = [a for a in avail if a != "cash"]
        if len(avail_etfs) < 2:
            continue

        try:
            sim = compute_distances(z_data, dt, config)
        except ValueError:
            continue

        # Expected returns from similar months
        sim_er = {}
        for a in avail:
            rets = [asset_ret.loc[d, a] for d in sim.similar_dates
                    if d in asset_ret.index and pd.notna(asset_ret.loc[d, a])]
            sim_er[a] = np.mean(rets) if rets else 0.0

        w = allocate_similarity_weighted(sim_er, max_single=0.40)
        actual = {a: asset_ret.loc[dt, a] for a in avail}
        ret = sum(w.get(a, 0) * actual.get(a, 0) for a in avail)

        # Classify similar months
        n_sim = len(sim.similar_dates)
        total_similar += n_sim

        # For each similar date, classify as proxy-era or ETF-era
        # Use the assets actually being allocated to determine which ETFs matter
        relevant_etfs = [a for a in avail if a in etf_starts]
        if not relevant_etfs:
            latest_etf_start = pd.Timestamp("1900-01-01")
        else:
            latest_etf_start = max(etf_starts[a] for a in relevant_etfs)

        proxy_dates = [d for d in sim.similar_dates if d < latest_etf_start]
        etf_dates = [d for d in sim.similar_dates if d >= latest_etf_start]
        n_proxy = len(proxy_dates)
        n_etf = len(etf_dates)

        total_proxy_era += n_proxy
        total_etf_era += n_etf

        proxy_frac = n_proxy / n_sim if n_sim > 0 else 0
        dominant = "proxy" if proxy_frac > 0.5 else "etf"

        # Compute separate expected returns for proxy-era vs ETF-era analogs
        proxy_er_avg = {}
        etf_er_avg = {}
        for a in avail:
            p_rets = [asset_ret.loc[d, a] for d in proxy_dates
                      if d in asset_ret.index and pd.notna(asset_ret.loc[d, a])]
            e_rets = [asset_ret.loc[d, a] for d in etf_dates
                      if d in asset_ret.index and pd.notna(asset_ret.loc[d, a])]
            proxy_er_avg[a] = np.mean(p_rets) if p_rets else None
            etf_er_avg[a] = np.mean(e_rets) if e_rets else None

        # Track for ER comparison
        if proxy_er_avg and etf_er_avg:
            shared = [a for a in avail if proxy_er_avg.get(a) is not None and etf_er_avg.get(a) is not None]
            if shared:
                proxy_era_ers.append(np.mean([proxy_er_avg[a] for a in shared]))
                etf_era_ers.append(np.mean([etf_er_avg[a] for a in shared]))

        # Would allocations differ?
        w_proxy = allocate_similarity_weighted(
            {a: (proxy_er_avg[a] if proxy_er_avg.get(a) is not None else 0.0) for a in avail},
            max_single=0.40
        ) if any(v is not None for v in proxy_er_avg.values()) else None
        w_etf = allocate_similarity_weighted(
            {a: (etf_er_avg[a] if etf_er_avg.get(a) is not None else 0.0) for a in avail},
            max_single=0.40
        ) if any(v is not None for v in etf_er_avg.values()) else None

        # Determine allocation differences
        diff_top = False
        diff_weights_material = False
        if w_proxy and w_etf:
            top_proxy = max(w_proxy, key=w_proxy.get)
            top_etf = max(w_etf, key=w_etf.get)
            diff_top = (top_proxy != top_etf)
            max_wt_delta = max(abs(w_proxy.get(a, 0) - w_etf.get(a, 0)) for a in avail)
            diff_weights_material = (max_wt_delta > 0.10)

        records.append({
            "date": dt,
            "ret": ret,
            "n_sim": n_sim,
            "n_proxy": n_proxy,
            "n_etf": n_etf,
            "proxy_frac": proxy_frac,
            "dominant": dominant,
            "diff_top": diff_top,
            "diff_weights_material": diff_weights_material,
        })

    rdf = pd.DataFrame(records).set_index("date")

    # ── Print era attribution report ──────────────────────────────────────────
    pr_fn(f"\n\nPROXY vs ETF ERA ATTRIBUTION")
    pr_fn("=" * 50)

    pr_fn(f"\n  Analog composition across backtest:")
    pr_fn(f"    Total similar months used:     {total_similar:>6}")
    pr_fn(f"    From proxy era (pre-ETF):      {total_proxy_era:>6} ({total_proxy_era/total_similar*100:.1f}%)")
    pr_fn(f"    From ETF era:                  {total_etf_era:>6} ({total_etf_era/total_similar*100:.1f}%)")

    n_proxy_dom = (rdf["dominant"] == "proxy").sum()
    n_etf_dom = (rdf["dominant"] == "etf").sum()
    pr_fn(f"\n  Backtest months by dominant analog source:")
    pr_fn(f"    Months where >50% analogs from proxy era: {n_proxy_dom:>4}")
    pr_fn(f"    Months where >50% analogs from ETF era:   {n_etf_dom:>4}")

    # Performance split
    pr_fn(f"\n\nPERFORMANCE SPLIT")
    pr_fn("-" * 70)

    for label, mask in [("Proxy-Dominated", rdf["dominant"] == "proxy"),
                         ("ETF-Dominated", rdf["dominant"] == "etf"),
                         ("Full System", pd.Series(True, index=rdf.index))]:
        sub = rdf.loc[mask, "ret"]
        if len(sub) < 12:
            pr_fn(f"\n  {label}: insufficient data ({len(sub)} months)")
            continue
        ann_ret = sub.mean() * 12
        ann_vol = sub.std() * np.sqrt(12)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        pr_fn(f"\n  {label} ({len(sub)} months)")
        pr_fn(f"    Ann. Return:    {ann_ret*100:.2f}%")
        pr_fn(f"    Volatility:     {ann_vol*100:.2f}%")
        pr_fn(f"    Sharpe:         {sharpe:.3f}")

    # ER comparison
    pr_fn(f"\n\nEXPECTED RETURN COMPARISON")
    pr_fn("-" * 50)
    if proxy_era_ers and etf_era_ers:
        pe = np.array(proxy_era_ers)
        ee = np.array(etf_era_ers)
        min_len = min(len(pe), len(ee))
        pe, ee = pe[:min_len], ee[:min_len]
        er_corr = np.corrcoef(pe, ee)[0, 1] if min_len > 3 else float("nan")
        pr_fn(f"  Mean Harvey ER (proxy-era analogs):  {pe.mean()*100:.3f}%")
        pr_fn(f"  Mean Harvey ER (ETF-era analogs):    {ee.mean()*100:.3f}%")
        pr_fn(f"  Correlation of ERs:                  {er_corr:.3f}")
        pr_fn(f"  Mean absolute difference:            {np.mean(np.abs(pe - ee))*100:.3f}%")

        if abs(er_corr) > 0.7:
            pr_fn(f"\n  Interpretation: Proxy-era and ETF-era analogs give SIMILAR signals (r={er_corr:.2f}).")
            pr_fn(f"  Proxy noise is unlikely to dramatically change allocation decisions.")
        elif abs(er_corr) > 0.4:
            pr_fn(f"\n  Interpretation: MODERATE agreement between eras (r={er_corr:.2f}).")
            pr_fn(f"  Some allocation decisions differ, but broad direction is preserved.")
        else:
            pr_fn(f"\n  Interpretation: WEAK agreement between eras (r={er_corr:.2f}).")
            pr_fn(f"  Proxy-era analogs produce materially different signals than ETF-era ones.")

    # Allocation decision impact
    pr_fn(f"\n\nALLOCATION DECISION IMPACT")
    pr_fn("-" * 50)
    n_diff_top = rdf["diff_top"].sum()
    n_diff_wt = rdf["diff_weights_material"].sum()
    n_total = len(rdf)
    pr_fn(f"  Different top asset:                       {n_diff_top:>4} months ({n_diff_top/n_total*100:.1f}%)")
    pr_fn(f"  Materially different weights (>10% delta):  {n_diff_wt:>4} months ({n_diff_wt/n_total*100:.1f}%)")

    # Time series: proxy fraction declining over time
    pr_fn(f"\n\nPROXY FRACTION OVER TIME (5-year rolling)")
    pr_fn("-" * 50)
    rolling_frac = rdf["proxy_frac"].rolling(60, min_periods=12).mean()
    for yr in range(1990, 2025, 5):
        ts = pd.Timestamp(f"{yr}-01-01")
        val = rolling_frac.asof(ts)
        if pd.notna(val):
            bar = "█" * int(val * 40)
            pr_fn(f"  {yr}: {val*100:>5.1f}% {bar}")

    # Final recommendation
    pr_fn(f"\n\nRECOMMENDATION")
    pr_fn("=" * 50)
    if n_proxy_dom > n_etf_dom * 0.5:
        pr_fn(f"  Proxy data still heavily influences allocation decisions.")
        pr_fn(f"  {n_proxy_dom} of {n_total} months ({n_proxy_dom/n_total*100:.0f}%) are proxy-dominated.")
        if n_diff_top / n_total > 0.2:
            pr_fn(f"  Top-asset disagreement in {n_diff_top/n_total*100:.0f}% of months — material impact.")
            pr_fn(f"  → Consider shortening Harvey lookback to ETF-era only (~2000+)")
        else:
            pr_fn(f"  But top-asset agreement is high — proxy noise may be acceptable.")
            pr_fn(f"  → Monitor but no urgent change needed.")
    else:
        pr_fn(f"  ETF-era analogs dominate — proxy noise has limited impact on recent decisions.")
        pr_fn(f"  → Current setup is acceptable for forward-looking use.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    lines = []
    def pr(s=""):
        print(s)
        lines.append(s)

    pr("=" * 80)
    pr("  COMPREHENSIVE PROXY QUALITY AUDIT")
    pr("=" * 80)

    print("\nBuilding proxy/ETF pairs...")
    pairs = build_all_pairs()
    if not pairs:
        pr("ERROR: No data fetched.")
        return

    all_results = []

    for label, asset, proxy, etf in pairs:
        pr(f"\n\n{asset} — {label}")
        pr("-" * 77)

        core = compute_core(proxy, etf)
        if core is None:
            pr("  INSUFFICIENT OVERLAP — skipping")
            continue

        pr(f"Overlap: {core['start'].strftime('%Y-%m')} to {core['end'].strftime('%Y-%m')} ({core['n']} months)")

        pr(f"\nCORE METRICS")
        pr(f"  Correlation:        {core['corr']:.4f}")
        pr(f"  Beta:               {core['beta']:.4f}")
        pr(f"  R²:                 {core['r2']:.4f}")
        pr(f"  Alpha (ann):        {core['alpha_ann']*100:.2f}%")
        pr(f"  Mean ret diff:      {core['ann_diff']*100:.2f}% annualized")
        pr(f"  Tracking error:     {core['te_ann']*100:.2f}% annualized")

        # Crisis analysis
        pr(f"\nCRISIS ANALYSIS")
        pr(f"  {'Period':<16} {'ETF Ret':>9} {'Proxy Ret':>10} {'Corr':>7} {'Beta':>7} {'MaxDiv':>7}")
        pr(f"  {'-'*16} {'-'*9} {'-'*10} {'-'*7} {'-'*7} {'-'*7}")

        crisis_corrs = []
        crisis_betas = []
        for cname, cs, ce in CRISES:
            cr = compute_crisis(proxy, etf, cs, ce)
            if cr is None:
                pr(f"  {cname:<16} {'—':>9} {'—':>10} {'—':>7} {'—':>7} {'—':>7}")
                continue
            if not np.isnan(cr["corr"]):
                crisis_corrs.append(cr["corr"])
            if not np.isnan(cr["beta"]):
                crisis_betas.append(cr["beta"])
            pr(f"  {cname:<16} {cr['cum_etf']*100:>+8.1f}% {cr['cum_proxy']*100:>+9.1f}% {cr['corr']:>7.3f} {cr['beta']:>7.3f} {cr['max_div']*100:>+6.1f}%")

        avg_crisis_corr = np.mean(crisis_corrs) if crisis_corrs else float("nan")
        avg_crisis_beta = np.mean(crisis_betas) if crisis_betas else float("nan")
        pr(f"\n  Crisis avg corr: {avg_crisis_corr:.3f}")
        pr(f"  Crisis avg beta: {avg_crisis_beta:.3f}")

        # Quality checklist
        pr(f"\nQUALITY CHECKLIST")
        checks = {
            "corr": core["corr"],
            "beta": core["beta"],
            "ann_diff": core["ann_diff"],
            "te_ann": core["te_ann"],
            "crisis_corr": avg_crisis_corr,
        }
        labels = {
            "corr": "Correlation > 0.90",
            "beta": "Beta 0.85-1.15",
            "ann_diff": "|Ret diff| < 2%",
            "te_ann": "Track err < 8%",
            "crisis_corr": "Crisis corr > 0.85",
        }
        n_pass = 0
        for k, v in checks.items():
            if np.isnan(v):
                mark = "—"
                detail = "N/A"
            elif PASS_CRITERIA[k](v):
                mark = "✓"
                n_pass += 1
                detail = f"{v:.4f}" if k in ("corr", "beta", "crisis_corr") else f"{v*100:.2f}%"
            else:
                mark = "✗"
                detail = f"{v:.4f}" if k in ("corr", "beta", "crisis_corr") else f"{v*100:.2f}%"
            pr(f"  [{mark}] {labels[k]:<24} {detail}")

        total_checks = sum(1 for v in checks.values() if not np.isnan(v))
        composite = "PASS" if n_pass == total_checks and total_checks > 0 else "FAIL"
        pr(f"\n  COMPOSITE: {composite} ({n_pass}/{total_checks} criteria met)")

        all_results.append({
            "label": label,
            "asset": asset,
            "corr": core["corr"],
            "beta": core["beta"],
            "r2": core["r2"],
            "ann_diff": core["ann_diff"],
            "te_ann": core["te_ann"],
            "crisis_corr": avg_crisis_corr,
            "n_pass": n_pass,
            "total_checks": total_checks,
            "composite": composite,
        })

    # ── Summary table ─────────────────────────────────────────────────────────
    pr(f"\n\n{'=' * 80}")
    pr(f"  SUMMARY TABLE")
    pr(f"{'=' * 80}")
    pr(f"\n  {'Asset':<6} {'Proxy':<16} {'Corr':>6} {'Beta':>6} {'R²':>6} {'RetDiff':>8} {'TE':>6} {'CrCorr':>7} {'PASS?':>6}")
    pr(f"  {'-'*6} {'-'*16} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*6} {'-'*7} {'-'*6}")

    for r in all_results:
        cc = f"{r['crisis_corr']:.3f}" if not np.isnan(r["crisis_corr"]) else "N/A"
        pr(f"  {r['asset']:<6} {r['label']:<16} {r['corr']:>6.3f} {r['beta']:>6.3f} {r['r2']:>6.3f} {r['ann_diff']*100:>+7.2f}% {r['te_ann']*100:>5.1f}% {cc:>7} {r['composite']:>6}")

    # ── Recommendations ───────────────────────────────────────────────────────
    pr(f"\n\n{'=' * 80}")
    pr(f"  RECOMMENDATIONS")
    pr(f"{'=' * 80}")

    pass_list = [r for r in all_results if r["composite"] == "PASS"]
    fail_list = [r for r in all_results if r["composite"] == "FAIL"]

    pr(f"\nPROXIES THAT PASS (use these):")
    if pass_list:
        for r in pass_list:
            pr(f"  - {r['asset']} ← {r['label']} (corr={r['corr']:.3f}, β={r['beta']:.3f}, TE={r['te_ann']*100:.1f}%)")
    else:
        pr(f"  (none)")

    pr(f"\nPROXIES THAT FAIL (need replacement or shorter lookback):")
    for r in fail_list:
        reasons = []
        if abs(r["corr"]) <= 0.90:
            reasons.append(f"low corr ({r['corr']:.3f})")
        if not (0.85 <= r["beta"] <= 1.15):
            reasons.append(f"beta out of range ({r['beta']:.3f})")
        if abs(r["ann_diff"]) >= 0.02:
            reasons.append(f"ret diff {r['ann_diff']*100:+.1f}%")
        if r["te_ann"] >= 0.08:
            reasons.append(f"TE {r['te_ann']*100:.1f}%")
        if not np.isnan(r["crisis_corr"]) and r["crisis_corr"] <= 0.85:
            reasons.append(f"crisis corr {r['crisis_corr']:.3f}")
        pr(f"  - {r['asset']} ← {r['label']}: {'; '.join(reasons)}")

    # Group by asset — which assets have no passing proxy?
    assets_with_pass = set(r["asset"] for r in pass_list)
    all_assets = set(r["asset"] for r in all_results)
    no_proxy = all_assets - assets_with_pass
    if no_proxy:
        pr(f"\nASSETS WITH NO GOOD PROXY:")
        for a in sorted(no_proxy):
            best = max((r for r in all_results if r["asset"] == a), key=lambda r: abs(r["corr"]))
            pr(f"  - {a}: best available is {best['label']} (corr={best['corr']:.3f})")
            pr(f"    Recommendation: shorten lookback to ETF-only era or accept proxy noise")

    # ── Impact on Harvey similarity ───────────────────────────────────────────
    pr(f"\n\n{'=' * 80}")
    pr(f"  IMPACT ON HARVEY SIMILARITY")
    pr(f"{'=' * 80}")

    etf_starts = {
        "IVV": pd.Timestamp("2000-05-01"),
        "QQQ": pd.Timestamp("1999-03-01"),
        "VGLT": pd.Timestamp("2009-11-01"),
        "IAU": pd.Timestamp("2004-11-01"),
        "DBC": pd.Timestamp("2006-02-01"),
    }
    # With recommended best proxies
    best_proxies = {}
    for a in all_assets:
        candidates = [r for r in all_results if r["asset"] == a]
        best_proxies[a] = max(candidates, key=lambda r: (r["composite"] == "PASS", abs(r["corr"])))

    pr(f"\n  With recommended proxy changes:")
    for a in sorted(best_proxies):
        r = best_proxies[a]
        etf_start = etf_starts.get(a, pd.Timestamp("2000-01-01"))
        pr(f"    {a}: {r['label']} ({r['composite']}) — proxy needed before {etf_start.strftime('%Y-%m')}")

    earliest_reliable = min(etf_starts.values())
    pr(f"\n  Earliest date with ALL assets having ETF data: {max(etf_starts.values()).strftime('%Y-%m')}")
    pr(f"  Recommended Harvey lookback start: {earliest_reliable.strftime('%Y-%m')} (earliest ETF)")
    bt_start = pd.Timestamp("1985-01-01")
    bt_end = pd.Timestamp("2024-12-01")
    total_bt = (bt_end.year - bt_start.year) * 12
    etf_only = (bt_end.year - max(etf_starts.values()).year) * 12
    pr(f"  Current backtest: {total_bt} months (1985-2024)")
    pr(f"  ETF-only window:  {etf_only} months ({max(etf_starts.values()).strftime('%Y')}-2024)")

    # ── Phase 2: Era attribution ──────────────────────────────────────────────
    run_era_attribution(pr)

    # Save report
    report_path = os.path.join(OUTDIR, "comprehensive_proxy_audit.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n\n  Report saved: {report_path}")


if __name__ == "__main__":
    main()

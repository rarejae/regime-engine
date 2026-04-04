"""Treasury data pipeline: source, validate, integrate.

Data layers:
  1970-2000: ^TNX/^TYX yield indices → duration model → synthetic returns
  2000-2002: ZN/ZB futures → price returns
  2002+:     IEF/TLT ETFs → adjusted close returns

Validates duration model against futures during 2000-2026 overlap,
then validates futures against ETFs during 2002-2026 overlap.
"""

import sys, os, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv; load_dotenv()
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from scipy import stats as sp_stats

OUTDIR = Path(__file__).resolve().parent / "output"
DATA_DIR = Path("data/macro")


def yf_daily(ticker, start="1970-01-01"):
    import yfinance as yf
    d = yf.download(ticker, start=start, progress=False)
    if d is None or d.empty:
        return None
    p = d["Close"]
    if hasattr(p, "columns"):
        p = p.iloc[:, 0]
    p.index = pd.to_datetime(p.index).tz_localize(None)
    return p


def duration_model(yields_monthly, duration):
    """Monthly bond return from yield changes: -dur × (Δy/100) + coupon."""
    dy = yields_monthly.diff()
    coupon = yields_monthly.shift(1) / 1200
    return (-duration * dy / 100 + coupon).dropna()


def monthly_price_return(prices):
    """Convert daily prices to monthly returns."""
    m = prices.resample("MS").last().dropna()
    ret = m.pct_change().dropna()
    ret.index = ret.index.to_period("M").to_timestamp()
    return ret


def validate(proxy, etf, name):
    """Compute overlap metrics between proxy and ETF monthly returns."""
    common = proxy.index.intersection(etf.index).sort_values()
    p = proxy.reindex(common).dropna()
    e = etf.reindex(common).dropna()
    common = p.index.intersection(e.index)
    if len(common) < 12:
        return None
    p, e = p.reindex(common), e.reindex(common)
    slope, intercept, r_value, _, _ = sp_stats.linregress(e, p)
    diff = p - e
    return {
        "name": name, "n": len(common),
        "start": common.min().strftime("%Y-%m"),
        "end": common.max().strftime("%Y-%m"),
        "corr": p.corr(e), "beta": slope, "r2": r_value ** 2,
        "alpha_ann": intercept * 12,
        "ret_proxy": p.mean() * 12, "ret_etf": e.mean() * 12,
        "ret_diff": diff.mean() * 12,
        "te": diff.std() * np.sqrt(12),
    }


def fmt_validation(m):
    if m is None:
        return "  INSUFFICIENT DATA\n"
    passed = m["corr"] > 0.90
    lines = [
        f"  Overlap:          {m['start']} to {m['end']} ({m['n']} months)",
        f"  Correlation:      {m['corr']:.4f}",
        f"  Beta:             {m['beta']:.4f}",
        f"  R²:               {m['r2']:.4f}",
        f"  Ann. Return Proxy:{m['ret_proxy']*100:.2f}%",
        f"  Ann. Return ETF:  {m['ret_etf']*100:.2f}%",
        f"  Return diff:      {m['ret_diff']*100:.2f}%",
        f"  Tracking error:   {m['te']*100:.2f}% ann.",
        f"  PASS:             {'YES' if passed else 'NO'} (threshold: corr > 0.90)",
    ]
    return "\n".join(lines)


def main():
    lines = []
    def pr(s=""):
        print(s); lines.append(s)

    pr("=" * 80)
    pr("  TREASURY DATA PIPELINE")
    pr("=" * 80)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1: DATA AVAILABILITY
    # ══════════════════════════════════════════════════════════════════════════

    pr(f"\n\nPHASE 1: DATA AVAILABILITY")
    pr("=" * 50)

    print("Downloading data...")

    # Yield indices (long history)
    tnx = yf_daily("^TNX", "1970-01-01")  # 10Y yield
    tyx = yf_daily("^TYX", "1977-01-01")  # 30Y yield

    # Futures (medium history)
    zn = yf_daily("ZN=F", "1998-01-01")   # 10Y futures
    zb = yf_daily("ZB=F", "1998-01-01")   # 30Y futures

    # ETFs (short history, tradeable)
    ief = yf_daily("IEF", "2002-01-01")   # 7-10Y ETF
    tlt = yf_daily("TLT", "2002-01-01")   # 20+Y ETF

    sources = [
        ("^TNX", "10Y Yield Index", tnx),
        ("^TYX", "30Y Yield Index", tyx),
        ("ZN=F", "10Y T-Note Futures", zn),
        ("ZB=F", "30Y T-Bond Futures", zb),
        ("IEF", "7-10Y Treasury ETF", ief),
        ("TLT", "20+Y Treasury ETF", tlt),
    ]

    pr(f"\n  {'Symbol':<8} {'Description':<25} {'Start':>12} {'End':>12} {'Rows':>7}")
    pr(f"  {'-'*8} {'-'*25} {'-'*12} {'-'*12} {'-'*7}")
    for sym, desc, data in sources:
        if data is not None:
            pr(f"  {sym:<8} {desc:<25} {str(data.index.min().date()):>12} {str(data.index.max().date()):>12} {len(data):>7}")
        else:
            pr(f"  {sym:<8} {desc:<25} {'—':>12} {'—':>12} {'—':>7}")

    pr(f"\n  Key finding: ^TNX goes back to 1970 (yields, not prices)")
    pr(f"  Can synthesize bond returns via duration model, validated against futures.")

    # Check for gaps
    pr(f"\n  Gap analysis:")
    for sym, _, data in sources:
        if data is None: continue
        monthly = data.resample("MS").last().dropna()
        full = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
        gaps = full.difference(monthly.index)
        pr(f"    {sym}: {len(gaps)} missing months" + (f" ({', '.join(d.strftime('%Y-%m') for d in gaps[:5])}{'...' if len(gaps)>5 else ''})" if len(gaps) > 0 else ""))

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2: BUILD CANDIDATE RETURN SERIES
    # ══════════════════════════════════════════════════════════════════════════

    pr(f"\n\n{'='*80}")
    pr(f"  PHASE 2: BUILD RETURN SERIES")
    pr("=" * 50)

    # 10-YEAR CANDIDATES

    # ^TNX duration model: optimize duration against ZN futures
    pr(f"\n  10-YEAR DURATION OPTIMIZATION (^TNX model vs ZN=F)")
    pr(f"  Scanning duration 4.0 to 10.0...")

    tnx_monthly = tnx.resample("MS").last().dropna() if tnx is not None else None
    zn_ret = monthly_price_return(zn) if zn is not None else None
    ief_ret = monthly_price_return(ief) if ief is not None else None

    best_10y = {"dur": 7.5, "corr": 0}
    dur_scan_10 = {}
    if tnx_monthly is not None and zn_ret is not None:
        for dur in np.arange(4.0, 10.1, 0.5):
            proxy = duration_model(tnx_monthly, dur)
            m = validate(proxy, zn_ret, f"^TNX dur={dur:.1f} vs ZN")
            if m is not None:
                dur_scan_10[dur] = m
                if m["corr"] > best_10y["corr"]:
                    best_10y = {"dur": dur, "corr": m["corr"], "m": m}

        pr(f"\n  {'Duration':>8} {'Corr':>7} {'Beta':>7} {'TE':>7}")
        pr(f"  {'-'*8} {'-'*7} {'-'*7} {'-'*7}")
        for dur in sorted(dur_scan_10):
            m = dur_scan_10[dur]
            marker = " ← best" if dur == best_10y["dur"] else ""
            pr(f"  {dur:>8.1f} {m['corr']:>7.4f} {m['beta']:>7.3f} {m['te']*100:>6.1f}%{marker}")

    # 30-YEAR CANDIDATES

    pr(f"\n  30-YEAR DURATION OPTIMIZATION (^TYX model vs ZB=F)")
    pr(f"  Scanning duration 12.0 to 22.0...")

    tyx_monthly = tyx.resample("MS").last().dropna() if tyx is not None else None
    zb_ret = monthly_price_return(zb) if zb is not None else None
    tlt_ret = monthly_price_return(tlt) if tlt is not None else None

    best_30y = {"dur": 17.0, "corr": 0}
    dur_scan_30 = {}
    if tyx_monthly is not None and zb_ret is not None:
        for dur in np.arange(12.0, 22.1, 0.5):
            proxy = duration_model(tyx_monthly, dur)
            m = validate(proxy, zb_ret, f"^TYX dur={dur:.1f} vs ZB")
            if m is not None:
                dur_scan_30[dur] = m
                if m["corr"] > best_30y["corr"]:
                    best_30y = {"dur": dur, "corr": m["corr"], "m": m}

        pr(f"\n  {'Duration':>8} {'Corr':>7} {'Beta':>7} {'TE':>7}")
        pr(f"  {'-'*8} {'-'*7} {'-'*7} {'-'*7}")
        for dur in sorted(dur_scan_30):
            m = dur_scan_30[dur]
            marker = " ← best" if dur == best_30y["dur"] else ""
            pr(f"  {dur:>8.1f} {m['corr']:>7.4f} {m['beta']:>7.3f} {m['te']*100:>6.1f}%{marker}")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 3: VALIDATE FUTURES vs ETFs
    # ══════════════════════════════════════════════════════════════════════════

    pr(f"\n\n{'='*80}")
    pr(f"  PHASE 3: FUTURES vs ETF VALIDATION")
    pr("=" * 50)

    pr(f"\n  ZN=F vs IEF (10-Year)")
    pr(f"  ---------------------")
    zn_ief = validate(zn_ret, ief_ret, "ZN vs IEF") if zn_ret is not None and ief_ret is not None else None
    pr(fmt_validation(zn_ief))

    pr(f"\n  ZB=F vs TLT (30-Year)")
    pr(f"  ---------------------")
    zb_tlt = validate(zb_ret, tlt_ret, "ZB vs TLT") if zb_ret is not None and tlt_ret is not None else None
    pr(fmt_validation(zb_tlt))

    # Also validate full chain: duration model → ETF (skipping futures)
    pr(f"\n  ^TNX dur={best_10y['dur']:.1f} vs IEF (end-to-end 10Y)")
    pr(f"  " + "-" * 40)
    if tnx_monthly is not None and ief_ret is not None:
        proxy_10 = duration_model(tnx_monthly, best_10y["dur"])
        m_10_ief = validate(proxy_10, ief_ret, f"^TNX dur={best_10y['dur']:.1f} vs IEF")
        pr(fmt_validation(m_10_ief))

    pr(f"\n  ^TYX dur={best_30y['dur']:.1f} vs TLT (end-to-end 30Y)")
    pr(f"  " + "-" * 40)
    if tyx_monthly is not None and tlt_ret is not None:
        proxy_30 = duration_model(tyx_monthly, best_30y["dur"])
        m_30_tlt = validate(proxy_30, tlt_ret, f"^TYX dur={best_30y['dur']:.1f} vs TLT")
        pr(fmt_validation(m_30_tlt))

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 4: SELECT INSTRUMENT AND BUILD COMBINED SERIES
    # ══════════════════════════════════════════════════════════════════════════

    pr(f"\n\n{'='*80}")
    pr(f"  PHASE 4: INSTRUMENT SELECTION & COMBINED SERIES")
    pr("=" * 50)

    # Compare 10Y vs 30Y
    pr(f"\n  HEAD-TO-HEAD COMPARISON")
    pr(f"  {'Criterion':<30} {'10-Year (ZN/IEF)':>18} {'30-Year (ZB/TLT)':>18}")
    pr(f"  {'-'*30} {'-'*18} {'-'*18}")

    zn_corr = zn_ief["corr"] if zn_ief else 0
    zb_corr = zb_tlt["corr"] if zb_tlt else 0
    m10_corr = m_10_ief["corr"] if m_10_ief else 0
    m30_corr = m_30_tlt["corr"] if m_30_tlt else 0

    pr(f"  {'Futures-ETF corr':<30} {zn_corr:>18.4f} {zb_corr:>18.4f}")
    pr(f"  {'DurModel-ETF corr':<30} {m10_corr:>18.4f} {m30_corr:>18.4f}")
    pr(f"  {'DurModel-Futures corr':<30} {best_10y['corr']:>18.4f} {best_30y['corr']:>18.4f}")
    pr(f"  {'Optimal duration':<30} {best_10y['dur']:>18.1f} {best_30y['dur']:>18.1f}")

    tnx_start = tnx_monthly.index.min().strftime("%Y-%m") if tnx_monthly is not None else "N/A"
    tyx_start = tyx_monthly.index.min().strftime("%Y-%m") if tyx_monthly is not None else "N/A"
    pr(f"  {'Yield data starts':<30} {tnx_start:>18} {tyx_start:>18}")

    # Equity correlation (diversification value)
    spy_d = yf_daily("SPY", "2000-01-01")
    if spy_d is not None:
        spy_ret = monthly_price_return(spy_d)
        for label, fut_ret in [("ZN-SPY corr", zn_ret), ("ZB-SPY corr", zb_ret)]:
            if fut_ret is not None:
                common = fut_ret.index.intersection(spy_ret.index)
                c = fut_ret.reindex(common).corr(spy_ret.reindex(common))
                pr(f"  {label:<30} {c:>18.3f}" + (" " * 18 if "ZN" in label else ""))

    # Decision
    # Prefer the one with highest end-to-end (duration model → ETF) correlation
    # because that's what matters for the pre-2000 era
    if m10_corr > m30_corr:
        choice = "10Y"
        choice_yield = tnx_monthly
        choice_dur = best_10y["dur"]
        choice_fut_ret = zn_ret
        choice_etf_ret = ief_ret
        choice_etf_name = "IEF"
        choice_yield_sym = "^TNX"
        choice_fut_sym = "ZN=F"
        choice_etf_daily = ief
    else:
        choice = "30Y"
        choice_yield = tyx_monthly
        choice_dur = best_30y["dur"]
        choice_fut_ret = zb_ret
        choice_etf_ret = tlt_ret
        choice_etf_name = "TLT"
        choice_yield_sym = "^TYX"
        choice_fut_sym = "ZB=F"
        choice_etf_daily = tlt

    pr(f"\n  SELECTION: {choice} ({choice_yield_sym} → {choice_fut_sym} → {choice_etf_name})")
    pr(f"  Duration model parameter: {choice_dur:.1f}")

    # Build 3-layer combined series
    pr(f"\n  Building combined return series...")

    # Layer 1: Duration model from yield index (earliest history)
    layer1 = duration_model(choice_yield, choice_dur)
    layer1_start = layer1.index.min()
    pr(f"    Layer 1 (duration model): {layer1.index.min().strftime('%Y-%m')} to {layer1.index.max().strftime('%Y-%m')} ({len(layer1)} months)")

    # Layer 2: Futures returns (2000+)
    layer2 = choice_fut_ret
    if layer2 is not None:
        layer2_start = layer2.index.min()
        pr(f"    Layer 2 (futures):        {layer2.index.min().strftime('%Y-%m')} to {layer2.index.max().strftime('%Y-%m')} ({len(layer2)} months)")
    else:
        layer2_start = None
        pr(f"    Layer 2 (futures):        NOT AVAILABLE")

    # Layer 3: ETF returns (2002+)
    layer3 = choice_etf_ret
    if layer3 is not None:
        layer3_start = layer3.index.min()
        pr(f"    Layer 3 (ETF):           {layer3.index.min().strftime('%Y-%m')} to {layer3.index.max().strftime('%Y-%m')} ({len(layer3)} months)")
    else:
        layer3_start = None
        pr(f"    Layer 3 (ETF):           NOT AVAILABLE")

    # Splice: each successive layer overwrites
    combined = layer1.copy()
    splice_points = []
    if layer2 is not None and len(layer2) > 0:
        cutoff2 = layer2.index.min()
        combined = pd.concat([combined[combined.index < cutoff2], layer2]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        splice_points.append(f"Futures from {cutoff2.strftime('%Y-%m')}")
    if layer3 is not None and len(layer3) > 0:
        cutoff3 = layer3.index.min()
        combined = pd.concat([combined[combined.index < cutoff3], layer3]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        splice_points.append(f"ETF from {cutoff3.strftime('%Y-%m')}")

    combined = combined.dropna()
    pr(f"\n    Combined series: {combined.index.min().strftime('%Y-%m')} to {combined.index.max().strftime('%Y-%m')} ({len(combined)} months)")
    pr(f"    Splice points: {'; '.join(splice_points)}")
    pr(f"    Mean return: {combined.mean()*100:.3f}%/mo ({combined.mean()*12*100:.2f}% ann)")
    pr(f"    Volatility: {combined.std()*np.sqrt(12)*100:.1f}% ann")

    # Build price series for Faber SMAs
    # Use yield index → synthetic price via cumulative return
    # Then splice with ETF price
    synth_price = (1 + combined).cumprod() * 100  # normalize to 100 at start
    pr(f"    Synthetic price range: {synth_price.min():.1f} to {synth_price.max():.1f}")

    # For Faber, also need daily prices (use ETF daily where available, futures before)
    daily_price_combined = pd.Series(dtype=float)
    if choice == "10Y":
        # Use ZN daily prices pre-ETF, IEF daily after
        if zn is not None:
            daily_price_combined = zn.copy()
        if ief is not None:
            cutoff = ief.index.min()
            daily_price_combined = pd.concat([
                daily_price_combined[daily_price_combined.index < cutoff],
                ief
            ]).sort_index()
    else:
        if zb is not None:
            daily_price_combined = zb.copy()
        if tlt is not None:
            cutoff = tlt.index.min()
            daily_price_combined = pd.concat([
                daily_price_combined[daily_price_combined.index < cutoff],
                tlt
            ]).sort_index()

    daily_price_combined = daily_price_combined[~daily_price_combined.index.duplicated(keep="last")]

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 5: INTEGRATE INTO PARQUET PIPELINE
    # ══════════════════════════════════════════════════════════════════════════

    pr(f"\n\n{'='*80}")
    pr(f"  PHASE 5: PARQUET INTEGRATION")
    pr("=" * 50)

    # Load existing asset returns
    existing = pd.read_parquet(DATA_DIR / "roth_asset_returns.parquet")
    pr(f"\n  Existing file: {existing.shape[0]} rows × {existing.shape[1]} cols")
    pr(f"  Columns: {list(existing.columns)}")
    pr(f"  Date range: {existing.index.min().strftime('%Y-%m')} to {existing.index.max().strftime('%Y-%m')}")

    # Align combined treasury returns to existing index
    combined.name = "TSY"
    combined_aligned = combined.reindex(existing.index)

    # Create new asset returns with TSY
    new_asset_ret = existing.copy()
    new_asset_ret["TSY"] = combined_aligned

    tsy_coverage = new_asset_ret["TSY"].notna().sum()
    tsy_nulls = new_asset_ret["TSY"].isna().sum()
    tsy_first = new_asset_ret["TSY"].first_valid_index()

    pr(f"\n  TSY column added:")
    pr(f"    Coverage: {tsy_coverage} months with data, {tsy_nulls} NaN")
    pr(f"    First valid: {tsy_first.strftime('%Y-%m') if tsy_first else 'N/A'}")
    pr(f"    Last valid:  {new_asset_ret['TSY'].last_valid_index().strftime('%Y-%m')}")

    # Save new parquet
    out_path = DATA_DIR / "roth_asset_returns_5asset.parquet"
    new_asset_ret.to_parquet(out_path)
    pr(f"\n  Saved: {out_path}")
    pr(f"    Shape: {new_asset_ret.shape}")

    # Save daily treasury prices for Faber SMAs
    tsy_price_path = DATA_DIR / "treasury_daily_prices.parquet"
    pd.DataFrame({"TSY": daily_price_combined}).to_parquet(tsy_price_path)
    pr(f"  Saved: {tsy_price_path}")
    pr(f"    Daily prices: {daily_price_combined.index.min().date()} to {daily_price_combined.index.max().date()}")

    # ══════════════════════════════════════════════════════════════════════════
    # VALIDATION REPORT
    # ══════════════════════════════════════════════════════════════════════════

    pr(f"\n\n{'='*80}")
    pr(f"  VALIDATION REPORT")
    pr("=" * 50)

    pr(f"\n  DATA SOURCE")
    pr(f"  Instrument:     {choice} Treasury")
    pr(f"  Yield index:    {choice_yield_sym} (yfinance, {choice_yield.index.min().strftime('%Y-%m')} to present)")
    pr(f"  Futures:        {choice_fut_sym} (yfinance, 2000-09 to present)")
    pr(f"  ETF:            {choice_etf_name} (yfinance, 2002-07 to present)")
    pr(f"  Duration param: {choice_dur:.1f}")

    pr(f"\n  VALIDATION CHAIN")
    pr(f"  Duration model → Futures:  corr = {best_10y['corr'] if choice=='10Y' else best_30y['corr']:.4f} {'PASS' if (best_10y['corr'] if choice=='10Y' else best_30y['corr']) > 0.90 else 'FAIL'}")
    pr(f"  Futures → ETF:             corr = {zn_corr if choice=='10Y' else zb_corr:.4f} {'PASS' if (zn_corr if choice=='10Y' else zb_corr) > 0.90 else 'FAIL'}")
    pr(f"  Duration model → ETF:      corr = {m10_corr if choice=='10Y' else m30_corr:.4f} {'PASS' if (m10_corr if choice=='10Y' else m30_corr) > 0.90 else 'FAIL'}")

    pr(f"\n  COMBINED SERIES")
    pr(f"  Pre-2000:  {choice_yield_sym} duration model (dur={choice_dur:.1f})")
    pr(f"  2000-2002: {choice_fut_sym} futures returns")
    pr(f"  2002+:     {choice_etf_name} ETF returns")
    pr(f"  Splice:    {'; '.join(splice_points)}")
    pr(f"  Coverage:  {combined.index.min().strftime('%Y-%m')} to {combined.index.max().strftime('%Y-%m')} ({len(combined)} months)")
    pr(f"  Gaps:      {combined.isna().sum()}")

    pr(f"\n  OUTPUT FILES")
    pr(f"  Asset returns: {out_path}")
    pr(f"  Daily prices:  {tsy_price_path}")

    all_pass = all([
        (best_10y["corr"] if choice == "10Y" else best_30y["corr"]) > 0.90,
        (zn_corr if choice == "10Y" else zb_corr) > 0.90,
    ])
    pr(f"\n  READY FOR BACKTEST: {'YES' if all_pass else 'NO — validation failed'}")

    # Save report
    report_path = OUTDIR / "treasury_pipeline_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Report saved: {report_path}")


if __name__ == "__main__":
    main()

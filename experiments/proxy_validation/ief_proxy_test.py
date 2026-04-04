"""Test IEF proxy quality with GS10 duration model."""

import os, sys, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dotenv import load_dotenv; load_dotenv()
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

OUTDIR = os.path.join(os.path.dirname(__file__), "output")


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


def duration_model(yields_monthly, duration):
    """Exact pipeline formula: -duration * (dy/100) + lagged_yield/1200."""
    dy = yields_monthly.diff()
    coupon = yields_monthly.shift(1) / 1200
    return (-duration * dy / 100 + coupon).dropna()


def metrics(proxy, etf):
    common = proxy.index.intersection(etf.index).sort_values()
    p = proxy.reindex(common).dropna()
    e = etf.reindex(common).dropna()
    common = p.index.intersection(e.index)
    p, e = p.reindex(common), e.reindex(common)
    diff = p - e
    corr = p.corr(e)
    te_ann = diff.std() * np.sqrt(12)
    ann_diff = diff.mean() * 12
    max_idx = diff.abs().idxmax()
    return {
        "n": len(common),
        "start": common.min().strftime("%Y-%m"),
        "end": common.max().strftime("%Y-%m"),
        "corr": corr,
        "mean_etf": e.mean(),
        "mean_proxy": p.mean(),
        "ann_diff": ann_diff,
        "te_monthly": diff.std(),
        "te_ann": te_ann,
        "max_div": diff.loc[max_idx],
        "max_div_date": max_idx.strftime("%Y-%m"),
    }


def rate_metric(val, thresholds):
    """thresholds = (good_bound, accept_bound) for abs comparison."""
    if abs(val) > thresholds[0]:
        return "Good"
    elif abs(val) > thresholds[1]:
        return "Acceptable"
    return "Poor"


def rate_all(corr, ann_diff, te_ann):
    c = "Good" if abs(corr) > 0.95 else ("Acceptable" if abs(corr) > 0.85 else "Poor")
    d = "Good" if abs(ann_diff) < 0.01 else ("Acceptable" if abs(ann_diff) < 0.03 else "Poor")
    t = "Good" if te_ann < 0.05 else ("Acceptable" if te_ann < 0.10 else "Poor")
    scores = [c, d, t]
    overall = "Poor" if "Poor" in scores else ("Acceptable" if "Acceptable" in scores else "Good")
    return overall, scores


def main():
    lines = []
    def pr(s=""):
        print(s)
        lines.append(s)

    pr("=" * 70)
    pr("  IEF PROXY VALIDATION")
    pr("=" * 70)

    # Fetch data
    print("\nFetching GS10 from FRED...")
    gs10 = fred("GS10", "1960-01-01")
    if gs10 is None:
        pr("ERROR: Could not fetch GS10"); return
    gs10_m = gs10.resample("MS").last().dropna()
    pr(f"\nGS10: {gs10_m.index.min().date()} to {gs10_m.index.max().date()} ({len(gs10_m)} months)")

    print("Fetching IEF from yfinance...")
    ief_ret = yf_monthly_ret("IEF", "2002-07-01")
    if ief_ret is None:
        pr("ERROR: Could not fetch IEF"); return
    pr(f"IEF:  {ief_ret.index.min().date()} to {ief_ret.index.max().date()} ({len(ief_ret)} months)")

    print("Fetching VGLT from yfinance...")
    vglt_ret = yf_monthly_ret("VGLT", "2009-11-01")
    vglt_label = "VGLT"
    if vglt_ret is None or len(vglt_ret) < 10:
        print("VGLT unavailable, fetching TLT...")
        vglt_ret = yf_monthly_ret("TLT", "2002-07-01")
        vglt_label = "TLT"

    # ── Primary test: IEF vs GS10 dur=7.5 ────────────────────────────────────
    dur = 7.5
    proxy = duration_model(gs10_m, dur)
    m = metrics(proxy, ief_ret)
    overall, sub = rate_all(m["corr"], m["ann_diff"], m["te_ann"])

    pr(f"\n\nIEF vs GS10 duration model (duration={dur})")
    pr("-" * 50)
    pr(f"  Overlap period:       {m['start']} to {m['end']} ({m['n']} months)")
    pr(f"  Correlation:          {m['corr']:.4f}    [{sub[0]}]")
    pr(f"  Mean monthly return — ETF: {m['mean_etf']*100:.3f}%  Proxy: {m['mean_proxy']*100:.3f}%  Diff: {(m['mean_proxy']-m['mean_etf'])*100:.3f}%")
    dir_ = "overstates" if m["ann_diff"] > 0 else "understates"
    pr(f"  Annualized ret diff:  {m['ann_diff']*100:.2f}% (proxy {dir_})    [{sub[1]}]")
    pr(f"  Tracking error:       {m['te_monthly']*100:.2f}% monthly, {m['te_ann']*100:.2f}% annualized    [{sub[2]}]")
    pr(f"  Max single-mo diverge:{m['max_div']*100:.2f}% ({m['max_div_date']})")
    pr(f"  ── Overall rating:    {overall}")

    # ── Duration sensitivity ──────────────────────────────────────────────────
    pr(f"\n\nDURATION SENSITIVITY")
    pr("-" * 50)
    pr(f"  {'Duration':>8}  {'Correlation':>11}  {'Ann.Ret.Diff':>12}  {'Track.Err(ann)':>14}")
    pr(f"  {'--------':>8}  {'-----------':>11}  {'------------':>12}  {'--------------':>14}")

    best_dur, best_corr = dur, m["corr"]
    scan_results = {}

    for d in np.arange(5.0, 12.1, 0.5):
        p = duration_model(gs10_m, d)
        r = metrics(p, ief_ret)
        scan_results[d] = r
        if abs(r["corr"]) > abs(best_corr):
            best_corr = r["corr"]
            best_dur = d
        pr(f"  {d:>8.1f}  {r['corr']:>11.4f}  {r['ann_diff']*100:>11.2f}%  {r['te_ann']*100:>13.2f}%")

    pr(f"\n  Optimal duration: {best_dur:.1f} (correlation = {best_corr:.4f})")

    # Show optimal if different from 7.5
    if best_dur != 7.5:
        opt_m = scan_results[best_dur]
        opt_overall, opt_sub = rate_all(opt_m["corr"], opt_m["ann_diff"], opt_m["te_ann"])
        pr(f"\n\nIEF vs GS10 duration model (OPTIMAL duration={best_dur})")
        pr("-" * 50)
        pr(f"  Overlap period:       {opt_m['start']} to {opt_m['end']} ({opt_m['n']} months)")
        pr(f"  Correlation:          {opt_m['corr']:.4f}    [{opt_sub[0]}]")
        dir_ = "overstates" if opt_m["ann_diff"] > 0 else "understates"
        pr(f"  Annualized ret diff:  {opt_m['ann_diff']*100:.2f}% (proxy {dir_})    [{opt_sub[1]}]")
        pr(f"  Tracking error:       {opt_m['te_ann']*100:.2f}% annualized    [{opt_sub[2]}]")
        pr(f"  Max single-mo diverge:{opt_m['max_div']*100:.2f}% ({opt_m['max_div_date']})")
        pr(f"  ── Overall rating:    {opt_overall}")

    # ── Compare IEF vs VGLT proxy quality ─────────────────────────────────────
    pr(f"\n\nCOMPARISON: IEF vs VGLT PROXY QUALITY")
    pr("-" * 50)

    # Use optimal duration for IEF
    ief_proxy = duration_model(gs10_m, best_dur)
    ief_m = metrics(ief_proxy, ief_ret)
    ief_overall, _ = rate_all(ief_m["corr"], ief_m["ann_diff"], ief_m["te_ann"])

    # VGLT with dur=17 (original pipeline)
    vglt_proxy = duration_model(gs10_m, 17.0)
    if vglt_ret is not None:
        vglt_m = metrics(vglt_proxy, vglt_ret)
        vglt_overall, _ = rate_all(vglt_m["corr"], vglt_m["ann_diff"], vglt_m["te_ann"])
    else:
        vglt_m = None
        vglt_overall = "N/A"

    pr(f"  {'Metric':<22} {'IEF (dur='+str(best_dur)+')':>16}    {vglt_label+' (dur=17)':>16}")
    pr(f"  {'------':<22} {'-------------':>16}    {'-------------':>16}")
    pr(f"  {'Correlation':<22} {ief_m['corr']:>16.4f}    {vglt_m['corr'] if vglt_m else float('nan'):>16.4f}")
    pr(f"  {'Ann. return diff':<22} {ief_m['ann_diff']*100:>15.2f}%    {vglt_m['ann_diff']*100 if vglt_m else float('nan'):>15.2f}%")
    pr(f"  {'Tracking error (ann)':<22} {ief_m['te_ann']*100:>15.2f}%    {vglt_m['te_ann']*100 if vglt_m else float('nan'):>15.2f}%")
    pr(f"  {'Overlap months':<22} {ief_m['n']:>16}    {vglt_m['n'] if vglt_m else 0:>16}")
    pr(f"  {'Rating':<22} {ief_overall:>16}    {vglt_overall:>16}")

    if vglt_m:
        corr_imp = ief_m["corr"] - vglt_m["corr"]
        te_imp = vglt_m["te_ann"] - ief_m["te_ann"]
        pr(f"\n  Correlation improvement:     {'+' if corr_imp > 0 else ''}{corr_imp:.4f}")
        pr(f"  Tracking error improvement:  {'+' if te_imp > 0 else ''}{te_imp*100:.2f}% (lower is better)")

    # ── Recommendation ────────────────────────────────────────────────────────
    pr(f"\n\nRECOMMENDATION")
    pr("=" * 50)

    if ief_overall in ("Good", "Acceptable"):
        pr(f"  REPLACE VGLT with IEF")
        pr(f"")
        pr(f"  Rationale: IEF (7-10Y treasuries) matches GS10 duration model")
        pr(f"  far better than VGLT (20-30Y). The proxy-ETF correlation jumps")
        pr(f"  from {vglt_m['corr']:.3f} to {ief_m['corr']:.3f}, and tracking error drops from")
        pr(f"  {vglt_m['te_ann']*100:.1f}% to {ief_m['te_ann']*100:.1f}% annualized.")
        pr(f"")
        pr(f"  Implementation:")
        pr(f"  - Pre-2002 proxy: GS10 with duration={best_dur}")
        pr(f"  - ETF: IEF from 2002-07 onward")
        pr(f"  - IEF inception: 2002-07-22 — covers 22+ years of ETF data")
        pr(f"  - Pre-2002 proxy needed: ~17 years (1985–2002) for Harvey lookback")
        pr(f"  - Role in portfolio: intermediate bond hedge (lower vol than VGLT)")
        pr(f"  - Trade-off: lower duration = less equity crash hedge, but higher")
        pr(f"    proxy fidelity means Harvey similarity estimates are trustworthy")
    else:
        pr(f"  KEEP VGLT — IEF proxy also rated {ief_overall}")
        pr(f"  Neither proxy achieves acceptable fidelity with free data.")

    # Save
    report_path = os.path.join(OUTDIR, "ief_proxy_test_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Report saved: {report_path}")


if __name__ == "__main__":
    main()

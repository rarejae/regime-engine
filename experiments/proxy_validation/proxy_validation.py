"""Proxy validation: compare pre-ETF proxies against actual ETF returns during overlap."""

import os, sys, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dotenv import load_dotenv; load_dotenv()
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# ── Fetchers (replicate exact pipeline logic) ────────────────────────────────

def _fred(sid, start="1955-01-01"):
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


def _yf_monthly(ticker, start="1985-01-01"):
    import yfinance as yf
    d = yf.download(ticker, start=start, interval="1mo", progress=False)
    if d is None or d.empty:
        return None
    p = d["Close"]
    if hasattr(p, "columns"):
        p = p.iloc[:, 0]
    p.index = pd.to_datetime(p.index).tz_localize(None).to_period("M").to_timestamp()
    return p.pct_change().dropna()


def _yf_daily(ticker, start="1985-01-01"):
    import yfinance as yf
    d = yf.download(ticker, start=start, progress=False)
    if d is None or d.empty:
        return None
    p = d["Close"]
    if hasattr(p, "columns"):
        p = p.iloc[:, 0]
    p.index = pd.to_datetime(p.index).tz_localize(None)
    return p


# ── Build proxy and ETF return series (exact pipeline transforms) ─────────────

def build_pairs():
    """Return dict of {name: (proxy_returns, etf_returns, etf_start_label)}."""
    pairs = {}

    # 1. IVV vs SPASTT01USM661N
    print("Fetching IVV proxy + ETF...")
    sp_fred = _fred("SPASTT01USM661N", "1957-01-01")
    if sp_fred is not None:
        proxy = sp_fred.resample("MS").last().pct_change().dropna()
        etf = _yf_monthly("IVV", "2000-05-01")
        if etf is not None:
            pairs["IVV"] = (proxy, etf, "2000-05")

    # 2. QQQ vs ^NDX
    print("Fetching QQQ proxy + ETF...")
    ndx = _yf_daily("^NDX", "1985-01-01")
    if ndx is not None:
        proxy = ndx.resample("MS").last().pct_change().dropna()
        proxy.index = proxy.index.to_period("M").to_timestamp()
        etf = _yf_monthly("QQQ", "1999-03-01")
        if etf is not None:
            pairs["QQQ"] = (proxy, etf, "1999-03")

    # 3. VGLT vs GS10 duration model
    print("Fetching VGLT proxy + ETF...")
    gs10 = _fred("GS10", "1960-01-01")
    if gs10 is not None:
        gs10_m = gs10.resample("MS").last().dropna()
        dy = gs10_m.diff()
        coupon = gs10_m.shift(1) / 1200
        proxy = (-17.0 * dy / 100 + coupon).dropna()
        # Try VGLT first, then TLT
        etf = _yf_monthly("VGLT", "2009-11-01")
        label = "2009-11"
        if etf is None or len(etf) < 10:
            etf = _yf_monthly("TLT", "2002-07-01")
            label = "2002-07"
        if etf is not None:
            pairs["VGLT"] = (proxy, etf, label)

    # 4. IAU vs WPU1017
    print("Fetching IAU proxy + ETF...")
    gold_ppi = _fred("WPU1017", "1960-01-01")
    if gold_ppi is not None and len(gold_ppi) > 200:
        proxy = gold_ppi.resample("MS").last().pct_change().dropna()
        etf = _yf_monthly("GLD", "2004-11-01")
        label = "2004-11"
        if etf is None or len(etf) < 10:
            etf = _yf_monthly("IAU", "2005-01-01")
            label = "2005-01"
        if etf is not None:
            pairs["IAU"] = (proxy, etf, label)

    # 5. DBC vs PPIACO
    print("Fetching DBC proxy + ETF...")
    ppi = _fred("PPIACO", "1960-01-01")
    if ppi is not None:
        proxy = ppi.resample("MS").last().pct_change().dropna()
        etf = _yf_monthly("DBC", "2006-02-01")
        if etf is not None:
            pairs["DBC"] = (proxy, etf, "2006-02")

    return pairs


def compute_metrics(proxy, etf, name, etf_start_label):
    """Align to overlap and compute all comparison metrics."""
    # Align on common dates
    common = proxy.index.intersection(etf.index)
    common = common.sort_values()
    if len(common) < 12:
        return None

    p = proxy.reindex(common).dropna()
    e = etf.reindex(common).dropna()
    common = p.index.intersection(e.index)
    p = p.reindex(common)
    e = e.reindex(common)

    diff = p - e
    corr = p.corr(e)
    mean_etf = e.mean()
    mean_proxy = p.mean()
    mean_diff = diff.mean()
    ann_diff = mean_diff * 12
    te_monthly = diff.std()
    te_ann = te_monthly * np.sqrt(12)
    max_div_idx = diff.abs().idxmax()
    max_div = diff.loc[max_div_idx]

    # Cumulative return divergence
    cum_etf = (1 + e).cumprod()
    cum_proxy = (1 + p).cumprod()

    return {
        "name": name,
        "etf_start": etf_start_label,
        "n_months": len(common),
        "overlap_start": common.min().strftime("%Y-%m"),
        "overlap_end": common.max().strftime("%Y-%m"),
        "correlation": corr,
        "mean_etf": mean_etf,
        "mean_proxy": mean_proxy,
        "mean_diff": mean_diff,
        "ann_return_diff": ann_diff,
        "te_monthly": te_monthly,
        "te_ann": te_ann,
        "max_divergence": max_div,
        "max_divergence_date": max_div_idx.strftime("%Y-%m"),
        "proxy": p,
        "etf": e,
        "cum_etf": cum_etf,
        "cum_proxy": cum_proxy,
    }


def rate(corr, ann_diff, te_ann):
    """Rate proxy quality."""
    scores = []
    if abs(corr) > 0.95: scores.append("Good")
    elif abs(corr) > 0.85: scores.append("Acceptable")
    else: scores.append("Poor")

    if abs(ann_diff) < 0.01: scores.append("Good")
    elif abs(ann_diff) < 0.03: scores.append("Acceptable")
    else: scores.append("Poor")

    if te_ann < 0.05: scores.append("Good")
    elif te_ann < 0.10: scores.append("Acceptable")
    else: scores.append("Poor")

    # Overall: worst of the three
    if "Poor" in scores: return "Poor", scores
    if "Acceptable" in scores: return "Acceptable", scores
    return "Good", scores


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_scatter(results, outpath):
    """Scatter plot: proxy return vs ETF return, one subplot per asset."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.2), squeeze=False)
    axes = axes[0]

    for i, r in enumerate(results):
        ax = axes[i]
        p, e = r["proxy"], r["etf"]
        ax.scatter(e * 100, p * 100, alpha=0.35, s=12, edgecolors="none")

        # Fit line
        m, b = np.polyfit(e, p, 1)
        x_line = np.linspace(e.min(), e.max(), 50)
        ax.plot(x_line * 100, (m * x_line + b) * 100, "r-", linewidth=1.2,
                label=f"β={m:.2f}, r={r['correlation']:.3f}")

        # 45-degree reference
        lim = max(abs(e.min()), abs(e.max()), abs(p.min()), abs(p.max())) * 100 * 1.1
        ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3, linewidth=0.8)

        ax.set_xlabel("ETF return (%)")
        ax.set_ylabel("Proxy return (%)")
        ax.set_title(f"{r['name']}\nr={r['correlation']:.3f}, n={r['n_months']}")
        ax.legend(fontsize=7, loc="upper left")
        ax.set_aspect("equal", adjustable="datalim")

    fig.suptitle("Proxy vs ETF Monthly Returns — Overlap Period", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    plt.close(fig)


def plot_cumulative_divergence(results, outpath):
    """Cumulative return gap over time."""
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.2 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for i, r in enumerate(results):
        ax = axes[i]
        cum_gap = r["cum_proxy"] - r["cum_etf"]
        ax.fill_between(cum_gap.index, 0, cum_gap.values,
                        where=cum_gap.values >= 0, alpha=0.3, color="green", label="Proxy > ETF")
        ax.fill_between(cum_gap.index, 0, cum_gap.values,
                        where=cum_gap.values < 0, alpha=0.3, color="red", label="Proxy < ETF")
        ax.plot(cum_gap.index, cum_gap.values, "k-", linewidth=0.7)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_title(f"{r['name']} — Cumulative Return Divergence (proxy − ETF)")
        ax.set_ylabel("Cumulative gap")
        ax.legend(fontsize=7)

        # Also plot both cumulative lines on secondary axis inset
        ax2 = ax.twinx()
        ax2.plot(r["cum_etf"].index, r["cum_etf"].values, "b-", alpha=0.3, linewidth=0.8, label="ETF")
        ax2.plot(r["cum_proxy"].index, r["cum_proxy"].values, "orange", alpha=0.3, linewidth=0.8, label="Proxy")
        ax2.set_ylabel("Cumulative return ($1)", fontsize=7)
        ax2.tick_params(axis="y", labelsize=7)

    fig.suptitle("Proxy vs ETF — Cumulative Return Divergence", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    outdir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(outdir, exist_ok=True)

    print("=" * 80)
    print("  PRE-ETF PROXY VALIDATION AUDIT")
    print("=" * 80)
    print()

    pairs = build_pairs()
    if not pairs:
        print("ERROR: No data fetched. Check FRED_API_KEY and network.")
        return

    results = []
    for name in ["IVV", "QQQ", "VGLT", "IAU", "DBC"]:
        if name not in pairs:
            print(f"\n⚠ {name}: DATA NOT AVAILABLE — skipping")
            continue
        proxy, etf, label = pairs[name]
        m = compute_metrics(proxy, etf, name, label)
        if m is None:
            print(f"\n⚠ {name}: Insufficient overlap — skipping")
            continue
        results.append(m)

    if not results:
        print("ERROR: No valid overlap periods found.")
        return

    # ── Print detailed report ─────────────────────────────────────────────────
    report_lines = []
    def pr(line=""):
        print(line)
        report_lines.append(line)

    pr()
    pr("=" * 80)
    pr("  PROXY VALIDATION REPORT")
    pr("=" * 80)

    proxy_descriptions = {
        "IVV": "SPASTT01USM661N (OECD US Share Prices)",
        "QQQ": "^NDX (Nasdaq-100 Index, daily → monthly)",
        "VGLT": "GS10 duration model: −17 × (Δyield/100) + coupon",
        "IAU": "WPU1017 (PPI Precious Metals & Alloys)",
        "DBC": "PPIACO (PPI All Commodities)",
    }

    for r in results:
        overall, sub = rate(r["correlation"], r["ann_return_diff"], r["te_ann"])
        pr()
        pr(f"{r['name']} vs {proxy_descriptions[r['name']]}")
        pr("-" * 70)
        pr(f"  Overlap period:       {r['overlap_start']} to {r['overlap_end']} ({r['n_months']} months)")
        pr(f"  Correlation:          {r['correlation']:.4f}    [{sub[0]}]")
        pr(f"  Mean monthly return — ETF: {r['mean_etf']*100:.3f}%  Proxy: {r['mean_proxy']*100:.3f}%  Diff: {r['mean_diff']*100:.3f}%")
        direction = "overstates" if r["ann_return_diff"] > 0 else "understates"
        pr(f"  Annualized ret diff:  {r['ann_return_diff']*100:.2f}% (proxy {direction} by this much)    [{sub[1]}]")
        pr(f"  Tracking error:       {r['te_monthly']*100:.2f}% monthly, {r['te_ann']*100:.2f}% annualized    [{sub[2]}]")
        pr(f"  Max single-mo diverge:{r['max_divergence']*100:.2f}% ({r['max_divergence_date']})")
        pr(f"  ── Overall rating:    {overall.upper()}")

    # ── Summary table ─────────────────────────────────────────────────────────
    pr()
    pr()
    pr("=" * 80)
    pr("  PROXY QUALITY SUMMARY")
    pr("=" * 80)
    pr()
    pr(f"{'Asset':<8} {'Correlation':>12} {'Ann.Ret.Diff':>13} {'Track.Err(ann)':>15} {'Rating':>10}")
    pr(f"{'-'*8} {'-'*12} {'-'*13} {'-'*15} {'-'*10}")
    for r in results:
        overall, _ = rate(r["correlation"], r["ann_return_diff"], r["te_ann"])
        pr(f"{r['name']:<8} {r['correlation']:>12.4f} {r['ann_return_diff']*100:>12.2f}% {r['te_ann']*100:>14.2f}% {overall:>10}")

    # ── Impact assessment ─────────────────────────────────────────────────────
    pr()
    pr()
    pr("=" * 80)
    pr("  IMPACT ON HARVEY SIMILARITY")
    pr("=" * 80)

    # Load actual roth returns to count proxy-only months
    roth_path = os.path.join(os.path.dirname(__file__), "..", "data", "macro", "roth_asset_returns.parquet")
    try:
        roth = pd.read_parquet(roth_path)
        bt_start = pd.Timestamp("1985-01-01")
        bt_end = pd.Timestamp("2024-12-01")
        bt = roth[(roth.index >= bt_start) & (roth.index <= bt_end)]
        total_months = len(bt)
        pr(f"\n  Backtest period: 1985-01 to 2024-12 ({total_months} months)")
        pr()

        etf_starts = {"IVV": "2000-05", "QQQ": "1999-03", "VXUS": "2001-08",
                       "VGLT": "2009-11", "IAU": "2004-11", "DBC": "2006-02", "VNQ": "2004-09"}

        for name in ["IVV", "QQQ", "VGLT", "IAU", "DBC"]:
            if name in etf_starts:
                etf_start = pd.Timestamp(etf_starts[name] + "-01")
                proxy_months = len(bt[bt.index < etf_start])
                etf_months = len(bt[bt.index >= etf_start])
                pct = proxy_months / total_months * 100 if total_months > 0 else 0
                r_match = next((r for r in results if r["name"] == name), None)
                rating = rate(r_match["correlation"], r_match["ann_return_diff"], r_match["te_ann"])[0] if r_match else "N/A"
                pr(f"  {name:<6}: {proxy_months:>3} proxy-only months ({pct:.0f}%) + {etf_months:>3} ETF months  [{rating}]")
    except Exception as e:
        pr(f"\n  Could not load roth_asset_returns.parquet: {e}")

    # ── Recommendations ───────────────────────────────────────────────────────
    pr()
    pr()
    pr("=" * 80)
    pr("  RECOMMENDATIONS")
    pr("=" * 80)
    pr()

    for r in results:
        overall, sub = rate(r["correlation"], r["ann_return_diff"], r["te_ann"])
        if overall == "Good":
            pr(f"  {r['name']}: ✓ Keep proxy. High fidelity during overlap.")
        elif overall == "Acceptable":
            if abs(r["ann_return_diff"]) > 0.01:
                pr(f"  {r['name']}: ~ Acceptable but annualized return diff of {r['ann_return_diff']*100:.1f}% warrants monitoring.")
                pr(f"          Consider: level-shift adjustment or scaling factor on pre-ETF returns.")
            else:
                pr(f"  {r['name']}: ~ Acceptable. Tracking error {r['te_ann']*100:.1f}% ann. is elevated but correlation is strong.")
        else:
            pr(f"  {r['name']}: ✗ POOR proxy. Correlation {r['correlation']:.3f}, TE {r['te_ann']*100:.1f}%.")
            pr(f"          Regime similarity estimates using pre-{etf_starts.get(r['name'], '?')} data are unreliable for this asset.")
            pr(f"          Consider: replace proxy, shorten lookback, or downweight this asset in pre-ETF era.")

    # ── Save report ───────────────────────────────────────────────────────────
    report_path = os.path.join(outdir, "proxy_validation_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\n  Report saved: {report_path}")

    # ── Generate plots ────────────────────────────────────────────────────────
    print("\nGenerating visualizations...")
    plot_scatter(results, os.path.join(outdir, "proxy_correlation_scatter.png"))
    plot_cumulative_divergence(results, os.path.join(outdir, "proxy_cumulative_divergence.png"))

    print("\n✓ Proxy validation complete.")


if __name__ == "__main__":
    main()

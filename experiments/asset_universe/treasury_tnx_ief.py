"""Treasury data pipeline — ^TNX duration model pre-2002, IEF ETF post-2002."""

import sys, os, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv; load_dotenv()
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from scipy import stats as sp_stats

DATA_DIR = Path("data/macro")
OUTDIR = Path(__file__).resolve().parent / "output"
DURATION = 10.0
IEF_START = "2002-08-01"  # First full month of IEF data


def main():
    lines = []
    def pr(s=""):
        print(s); lines.append(s)

    import yfinance as yf

    pr("=" * 80)
    pr("  TREASURY DATA PIPELINE — ^TNX + IEF SPLICE")
    pr("=" * 80)

    # ── Download ──────────────────────────────────────────────────────────────

    pr(f"\nDATA SOURCES")
    pr("-" * 40)

    d = yf.download("^TNX", start="1970-01-01", progress=False)
    tnx_daily = d["Close"]
    if hasattr(tnx_daily, "columns"):
        tnx_daily = tnx_daily.iloc[:, 0]
    tnx_daily.index = pd.to_datetime(tnx_daily.index).tz_localize(None)

    ief_d = yf.download("IEF", start="2002-01-01", progress=False)
    ief_daily = ief_d["Close"]
    if hasattr(ief_daily, "columns"):
        ief_daily = ief_daily.iloc[:, 0]
    ief_daily.index = pd.to_datetime(ief_daily.index).tz_localize(None)

    pr(f"  ^TNX (10Y yield): {tnx_daily.index.min().date()} to {tnx_daily.index.max().date()} ({len(tnx_daily)} daily)")
    pr(f"  IEF (7-10Y ETF):  {ief_daily.index.min().date()} to {ief_daily.index.max().date()} ({len(ief_daily)} daily)")
    pr(f"  Splice date:       {IEF_START} (IEF takes over)")

    # ── Monthly returns ───────────────────────────────────────────────────────

    pr(f"\nMONTHLY RETURN SERIES")
    pr("-" * 40)

    # ^TNX duration model
    tnx_monthly = tnx_daily.resample("MS").last().dropna()
    dy_m = tnx_monthly.diff() / 100
    coupon_m = tnx_monthly.shift(1) / 100 / 12
    tnx_ret = (-DURATION * dy_m + coupon_m).dropna()

    # IEF monthly returns
    ief_monthly_price = ief_daily.resample("MS").last().dropna()
    ief_ret = ief_monthly_price.pct_change().dropna()
    ief_ret.index = ief_ret.index.to_period("M").to_timestamp()

    # Splice: ^TNX before IEF_START, IEF from IEF_START onward
    cutoff = pd.Timestamp(IEF_START)
    pre = tnx_ret[tnx_ret.index < cutoff]
    post = ief_ret[ief_ret.index >= cutoff]
    combined_ret = pd.concat([pre, post]).sort_index()
    combined_ret = combined_ret[~combined_ret.index.duplicated(keep="last")]

    pr(f"  Pre-ETF (^TNX dur={DURATION}): {pre.index.min().strftime('%Y-%m')} to {pre.index.max().strftime('%Y-%m')} ({len(pre)} months)")
    pr(f"  ETF era (IEF):                 {post.index.min().strftime('%Y-%m')} to {post.index.max().strftime('%Y-%m')} ({len(post)} months)")
    pr(f"  Combined:                      {combined_ret.index.min().strftime('%Y-%m')} to {combined_ret.index.max().strftime('%Y-%m')} ({len(combined_ret)} months)")
    pr(f"  Mean return: {combined_ret.mean()*100:.3f}%/mo ({combined_ret.mean()*12*100:.2f}% ann)")
    pr(f"  Volatility:  {combined_ret.std()*np.sqrt(12)*100:.1f}% ann")

    # ── Daily prices for Faber ────────────────────────────────────────────────

    pr(f"\nDAILY PRICE SERIES (for Faber SMAs)")
    pr("-" * 40)

    # ^TNX → synthetic daily prices (pre-ETF)
    dy_d = tnx_daily.diff() / 100
    coupon_d = tnx_daily.shift(1) / 100 / 252
    tnx_daily_ret = (-DURATION * dy_d + coupon_d).dropna()
    tnx_daily_price = (1 + tnx_daily_ret.fillna(0)).cumprod() * 100

    # Scale IEF price to match synthetic price at splice point
    # Find the last synthetic price before IEF starts
    cutoff_daily = ief_daily.index.min()
    synth_before = tnx_daily_price[tnx_daily_price.index < cutoff_daily]
    if len(synth_before) > 0:
        splice_level = synth_before.iloc[-1]
        ief_first = ief_daily.iloc[0]
        ief_scaled = ief_daily * (splice_level / ief_first)
    else:
        ief_scaled = ief_daily

    # Splice daily prices
    daily_pre = tnx_daily_price[tnx_daily_price.index < cutoff_daily]
    daily_post = ief_scaled[ief_scaled.index >= cutoff_daily]
    combined_daily_price = pd.concat([daily_pre, daily_post]).sort_index()
    combined_daily_price = combined_daily_price[~combined_daily_price.index.duplicated(keep="last")]

    pr(f"  Synthetic (^TNX): {daily_pre.index.min().date()} to {daily_pre.index.max().date()} ({len(daily_pre)} days)")
    pr(f"  IEF (scaled):     {daily_post.index.min().date()} to {daily_post.index.max().date()} ({len(daily_post)} days)")
    pr(f"  Combined:         {combined_daily_price.index.min().date()} to {combined_daily_price.index.max().date()} ({len(combined_daily_price)} days)")
    pr(f"  Splice level:     {splice_level:.1f} (synthetic) = {ief_first:.2f} (IEF raw)")

    # ── Validation ────────────────────────────────────────────────────────────

    pr(f"\nVALIDATION — ^TNX vs IEF during overlap (2002-2026)")
    pr("-" * 40)

    # Align ^TNX returns with IEF returns during overlap
    tnx_aligned = tnx_ret.copy()
    tnx_aligned.index = tnx_aligned.index.to_period("M").to_timestamp()
    common = tnx_aligned.index.intersection(ief_ret.index).sort_values()
    t = tnx_aligned.reindex(common).dropna()
    e = ief_ret.reindex(common).dropna()
    common = t.index.intersection(e.index)
    t, e = t.reindex(common), e.reindex(common)

    corr = t.corr(e)
    slope, intercept, r_value, _, _ = sp_stats.linregress(e, t)
    diff = t - e

    pr(f"  Overlap:         {common.min().strftime('%Y-%m')} to {common.max().strftime('%Y-%m')} ({len(common)} months)")
    pr(f"  Correlation:     {corr:.4f}")
    pr(f"  Beta:            {slope:.4f}")
    pr(f"  R²:              {r_value**2:.4f}")
    pr(f"  ^TNX ann return: {t.mean()*12*100:.2f}%")
    pr(f"  IEF ann return:  {e.mean()*12*100:.2f}%")
    pr(f"  Return diff:     {diff.mean()*12*100:.2f}% ann")
    pr(f"  Tracking error:  {diff.std()*np.sqrt(12)*100:.2f}% ann")
    passed = corr > 0.95
    pr(f"  PASS:            {'YES' if passed else 'NO'} (target: >0.95)")

    # ── Save parquets ─────────────────────────────────────────────────────────

    pr(f"\nOUTPUT FILES")
    pr("-" * 40)

    # Monthly returns
    existing = pd.read_parquet(DATA_DIR / "roth_asset_returns.parquet")
    new_ret = existing.copy()
    new_ret["TSY"] = combined_ret
    out_path = DATA_DIR / "roth_asset_returns_5asset.parquet"
    new_ret.to_parquet(out_path)

    tsy_valid = new_ret["TSY"].notna().sum()
    tsy_first = new_ret["TSY"].first_valid_index()
    pr(f"  Monthly returns: {out_path}")
    pr(f"    Columns:       {list(new_ret.columns)}")
    pr(f"    TSY coverage:  {tsy_valid} months (first: {tsy_first.strftime('%Y-%m')})")
    pr(f"    Pre-ETF (^TNX): {len(pre)} months")
    pr(f"    ETF era (IEF):  {len(post)} months")

    # Daily prices
    daily_df = pd.DataFrame({"TSY": combined_daily_price})
    daily_path = DATA_DIR / "treasury_daily_prices.parquet"
    daily_df.to_parquet(daily_path)
    pr(f"  Daily prices:    {daily_path}")
    pr(f"    Coverage:      {combined_daily_price.index.min().date()} to {combined_daily_price.index.max().date()}")

    pr(f"\nREADY FOR 5-ASSET BACKTEST: {'YES' if passed else 'NO'}")

    # Save report
    rp = OUTDIR / "treasury_tnx_ief_report.txt"
    with open(rp, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Report saved: {rp}")


if __name__ == "__main__":
    main()

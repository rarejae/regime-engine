"""Treasury data pipeline — ^TNX duration model only, no splicing."""

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


def main():
    lines = []
    def pr(s=""):
        print(s); lines.append(s)

    import yfinance as yf

    pr("=" * 80)
    pr("  TREASURY DATA PIPELINE — ^TNX ONLY")
    pr("=" * 80)

    # ── Step 1: Download ^TNX ─────────────────────────────────────────────────

    pr(f"\nDATA SOURCE")
    pr("-" * 40)

    d = yf.download("^TNX", start="1970-01-01", progress=False)
    tnx_daily = d["Close"]
    if hasattr(tnx_daily, "columns"):
        tnx_daily = tnx_daily.iloc[:, 0]
    tnx_daily.index = pd.to_datetime(tnx_daily.index).tz_localize(None)

    pr(f"  Instrument:     10Y Treasury")
    pr(f"  Source:          ^TNX (yfinance)")
    pr(f"  Method:          Duration model (dur={DURATION})")
    pr(f"  Raw daily data:  {tnx_daily.index.min().date()} to {tnx_daily.index.max().date()} ({len(tnx_daily)} obs)")

    # Check for gaps in monthly
    tnx_monthly = tnx_daily.resample("MS").last().dropna()
    full_idx = pd.date_range(tnx_monthly.index.min(), tnx_monthly.index.max(), freq="MS")
    gaps = full_idx.difference(tnx_monthly.index)
    pr(f"  Monthly obs:     {len(tnx_monthly)}")
    pr(f"  Gaps:            {len(gaps)}")

    # ── Step 2: Compute monthly returns ───────────────────────────────────────

    pr(f"\nRETURN SERIES")
    pr("-" * 40)

    dy_m = tnx_monthly.diff() / 100  # yield change in decimal
    coupon_m = tnx_monthly.shift(1) / 100 / 12  # monthly coupon approximation
    tsy_monthly_ret = (-DURATION * dy_m + coupon_m).dropna()

    pr(f"  Monthly returns: {len(tsy_monthly_ret)} observations")
    pr(f"  Coverage:        {tsy_monthly_ret.index.min().strftime('%Y-%m')} to {tsy_monthly_ret.index.max().strftime('%Y-%m')}")
    pr(f"  Mean return:     {tsy_monthly_ret.mean()*100:.3f}%/mo ({tsy_monthly_ret.mean()*12*100:.2f}% ann)")
    pr(f"  Volatility:      {tsy_monthly_ret.std()*np.sqrt(12)*100:.1f}% ann")
    pr(f"  Min month:       {tsy_monthly_ret.min()*100:.1f}%")
    pr(f"  Max month:       {tsy_monthly_ret.max()*100:.1f}%")

    # ── Step 3: Build synthetic price series ──────────────────────────────────

    pr(f"\nSYNTHETIC PRICE SERIES")
    pr("-" * 40)

    tsy_monthly_price = (1 + tsy_monthly_ret.fillna(0)).cumprod() * 100
    pr(f"  Start price:     100")
    pr(f"  End price:       {tsy_monthly_price.iloc[-1]:.1f}")
    pr(f"  For Faber SMAs:  YES")

    # Daily prices
    dy_d = tnx_daily.diff() / 100
    coupon_d = tnx_daily.shift(1) / 100 / 252
    tsy_daily_ret = (-DURATION * dy_d + coupon_d).dropna()
    tsy_daily_price = (1 + tsy_daily_ret.fillna(0)).cumprod() * 100

    pr(f"  Daily prices:    {tsy_daily_price.index.min().date()} to {tsy_daily_price.index.max().date()} ({len(tsy_daily_price)} obs)")

    # ── Step 4: Validate vs IEF ──────────────────────────────────────────────

    pr(f"\nVALIDATION vs IEF (2002-2026)")
    pr("-" * 40)

    ief_d = yf.download("IEF", start="2002-01-01", progress=False)
    ief_p = ief_d["Close"]
    if hasattr(ief_p, "columns"):
        ief_p = ief_p.iloc[:, 0]
    ief_p.index = pd.to_datetime(ief_p.index).tz_localize(None)
    ief_monthly_ret = ief_p.resample("MS").last().pct_change().dropna()
    ief_monthly_ret.index = ief_monthly_ret.index.to_period("M").to_timestamp()

    # Align
    tsy_aligned = tsy_monthly_ret.copy()
    tsy_aligned.index = tsy_aligned.index.to_period("M").to_timestamp()
    common = tsy_aligned.index.intersection(ief_monthly_ret.index).sort_values()
    t = tsy_aligned.reindex(common).dropna()
    e = ief_monthly_ret.reindex(common).dropna()
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

    # ── Step 5: Update parquets ───────────────────────────────────────────────

    pr(f"\nOUTPUT FILES")
    pr("-" * 40)

    # Monthly returns → roth_asset_returns_5asset.parquet
    existing = pd.read_parquet(DATA_DIR / "roth_asset_returns.parquet")
    new_ret = existing.copy()
    new_ret["TSY"] = tsy_monthly_ret
    new_ret.to_parquet(DATA_DIR / "roth_asset_returns_5asset.parquet")

    tsy_valid = new_ret["TSY"].notna().sum()
    tsy_first = new_ret["TSY"].first_valid_index()
    pr(f"  Monthly returns: {DATA_DIR / 'roth_asset_returns_5asset.parquet'}")
    pr(f"    TSY coverage:  {tsy_valid} months (first: {tsy_first.strftime('%Y-%m')})")

    # Daily prices → treasury_daily_prices.parquet
    daily_df = pd.DataFrame({"TSY": tsy_daily_price})
    daily_df.to_parquet(DATA_DIR / "treasury_daily_prices.parquet")
    pr(f"  Daily prices:    {DATA_DIR / 'treasury_daily_prices.parquet'}")
    pr(f"    Coverage:      {tsy_daily_price.index.min().date()} to {tsy_daily_price.index.max().date()}")

    pr(f"\nREADY FOR 5-ASSET BACKTEST: {'YES' if passed else 'NO'}")

    # Save report
    rp = OUTDIR / "treasury_tnx_only_report.txt"
    with open(rp, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Report saved: {rp}")


if __name__ == "__main__":
    main()

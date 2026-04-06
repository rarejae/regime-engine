"""Universe expansion experiment: EFA, VNQ, DBA added to Faber-only system.

Steps:
1. Proxy validation (hard gate: annual return corr < 0.90 excludes asset)
2. Correlation analysis (full period + crisis period)
3. Baseline weight recalibration
4. Backtest: Faber-Original, Faber-Expanded-Full, Faber-Expanded-EFA-VNQ, Faber-Expanded-EFA-Only
5. Full performance report

Proxy sourcing notes:
- EFA: ETF starts Aug 2001, before backtest start (2002). No proxy needed.
- VNQ: ETF starts Sep 2004. RWR (SPDR DJ Wilshire REIT) starts Aug 2001 as proxy.
- DBA: ETF starts Jan 2007. No reliable proxy available from FRED or yfinance.
  FRED tickers SPGSAGSP and WILLREITIND no longer exist. DBA uses ETF-only data.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv; load_dotenv()

import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from taa.faber import compute_trend_scores, SMA_PERIODS

# ── Configuration ────────────────────────────────────────────────────────────

ORIGINAL_BASELINE = {
    "IVV": 0.45, "QQQ": 0.25, "VGLT": 0.05,
    "IAU": 0.10, "DBC": 0.05, "cash": 0.10,
}

BACKTEST_START = "2002-01-01"
PROXY_CORR_GATE = 0.90


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: PROXY VALIDATION
# ��═════════════════════════════════════════════════════════════════════════════

def _yf_monthly_prices(ticker, start="1990-01-01"):
    import yfinance as yf
    d = yf.download(ticker, start=start, progress=False)
    if d is None or d.empty:
        return None
    p = d["Close"]
    if hasattr(p, "columns"):
        p = p.iloc[:, 0]
    p.index = pd.to_datetime(p.index).tz_localize(None)
    return p.resample("MS").last().dropna()


def _yf_monthly_returns(ticker, start="1990-01-01"):
    p = _yf_monthly_prices(ticker, start)
    if p is None:
        return None
    return p.pct_change().dropna()


def validate_proxies():
    """Validate proxy data for new assets.

    EFA: No proxy needed (ETF starts Aug 2001, before backtest 2002-01).
    VNQ: RWR (SPDR DJ Wilshire REIT ETF, Aug 2001) as proxy. Validate vs VNQ overlap.
    DBA: No proxy available. Specified FRED tickers (SPGSAGSP) don't exist.
         ETF-only from Jan 2007. Pre-2007 months score 0 → cash.

    Returns: (passed_assets, proxy_data_for_backfill, validation_results)
    """
    print(f"\n{'='*90}")
    print(f"  STEP 1: PROXY VALIDATION (gate: annual return corr >= {PROXY_CORR_GATE})")
    print(f"{'='*90}")

    passed = set()
    proxy_backfill = {}  # {asset: proxy_monthly_returns} for pre-ETF backfill
    results = {}

    # ── EFA: No proxy needed ─────────────────────────────────────────────
    print("\n  EFA: ETF starts Aug 2001, before backtest start (2002-01).")
    print("  No proxy needed — using actual ETF data for entire backtest period.")
    efa_ret = _yf_monthly_returns("EFA", "2001-01-01")
    if efa_ret is not None and len(efa_ret) > 12:
        passed.add("EFA")
        results["EFA"] = {
            "status": "PASS (no proxy needed)",
            "etf_start": efa_ret.index.min().strftime("%Y-%m"),
            "note": "ETF predates backtest start",
        }
        print(f"  EFA data: {efa_ret.index.min().date()} to {efa_ret.index.max().date()} ({len(efa_ret)} months)")
        print(f"  Status: PASS (no proxy needed)")
    else:
        print(f"  WARNING: Could not fetch EFA ETF data. EXCLUDED.")
        results["EFA"] = {"status": "FAIL", "reason": "ETF data unavailable"}

    # ── VNQ: RWR as proxy ────────────────────────────────────────────────
    print(f"\n  VNQ: ETF starts Sep 2004. Using RWR (SPDR DJ Wilshire REIT) as proxy.")
    print(f"  (Specified FRED ticker WILLREITIND no longer exists.)")

    vnq_ret = _yf_monthly_returns("VNQ", "2004-01-01")
    rwr_ret = _yf_monthly_returns("RWR", "2001-01-01")

    if vnq_ret is not None and rwr_ret is not None:
        # Validate: annual return correlation during overlap
        common = vnq_ret.index.intersection(rwr_ret.index).sort_values()
        p = rwr_ret.reindex(common).dropna()
        e = vnq_ret.reindex(common).dropna()
        common2 = p.index.intersection(e.index)
        p = p.reindex(common2)
        e = e.reindex(common2)

        # Annual returns
        p_ann = p.resample("YS").apply(lambda x: (1 + x).prod() - 1)
        e_ann = e.resample("YS").apply(lambda x: (1 + x).prod() - 1)
        ann_common = p_ann.index.intersection(e_ann.index)
        p_ann = p_ann.reindex(ann_common)
        e_ann = e_ann.reindex(ann_common)

        ann_corr = p_ann.corr(e_ann)
        monthly_corr = p.corr(e)

        overlap_start = common2.min().strftime("%Y")
        overlap_end = common2.max().strftime("%Y")

        status = "PASS" if ann_corr >= PROXY_CORR_GATE else "FAIL"
        results["VNQ"] = {
            "status": status,
            "proxy_name": "RWR (SPDR DJ Wilshire REIT ETF)",
            "overlap_period": f"{overlap_start}-{overlap_end}",
            "n_months": len(common2),
            "n_years": len(ann_common),
            "annual_corr": ann_corr,
            "monthly_corr": monthly_corr,
        }

        print(f"  Overlap: {overlap_start}-{overlap_end} ({len(common2)} months, {len(ann_common)} years)")
        print(f"  Monthly return correlation: {monthly_corr:.3f}")
        print(f"  Annual return correlation:  {ann_corr:.3f}")
        print(f"  Status: {status}")

        if status == "PASS":
            passed.add("VNQ")
            # Use RWR for pre-VNQ backfill
            vnq_start = vnq_ret.index.min()
            rwr_pre = rwr_ret[rwr_ret.index < vnq_start]
            proxy_backfill["VNQ"] = rwr_pre
            print(f"  RWR backfill: {rwr_pre.index.min().date()} to {rwr_pre.index.max().date()} ({len(rwr_pre)} months)")
        else:
            print(f"  VNQ EXCLUDED: annual return correlation {ann_corr:.3f} < {PROXY_CORR_GATE}")
    else:
        print(f"  WARNING: Could not fetch VNQ or RWR data.")
        results["VNQ"] = {"status": "FAIL", "reason": "data unavailable"}

    # ── DBA: No proxy available ───��──────────────────────────────────────
    print(f"\n  DBA: ETF starts Jan 2007. No proxy available.")
    print(f"  (Specified FRED ticker SPGSAGSP no longer exists. No yfinance alternative.)")

    dba_ret = _yf_monthly_returns("DBA", "2007-01-01")
    if dba_ret is not None and len(dba_ret) > 12:
        # DBA has no proxy to validate — it enters the backtest ETF-only from 2007.
        # Before 2007+12mo (SMA warmup), DBA scores 0 → weight goes to cash.
        # This is not a proxy failure — it's a data limitation.
        # We pass DBA through without a proxy gate since there's no proxy to test.
        passed.add("DBA")
        results["DBA"] = {
            "status": "PASS (ETF-only, no proxy)",
            "etf_start": dba_ret.index.min().strftime("%Y-%m"),
            "note": "No proxy available. Pre-2008 months score 0 (NaN → cash). ~5 years missing.",
        }
        print(f"  DBA data: {dba_ret.index.min().date()} to {dba_ret.index.max().date()} ({len(dba_ret)} months)")
        print(f"  Pre-2008 months: DBA has no price data → Faber score 0 → weight to cash.")
        print(f"  Status: PASS (ETF-only, no proxy to validate)")
    else:
        print(f"  WARNING: Could not fetch DBA ETF data. EXCLUDED.")
        results["DBA"] = {"status": "FAIL", "reason": "ETF data unavailable"}

    # ── Summary table ─────��──────────────────────────────────────────────
    print(f"\n  {'Asset':<8} {'ETF':<8} {'Proxy':<32} {'Overlap':<14} {'Ann Ret Corr':>13} {'Status':>8}")
    print(f"  {'-'*8} {'-'*8} {'-'*32} {'-'*14} {'-'*13} {'-'*8}")

    r = results.get("EFA", {})
    print(f"  {'EFA':<8} {'actual':<8} {'(no proxy needed)':<32} {'N/A':<14} {'N/A':>13} {'PASS':>8}")

    r = results.get("VNQ", {})
    if "annual_corr" in r:
        print(f"  {'VNQ':<8} {'actual':<8} {'RWR (SPDR DJ Wilshire REIT)':<32} {r['overlap_period']:<14} {r['annual_corr']:>13.3f} {r['status']:>8}")
    else:
        print(f"  {'VNQ':<8} {'actual':<8} {'RWR':<32} {'N/A':<14} {'N/A':>13} {r.get('status','N/A'):>8}")

    r = results.get("DBA", {})
    print(f"  {'DBA':<8} {'actual':<8} {'(none available)':<32} {'N/A':<14} {'N/A':>13} {'PASS*':>8}")

    print(f"\n  * DBA passes without proxy gate (no proxy to test). ETF-only from 2007.")
    print(f"    Pre-2008 months: Faber SMA needs 12mo warmup → DBA ineligible until ~Jan 2008.")

    print(f"\n  MSCI EAFE NOTE: EFA represents large/mid-cap developed markets ex US/Canada.")
    print(f"  Composition has evolved (Japan ~60% in 1990, ~20% now). No pre-2001 proxy")
    print(f"  was needed since the ETF predates the backtest start.")

    print(f"\n  Assets passing proxy gate: {sorted(passed)}")

    return passed, proxy_backfill, results


# ════════��═════════════════════════════��═══════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════���═══════════════════════

def load_all_monthly_prices(passed_assets):
    """Load monthly closing prices for all assets (existing + new)."""
    import yfinance as yf

    ticker_map = {
        "IVV": "SPY", "QQQ": "QQQ", "VGLT": "TLT",
        "IAU": "GLD", "DBC": "DBC", "EFA": "EFA",
        "VNQ": "VNQ", "DBA": "DBA",
    }

    all_assets = ["IVV", "QQQ", "VGLT", "IAU", "DBC"] + sorted(passed_assets)
    all_assets = list(dict.fromkeys(all_assets))  # deduplicate preserving order

    frames = {}
    for asset in all_assets:
        ticker = ticker_map.get(asset, asset)
        d = yf.download(ticker, start="1990-01-01", progress=False)
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p, "columns"):
                p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None)
            frames[asset] = p

    df = pd.DataFrame(frames).sort_index()
    return df.resample("MS").last()


def load_all_monthly_returns(passed_assets, proxy_backfill):
    """Load monthly returns. Uses proxy backfill for pre-ETF periods where available."""
    import yfinance as yf

    ticker_map = {
        "IVV": "SPY", "QQQ": "QQQ", "VGLT": "TLT",
        "IAU": "GLD", "DBC": "DBC", "EFA": "EFA",
        "VNQ": "VNQ", "DBA": "DBA",
    }

    frames = {}

    # Existing assets from parquet
    asset_ret = pd.read_parquet("data/macro/roth_asset_returns.parquet")
    for c in ["IVV", "QQQ", "VGLT", "IAU", "DBC", "cash"]:
        if c in asset_ret.columns:
            frames[c] = asset_ret[c]

    # New assets: ETF + proxy backfill
    for asset in sorted(passed_assets):
        if asset in frames:
            continue
        ticker = ticker_map.get(asset, asset)
        d = yf.download(ticker, start="1990-01-01", progress=False)
        etf_ret = None
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p, "columns"):
                p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None)
            etf_monthly = p.resample("MS").last()
            etf_ret = etf_monthly.pct_change().dropna()

        if asset in proxy_backfill and etf_ret is not None:
            pre_etf = proxy_backfill[asset]
            combined = pd.concat([pre_etf, etf_ret]).sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
            frames[asset] = combined
        elif etf_ret is not None:
            frames[asset] = etf_ret

    df = pd.DataFrame(frames).sort_index()
    return df


# Also need prices with proxy backfill for SMA computation
def load_all_monthly_prices_with_backfill(passed_assets, proxy_backfill):
    """Load monthly prices with proxy backfill for SMA computation."""
    import yfinance as yf

    ticker_map = {
        "IVV": "SPY", "QQQ": "QQQ", "VGLT": "TLT",
        "IAU": "GLD", "DBC": "DBC", "EFA": "EFA",
        "VNQ": "VNQ", "DBA": "DBA",
    }

    all_assets = ["IVV", "QQQ", "VGLT", "IAU", "DBC"] + sorted(passed_assets)
    all_assets = list(dict.fromkeys(all_assets))

    frames = {}
    for asset in all_assets:
        ticker = ticker_map.get(asset, asset)
        d = yf.download(ticker, start="1990-01-01", progress=False)
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p, "columns"):
                p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None)
            etf_prices = p.resample("MS").last()

            # Backfill with proxy prices (reconstructed from proxy returns)
            if asset in proxy_backfill:
                proxy_ret = proxy_backfill[asset]
                etf_start = etf_prices.dropna().index.min()
                anchor = etf_prices.loc[etf_start]

                # Reconstruct proxy prices backward from ETF anchor
                pre_dates = proxy_ret.index[proxy_ret.index < etf_start].sort_values()
                if len(pre_dates) > 0:
                    proxy_prices = pd.Series(dtype=float)
                    # Work backward: anchor / (1 + ret) for each prior month
                    price = anchor
                    proxy_price_list = []
                    for dt in reversed(pre_dates):
                        ret = proxy_ret.loc[dt]
                        if pd.notna(ret) and ret > -1:
                            price = price / (1 + ret)
                            proxy_price_list.append((dt, price))

                    if proxy_price_list:
                        proxy_prices = pd.Series(
                            [v for _, v in reversed(proxy_price_list)],
                            index=[d for d, _ in reversed(proxy_price_list)]
                        )
                        combined = pd.concat([proxy_prices, etf_prices.dropna()]).sort_index()
                        combined = combined[~combined.index.duplicated(keep="last")]
                        frames[asset] = combined
                        continue

            frames[asset] = etf_prices

    df = pd.DataFrame(frames).sort_index()
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: CORRELATION ANALYSIS
# ���══════════════════════════���══════════════════════════════��═══════════════════

def correlation_analysis(all_returns, passed_assets):
    print(f"\n{'='*90}")
    print(f"  STEP 2: CORRELATION ANALYSIS")
    print(f"{'='*90}")

    all_risky = ["IVV", "QQQ", "VGLT", "IAU", "DBC"] + sorted(passed_assets)
    all_risky = list(dict.fromkeys(all_risky))  # deduplicate
    cols = [c for c in all_risky if c in all_returns.columns]

    bt_start = pd.Timestamp(BACKTEST_START)

    # Full period
    full = all_returns.loc[bt_start:, cols].dropna(how="all")
    full_corr = full.corr()

    print(f"\n  Full-period pairwise correlation (2002-2026, monthly returns):")
    print(f"  {'':>8}", end="")
    for c in cols:
        print(f" {c:>8}", end="")
    print()
    for r in cols:
        print(f"  {r:>8}", end="")
        for c in cols:
            val = full_corr.loc[r, c] if r in full_corr.index and c in full_corr.columns else np.nan
            if pd.isna(val):
                print(f" {'N/A':>8}", end="")
            else:
                marker = " *" if abs(val) > 0.70 and r != c else "  "
                print(f" {val:>6.3f}{marker}", end="")
        print()
    print(f"\n  * = correlation > 0.70 (potentially redundant)")

    # Crisis period (2008-2009)
    crisis = all_returns.loc["2008-01":"2009-12", cols].dropna(how="all")
    crisis_corr = crisis.corr()

    print(f"\n  Crisis-period correlation (2008-2009 only — this is the one that matters):")
    print(f"  {'':>8}", end="")
    for c in cols:
        print(f" {c:>8}", end="")
    print()
    for r in cols:
        print(f"  {r:>8}", end="")
        for c in cols:
            val = crisis_corr.loc[r, c] if r in crisis_corr.index and c in crisis_corr.columns else np.nan
            if pd.isna(val):
                print(f" {'N/A':>8}", end="")
            else:
                marker = " *" if abs(val) > 0.70 and r != c else "  "
                print(f" {val:>6.3f}{marker}", end="")
        print()

    return full_corr, crisis_corr


# ═════════════════════���════════════════════════════════��═══════════════════════
# STEP 3: WEIGHT CALIBRATION
# ═════════════════���════════════════════════════════════════════════════════════

def calibrate_weights(passed_assets):
    print(f"\n{'='*90}")
    print(f"  STEP 3: BASELINE WEIGHT RECALIBRATION")
    print(f"{'='*90}")

    has_efa = "EFA" in passed_assets
    has_vnq = "VNQ" in passed_assets
    has_dba = "DBA" in passed_assets

    # Full expansion (all 3)
    full_bl = {
        "IVV": 0.30, "QQQ": 0.15, "VGLT": 0.05,
        "IAU": 0.07, "DBC": 0.03, "cash": 0.10,
    }
    if has_efa: full_bl["EFA"] = 0.15
    if has_vnq: full_bl["VNQ"] = 0.10
    if has_dba: full_bl["DBA"] = 0.05
    # Redistribute missing asset weight to cash
    target_sum = sum(full_bl.values())
    if abs(target_sum - 1.0) > 0.001:
        full_bl["cash"] += (1.0 - target_sum)

    # EFA + VNQ only
    efa_vnq_bl = {
        "IVV": 0.32, "QQQ": 0.18, "VGLT": 0.05,
        "IAU": 0.09, "DBC": 0.04, "cash": 0.12,
    }
    if has_efa: efa_vnq_bl["EFA"] = 0.13
    if has_vnq: efa_vnq_bl["VNQ"] = 0.07
    target_sum = sum(efa_vnq_bl.values())
    if abs(target_sum - 1.0) > 0.001:
        efa_vnq_bl["cash"] += (1.0 - target_sum)

    # EFA only
    efa_only_bl = {
        "IVV": 0.35, "QQQ": 0.20, "VGLT": 0.05,
        "IAU": 0.10, "DBC": 0.05, "cash": 0.13,
    }
    if has_efa: efa_only_bl["EFA"] = 0.12
    target_sum = sum(efa_only_bl.values())
    if abs(target_sum - 1.0) > 0.001:
        efa_only_bl["cash"] += (1.0 - target_sum)

    print(f"\n  Passed proxy gate: {sorted(passed_assets)}")
    print(f"\n  Weight rationale:")
    print(f"  - Maintain ~70% total equity (IVV + QQQ + EFA + VNQ)")
    print(f"  - Maintain ~10-15% real assets (IAU + DBC + DBA)")
    print(f"  - Maintain ~5% bonds (VGLT)")
    print(f"  - Maintain ~10-13% cash")

    def print_weights(name, w):
        items = [f"{a} {v:.0%}" for a, v in sorted(w.items(), key=lambda x: -x[1] if x[0] != 'cash' else 0)]
        eq = sum(v for a, v in w.items() if a in ["IVV", "QQQ", "EFA", "VNQ"])
        real = sum(v for a, v in w.items() if a in ["IAU", "DBC", "DBA"])
        print(f"\n  {name}:")
        print(f"    Weights: {', '.join(items)}")
        print(f"    Equity: {eq:.0%} | Real assets: {real:.0%} | Bonds: {w.get('VGLT',0):.0%} | Cash: {w.get('cash',0):.0%}")

    print_weights("Original (control)", ORIGINAL_BASELINE)
    print_weights("Expanded-Full", full_bl)
    print_weights("Expanded-EFA-VNQ", efa_vnq_bl)
    print_weights("Expanded-EFA-Only", efa_only_bl)

    return full_bl, efa_vnq_bl, efa_only_bl


# ══════════��════════════════════════════���══════════════════════════════════════
# FABER FILTER (generalized)
# ════════════════════════════════════════════════════���═════════════════════════

def apply_faber_filter(trend_scores, baseline):
    """Apply Faber trend filter. Freed capital to cash. Returns (weights, pool, eligible)."""
    w = {}
    pool = 0.0
    eligible = []

    for a, base_w in baseline.items():
        if a == "cash":
            w[a] = base_w
            continue
        score = trend_scores.get(a, 0)
        if score >= 3:
            w[a] = base_w
            eligible.append(a)
        elif score == 2:
            w[a] = base_w * 0.70
            pool += base_w * 0.30
            eligible.append(a)
        else:
            w[a] = 0.0
            pool += base_w

    return w, pool, eligible


# ══════���════════════════════════���══════════════════════════════════��═══════════
# STEP 4: BACKTEST
# ══════════════════════════════════════════════════════════════���═══════════════

def run_backtest(all_prices, all_returns, baselines, passed_assets):
    print(f"\n{'='*90}")
    print(f"  STEP 4: BACKTEST")
    print(f"{'='*90}")

    orig_bl, full_bl, efa_vnq_bl, efa_only_bl = baselines

    strategies = {
        "Faber-Original": orig_bl,
        "Faber-Expanded-Full": full_bl,
        "Faber-Expanded-EFA-VNQ": efa_vnq_bl,
        "Faber-Expanded-EFA-Only": efa_only_bl,
    }

    # Compute trend scores for all assets
    trend_df = compute_trend_scores(all_prices)

    bt_start = pd.Timestamp(BACKTEST_START)
    bt_dates = all_returns.index[all_returns.index >= bt_start]
    bt_dates = bt_dates[bt_dates.isin(trend_df.index)]

    strat_names = list(strategies.keys()) + ["IVV B&H", "60/40"]
    results = {s: {} for s in strat_names}
    weight_log = {s: [] for s in strategies.keys()}

    print(f"\n  Backtest: {bt_dates.min().date()} to {bt_dates.max().date()} ({len(bt_dates)} months)")
    print(f"  Signal alignment: trend_df shifted by 1 (PIT). Month T scores from T-1 data.")

    # Report data availability for new assets
    for asset in sorted(passed_assets):
        if asset in trend_df.columns:
            valid = trend_df[asset].dropna()
            first_valid = valid.index.min() if len(valid) > 0 else None
            first_nonzero = valid[valid > 0].index.min() if (valid > 0).any() else None
            print(f"  {asset}: first trend data {first_valid.date() if first_valid else 'N/A'}, "
                  f"first eligible (score>0) {first_nonzero.date() if first_nonzero and pd.notna(first_nonzero) else 'N/A'}")

    for dt in bt_dates:
        # Get trend scores for ALL assets in the expanded universe
        cur_trends = {}
        for a in trend_df.columns:
            v = trend_df.loc[dt, a]
            cur_trends[a] = int(v) if pd.notna(v) else 0

        for strat_name, baseline in strategies.items():
            assets_in_strat = list(baseline.keys())

            # Get actual returns for available assets
            actual = {}
            for a in assets_in_strat:
                if a == "cash":
                    val = all_returns.loc[dt, "cash"] if "cash" in all_returns.columns and dt in all_returns.index and pd.notna(all_returns.loc[dt, "cash"]) else 0.0
                    actual[a] = float(val)
                elif a in all_returns.columns and dt in all_returns.index and pd.notna(all_returns.loc[dt, a]):
                    actual[a] = float(all_returns.loc[dt, a])
                else:
                    actual[a] = 0.0  # missing data → 0 return (weight will be 0 anyway due to Faber)

            # Apply Faber filter
            faber_w, pool, eligible = apply_faber_filter(cur_trends, baseline)

            # Freed capital to cash
            w = dict(faber_w)
            w["cash"] = w.get("cash", 0) + pool

            # Weight sum assertion
            total_w = sum(w.get(a, 0) for a in assets_in_strat)
            assert abs(total_w - 1.0) < 0.01, f"{strat_name} weights sum to {total_w:.4f} at {dt}"

            # Portfolio return
            port_ret = sum(w.get(a, 0) * actual.get(a, 0) for a in assets_in_strat)
            results[strat_name][dt] = port_ret
            weight_log[strat_name].append({"date": dt, **w})

        # Benchmarks
        ivv_ret = float(all_returns.loc[dt, "IVV"]) if "IVV" in all_returns.columns and pd.notna(all_returns.loc[dt, "IVV"]) else 0
        vglt_ret = float(all_returns.loc[dt, "VGLT"]) if "VGLT" in all_returns.columns and pd.notna(all_returns.loc[dt, "VGLT"]) else 0
        results["IVV B&H"][dt] = ivv_ret
        results["60/40"][dt] = 0.60 * ivv_ret + 0.40 * vglt_ret

    ret_series = {s: pd.Series(d).sort_index() for s, d in results.items()}
    return ret_series, weight_log, strat_names, bt_dates, trend_df


# ���═════════════════════════════════════════��═══════════════════════════════════
# METRICS
# ═════════════════════════════��════════════════════════════════════════════════

def compute_metrics(s):
    s = s.dropna()
    if len(s) < 12:
        return None
    ar = s.mean() * 12
    av = s.std() * np.sqrt(12)
    sh = ar / av if av > 0 else 0
    neg = s[s < 0]
    ds = neg.std() * np.sqrt(12) if len(neg) > 10 else av
    sortino = ar / ds if ds > 0 else 0
    cum = (1 + s).cumprod()
    dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
    calmar = ar / abs(dd) if dd != 0 else 0
    final = cum.iloc[-1]
    return {"ar": ar, "av": av, "sh": sh, "sortino": sortino, "dd": dd, "calmar": calmar, "final": final}


# ════���══════════════════════════════════════════════════════���══════════════════
# STEP 5: OUTPUT
# ══════════════════════════════════════════════════════��═══════════════════════

def print_report(ret_series, weight_log, strat_names, bt_dates, trend_df,
                 baselines, passed_assets, all_returns):
    orig_bl, full_bl, efa_vnq_bl, efa_only_bl = baselines

    perfs = {}
    for s in strat_names:
        m = compute_metrics(ret_series[s])
        if m:
            perfs[s] = m

    # ── Performance table ────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"  PERFORMANCE")
    print(f"{'='*100}")
    print(f"  {'Strategy':<28} {'Return':>8} {'Vol':>7} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'Calmar':>8} {'Terminal':>10}")
    print(f"  {'-'*28} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for s in strat_names:
        if s not in perfs:
            continue
        p = perfs[s]
        print(f"  {s:<28} {p['ar']:>7.1%} {p['av']:>6.1%} {p['sh']:>8.3f} {p['sortino']:>8.3f} {p['dd']:>7.1%} {p['calmar']:>8.2f} ${p['final']:>9.2f}")

    # ── Crisis analysis ──────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"  CRISIS ANALYSIS")
    print(f"{'='*100}")
    for cname, cs, ce in [("GFC (2008-2009)", "2008-09", "2009-03"),
                           ("COVID (2020)", "2020-02", "2020-04"),
                           ("2022 Bear", "2022-01", "2022-10")]:
        print(f"\n  {cname}:")
        print(f"  {'Strategy':<28} {'Return':>10} {'MaxDD':>10}")
        print(f"  {'-'*28} {'-'*10} {'-'*10}")
        for s in strat_names:
            sr = ret_series.get(s)
            if sr is None:
                continue
            c = sr[(sr.index >= pd.Timestamp(cs)) & (sr.index <= pd.Timestamp(ce))]
            if len(c) > 0:
                cum = (1 + c).cumprod()
                mdd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
                print(f"  {s:<28} {(1+c).prod()-1:>+9.1%} {mdd:>9.1%}")

    # ── Weight diagnostics ───────────���───────────────────────────────────
    print(f"\n{'='*100}")
    print(f"  AVERAGE WEIGHT DIAGNOSTICS")
    print(f"{'='*100}")

    strategy_baselines = {
        "Faber-Original": orig_bl,
        "Faber-Expanded-Full": full_bl,
        "Faber-Expanded-EFA-VNQ": efa_vnq_bl,
        "Faber-Expanded-EFA-Only": efa_only_bl,
    }

    for strat_name in strategy_baselines:
        wl = weight_log.get(strat_name, [])
        if not wl:
            continue
        wdf = pd.DataFrame(wl).set_index("date")
        bl = strategy_baselines[strat_name]
        print(f"\n  {strat_name}:")
        for a in bl.keys():
            if a in wdf.columns:
                avg = wdf[a].mean()
                mx = wdf[a].max()
                pct_active = (wdf[a] > 0.001).mean()
                print(f"    {a:<8}: avg {avg:>6.1%}  max {mx:>5.1%}  active {pct_active:>5.0%} of months")

    # ── Independent signal contribution ──────────────────────────────────
    print(f"\n{'='*100}")
    print(f"  INDEPENDENT SIGNAL CONTRIBUTION")
    print(f"{'='*100}")

    new_assets = sorted(passed_assets - {"IVV", "QQQ", "VGLT", "IAU", "DBC"})
    if new_assets:
        print(f"\n  {'Asset':<8} {'Pct Months Eligible':>20} {'Avg Alloc When Elig':>22} {'Corr with IVV':>16}")
        print(f"  {'-'*8} {'-'*20} {'-'*22} {'-'*16}")

        bt_start = pd.Timestamp(BACKTEST_START)
        full_ret = all_returns.loc[bt_start:]

        for asset in new_assets:
            if asset not in trend_df.columns:
                continue

            scores = trend_df.loc[bt_start:, asset].dropna()
            eligible_months = (scores >= 2).sum()
            total_months = len(scores)
            pct_eligible = eligible_months / total_months if total_months > 0 else 0

            # Avg allocation from Expanded-Full
            wl = weight_log.get("Faber-Expanded-Full", [])
            avg_alloc = 0
            if wl:
                wdf = pd.DataFrame(wl).set_index("date")
                if asset in wdf.columns:
                    active = wdf[asset][wdf[asset] > 0.001]
                    avg_alloc = active.mean() if len(active) > 0 else 0

            # Correlation with IVV
            if asset in full_ret.columns and "IVV" in full_ret.columns:
                common_idx = full_ret[[asset, "IVV"]].dropna().index
                corr_ivv = full_ret.loc[common_idx, asset].corr(full_ret.loc[common_idx, "IVV"])
            else:
                corr_ivv = np.nan

            print(f"  {asset:<8} {pct_eligible:>19.0%} {avg_alloc:>21.1%} {corr_ivv:>16.3f}")
    else:
        print(f"\n  No new assets passed proxy gate.")

    # ── Sharpe decomposition ───────────────────────���─────────────────────
    print(f"\n{'='*100}")
    print(f"  SHARPE DECOMPOSITION")
    print(f"{'='*100}")

    orig_sh = perfs.get("Faber-Original", {}).get("sh", 0)
    print(f"\n  Faber-Original Sharpe:          {orig_sh:.3f}")
    for s in ["Faber-Expanded-Full", "Faber-Expanded-EFA-VNQ", "Faber-Expanded-EFA-Only"]:
        if s in perfs:
            sh = perfs[s]["sh"]
            delta = sh - orig_sh
            print(f"  {s + ' Sharpe:':<34} {sh:.3f}  (delta: {delta:+.3f})")

    # ── Calendar year returns ─────���──────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"  CALENDAR YEAR RETURNS")
    print(f"{'='*100}")
    cal_strats = list(strategy_baselines.keys())
    header = f"  {'Year':>6}"
    for s in cal_strats:
        label = s.replace("Faber-", "").replace("Expanded-", "")[:10]
        header += f" {label:>11}"
    header += f" {'IVV':>10}"
    print(header)

    for yr in range(2002, 2027):
        row = f"  {yr:>6}"
        for s in cal_strats:
            sr = ret_series.get(s, pd.Series(dtype=float))
            y = sr[sr.index.year == yr]
            if len(y) > 3:
                row += f" {(1+y).prod()-1:>+10.1%}"
            else:
                row += f" {'--':>11}"
        ivv_sr = ret_series.get("IVV B&H", pd.Series(dtype=float))
        y = ivv_sr[ivv_sr.index.year == yr]
        if len(y) > 3:
            row += f" {(1+y).prod()-1:>+9.1%}"
        else:
            row += f" {'--':>10}"
        print(row)

    print()
    return perfs


# ═══════════��════════════════════════════════════════════════════════════════��═
# MAIN
# ════════════════════════════════════════════════════════════════���═════════════

if __name__ == "__main__":
    print("=" * 90)
    print("  UNIVERSE EXPANSION EXPERIMENT: EFA, VNQ, DBA")
    print("=" * 90)

    # Step 1: Proxy validation
    passed_assets, proxy_backfill, proxy_results = validate_proxies()

    # Load data
    print(f"\n  Loading price and return data for expanded universe...")
    all_prices = load_all_monthly_prices_with_backfill(passed_assets, proxy_backfill)
    all_returns = load_all_monthly_returns(passed_assets, proxy_backfill)

    print(f"  Price columns: {sorted(all_prices.columns.tolist())}")
    print(f"  Return columns: {sorted(all_returns.columns.tolist())}")
    for asset in sorted(passed_assets):
        if asset in all_prices.columns:
            valid = all_prices[asset].dropna()
            print(f"  {asset} prices: {valid.index.min().date()} to {valid.index.max().date()} ({len(valid)} months)")
        if asset in all_returns.columns:
            valid = all_returns[asset].dropna()
            print(f"  {asset} returns: {valid.index.min().date()} to {valid.index.max().date()} ({len(valid)} months)")

    # Step 2: Correlation analysis
    correlation_analysis(all_returns, passed_assets)

    # Step 3: Weight calibration
    full_bl, efa_vnq_bl, efa_only_bl = calibrate_weights(passed_assets)
    baselines = (ORIGINAL_BASELINE, full_bl, efa_vnq_bl, efa_only_bl)

    # Step 4: Backtest
    ret_series, weight_log, strat_names, bt_dates, trend_df = run_backtest(
        all_prices, all_returns, baselines, passed_assets
    )

    # Step 5: Report
    perfs = print_report(
        ret_series, weight_log, strat_names, bt_dates, trend_df,
        baselines, passed_assets, all_returns
    )

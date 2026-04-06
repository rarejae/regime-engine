"""VRP proxy validation: data quality gate for the VRP backtest.

Validates SVXY, PUTW, QYLD, XYLD ETFs and their pre-ETF proxy indices.
Must pass before running any VRP strategy backtest.
"""

import sys, os, io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv; load_dotenv()

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd


# ── Helpers ──────────────────────────────────────────────────────────────────

def yf_adj_close(ticker, start="2000-01-01"):
    """Download adjusted close prices from yfinance. Returns monthly prices."""
    import yfinance as yf
    d = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    if d is None or d.empty:
        return None
    p = d["Close"]
    if hasattr(p, "columns"):
        p = p.iloc[:, 0]
    p.index = pd.to_datetime(p.index).tz_localize(None)
    return p


def yf_adj_monthly_ret(ticker, start="2000-01-01"):
    """Monthly returns from adjusted close."""
    p = yf_adj_close(ticker, start)
    if p is None:
        return None, None
    monthly = p.resample("MS").last().dropna()
    return monthly, monthly.pct_change().dropna()


def monthly_corr(a, b):
    """Pearson correlation on aligned monthly returns."""
    common = a.index.intersection(b.index).sort_values()
    if len(common) < 12:
        return np.nan, 0
    return float(a.reindex(common).corr(b.reindex(common))), len(common)


def annual_corr(a, b):
    """Pearson correlation on aligned annual returns."""
    common = a.index.intersection(b.index).sort_values()
    a_ann = a.reindex(common).resample("YS").apply(lambda x: (1 + x).prod() - 1)
    b_ann = b.reindex(common).resample("YS").apply(lambda x: (1 + x).prod() - 1)
    ac = a_ann.index.intersection(b_ann.index)
    if len(ac) < 3:
        return np.nan, 0
    return float(a_ann.reindex(ac).corr(b_ann.reindex(ac))), len(ac)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: ETF PRICE SERIES
# ══════════════════════════════════════════════════════════════════════════════

def step1_etf_data():
    print(f"\n{'='*100}")
    print(f"  STEP 1: DOWNLOAD ETF PRICE SERIES (adjusted close)")
    print(f"{'='*100}")

    etfs = {
        "SVXY": ("Oct 2011", "2011-10-01"),
        "PUTW": ("Feb 2016", "2016-02-01"),
        "QYLD": ("Dec 2013", "2013-12-01"),
        "XYLD": ("Jun 2013", "2013-06-01"),
    }

    etf_data = {}
    for ticker, (inception, start) in etfs.items():
        prices, monthly_ret = yf_adj_monthly_ret(ticker, start)
        if prices is not None:
            # Spot check: compute 2024 calendar year total return
            daily_p = yf_adj_close(ticker, "2024-01-01")
            yr_ret = None
            if daily_p is not None and len(daily_p) > 200:
                yr_start = daily_p.iloc[0]
                yr_end_idx = daily_p.index[daily_p.index.year == 2024]
                if len(yr_end_idx) > 0:
                    yr_end = daily_p.loc[yr_end_idx[-1]]
                    yr_ret = (yr_end / yr_start) - 1

            etf_data[ticker] = {
                "prices": prices,
                "returns": monthly_ret,
                "inception": inception,
                "first": prices.index.min(),
                "last": prices.index.max(),
                "n_months": len(prices),
                "yr_2024_ret": yr_ret,
            }

    print(f"\n  {'ETF':<8} {'Inception':<12} {'First Avail':<14} {'Last Date':<14} "
          f"{'Months':>8} {'Div-Adj':>9} {'2024 TotRet':>12}")
    print(f"  {'-'*8} {'-'*12} {'-'*14} {'-'*14} {'-'*8} {'-'*9} {'-'*12}")

    for ticker in ["SVXY", "PUTW", "QYLD", "XYLD"]:
        d = etf_data.get(ticker)
        if d is None:
            print(f"  {ticker:<8} {'N/A':<12} {'DOWNLOAD FAILED'}")
            continue
        yr_str = f"{d['yr_2024_ret']:+.1%}" if d['yr_2024_ret'] is not None else "N/A"
        print(f"  {ticker:<8} {d['inception']:<12} {d['first'].strftime('%Y-%m-%d'):<14} "
              f"{d['last'].strftime('%Y-%m-%d'):<14} {d['n_months']:>8} {'Yes':>9} {yr_str:>12}")

    print(f"\n  Note: yfinance auto_adjust=True returns dividend-adjusted prices.")
    print(f"  QYLD/XYLD pay large monthly distributions (~1%/mo); adjusted close captures these.")

    return etf_data


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: SVXY PROXY CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def step2_svxy_proxy(etf_data):
    print(f"\n{'='*100}")
    print(f"  STEP 2: SVXY PROXY CONSTRUCTION")
    print(f"{'='*100}")

    # Step 2a: Try SPVXSTR from yfinance
    print(f"\n  Step 2a: Attempting ^SPVXSTR from yfinance...")
    spvxstr_prices = yf_adj_close("^SPVXSTR", "2004-01-01")
    spvxstr_source = None

    if spvxstr_prices is not None and len(spvxstr_prices) > 100:
        print(f"    SUCCESS: ^SPVXSTR available from {spvxstr_prices.index.min().date()} "
              f"to {spvxstr_prices.index.max().date()} ({len(spvxstr_prices)} daily obs)")
        spvxstr_source = "yfinance"
    else:
        print(f"    FAILED: ^SPVXSTR not available from yfinance")

    # Step 2b: Try ^VIX futures / SPVXSP as alternative
    if spvxstr_source is None:
        print(f"\n  Step 2b: Attempting ^SPVXSP (S&P VIX Short-Term Futures Total Return)...")
        spvxsp = yf_adj_close("^SPVXSP", "2004-01-01")
        if spvxsp is not None and len(spvxsp) > 100:
            spvxstr_prices = spvxsp
            spvxstr_source = "yfinance ^SPVXSP"
            print(f"    SUCCESS: ^SPVXSP available from {spvxstr_prices.index.min().date()} "
                  f"to {spvxstr_prices.index.max().date()} ({len(spvxstr_prices)} daily obs)")
        else:
            print(f"    FAILED: ^SPVXSP not available from yfinance")

    # Try CBOE VIX futures CSV
    if spvxstr_source is None:
        print(f"\n  Step 2b (alt): Attempting CBOE VIX futures CSV download...")
        try:
            import urllib.request
            url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VX_History.csv"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
            vx_df = pd.read_csv(io.StringIO(raw))
            print(f"    Downloaded VX_History.csv: {len(vx_df)} rows, columns: {vx_df.columns.tolist()[:6]}")
            spvxstr_source = "CBOE VX futures (raw — would need index construction)"
            # Mark as needing construction — complex, will note in report
        except Exception as e:
            print(f"    FAILED: {e}")

    # Construct synthetic SVXY proxy if we have index data
    svxy_proxy_ret = None
    if spvxstr_prices is not None and spvxstr_source and "yfinance" in spvxstr_source:
        # SPVXSTR/SPVXSP is the VIX short-term futures index (long VIX futures)
        # SVXY is -0.5x short of this index
        spvx_monthly = spvxstr_prices.resample("MS").last().dropna()
        spvx_ret = spvx_monthly.pct_change().dropna()

        # Synthetic SVXY = -0.5x of the index return
        svxy_proxy_ret = -0.5 * spvx_ret
        svxy_proxy_ret.name = "SVXY_proxy"

        print(f"\n  Synthetic SVXY proxy constructed: -0.5 * {spvxstr_source} monthly returns")
        print(f"    Date range: {svxy_proxy_ret.index.min().date()} to {svxy_proxy_ret.index.max().date()} "
              f"({len(svxy_proxy_ret)} months)")

    # Step 2c: Feb 2018 structural break
    print(f"\n  Step 2c: SVXY February 2018 Structural Break")
    print(f"  " + "-" * 60)

    svxy_daily = yf_adj_close("SVXY", "2018-01-01")
    if svxy_daily is not None:
        # Feb 5, 2018
        feb5 = pd.Timestamp("2018-02-05")
        feb6 = pd.Timestamp("2018-02-06")
        jan_peak = svxy_daily.loc["2018-01-01":"2018-01-31"].max()
        feb_trough = svxy_daily.loc["2018-02-01":"2018-02-28"].min()
        peak_to_trough = (feb_trough / jan_peak) - 1

        # Single day return around Feb 5
        if feb5 in svxy_daily.index:
            prev_day_idx = svxy_daily.index[svxy_daily.index < feb5]
            if len(prev_day_idx) > 0:
                prev_close = svxy_daily.loc[prev_day_idx[-1]]
                feb5_close = svxy_daily.loc[feb5]
                feb5_ret = (feb5_close / prev_close) - 1
                print(f"    Feb 5 2018 single-day SVXY return: {feb5_ret:+.1%}")
        if feb6 in svxy_daily.index:
            prev_close = svxy_daily.loc[feb5] if feb5 in svxy_daily.index else None
            if prev_close:
                feb6_ret = (svxy_daily.loc[feb6] / prev_close) - 1
                print(f"    Feb 6 2018 single-day SVXY return: {feb6_ret:+.1%}")

        print(f"    Peak-to-trough drawdown (Jan-Feb 2018): {peak_to_trough:+.1%}")
        print(f"    Restructuring date: Feb 27, 2018 (SVXY reduced from -1x to -0.5x)")
        print(f"    Pre-2018 returns in proxy: computed as -0.5x of index (matching -0.5x structure)")
        print(f"    Note: XIV (-1x version) suffered ~-93% and was terminated.")
        print(f"    At -0.5x, the Feb 2018 event produces ~half the drawdown of -1x.")

    return svxy_proxy_ret, spvxstr_source


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: PROXY INDEX SERIES
# ══════════════════════════════════════════════════════════════════════════════

def step3_proxy_indices():
    print(f"\n{'='*100}")
    print(f"  STEP 3: DOWNLOAD PROXY INDEX SERIES")
    print(f"{'='*100}")

    proxies = {}

    # PUT Index → PUTW proxy
    for name, ticker, alt_tickers in [
        ("PUT",  "^PUT",  ["PUT"]),
        ("BXM",  "^BXM",  ["BXM"]),
        ("BXN",  "^BXN",  ["BXN"]),
    ]:
        prices = yf_adj_close(ticker, "2000-01-01")
        source = f"yfinance {ticker}"

        if prices is None or len(prices) < 50:
            for alt in alt_tickers:
                prices = yf_adj_close(alt, "2000-01-01")
                if prices is not None and len(prices) > 50:
                    source = f"yfinance {alt}"
                    break

        if prices is not None and len(prices) > 50:
            monthly = prices.resample("MS").last().dropna()
            monthly_ret = monthly.pct_change().dropna()
            proxies[name] = {
                "prices": monthly,
                "returns": monthly_ret,
                "source": source,
                "first": monthly.index.min(),
                "last": monthly.index.max(),
                "n_months": len(monthly_ret),
            }
        else:
            proxies[name] = None

    # Print status
    print(f"\n  {'Proxy':<10} {'Source':<25} {'First Date':<14} {'Last Date':<14} {'Months':>8} {'Data Gaps':>10}")
    print(f"  {'-'*10} {'-'*25} {'-'*14} {'-'*14} {'-'*8} {'-'*10}")

    for name in ["PUT", "BXM", "BXN"]:
        p = proxies.get(name)
        if p is None:
            print(f"  {name:<10} {'NOT AVAILABLE':<25}")
        else:
            # Check for gaps > 2 months
            diffs = p["returns"].index.to_series().diff().dt.days
            gaps = diffs[diffs > 45]
            gap_str = "None" if len(gaps) == 0 else f"{len(gaps)} gaps"
            print(f"  {name:<10} {p['source']:<25} {p['first'].strftime('%Y-%m-%d'):<14} "
                  f"{p['last'].strftime('%Y-%m-%d'):<14} {p['n_months']:>8} {gap_str:>10}")

    return proxies


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: OVERLAP CORRELATION VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def step4_overlap_correlation(etf_data, proxies, svxy_proxy_ret):
    print(f"\n{'='*100}")
    print(f"  STEP 4: OVERLAP CORRELATION VALIDATION")
    print(f"{'='*100}")

    pairs = [
        ("PUTW", "PUT",  "PUTW vs PUT Index"),
        ("XYLD", "BXM",  "XYLD vs BXM Index"),
        ("QYLD", "BXN",  "QYLD vs BXN Index"),
    ]

    results = {}

    for etf_ticker, proxy_name, label in pairs:
        print(f"\n  {label}")
        print(f"  {'-'*60}")

        etf_d = etf_data.get(etf_ticker)
        proxy_d = proxies.get(proxy_name)

        if etf_d is None or proxy_d is None:
            print(f"    SKIP — data unavailable (ETF: {'ok' if etf_d else 'missing'}, "
                  f"Proxy: {'ok' if proxy_d else 'missing'})")
            results[etf_ticker] = {"status": "SKIP", "reason": "data unavailable"}
            continue

        etf_ret = etf_d["returns"]
        proxy_ret = proxy_d["returns"]

        m_corr, n_months = monthly_corr(etf_ret, proxy_ret)
        a_corr, n_years = annual_corr(etf_ret, proxy_ret)

        # Aligned stats
        common = etf_ret.index.intersection(proxy_ret.index).sort_values()
        e = etf_ret.reindex(common).dropna()
        p = proxy_ret.reindex(common).dropna()
        common2 = e.index.intersection(p.index)
        e = e.reindex(common2)
        p = p.reindex(common2)

        mean_etf = e.mean()
        mean_proxy = p.mean()
        diff = p - e
        te = diff.std()
        ann_ret_diff = (p.mean() - e.mean()) * 12

        overlap_start = common2.min().strftime("%Y-%m")
        overlap_end = common2.max().strftime("%Y-%m")

        if m_corr >= 0.95:
            status = "PASS"
        elif m_corr >= 0.90:
            status = "MARGINAL"
        else:
            status = "FAIL"

        print(f"    Overlap period: {overlap_start} to {overlap_end} ({n_months} months)")
        print(f"    Monthly return correlation:  {m_corr:.3f}")
        print(f"    Annual return correlation:   {a_corr:.3f}")
        print(f"    Mean monthly return ETF:     {mean_etf*100:.2f}%")
        print(f"    Mean monthly return proxy:   {mean_proxy*100:.2f}%")
        print(f"    Annual return diff (proxy - ETF): {ann_ret_diff*100:.2f}%")
        print(f"    Monthly tracking error:      {te*100:.2f}%")
        print(f"    Status: {status} (threshold: >0.95 PASS, 0.90-0.95 MARGINAL, <0.90 FAIL)")

        results[etf_ticker] = {
            "status": status,
            "proxy": proxy_name,
            "m_corr": m_corr,
            "a_corr": a_corr,
            "n_months": n_months,
            "ann_ret_diff": ann_ret_diff,
            "te": te,
            "overlap": f"{overlap_start} to {overlap_end}",
        }

    # SVXY vs synthetic proxy
    print(f"\n  SVXY vs synthetic proxy (both at -0.5x)")
    print(f"  {'-'*60}")

    svxy_d = etf_data.get("SVXY")
    if svxy_d is not None and svxy_proxy_ret is not None:
        svxy_ret = svxy_d["returns"]

        # Use post-restructuring period only (Mar 2018+) for clean -0.5x comparison
        post_restructure = pd.Timestamp("2018-03-01")
        svxy_post = svxy_ret[svxy_ret.index >= post_restructure]
        proxy_post = svxy_proxy_ret[svxy_proxy_ret.index >= post_restructure]

        m_corr_post, n_post = monthly_corr(svxy_post, proxy_post)
        a_corr_post, _ = annual_corr(svxy_post, proxy_post)

        # Also pre-restructure (need to scale SVXY from -1x to -0.5x)
        pre_restructure = pd.Timestamp("2018-02-01")
        svxy_pre = svxy_ret[svxy_ret.index < pre_restructure]
        proxy_pre = svxy_proxy_ret[svxy_proxy_ret.index < pre_restructure]
        # SVXY was -1x, proxy is -0.5x, so scale SVXY returns by 0.5
        svxy_pre_scaled = svxy_pre * 0.5
        m_corr_pre, n_pre = monthly_corr(svxy_pre_scaled, proxy_pre)

        common = svxy_post.index.intersection(proxy_post.index).sort_values()
        if len(common) > 0:
            e = svxy_post.reindex(common).dropna()
            p = proxy_post.reindex(common).dropna()
            common2 = e.index.intersection(p.index)
            if len(common2) > 0:
                e = e.reindex(common2)
                p = p.reindex(common2)
                mean_etf = e.mean()
                mean_proxy = p.mean()
                ann_ret_diff = (p.mean() - e.mean()) * 12
                te = (p - e).std()

        if m_corr_post >= 0.95:
            status = "PASS"
        elif m_corr_post >= 0.90:
            status = "MARGINAL"
        else:
            status = "FAIL"

        print(f"    Post-restructuring overlap (Mar 2018+): {n_post} months")
        print(f"    Monthly return correlation (post):  {m_corr_post:.3f}")
        print(f"    Annual return correlation (post):   {a_corr_post:.3f}")
        if len(common2) > 0:
            print(f"    Mean monthly return SVXY:           {mean_etf*100:.2f}%")
            print(f"    Mean monthly return proxy:          {mean_proxy*100:.2f}%")
            print(f"    Annual return diff (proxy - ETF):   {ann_ret_diff*100:.2f}%")
            print(f"    Monthly tracking error:             {te*100:.2f}%")
        print(f"    Pre-restructuring corr (scaled):    {m_corr_pre:.3f} ({n_pre} months, SVXY scaled 0.5x)")
        print(f"    Status: {status}")

        results["SVXY"] = {
            "status": status,
            "proxy": "SPVXSTR -0.5x",
            "m_corr": m_corr_post,
            "a_corr": a_corr_post,
            "n_months": n_post,
            "ann_ret_diff": ann_ret_diff if len(common2) > 0 else np.nan,
            "te": te if len(common2) > 0 else np.nan,
            "overlap": f"Mar 2018 to present",
            "pre_corr": m_corr_pre,
        }
    else:
        print(f"    SKIP — SVXY ETF or proxy data unavailable")
        results["SVXY"] = {"status": "SKIP", "reason": "data unavailable"}

    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: REGIME STABILITY TEST
# ══════════════════════════════════════════════════════════════════════════════

def step5_regime_stability(etf_data, proxies, svxy_proxy_ret, corr_results):
    print(f"\n{'='*100}")
    print(f"  STEP 5: REGIME STABILITY TEST (correlation by VIX level)")
    print(f"{'='*100}")

    # Download VIX
    vix = yf_adj_close("^VIX", "2010-01-01")
    if vix is None:
        print(f"  ERROR: Could not download VIX data.")
        return {}

    vix_monthly = vix.resample("MS").last().dropna()

    regime_results = {}

    pairs = [
        ("PUTW", "PUT", etf_data.get("PUTW", {}).get("returns"), proxies.get("PUT", {}).get("returns") if proxies.get("PUT") else None),
        ("XYLD", "BXM", etf_data.get("XYLD", {}).get("returns"), proxies.get("BXM", {}).get("returns") if proxies.get("BXM") else None),
        ("QYLD", "BXN", etf_data.get("QYLD", {}).get("returns"), proxies.get("BXN", {}).get("returns") if proxies.get("BXN") else None),
    ]

    # SVXY pair
    svxy_ret = etf_data.get("SVXY", {}).get("returns")
    if svxy_ret is not None and svxy_proxy_ret is not None:
        # Post-restructuring only
        post = pd.Timestamp("2018-03-01")
        pairs.append(("SVXY", "proxy", svxy_ret[svxy_ret.index >= post],
                       svxy_proxy_ret[svxy_proxy_ret.index >= post]))

    print(f"\n  {'Instrument':<12} {'Normal (<20)':>13} {'Elevated (20-30)':>17} {'Stress (>30)':>13} {'Interpretation'}")
    print(f"  {'-'*12} {'-'*13} {'-'*17} {'-'*13} {'-'*30}")

    for etf_name, proxy_name, etf_ret, proxy_ret in pairs:
        if etf_ret is None or proxy_ret is None:
            print(f"  {etf_name + '/' + proxy_name:<12} {'N/A':>13} {'N/A':>17} {'N/A':>13} Data unavailable")
            continue

        common = etf_ret.index.intersection(proxy_ret.index).intersection(vix_monthly.index).sort_values()
        if len(common) < 12:
            print(f"  {etf_name + '/' + proxy_name:<12} {'N/A':>13} {'N/A':>17} {'N/A':>13} Insufficient overlap")
            continue

        e = etf_ret.reindex(common)
        p = proxy_ret.reindex(common)
        v = vix_monthly.reindex(common)

        normal = v < 20
        elevated = (v >= 20) & (v < 30)
        stress = v >= 30

        def bucket_corr(mask):
            if mask.sum() < 5:
                return np.nan
            return float(e[mask].corr(p[mask]))

        c_normal = bucket_corr(normal)
        c_elevated = bucket_corr(elevated)
        c_stress = bucket_corr(stress)

        # Interpretation
        if pd.isna(c_stress) or stress.sum() < 5:
            interp = "Insufficient stress data"
        elif c_stress < 0.80:
            interp = "DEGRADES in stress — CAUTION"
        elif c_stress < c_normal - 0.10:
            interp = "Weakens in stress"
        else:
            interp = "Stable across regimes"

        c_norm_str = f"{c_normal:.3f}" if not pd.isna(c_normal) else "N/A"
        c_elev_str = f"{c_elevated:.3f}" if not pd.isna(c_elevated) else "N/A"
        c_stress_str = f"{c_stress:.3f}" if not pd.isna(c_stress) else "N/A"
        n_str = f"(n={int(normal.sum())}/{int(elevated.sum())}/{int(stress.sum())})"

        print(f"  {etf_name + '/' + proxy_name:<12} {c_norm_str:>13} {c_elev_str:>17} {c_stress_str:>13} {interp}  {n_str}")

        regime_results[etf_name] = {
            "normal": c_normal, "elevated": c_elevated, "stress": c_stress,
            "n_normal": int(normal.sum()), "n_elevated": int(elevated.sum()),
            "n_stress": int(stress.sum()), "interp": interp,
        }

    return regime_results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: SVXY BACKTEST PERIOD ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════

def step6_svxy_assessment(svxy_proxy_ret, spvxstr_source):
    print(f"\n{'='*100}")
    print(f"  STEP 6: SVXY BACKTEST PERIOD ASSESSMENT")
    print(f"{'='*100}")

    if svxy_proxy_ret is not None:
        first = svxy_proxy_ret.index.min()
        print(f"\n  SVXY proxy available from: {first.date()} ({spvxstr_source})")
    else:
        print(f"\n  SVXY proxy: NOT AVAILABLE")
        first = pd.Timestamp("2011-10-01")
        print(f"  Fallback: ETF-only from Oct 2011")

    print(f"  Coverage gap: Jan 2002 - {first.strftime('%b %Y')} ({(first - pd.Timestamp('2002-01-01')).days // 30} months unavailable)")
    print(f"  Missing periods: Early 2002 bear market recovery, pre-{first.year} era")
    if svxy_proxy_ret is not None:
        remaining_years = (svxy_proxy_ret.index.max() - first).days / 365.25
        print(f"  Available backtest: {first.date()} to {svxy_proxy_ret.index.max().date()} ({remaining_years:.1f} years)")
    else:
        print(f"  Available backtest: Oct 2011 to present (ETF-only, ~14 years)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def step7_summary(corr_results, regime_results, proxies, svxy_proxy_ret, spvxstr_source):
    print(f"\n{'='*100}")
    print(f"  STEP 7: FINAL VALIDATION SUMMARY")
    print(f"{'='*100}")

    instruments = [
        ("SVXY", "SPVXSTR -0.5x" if svxy_proxy_ret is not None else "ETF only",
         "2004-03" if svxy_proxy_ret is not None else "2011-10"),
        ("PUTW", "PUT Index", "2002-01" if proxies.get("PUT") else "2016-02"),
        ("QYLD", "BXN Index", "2002-01" if proxies.get("BXN") else "2013-12"),
        ("XYLD", "BXM Index", "2002-01" if proxies.get("BXM") else "2013-06"),
    ]

    print(f"\n  {'Instrument':<12} {'Proxy Source':<20} {'Overlap Corr':>13} {'Regime Stable?':>15} "
          f"{'Reliable From':>14} {'BT Start':>10} {'Verdict':>10}")
    print(f"  {'-'*12} {'-'*20} {'-'*13} {'-'*15} {'-'*14} {'-'*10} {'-'*10}")

    verdicts = {}
    for ticker, proxy_src, reliable_from in instruments:
        cr = corr_results.get(ticker, {})
        rr = regime_results.get(ticker, {})

        m_corr = cr.get("m_corr", np.nan)
        m_corr_str = f"{m_corr:.3f}" if not pd.isna(m_corr) else "N/A"

        regime_ok = rr.get("interp", "N/A")
        regime_short = "Yes" if "Stable" in regime_ok else ("CAUTION" if "CAUTION" in regime_ok else "N/A")

        status = cr.get("status", "SKIP")
        if status == "PASS" and regime_short == "Yes":
            verdict = "PROCEED"
        elif status == "PASS" or status == "MARGINAL":
            verdict = "CAUTION"
        elif status == "SKIP":
            # Check if ETF data exists at least
            verdict = "ETF-ONLY"
        else:
            verdict = "SKIP"

        bt_start = reliable_from
        if status == "FAIL":
            # Restrict to ETF-only
            if ticker == "SVXY":
                bt_start = "2018-03"
            elif ticker == "PUTW":
                bt_start = "2016-02"
            elif ticker == "QYLD":
                bt_start = "2013-12"
            elif ticker == "XYLD":
                bt_start = "2013-06"

        print(f"  {ticker:<12} {proxy_src:<20} {m_corr_str:>13} {regime_short:>15} "
              f"{reliable_from:>14} {bt_start:>10} {verdict:>10}")

        verdicts[ticker] = {"verdict": verdict, "bt_start": bt_start, "proxy_src": proxy_src}

    # Explicit conclusions
    print(f"\n  CONCLUSIONS:")

    cleared = [t for t, v in verdicts.items() if v["verdict"] in ("PROCEED", "CAUTION")]
    etf_only = [t for t, v in verdicts.items() if v["verdict"] == "ETF-ONLY"]
    dropped = [t for t, v in verdicts.items() if v["verdict"] == "SKIP"]

    print(f"\n  1. Instruments cleared for full VRP backtest (proxy + ETF):")
    for t in cleared:
        v = verdicts[t]
        print(f"     {t}: {v['proxy_src']}, backtest from {v['bt_start']}")
    if not cleared:
        print(f"     NONE")

    print(f"\n  2. Instruments restricted to ETF-only periods:")
    for t in etf_only:
        v = verdicts[t]
        print(f"     {t}: ETF only from {v['bt_start']}")
    if not etf_only:
        print(f"     NONE")

    print(f"\n  3. Instruments dropped:")
    for t in dropped:
        print(f"     {t}")
    if not dropped:
        print(f"     NONE")

    # Recommended backtest start
    all_starts = [v["bt_start"] for v in verdicts.values() if v["verdict"] != "SKIP"]
    if all_starts:
        latest = max(all_starts)
        earliest = min(all_starts)
        print(f"\n  4. Recommended VRP backtest start date:")
        print(f"     To include ALL cleared instruments: {latest}")
        print(f"     Earliest available data (longest history): {earliest}")
        print(f"     Suggested: run from {earliest}, note instrument entry dates in results")

    return verdicts


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 100)
    print("  VRP PROXY VALIDATION — DATA QUALITY GATE")
    print("=" * 100)
    print("  Validates SVXY, PUTW, QYLD, XYLD and their pre-ETF proxy indices.")
    print("  Must pass before running any VRP strategy backtest.")

    etf_data = step1_etf_data()
    svxy_proxy_ret, spvxstr_source = step2_svxy_proxy(etf_data)
    proxies = step3_proxy_indices()
    corr_results = step4_overlap_correlation(etf_data, proxies, svxy_proxy_ret)
    regime_results = step5_regime_stability(etf_data, proxies, svxy_proxy_ret, corr_results)
    step6_svxy_assessment(svxy_proxy_ret, spvxstr_source)
    verdicts = step7_summary(corr_results, regime_results, proxies, svxy_proxy_ret, spvxstr_source)

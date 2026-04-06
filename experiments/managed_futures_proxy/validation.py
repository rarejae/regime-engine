"""Managed futures proxy validation: data quality gate for Pod 3 backtest.

Validates DBMF/KMLM ETFs and attempts multiple proxy sources for pre-2019 history:
AQR TSMOM, SG Trend Index, Barclay CTA, BTOP50, mutual fund fallbacks.
"""

import sys, os, io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv; load_dotenv()

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd


# ── Helpers ──────────────────────────────────────────────────────────────────

def yf_adj_close(ticker, start="2000-01-01"):
    import yfinance as yf
    d = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    if d is None or d.empty:
        return None
    p = d["Close"]
    if hasattr(p, "columns"):
        p = p.iloc[:, 0]
    p.index = pd.to_datetime(p.index).tz_localize(None)
    return p


def yf_monthly_ret(ticker, start="2000-01-01"):
    p = yf_adj_close(ticker, start)
    if p is None:
        return None
    monthly = p.resample("MS").last().dropna()
    return monthly.pct_change().dropna()


def monthly_corr(a, b):
    common = a.dropna().index.intersection(b.dropna().index).sort_values()
    if len(common) < 6:
        return np.nan, 0
    return float(a.reindex(common).corr(b.reindex(common))), len(common)


def annual_corr(a, b):
    common = a.dropna().index.intersection(b.dropna().index).sort_values()
    a_ann = a.reindex(common).resample("YS").apply(lambda x: (1 + x).prod() - 1)
    b_ann = b.reindex(common).resample("YS").apply(lambda x: (1 + x).prod() - 1)
    ac = a_ann.dropna().index.intersection(b_ann.dropna().index)
    if len(ac) < 3:
        return np.nan, 0
    return float(a_ann.reindex(ac).corr(b_ann.reindex(ac))), len(ac)


def calendar_year_return(s, year):
    y = s[s.index.year == year]
    if len(y) < 6:
        return np.nan
    return float((1 + y).prod() - 1)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: ETF DATA
# ══════════════════════════════════════════════════════════════════════════════

def step1_etf_data():
    print(f"\n{'='*100}")
    print(f"  STEP 1: DOWNLOAD DBMF AND KMLM ETF DATA")
    print(f"{'='*100}")

    etf_data = {}

    for ticker, name, inception in [("DBMF", "DBMF (iMGP DBi Managed Futures)", "May 2019"),
                                     ("KMLM", "KMLM (KFA Mount Lucas)", "Nov 2020")]:
        ret = yf_monthly_ret(ticker, "2019-01-01")
        if ret is not None and len(ret) > 6:
            prices = yf_adj_close(ticker, "2019-01-01")
            cum = (1 + ret).cumprod()
            dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()

            etf_data[ticker] = ret
            print(f"\n  {name}:")
            print(f"    First date:  {ret.index.min().strftime('%Y-%m-%d')}")
            print(f"    Last date:   {ret.index.max().strftime('%Y-%m-%d')}")
            print(f"    Total months: {len(ret)}")
            for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
                yr_ret = calendar_year_return(ret, yr)
                if not np.isnan(yr_ret):
                    print(f"    {yr} return: {yr_ret:+.1%}")
            print(f"    Max drawdown since inception: {dd:.1%}")
        else:
            print(f"\n  {name}: DOWNLOAD FAILED")

    return etf_data


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: PROXY DOWNLOADS
# ══════════════════════════════════════════════════════════════════════════════

def step2_proxy_downloads():
    print(f"\n{'='*100}")
    print(f"  STEP 2: ATTEMPT PROXY DOWNLOADS")
    print(f"{'='*100}")

    proxies = {}

    # ── Source A: AQR TSMOM ──────────────────────────────────────────────
    print(f"\n  Source A: AQR Time Series Momentum (TSMOM) Factor Data")
    print(f"  " + "-" * 70)

    aqr_urls = [
        "https://images.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Time-Series-Momentum-Factors-Monthly.xlsx",
    ]

    aqr_success = False
    for url in aqr_urls:
        try:
            print(f"    Trying: {url[:80]}...")
            import urllib.request
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            })
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()

            # AQR TSMOM Excel: skip 17 metadata rows, then columns are
            # Date, TSMOM, TSMOM^CM, TSMOM^EQ, TSMOM^FI, TSMOM^FX
            df = pd.read_excel(io.BytesIO(data), sheet_name=0, skiprows=17)
            df.columns = ["Date", "TSMOM", "TSMOM_CM", "TSMOM_EQ", "TSMOM_FI", "TSMOM_FX"]
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").dropna(subset=["TSMOM"])
            df.index = df.index.to_period("M").to_timestamp()
            tsmom = df["TSMOM"].astype(float)

            if len(tsmom) > 100:
                proxies["AQR_TSMOM"] = tsmom
                aqr_success = True
                print(f"    SUCCESS: AQR TSMOM parsed ({len(tsmom)} months)")
                print(f"    Date range: {tsmom.index.min().strftime('%Y-%m')} to {tsmom.index.max().strftime('%Y-%m')}")
                print(f"    Mean monthly return: {tsmom.mean()*100:.2f}%")
                print(f"    Std dev monthly: {tsmom.std()*100:.2f}%")
                yr22 = tsmom[tsmom.index.year == 2022]
                if len(yr22) > 6:
                    print(f"    2022 return: {(1+yr22).prod()-1:+.1%}")

        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {str(e)[:100]}")

    if not aqr_success:
        print(f"    STATUS: FAILED — AQR TSMOM data not available")

    # ── Source B: SG Trend Index ─────────────────────────────────────────
    print(f"\n  Source B: SG Trend Index")
    print(f"  " + "-" * 70)

    sg_success = False
    # Try sgindex.com
    try:
        print(f"    Trying sgindex.com...")
        import urllib.request
        req = urllib.request.Request("https://sgindex.com/index/sgtrnd",
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        if len(html) > 1000:
            print(f"    Page downloaded ({len(html)} bytes) — would need HTML parsing for monthly data")
            print(f"    Requires registration/login for CSV download — not automated")
        else:
            print(f"    Insufficient content")
    except Exception as e:
        print(f"    FAILED: {type(e).__name__}: {str(e)[:80]}")

    if not sg_success:
        print(f"    STATUS: FAILED — SG Trend Index requires registration")

    # ── Source C: Barclay CTA Index ──────────────────────────────────────
    print(f"\n  Source C: Barclay CTA Index")
    print(f"  " + "-" * 70)

    barclay_success = False
    try:
        print(f"    Trying barclayhedge.com CTA Index page...")
        import urllib.request
        req = urllib.request.Request(
            "https://www.barclayhedge.com/solutions/assets-under-management/cta-fund-assets/barclay-cta-index/",
            headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        # Try to find table data in HTML
        if "table" in html.lower():
            # Simple extraction attempt: look for year/month return tables
            import re
            # Look for patterns like "2022" followed by percentage-like numbers
            tables = pd.read_html(io.StringIO(html))
            if tables:
                print(f"    Found {len(tables)} HTML tables")
                for i, t in enumerate(tables[:3]):
                    if len(t) > 5 and len(t.columns) > 5:
                        print(f"    Table {i}: {t.shape}, columns: {list(t.columns)[:6]}")
                        # Check if this looks like a year x month return table
                        if any("Jan" in str(c) or "Feb" in str(c) for c in t.columns):
                            print(f"    → Looks like monthly return table!")
                            # Parse: rows are years, columns are months
                            try:
                                months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                                month_cols = []
                                for mc in months:
                                    matching = [c for c in t.columns if mc in str(c)]
                                    if matching:
                                        month_cols.append(matching[0])
                                    else:
                                        month_cols.append(None)

                                # First column is likely year
                                year_col = t.columns[0]
                                returns = []
                                for _, row in t.iterrows():
                                    try:
                                        yr = int(float(str(row[year_col]).strip()))
                                    except (ValueError, TypeError):
                                        continue
                                    if yr < 1980 or yr > 2030:
                                        continue
                                    for mi, mc in enumerate(month_cols):
                                        if mc is None:
                                            continue
                                        val = row[mc]
                                        try:
                                            r = float(str(val).replace("%", "").strip())
                                            dt = pd.Timestamp(f"{yr}-{mi+1:02d}-01")
                                            returns.append((dt, r / 100))
                                        except (ValueError, TypeError):
                                            continue

                                if len(returns) > 50:
                                    barclay_series = pd.Series(
                                        {d: r for d, r in returns}).sort_index()
                                    proxies["Barclay_CTA"] = barclay_series
                                    barclay_success = True
                                    print(f"    SUCCESS: Parsed {len(barclay_series)} monthly returns")
                                    print(f"    Date range: {barclay_series.index.min().strftime('%Y-%m')} to {barclay_series.index.max().strftime('%Y-%m')}")
                                    print(f"    Mean monthly: {barclay_series.mean()*100:.2f}%")
                                    print(f"    Std monthly: {barclay_series.std()*100:.2f}%")
                            except Exception as e2:
                                print(f"    Parse failed: {e2}")

        if not barclay_success:
            print(f"    Could not extract return data from page")
    except Exception as e:
        print(f"    FAILED: {type(e).__name__}: {str(e)[:80]}")

    if not barclay_success:
        print(f"    STATUS: FAILED — Barclay CTA data not parseable")

    # ── Source D: BTOP50 ─────────────────────────────────────────────────
    print(f"\n  Source D: BTOP50 Index (Barclay)")
    print(f"  " + "-" * 70)

    btop_success = False
    try:
        print(f"    Trying barclayhedge.com BTOP50 page...")
        import urllib.request
        req = urllib.request.Request(
            "https://www.barclayhedge.com/solutions/assets-under-management/cta-fund-assets/btop50-index/",
            headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        if "table" in html.lower():
            tables = pd.read_html(io.StringIO(html))
            if tables:
                print(f"    Found {len(tables)} HTML tables")
                for i, t in enumerate(tables[:3]):
                    if len(t) > 5 and len(t.columns) > 5:
                        print(f"    Table {i}: {t.shape}, columns: {list(t.columns)[:6]}")
                        if any("Jan" in str(c) or "Feb" in str(c) for c in t.columns):
                            print(f"    → Looks like monthly return table!")
                            try:
                                months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                                month_cols = []
                                for mc in months:
                                    matching = [c for c in t.columns if mc in str(c)]
                                    month_cols.append(matching[0] if matching else None)

                                year_col = t.columns[0]
                                returns = []
                                for _, row in t.iterrows():
                                    try:
                                        yr = int(float(str(row[year_col]).strip()))
                                    except (ValueError, TypeError):
                                        continue
                                    if yr < 1980 or yr > 2030:
                                        continue
                                    for mi, mc in enumerate(month_cols):
                                        if mc is None:
                                            continue
                                        val = row[mc]
                                        try:
                                            r = float(str(val).replace("%", "").strip())
                                            dt = pd.Timestamp(f"{yr}-{mi+1:02d}-01")
                                            returns.append((dt, r / 100))
                                        except (ValueError, TypeError):
                                            continue

                                if len(returns) > 50:
                                    btop_series = pd.Series(
                                        {d: r for d, r in returns}).sort_index()
                                    proxies["BTOP50"] = btop_series
                                    btop_success = True
                                    print(f"    SUCCESS: Parsed {len(btop_series)} monthly returns")
                                    print(f"    Date range: {btop_series.index.min().strftime('%Y-%m')} to {btop_series.index.max().strftime('%Y-%m')}")
                                    print(f"    Mean monthly: {btop_series.mean()*100:.2f}%")
                                    print(f"    Std monthly: {btop_series.std()*100:.2f}%")
                            except Exception as e2:
                                print(f"    Parse failed: {e2}")

        if not btop_success:
            print(f"    Could not extract return data from page")
    except Exception as e:
        print(f"    FAILED: {type(e).__name__}: {str(e)[:80]}")

    if not btop_success:
        print(f"    STATUS: FAILED — BTOP50 data not parseable")

    # ── Source E: Managed Futures Mutual Funds ───────────────────────────
    print(f"\n  Source E: Managed Futures Mutual Fund Proxies")
    print(f"  " + "-" * 70)

    fund_candidates = [
        ("AQMIX", "AQR Managed Futures Fund I"),
        ("AMFAX", "Abbey Capital Futures Strategy Fund A"),
        ("MFTFX", "Pimco TRENDS Managed Futures A"),
        ("RYMFX", "Rydex|SGI Managed Futures Fund A"),
    ]

    for ticker, name in fund_candidates:
        ret = yf_monthly_ret(ticker, "1998-01-01")
        if ret is not None and len(ret) > 24:
            proxies[f"MF_{ticker}"] = ret
            print(f"    {ticker} ({name}):")
            print(f"      Date range: {ret.index.min().strftime('%Y-%m')} to {ret.index.max().strftime('%Y-%m')}")
            print(f"      Monthly obs: {len(ret)}")
            print(f"      Mean monthly: {ret.mean()*100:.2f}%")
            print(f"      2022 return: {calendar_year_return(ret, 2022):+.1%}" if not np.isnan(calendar_year_return(ret, 2022)) else "      2022: N/A")
            print(f"      STATUS: Available")
        else:
            print(f"    {ticker}: Not available or insufficient data")

    # Summary
    print(f"\n  PROXY DOWNLOAD SUMMARY:")
    print(f"  {'Source':<20} {'Status':<12} {'Date Range':<24} {'Months':>8}")
    print(f"  {'-'*20} {'-'*12} {'-'*24} {'-'*8}")
    sources = [
        ("AQR TSMOM", "AQR_TSMOM"),
        ("SG Trend Index", None),
        ("Barclay CTA", "Barclay_CTA"),
        ("BTOP50", "BTOP50"),
    ]
    for label, key in sources:
        if key and key in proxies:
            s = proxies[key]
            print(f"  {label:<20} {'SUCCESS':<12} {s.index.min().strftime('%Y-%m')+' to '+s.index.max().strftime('%Y-%m'):<24} {len(s):>8}")
        else:
            print(f"  {label:<20} {'FAILED':<12}")

    for key in proxies:
        if key.startswith("MF_"):
            ticker = key[3:]
            s = proxies[key]
            print(f"  {ticker:<20} {'SUCCESS':<12} {s.index.min().strftime('%Y-%m')+' to '+s.index.max().strftime('%Y-%m'):<24} {len(s):>8}")

    return proxies


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: SELECT BEST PROXY
# ══════════════════════════════════════════════════════════════════════════════

def step3_select_proxy(proxies):
    print(f"\n{'='*100}")
    print(f"  STEP 3: SELECT BEST AVAILABLE PROXY")
    print(f"{'='*100}")

    if not proxies:
        print(f"\n  NO PROXIES AVAILABLE. Cannot proceed with pre-2019 backtest.")
        return None, None

    # Rank by: (1) reaches back to 2002, (2) broadest asset coverage, (3) most months
    candidates = []
    for key, series in proxies.items():
        start_year = series.index.min().year
        n_months = len(series)
        reaches_2002 = start_year <= 2002
        candidates.append((key, start_year, n_months, reaches_2002))

    candidates.sort(key=lambda x: (-x[3], x[1], -x[2]))

    print(f"\n  Ranked candidates (priority: reaches 2002, then earliest start, then most data):")
    for key, start_yr, n, r2002 in candidates:
        marker = " <-- SELECTED" if key == candidates[0][0] else ""
        print(f"    {key:<20} start={start_yr}  months={n:<5}  reaches_2002={'Yes' if r2002 else 'No'}{marker}")

    selected_key = candidates[0][0]
    selected_series = proxies[selected_key]

    # Describe why
    if "AQR" in selected_key:
        reason = "AQR TSMOM: diversified time-series momentum across 58 instruments (equities, bonds, currencies, commodities). Academic factor from Moskowitz et al. (2012). Broadest asset class coverage."
    elif "Barclay" in selected_key:
        reason = "Barclay CTA: tracks ~500 CTAs equally weighted. Broad industry benchmark."
    elif "BTOP50" in selected_key:
        reason = "BTOP50: tracks 50 largest CTAs. More comparable to DBMF (which replicates large CTA exposures)."
    elif "MF_" in selected_key:
        ticker = selected_key[3:]
        reason = f"Mutual fund {ticker}: live returns from regulated fund. May not perfectly match DBMF strategy but captures managed futures return stream."
    else:
        reason = "Best available option."

    print(f"\n  SELECTED: {selected_key}")
    print(f"  Rationale: {reason}")
    print(f"  Date range: {selected_series.index.min().strftime('%Y-%m')} to {selected_series.index.max().strftime('%Y-%m')}")

    return selected_key, selected_series


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: OVERLAP CORRELATION VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def step4_overlap_validation(proxies, etf_data):
    print(f"\n{'='*100}")
    print(f"  STEP 4: OVERLAP CORRELATION VALIDATION")
    print(f"{'='*100}")

    dbmf = etf_data.get("DBMF")
    kmlm = etf_data.get("KMLM")

    results = {}

    if dbmf is None:
        print(f"\n  DBMF ETF data not available — cannot validate.")
        return results

    # Validate each proxy against DBMF
    for key, proxy_series in proxies.items():
        m_corr, n_months = monthly_corr(proxy_series, dbmf)
        a_corr, n_years = annual_corr(proxy_series, dbmf)

        if n_months < 6:
            print(f"\n  {key} vs DBMF: insufficient overlap ({n_months} months)")
            results[key] = {"status": "SKIP", "reason": "insufficient overlap", "n_months": n_months}
            continue

        common = proxy_series.dropna().index.intersection(dbmf.dropna().index).sort_values()
        p = proxy_series.reindex(common).dropna()
        e = dbmf.reindex(common).dropna()
        common2 = p.index.intersection(e.index)
        p = p.reindex(common2)
        e = e.reindex(common2)

        te = (p - e).std()
        ann_diff = (p.mean() - e.mean()) * 12

        if m_corr >= 0.85:
            status = "PASS"
        elif m_corr >= 0.75:
            status = "MARGINAL"
        else:
            status = "FAIL"

        overlap_str = f"{common2.min().strftime('%Y-%m')} to {common2.max().strftime('%Y-%m')}"

        print(f"\n  {key} vs DBMF (overlap: {overlap_str}, {n_months} months):")
        print(f"    Monthly return correlation:  {m_corr:.3f}")
        print(f"    Annual return correlation:   {a_corr:.3f}" if not np.isnan(a_corr) else "    Annual return correlation:   N/A (insufficient years)")
        print(f"    Mean monthly return (proxy): {p.mean()*100:.2f}%")
        print(f"    Mean monthly return (DBMF):  {e.mean()*100:.2f}%")
        print(f"    Annual return diff:          {ann_diff*100:.2f}%")
        print(f"    Monthly tracking error:      {te*100:.2f}%")
        print(f"    Status: {status} (threshold: >0.85 PASS, 0.75-0.85 MARGINAL, <0.75 FAIL)")

        results[key] = {
            "status": status, "m_corr": m_corr, "a_corr": a_corr,
            "n_months": n_months, "ann_diff": ann_diff, "te": te,
            "overlap": overlap_str,
        }

    # KMLM vs DBMF supplementary
    if kmlm is not None:
        m_corr, n_months = monthly_corr(kmlm, dbmf)
        print(f"\n  KMLM vs DBMF (supplementary, {n_months} months):")
        print(f"    Monthly return correlation:  {m_corr:.3f}")
        if m_corr >= 0.85:
            print(f"    Status: PASS")
        elif m_corr >= 0.75:
            print(f"    Status: MARGINAL")
        else:
            print(f"    Status: FAIL")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: REGIME STABILITY TEST
# ══════════════════════════════════════════════════════════════════════════════

def step5_regime_stability(proxies, etf_data, selected_key):
    print(f"\n{'='*100}")
    print(f"  STEP 5: REGIME STABILITY TEST")
    print(f"{'='*100}")

    dbmf = etf_data.get("DBMF")
    if dbmf is None or selected_key is None or selected_key not in proxies:
        print(f"  Cannot run regime stability — missing data.")
        return {}

    proxy = proxies[selected_key]
    ivv_ret = yf_monthly_ret("SPY", "2018-01-01")
    vix = yf_adj_close("^VIX", "2018-01-01")

    if ivv_ret is None or vix is None:
        print(f"  Cannot run — IVV or VIX data unavailable.")
        return {}

    vix_monthly = vix.resample("MS").last().dropna()
    ivv_12m = ivv_ret.rolling(12).apply(lambda x: (1 + x).prod() - 1).dropna()

    common = proxy.dropna().index.intersection(dbmf.dropna().index)
    common = common.intersection(vix_monthly.dropna().index)
    common = common.intersection(ivv_12m.dropna().index).sort_values()

    if len(common) < 12:
        print(f"  Insufficient overlapping data ({len(common)} months).")
        return {}

    p = proxy.reindex(common)
    d = dbmf.reindex(common)
    v = vix_monthly.reindex(common)
    ivv12 = ivv_12m.reindex(common)

    regimes = {
        "Bull equity (12m>10%)": ivv12 > 0.10,
        "Bear equity (12m<0%)": ivv12 < 0,
        "High VIX (>25)": v > 25,
        "Low VIX (<15)": v < 15,
    }

    print(f"\n  {'Regime':<28} {'Proxy-DBMF Corr':>16} {'N months':>10} {'Interpretation'}")
    print(f"  {'-'*28} {'-'*16} {'-'*10} {'-'*20}")

    regime_results = {}
    for label, mask in regimes.items():
        n = mask.sum()
        if n < 5:
            print(f"  {label:<28} {'N/A':>16} {n:>10} Insufficient data")
            continue
        corr = float(p[mask].corr(d[mask]))
        if pd.isna(corr):
            print(f"  {label:<28} {'N/A':>16} {n:>10} Correlation undefined")
            continue

        if corr >= 0.75:
            interp = "Stable"
        elif corr >= 0.50:
            interp = "Moderate"
        else:
            interp = "WEAK"

        print(f"  {label:<28} {corr:>16.3f} {n:>10} {interp}")
        regime_results[label] = {"corr": corr, "n": int(n)}

    return regime_results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: CRISIS BEHAVIOR (DBMF LIVE DATA ONLY)
# ══════════════════════════════════════════════════════════════════════════════

def step6_crisis_behavior(etf_data):
    print(f"\n{'='*100}")
    print(f"  STEP 6: CRISIS BEHAVIOR CHECK (DBMF ETF PERIOD ONLY)")
    print(f"{'='*100}")

    dbmf = etf_data.get("DBMF")
    ivv_ret = yf_monthly_ret("SPY", "2019-01-01")

    if dbmf is None or ivv_ret is None:
        print(f"  Data unavailable.")
        return

    common = dbmf.dropna().index.intersection(ivv_ret.dropna().index).sort_values()
    d = dbmf.reindex(common)
    i = ivv_ret.reindex(common)

    print(f"\n  {'Period':<24} {'DBMF Return':>12} {'IVV Return':>12} {'DBMF-IVV Corr':>15}")
    print(f"  {'-'*24} {'-'*12} {'-'*12} {'-'*15}")

    crises = [
        ("COVID (Feb-Apr 2020)", "2020-02", "2020-04"),
        ("2022 Bear (Jan-Dec)", "2022-01", "2022-12"),
        ("2025 tariff (Feb-Apr)", "2025-02", "2025-04"),
    ]

    for label, cs, ce in crises:
        crisis_d = d[(d.index >= pd.Timestamp(cs)) & (d.index <= pd.Timestamp(ce))]
        crisis_i = i[(i.index >= pd.Timestamp(cs)) & (i.index <= pd.Timestamp(ce))]
        common_c = crisis_d.dropna().index.intersection(crisis_i.dropna().index)

        if len(common_c) < 2:
            print(f"  {label:<24} {'N/A':>12} {'N/A':>12} {'N/A':>15}")
            continue

        cd = crisis_d.reindex(common_c)
        ci = crisis_i.reindex(common_c)
        d_ret = (1 + cd).prod() - 1
        i_ret = (1 + ci).prod() - 1
        corr = cd.corr(ci) if len(common_c) >= 3 else np.nan
        corr_str = f"{corr:.3f}" if not np.isnan(corr) else "N/A (2mo)"

        print(f"  {label:<24} {d_ret:>+11.1%} {i_ret:>+11.1%} {corr_str:>15}")

    print(f"\n  Key question: Is DBMF negatively correlated with equities during stress?")
    # Check 2022 specifically
    d_2022 = d[d.index.year == 2022]
    i_2022 = i[i.index.year == 2022]
    if len(d_2022) > 6 and len(i_2022) > 6:
        corr_2022 = d_2022.reindex(i_2022.index).dropna().corr(i_2022.dropna())
        d_ret_2022 = (1 + d_2022).prod() - 1
        i_ret_2022 = (1 + i_2022).prod() - 1
        print(f"  2022 full year: DBMF {d_ret_2022:+.1%}, IVV {i_ret_2022:+.1%}, correlation {corr_2022:.3f}")
        if corr_2022 < -0.3:
            print(f"  → YES: DBMF shows negative equity correlation during 2022 — genuine crisis diversifier")
        elif corr_2022 < 0.1:
            print(f"  → PARTIALLY: DBMF uncorrelated (not negatively correlated) with equity in 2022")
        else:
            print(f"  → NO: DBMF positively correlated with equity in 2022 — not a crisis hedge")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def step7_summary(proxies, corr_results, regime_results, selected_key, etf_data):
    print(f"\n{'='*100}")
    print(f"  STEP 7: FINAL VALIDATION SUMMARY")
    print(f"{'='*100}")

    print(f"\n  {'Proxy':<20} {'Source':<16} {'Overlap Corr':>13} {'Regime Stable':>14} {'Date Range':<24} {'Verdict':>10}")
    print(f"  {'-'*20} {'-'*16} {'-'*13} {'-'*14} {'-'*24} {'-'*10}")

    for key, series in proxies.items():
        cr = corr_results.get(key, {})
        m_corr = cr.get("m_corr", np.nan)
        m_str = f"{m_corr:.3f}" if not np.isnan(m_corr) else "N/A"

        # Regime stability: check if any regime was WEAK
        regime_ok = "N/A"
        if regime_results:
            weak = any(v.get("corr", 1) < 0.50 for v in regime_results.values() if isinstance(v, dict))
            regime_ok = "WEAK" if weak else "Yes"

        status = cr.get("status", "N/A")
        dr = f"{series.index.min().strftime('%Y-%m')} to {series.index.max().strftime('%Y-%m')}"

        source = "AQR" if "AQR" in key else ("Barclay" if "Barclay" in key else
                 ("Barclay" if "BTOP" in key else ("yfinance" if "MF_" in key else "other")))

        marker = " <--" if key == selected_key else ""
        print(f"  {key:<20} {source:<16} {m_str:>13} {regime_ok:>14} {dr:<24} {status:>10}{marker}")

    # Conclusions
    print(f"\n  CONCLUSIONS:")

    cleared = [k for k, v in corr_results.items() if v.get("status") in ("PASS", "MARGINAL")]
    if cleared:
        print(f"\n  1. Proxies cleared for managed futures backtest:")
        for k in cleared:
            cr = corr_results[k]
            s = proxies[k]
            print(f"     {k}: corr {cr['m_corr']:.3f}, backtest from {s.index.min().strftime('%Y-%m')}")
    else:
        print(f"\n  1. No proxies cleared via overlap validation with DBMF.")
        if proxies:
            print(f"     Available proxies exist but may have insufficient DBMF overlap.")
            print(f"     Consider using best available proxy with appropriate caveats.")

    # DBMF crisis behavior
    dbmf = etf_data.get("DBMF")
    if dbmf is not None:
        d_2022 = dbmf[dbmf.index.year == 2022]
        if len(d_2022) > 6:
            ret_2022 = (1 + d_2022).prod() - 1
            print(f"\n  2. DBMF crisis behavior: 2022 return = {ret_2022:+.1%}")
            if ret_2022 > 0.10:
                print(f"     CONFIRMED: DBMF was strongly positive during 2022 equity bear — genuine diversifier")
            elif ret_2022 > 0:
                print(f"     POSITIVE: DBMF was modestly positive during 2022")
            else:
                print(f"     NEGATIVE: DBMF was negative during 2022 — not a reliable crisis hedge")

    if selected_key:
        s = proxies[selected_key]
        print(f"\n  3. Recommended proxy for full backtest: {selected_key}")
        print(f"     Backtest start: {s.index.min().strftime('%Y-%m')}")
        cr = corr_results.get(selected_key, {})
        if cr.get("m_corr"):
            print(f"     DBMF overlap correlation: {cr['m_corr']:.3f}")
    else:
        print(f"\n  3. No proxy selected. DBMF ETF-only from May 2019 (~7 years).")

    print(f"\n  4. Structural issues to flag:")
    print(f"     - All CTA indices have survivorship bias (only include funds that still exist)")
    print(f"     - DBMF uses weekly regression to estimate CTA positions — tracking error vs actual CTAs")
    print(f"     - Managed futures style has evolved: more systematic, faster signals, broader universes")
    print(f"     - Pre-2010 CTA returns may overstate go-forward performance due to less crowding")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 100)
    print("  MANAGED FUTURES PROXY VALIDATION — DATA QUALITY GATE (POD 3)")
    print("=" * 100)

    etf_data = step1_etf_data()
    proxies = step2_proxy_downloads()
    selected_key, selected_series = step3_select_proxy(proxies)
    corr_results = step4_overlap_validation(proxies, etf_data)
    regime_results = step5_regime_stability(proxies, etf_data, selected_key)
    step6_crisis_behavior(etf_data)
    step7_summary(proxies, corr_results, regime_results, selected_key, etf_data)

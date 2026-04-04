"""Search for better pre-ETF proxies for IVV, VGLT, IAU, DBC."""

import os, sys, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dotenv import load_dotenv; load_dotenv()
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = os.path.join(os.path.dirname(__file__), "output")

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
    """Monthly returns from yfinance (pct_change of monthly close)."""
    import yfinance as yf
    try:
        d = yf.download(ticker, start=start, interval="1mo", progress=False)
        if d is None or d.empty:
            return None
        p = d["Close"]
        if hasattr(p, "columns"):
            p = p.iloc[:, 0]
        p.index = pd.to_datetime(p.index).tz_localize(None).to_period("M").to_timestamp()
        return p.pct_change().dropna()
    except Exception:
        return None


def yf_daily(ticker, start="1955-01-01"):
    import yfinance as yf
    try:
        d = yf.download(ticker, start=start, progress=False)
        if d is None or d.empty:
            return None
        p = d["Close"]
        if hasattr(p, "columns"):
            p = p.iloc[:, 0]
        p.index = pd.to_datetime(p.index).tz_localize(None)
        return p
    except Exception:
        return None


# ── Metrics ───────────────────────────────────────────────────────────────────

def overlap_metrics(proxy, etf, name):
    """Compute correlation, return diff, TE on overlap period."""
    common = proxy.index.intersection(etf.index).sort_values()
    p = proxy.reindex(common).dropna()
    e = etf.reindex(common).dropna()
    common = p.index.intersection(e.index)
    if len(common) < 12:
        return None
    p, e = p.reindex(common), e.reindex(common)
    diff = p - e
    corr = p.corr(e)
    te_ann = diff.std() * np.sqrt(12)
    ann_diff = diff.mean() * 12
    return {
        "name": name,
        "n": len(common),
        "start": common.min().strftime("%Y-%m"),
        "end": common.max().strftime("%Y-%m"),
        "corr": corr,
        "ann_diff": ann_diff,
        "te_ann": te_ann,
        "max_div": diff.abs().max(),
        "max_div_date": diff.abs().idxmax().strftime("%Y-%m"),
        "proxy": p,
        "etf": e,
    }


def rate(corr, ann_diff, te_ann):
    c = "Good" if abs(corr) > 0.95 else ("Acceptable" if abs(corr) > 0.85 else "Poor")
    d = "Good" if abs(ann_diff) < 0.01 else ("Acceptable" if abs(ann_diff) < 0.03 else "Poor")
    t = "Good" if te_ann < 0.05 else ("Acceptable" if te_ann < 0.10 else "Poor")
    scores = [c, d, t]
    overall = "Poor" if "Poor" in scores else ("Acceptable" if "Acceptable" in scores else "Good")
    return overall, scores


def fmt_candidate(m, history_start=None):
    """Format a candidate result block."""
    if m is None:
        return "  Result: INSUFFICIENT OVERLAP (<12 months)\n"
    overall, sub = rate(m["corr"], m["ann_diff"], m["te_ann"])
    lines = []
    if history_start:
        lines.append(f"  Full history start: {history_start}")
    lines.append(f"  Overlap with ETF:   {m['start']} to {m['end']} ({m['n']} months)")
    lines.append(f"  Correlation:        {m['corr']:.4f}    [{sub[0]}]")
    dir_ = "overstates" if m["ann_diff"] > 0 else "understates"
    lines.append(f"  Ann. return diff:   {m['ann_diff']*100:.2f}% (proxy {dir_})    [{sub[1]}]")
    lines.append(f"  Tracking error:     {m['te_ann']*100:.2f}% annualized    [{sub[2]}]")
    lines.append(f"  Max divergence:     {m['max_div']*100:.2f}% ({m['max_div_date']})")
    lines.append(f"  ── Rating:          {overall.upper()}")
    return "\n".join(lines) + "\n"


# ── Asset-specific candidate searches ─────────────────────────────────────────

def search_ivv():
    """Test IVV proxy candidates."""
    print("\n  Fetching IVV ETF returns...")
    etf = yf_monthly_ret("IVV", "2000-05-01")
    if etf is None:
        return None, "ETF data unavailable"

    results = {}
    report = []

    # Current baseline
    print("  Testing current: SPASTT01USM661N...")
    sp_oecd = fred("SPASTT01USM661N", "1957-01-01")
    if sp_oecd is not None:
        proxy = sp_oecd.resample("MS").last().pct_change().dropna()
        m = overlap_metrics(proxy, etf, "SPASTT01USM661N (current)")
        results["current"] = m
        report.append(f"\nCurrent: SPASTT01USM661N (OECD US Share Prices)")
        report.append(fmt_candidate(m, sp_oecd.index.min().strftime("%Y-%m")))

    # Candidate 1: ^GSPC daily → monthly returns
    print("  Testing: ^GSPC daily → monthly...")
    gspc = yf_daily("^GSPC", "1927-01-01")
    if gspc is not None:
        proxy = gspc.resample("MS").last().pct_change().dropna()
        proxy.index = proxy.index.to_period("M").to_timestamp()
        m = overlap_metrics(proxy, etf, "^GSPC")
        results["gspc"] = m
        report.append(f"Candidate 1: ^GSPC (yfinance daily → monthly close)")
        report.append(f"  Available: YES — {gspc.index.min().date()} to {gspc.index.max().date()}")
        report.append(fmt_candidate(m, gspc.index.min().strftime("%Y-%m")))
    else:
        report.append(f"Candidate 1: ^GSPC — NOT AVAILABLE\n")

    # Candidate 2: ^SP500TR (total return)
    print("  Testing: ^SP500TR...")
    sp500tr = yf_daily("^SP500TR", "1988-01-01")
    if sp500tr is not None and len(sp500tr) > 100:
        proxy = sp500tr.resample("MS").last().pct_change().dropna()
        proxy.index = proxy.index.to_period("M").to_timestamp()
        m = overlap_metrics(proxy, etf, "^SP500TR")
        results["sp500tr"] = m
        report.append(f"Candidate 2: ^SP500TR (yfinance S&P 500 Total Return)")
        report.append(f"  Available: YES — {sp500tr.index.min().date()} to {sp500tr.index.max().date()}")
        report.append(fmt_candidate(m, sp500tr.index.min().strftime("%Y-%m")))
    else:
        report.append(f"Candidate 2: ^SP500TR — NOT AVAILABLE (yfinance may not carry this)\n")

    # Candidate 3: ^GSPC + dividend yield from FRED (S&P dividend yield or approximate)
    # We can try to add back ~2% annualized dividend return to ^GSPC
    print("  Testing: ^GSPC + FRED dividend yield adjustment...")
    if gspc is not None:
        # Try FRED SP dividend yield
        sp_divy = fred("MULTPL/SP500_DIV_YIELD_MONTH", "1955-01-01")  # may not exist
        if sp_divy is None:
            # Approximate: add TB3MS/2 as rough dividend proxy, or just a constant 1.5%/12
            gspc_m = gspc.resample("MS").last()
            price_ret = gspc_m.pct_change().dropna()
            # Constant dividend yield approximation: ~1.8% annualized in modern era
            div_monthly = 0.018 / 12
            proxy = price_ret + div_monthly
            proxy.index = proxy.index.to_period("M").to_timestamp()
            m = overlap_metrics(proxy, etf, "^GSPC+1.8%div")
            results["gspc_div"] = m
            report.append(f"Candidate 3: ^GSPC + 1.8% annualized dividend yield (constant approx)")
            report.append(fmt_candidate(m, gspc.index.min().strftime("%Y-%m")))

    # Candidate 4: FRED WILSHIRE 5000 (WILL5000INDFC) — check availability
    print("  Testing: WILL5000INDFC...")
    w5k = fred("WILL5000INDFC", "1970-01-01")
    if w5k is not None and len(w5k) > 200:
        proxy = w5k.resample("MS").last().pct_change().dropna()
        m = overlap_metrics(proxy, etf, "WILL5000INDFC")
        results["w5k"] = m
        report.append(f"Candidate 4: WILL5000INDFC (Wilshire 5000, FRED)")
        report.append(f"  Available: YES — {w5k.index.min().date()} to {w5k.index.max().date()}")
        report.append(fmt_candidate(m, w5k.index.min().strftime("%Y-%m")))
    else:
        report.append(f"Candidate 4: WILL5000INDFC — NOT AVAILABLE or too short\n")

    return results, "\n".join(report)


def search_vglt():
    """Test VGLT proxy candidates."""
    print("\n  Fetching VGLT ETF returns...")
    etf = yf_monthly_ret("VGLT", "2009-11-01")
    if etf is None or len(etf) < 10:
        print("  VGLT unavailable, trying TLT...")
        etf = yf_monthly_ret("TLT", "2002-07-01")
    if etf is None:
        return None, "ETF data unavailable"

    results = {}
    report = []

    # Current: GS10 duration=17
    print("  Testing current: GS10 duration=17...")
    gs10 = fred("GS10", "1960-01-01")
    if gs10 is not None:
        gs10_m = gs10.resample("MS").last().dropna()
        dy = gs10_m.diff()
        coupon = gs10_m.shift(1) / 1200
        proxy = (-17.0 * dy / 100 + coupon).dropna()
        m = overlap_metrics(proxy, etf, "GS10 dur=17 (current)")
        results["current"] = m
        report.append(f"\nCurrent: GS10 with duration=17, coupon=GS10/1200")
        report.append(fmt_candidate(m, gs10.index.min().strftime("%Y-%m")))

    # Candidate 1: GS30 duration=18
    print("  Testing: GS30 duration=18...")
    gs30 = fred("GS30", "1977-02-01")
    if gs30 is not None and len(gs30) > 100:
        gs30_m = gs30.resample("MS").last().dropna()
        dy = gs30_m.diff()
        coupon = gs30_m.shift(1) / 1200
        proxy = (-18.0 * dy / 100 + coupon).dropna()
        m = overlap_metrics(proxy, etf, "GS30 dur=18")
        results["gs30_18"] = m
        report.append(f"Candidate 1: GS30 (30Y yield) with duration=18")
        report.append(f"  Available: YES — {gs30.index.min().date()} to {gs30.index.max().date()}")
        report.append(fmt_candidate(m, gs30.index.min().strftime("%Y-%m")))
    else:
        report.append(f"Candidate 1: GS30 — NOT AVAILABLE\n")

    # Candidate 2: GS30 duration=20
    if gs30 is not None and len(gs30) > 100:
        print("  Testing: GS30 duration=20...")
        gs30_m = gs30.resample("MS").last().dropna()
        dy = gs30_m.diff()
        coupon = gs30_m.shift(1) / 1200
        proxy = (-20.0 * dy / 100 + coupon).dropna()
        m = overlap_metrics(proxy, etf, "GS30 dur=20")
        results["gs30_20"] = m
        report.append(f"Candidate 2: GS30 (30Y yield) with duration=20")
        report.append(fmt_candidate(m, gs30.index.min().strftime("%Y-%m")))

    # Candidate 3: GS20 duration=15
    print("  Testing: GS20 duration=15...")
    gs20 = fred("GS20", "1993-01-01")
    if gs20 is not None and len(gs20) > 100:
        gs20_m = gs20.resample("MS").last().dropna()
        dy = gs20_m.diff()
        coupon = gs20_m.shift(1) / 1200
        proxy = (-15.0 * dy / 100 + coupon).dropna()
        m = overlap_metrics(proxy, etf, "GS20 dur=15")
        results["gs20_15"] = m
        report.append(f"Candidate 3: GS20 (20Y yield) with duration=15")
        report.append(f"  Available: YES — {gs20.index.min().date()} to {gs20.index.max().date()}")
        report.append(fmt_candidate(m, gs20.index.min().strftime("%Y-%m")))
    else:
        report.append(f"Candidate 3: GS20 — NOT AVAILABLE or too short\n")

    # Candidate 4: GS10 with optimized duration (scan 5-25)
    print("  Testing: GS10 with optimized duration...")
    if gs10 is not None:
        gs10_m = gs10.resample("MS").last().dropna()
        dy_gs10 = gs10_m.diff()
        coupon_gs10 = gs10_m.shift(1) / 1200
        best_dur, best_corr = 17, 0
        for dur in np.arange(5, 26, 0.5):
            proxy_test = (-dur * dy_gs10 / 100 + coupon_gs10).dropna()
            m_test = overlap_metrics(proxy_test, etf, f"GS10 dur={dur}")
            if m_test and abs(m_test["corr"]) > abs(best_corr):
                best_corr = m_test["corr"]
                best_dur = dur
        proxy = (-best_dur * dy_gs10 / 100 + coupon_gs10).dropna()
        m = overlap_metrics(proxy, etf, f"GS10 dur={best_dur:.1f} (optimized)")
        results["gs10_opt"] = m
        report.append(f"Candidate 4: GS10 with optimized duration={best_dur:.1f}")
        report.append(fmt_candidate(m, gs10.index.min().strftime("%Y-%m")))

    # Candidate 5: GS30 with optimized duration (scan 10-25)
    if gs30 is not None and len(gs30) > 100:
        print("  Testing: GS30 with optimized duration...")
        gs30_m = gs30.resample("MS").last().dropna()
        dy_gs30 = gs30_m.diff()
        coupon_gs30 = gs30_m.shift(1) / 1200
        best_dur, best_corr = 18, 0
        for dur in np.arange(10, 26, 0.5):
            proxy_test = (-dur * dy_gs30 / 100 + coupon_gs30).dropna()
            m_test = overlap_metrics(proxy_test, etf, f"GS30 dur={dur}")
            if m_test and abs(m_test["corr"]) > abs(best_corr):
                best_corr = m_test["corr"]
                best_dur = dur
        proxy = (-best_dur * dy_gs30 / 100 + coupon_gs30).dropna()
        m = overlap_metrics(proxy, etf, f"GS30 dur={best_dur:.1f} (optimized)")
        results["gs30_opt"] = m
        report.append(f"Candidate 5: GS30 with optimized duration={best_dur:.1f}")
        report.append(fmt_candidate(m, gs30.index.min().strftime("%Y-%m")))

    # Candidate 6: Blend — GS30 for level, GS10 for convexity backup
    # Use GS30 where available, splice GS10*scale before
    if gs30 is not None and gs10 is not None:
        print("  Testing: GS10→GS30 splice (GS10*1.3 pre-1977, GS30 dur=18 after)...")
        gs10_m = gs10.resample("MS").last().dropna()
        gs30_m = gs30.resample("MS").last().dropna()
        # Pre-GS30: scale GS10 duration model to approximate long end
        dy10 = gs10_m.diff()
        coup10 = gs10_m.shift(1) / 1200
        pre = (-22.0 * dy10 / 100 + coup10).dropna()  # higher duration to approximate 30Y

        dy30 = gs30_m.diff()
        coup30 = gs30_m.shift(1) / 1200
        post = (-18.0 * dy30 / 100 + coup30).dropna()

        cutoff = post.index.min()
        spliced = pd.concat([pre[pre.index < cutoff], post]).sort_index()
        spliced = spliced[~spliced.index.duplicated(keep="last")]
        m = overlap_metrics(spliced, etf, "GS10(dur=22)→GS30(dur=18) splice")
        results["splice"] = m
        report.append(f"Candidate 6: Splice GS10(dur=22) pre-1977 → GS30(dur=18) post")
        report.append(f"  Full history: {gs10.index.min().date()} to present (spliced at {cutoff.date()})")
        report.append(fmt_candidate(m, gs10.index.min().strftime("%Y-%m")))

    return results, "\n".join(report)


def search_iau():
    """Test IAU/gold proxy candidates."""
    print("\n  Fetching GLD ETF returns...")
    etf = yf_monthly_ret("GLD", "2004-11-01")
    if etf is None or len(etf) < 10:
        print("  GLD unavailable, trying IAU...")
        etf = yf_monthly_ret("IAU", "2005-01-01")
    if etf is None:
        return None, "ETF data unavailable"

    results = {}
    report = []

    # Current: WPU1017
    print("  Testing current: WPU1017...")
    wpu = fred("WPU1017", "1960-01-01")
    if wpu is not None:
        proxy = wpu.resample("MS").last().pct_change().dropna()
        m = overlap_metrics(proxy, etf, "WPU1017 (current)")
        results["current"] = m
        report.append(f"\nCurrent: WPU1017 (PPI Precious Metals & Metal Products)")
        report.append(fmt_candidate(m, wpu.index.min().strftime("%Y-%m")))

    # Candidate 1: GOLDAMGBD228NLBM (London AM fix, daily)
    print("  Testing: GOLDAMGBD228NLBM...")
    gold_am = fred("GOLDAMGBD228NLBM", "1968-04-01")
    if gold_am is not None and len(gold_am) > 100:
        proxy = gold_am.resample("MS").last().pct_change().dropna()
        m = overlap_metrics(proxy, etf, "GOLDAMGBD228NLBM")
        results["gold_am"] = m
        report.append(f"Candidate 1: GOLDAMGBD228NLBM (London AM Gold Fix, FRED)")
        report.append(f"  Available: YES — {gold_am.index.min().date()} to {gold_am.index.max().date()}")
        report.append(fmt_candidate(m, gold_am.index.min().strftime("%Y-%m")))
    else:
        report.append(f"Candidate 1: GOLDAMGBD228NLBM — NOT AVAILABLE\n")

    # Candidate 2: GOLDPMGBD228NLBM (London PM fix, daily)
    print("  Testing: GOLDPMGBD228NLBM...")
    gold_pm = fred("GOLDPMGBD228NLBM", "1968-04-01")
    if gold_pm is not None and len(gold_pm) > 100:
        proxy = gold_pm.resample("MS").last().pct_change().dropna()
        m = overlap_metrics(proxy, etf, "GOLDPMGBD228NLBM")
        results["gold_pm"] = m
        report.append(f"Candidate 2: GOLDPMGBD228NLBM (London PM Gold Fix, FRED)")
        report.append(f"  Available: YES — {gold_pm.index.min().date()} to {gold_pm.index.max().date()}")
        report.append(fmt_candidate(m, gold_pm.index.min().strftime("%Y-%m")))
    else:
        report.append(f"Candidate 2: GOLDPMGBD228NLBM — NOT AVAILABLE\n")

    # Candidate 3: GC=F (gold futures, yfinance)
    print("  Testing: GC=F...")
    gc = yf_daily("GC=F", "2000-01-01")
    if gc is not None and len(gc) > 100:
        proxy = gc.resample("MS").last().pct_change().dropna()
        proxy.index = proxy.index.to_period("M").to_timestamp()
        m = overlap_metrics(proxy, etf, "GC=F")
        results["gc_f"] = m
        report.append(f"Candidate 3: GC=F (Gold Futures, yfinance)")
        report.append(f"  Available: YES — {gc.index.min().date()} to {gc.index.max().date()}")
        report.append(fmt_candidate(m, gc.index.min().strftime("%Y-%m")))
    else:
        report.append(f"Candidate 3: GC=F — NOT AVAILABLE\n")

    # Candidate 4: PCU2122212222 (Gold ore mining PPI)
    print("  Testing: PCU21222122221...")
    gold_ppi2 = fred("PCU21222122221", "1985-01-01")
    if gold_ppi2 is not None and len(gold_ppi2) > 50:
        proxy = gold_ppi2.resample("MS").last().pct_change().dropna()
        m = overlap_metrics(proxy, etf, "PCU21222122221")
        results["pcu_gold"] = m
        report.append(f"Candidate 4: PCU21222122221 (Gold Ore Mining PPI)")
        report.append(f"  Available: YES — {gold_ppi2.index.min().date()} to {gold_ppi2.index.max().date()}")
        report.append(fmt_candidate(m, gold_ppi2.index.min().strftime("%Y-%m")))
    else:
        report.append(f"Candidate 4: PCU21222122221 — NOT AVAILABLE or too short\n")

    # Candidate 5: WPU10170601 (PPI Gold Ores) — more targeted than WPU1017
    print("  Testing: WPU10170601...")
    gold_ore = fred("WPU10170601", "1960-01-01")
    if gold_ore is not None and len(gold_ore) > 50:
        proxy = gold_ore.resample("MS").last().pct_change().dropna()
        m = overlap_metrics(proxy, etf, "WPU10170601")
        results["wpu_gold_ore"] = m
        report.append(f"Candidate 5: WPU10170601 (PPI Gold Ores)")
        report.append(f"  Available: YES — {gold_ore.index.min().date()} to {gold_ore.index.max().date()}")
        report.append(fmt_candidate(m, gold_ore.index.min().strftime("%Y-%m")))
    else:
        report.append(f"Candidate 5: WPU10170601 — NOT AVAILABLE or too short\n")

    return results, "\n".join(report)


def search_dbc():
    """Test DBC proxy candidates."""
    print("\n  Fetching DBC ETF returns...")
    etf = yf_monthly_ret("DBC", "2006-02-01")
    if etf is None:
        return None, "ETF data unavailable"

    results = {}
    report = []

    # Current: PPIACO
    print("  Testing current: PPIACO...")
    ppiaco = fred("PPIACO", "1960-01-01")
    if ppiaco is not None:
        proxy = ppiaco.resample("MS").last().pct_change().dropna()
        m = overlap_metrics(proxy, etf, "PPIACO (current)")
        results["current"] = m
        report.append(f"\nCurrent: PPIACO (PPI All Commodities)")
        report.append(fmt_candidate(m, ppiaco.index.min().strftime("%Y-%m")))

    # Candidate 1: PPIENG (PPI Energy)
    print("  Testing: PPIENG...")
    ppieng = fred("PPIENG", "1960-01-01")
    if ppieng is not None and len(ppieng) > 100:
        proxy = ppieng.resample("MS").last().pct_change().dropna()
        m = overlap_metrics(proxy, etf, "PPIENG")
        results["ppieng"] = m
        report.append(f"Candidate 1: PPIENG (PPI Energy, FRED)")
        report.append(f"  Available: YES — {ppieng.index.min().date()} to {ppieng.index.max().date()}")
        report.append(fmt_candidate(m, ppieng.index.min().strftime("%Y-%m")))
    else:
        report.append(f"Candidate 1: PPIENG — NOT AVAILABLE\n")

    # Candidate 2: Oil-heavy composite (60% WTISPLC + 20% HG=F + 20% GOLDAMGBD228NLBM)
    # Approximates DBC weighting (~55% energy, ~15% industrial metals, ~10% precious)
    print("  Testing: Oil-weighted composite...")
    oil = fred("WTISPLC", "1960-01-01")
    copper_f = yf_daily("HG=F", "2000-01-01")
    gold_f = fred("GOLDAMGBD228NLBM", "1968-04-01")

    if oil is not None:
        oil_ret = oil.resample("MS").last().pct_change().dropna()
        # Start with just oil if metals not available
        composite_parts = {"oil": (oil_ret, 0.55)}

        if copper_f is not None:
            cu_ret = copper_f.resample("MS").last().pct_change().dropna()
            cu_ret.index = cu_ret.index.to_period("M").to_timestamp()
            composite_parts["copper"] = (cu_ret, 0.15)

        if gold_f is not None:
            au_ret = gold_f.resample("MS").last().pct_change().dropna()
            composite_parts["gold"] = (au_ret, 0.10)

        # Remaining weight to agriculture proxy — use PPIACO stripped of energy
        ppi_farm = fred("PPIFCF", "1960-01-01")  # PPI Farm Products
        if ppi_farm is not None:
            farm_ret = ppi_farm.resample("MS").last().pct_change().dropna()
            composite_parts["farm"] = (farm_ret, 0.20)

        # Build composite on common dates
        all_rets = pd.DataFrame({k: v[0] for k, v in composite_parts.items()})
        weights = {k: v[1] for k, v in composite_parts.items()}
        # Normalize weights
        total_w = sum(weights.values())
        composite = sum(all_rets[k] * (w / total_w) for k, w in weights.items() if k in all_rets.columns)
        composite = composite.dropna()

        m = overlap_metrics(composite, etf, "Oil-weighted composite")
        results["composite"] = m
        wt_str = ", ".join(f"{k}={w/total_w:.0%}" for k, w in weights.items())
        report.append(f"Candidate 2: Oil-weighted composite ({wt_str})")
        hist_start = composite.index.min().strftime("%Y-%m")
        report.append(f"  Available: YES — {hist_start} to present")
        report.append(fmt_candidate(m, hist_start))

    # Candidate 3: CL=F (crude oil futures alone — DBC is ~55% energy)
    print("  Testing: CL=F...")
    cl = yf_daily("CL=F", "2000-01-01")
    if cl is not None:
        proxy = cl.resample("MS").last().pct_change().dropna()
        proxy.index = proxy.index.to_period("M").to_timestamp()
        m = overlap_metrics(proxy, etf, "CL=F")
        results["cl_f"] = m
        report.append(f"Candidate 3: CL=F (Crude Oil Futures, yfinance)")
        report.append(f"  Available: YES — {cl.index.min().date()} to {cl.index.max().date()}")
        report.append(fmt_candidate(m, cl.index.min().strftime("%Y-%m")))
    else:
        report.append(f"Candidate 3: CL=F — NOT AVAILABLE\n")

    # Candidate 4: WTISPLC alone (FRED oil, longer history)
    print("  Testing: WTISPLC alone...")
    if oil is not None:
        proxy = oil.resample("MS").last().pct_change().dropna()
        m = overlap_metrics(proxy, etf, "WTISPLC")
        results["wtisplc"] = m
        report.append(f"Candidate 4: WTISPLC (WTI Spot, FRED)")
        report.append(f"  Available: YES — {oil.index.min().date()} to {oil.index.max().date()}")
        report.append(fmt_candidate(m, oil.index.min().strftime("%Y-%m")))

    # Candidate 5: PPIFCF (Farm Products) — test whether ag sector alone is better
    print("  Testing: PPIFCF...")
    if ppi_farm is not None and len(ppi_farm) > 100:
        proxy = ppi_farm.resample("MS").last().pct_change().dropna()
        m = overlap_metrics(proxy, etf, "PPIFCF")
        results["ppifcf"] = m
        report.append(f"Candidate 5: PPIFCF (PPI Farm Products, FRED)")
        report.append(f"  Available: YES — {ppi_farm.index.min().date()} to {ppi_farm.index.max().date()}")
        report.append(fmt_candidate(m, ppi_farm.index.min().strftime("%Y-%m")))
    else:
        report.append(f"Candidate 5: PPIFCF — NOT AVAILABLE\n")

    # Candidate 6: DCOILWTICO (daily WTI) → monthly
    print("  Testing: DCOILWTICO (daily)...")
    doil = fred("DCOILWTICO", "1986-01-01")
    if doil is not None and len(doil) > 200:
        proxy = doil.resample("MS").last().pct_change().dropna()
        m = overlap_metrics(proxy, etf, "DCOILWTICO")
        results["dcoilwtico"] = m
        report.append(f"Candidate 6: DCOILWTICO (Daily WTI, FRED)")
        report.append(f"  Available: YES — {doil.index.min().date()} to {doil.index.max().date()}")
        report.append(fmt_candidate(m, doil.index.min().strftime("%Y-%m")))
    else:
        report.append(f"Candidate 6: DCOILWTICO — NOT AVAILABLE\n")

    return results, "\n".join(report)


# ── Comparison plot ───────────────────────────────────────────────────────────

def plot_comparison(all_results, outpath):
    """Bar chart comparing all candidates per asset."""
    assets = ["IVV", "VGLT", "IAU", "DBC"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))

    for i, asset in enumerate(assets):
        ax = axes[i]
        res = all_results.get(asset, {})
        if not res:
            ax.set_title(f"{asset}\n(no data)")
            continue

        names = []
        corrs = []
        colors = []
        for key, m in res.items():
            if m is None:
                continue
            label = m["name"]
            if len(label) > 20:
                label = label[:18] + "…"
            names.append(label)
            corrs.append(m["corr"])
            overall, _ = rate(m["corr"], m["ann_diff"], m["te_ann"])
            colors.append({"Good": "#2ecc71", "Acceptable": "#f39c12", "Poor": "#e74c3c"}[overall])

        if not names:
            continue

        y_pos = np.arange(len(names))
        ax.barh(y_pos, corrs, color=colors, edgecolor="white", height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("Correlation with ETF")
        ax.set_title(f"{asset}", fontsize=12, fontweight="bold")
        ax.axvline(0.95, color="green", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.axvline(0.85, color="orange", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.set_xlim(-0.2, 1.1)

        # Annotate values
        for j, (c, n) in enumerate(zip(corrs, names)):
            ax.text(max(c + 0.02, 0.02), j, f"{c:.3f}", va="center", fontsize=7)

    fig.suptitle("Proxy Candidate Comparison — Correlation with ETF", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {outpath}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  PROXY REPLACEMENT SEARCH")
    print("=" * 80)

    full_report = []
    all_results = {}

    # IVV
    print("\n" + "─" * 40)
    print("IVV — S&P 500")
    print("─" * 40)
    ivv_res, ivv_report = search_ivv()
    all_results["IVV"] = ivv_res or {}
    full_report.append("\n" + "=" * 80)
    full_report.append("IVV — S&P 500")
    full_report.append("=" * 80)
    full_report.append(ivv_report)

    # VGLT
    print("\n" + "─" * 40)
    print("VGLT — Long-Term Treasuries")
    print("─" * 40)
    vglt_res, vglt_report = search_vglt()
    all_results["VGLT"] = vglt_res or {}
    full_report.append("\n" + "=" * 80)
    full_report.append("VGLT — Long-Term Treasuries")
    full_report.append("=" * 80)
    full_report.append(vglt_report)

    # IAU
    print("\n" + "─" * 40)
    print("IAU — Gold")
    print("─" * 40)
    iau_res, iau_report = search_iau()
    all_results["IAU"] = iau_res or {}
    full_report.append("\n" + "=" * 80)
    full_report.append("IAU — Gold")
    full_report.append("=" * 80)
    full_report.append(iau_report)

    # DBC
    print("\n" + "─" * 40)
    print("DBC — Commodities")
    print("─" * 40)
    dbc_res, dbc_report = search_dbc()
    all_results["DBC"] = dbc_res or {}
    full_report.append("\n" + "=" * 80)
    full_report.append("DBC — Commodities")
    full_report.append("=" * 80)
    full_report.append(dbc_report)

    # ── Pick winners ──────────────────────────────────────────────────────────
    full_report.append("\n\n" + "=" * 80)
    full_report.append("  FINAL RECOMMENDATIONS")
    full_report.append("=" * 80)

    recs = {}
    for asset in ["IVV", "VGLT", "IAU", "DBC"]:
        res = all_results.get(asset, {})
        if not res:
            continue
        # Find best non-current candidate by correlation
        current = res.get("current")
        best_key, best_m = None, None
        for k, m in res.items():
            if k == "current" or m is None:
                continue
            if best_m is None or abs(m["corr"]) > abs(best_m["corr"]):
                best_key, best_m = k, m
        recs[asset] = (current, best_key, best_m)

    full_report.append("")
    full_report.append(f"{'Asset':<8} {'Current Proxy':>30} {'Corr':>6} {'New Proxy':>35} {'Corr':>6} {'Improvement':>12}")
    full_report.append(f"{'-'*8} {'-'*30} {'-'*6} {'-'*35} {'-'*6} {'-'*12}")

    for asset in ["IVV", "VGLT", "IAU", "DBC"]:
        current, best_key, best_m = recs.get(asset, (None, None, None))
        c_corr = current["corr"] if current else float("nan")
        c_name = current["name"] if current else "N/A"
        if len(c_name) > 28: c_name = c_name[:26] + "…"
        if best_m:
            b_name = best_m["name"]
            if len(b_name) > 33: b_name = b_name[:31] + "…"
            b_corr = best_m["corr"]
            imp = b_corr - c_corr
            full_report.append(f"{asset:<8} {c_name:>30} {c_corr:>6.3f} {b_name:>35} {b_corr:>6.3f} {'+' if imp > 0 else ''}{imp:>.3f}")
        else:
            full_report.append(f"{asset:<8} {c_name:>30} {c_corr:>6.3f} {'NO IMPROVEMENT FOUND':>35}")

    # History coverage
    full_report.append("")
    full_report.append("\nHistory coverage with recommended proxies:")
    history_starts = {}
    for asset in ["IVV", "VGLT", "IAU", "DBC"]:
        current, best_key, best_m = recs.get(asset, (None, None, None))
        if best_m:
            # Estimate full history start from proxy series
            full_report.append(f"  {asset}: {best_m['name']} — overlap tested {best_m['start']} to {best_m['end']}")

    # Per-asset recommendation details
    full_report.append("")
    for asset in ["IVV", "VGLT", "IAU", "DBC"]:
        current, best_key, best_m = recs.get(asset, (None, None, None))
        if best_m is None:
            continue
        c_corr = current["corr"] if current else 0
        overall, _ = rate(best_m["corr"], best_m["ann_diff"], best_m["te_ann"])
        full_report.append(f"\n  {asset}: REPLACE with {best_m['name']}")
        full_report.append(f"    Correlation: {c_corr:.3f} → {best_m['corr']:.3f}")
        full_report.append(f"    Tracking error: {best_m['te_ann']*100:.1f}% ann.")
        full_report.append(f"    Rating: {overall}")
        if overall == "Poor":
            full_report.append(f"    ⚠ Still rated Poor — consider paid data (Bloomberg, GSCI) or shorter lookback")

    # Print and save
    report_text = "\n".join(full_report)
    print("\n" + report_text)

    report_path = os.path.join(OUTDIR, "proxy_replacement_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\n  Report saved: {report_path}")

    # Generate comparison plot
    plot_comparison(all_results, os.path.join(OUTDIR, "proxy_candidate_comparison.png"))

    print("\n✓ Proxy replacement search complete.")


if __name__ == "__main__":
    main()

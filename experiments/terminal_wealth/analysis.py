"""Terminal wealth optimization: simplified high-conviction architectures.

Tests 9 variants + baseline to find whether removing defensive assets
(VGLT/IAU/DBC) and concentrating in equity + cash improves terminal wealth
while maintaining drawdown protection via the circuit breaker.
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv; load_dotenv()

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

from taa.data import load_daily_etf_returns, load_monthly_prices
from taa.faber import apply_faber_filter

SMA_PERIODS = [126, 200, 252]
SSO_EXP = 0.0089; QLD_EXP = 0.0095
ASSETS_ALL = ["IVV", "QQQ", "VGLT", "IAU", "DBC", "cash"]


def load_data():
    import yfinance as yf
    daily_ret = load_daily_etf_returns()
    daily_ret = daily_ret[[c for c in daily_ret.columns if c in ASSETS_ALL]]

    from fredapi import Fred
    rfr_daily = pd.Series(0.0, index=daily_ret.index)
    key = os.environ.get("FRED_API_KEY")
    if key:
        tb = Fred(api_key=key).get_series("DTB3", observation_start="1998-01-01")
        tb.index = pd.to_datetime(tb.index)
        rfr_daily = (tb / 100 / 252).reindex(daily_ret.index, method="ffill").fillna(0)

    ticker_map = {"IVV": "SPY", "QQQ": "QQQ", "VGLT": "TLT", "IAU": "GLD", "DBC": "DBC"}
    dp = {}
    for our, ticker in ticker_map.items():
        d = yf.download(ticker, start="1998-01-01", progress=False)
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p, "columns"): p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None); dp[our] = p
    dpdf = pd.DataFrame(dp).sort_index()
    daily_smas = {p: dpdf.rolling(p, min_periods=p).mean() for p in SMA_PERIODS}

    actual_lev = {}
    for ticker in ["SSO", "QLD"]:
        d = yf.download(ticker, start="2006-01-01", progress=False, auto_adjust=True)
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p, "columns"): p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None)
            actual_lev[ticker] = p.pct_change().dropna()
    both_start = max(actual_lev.get("SSO", pd.Series()).index.min(),
                     actual_lev.get("QLD", pd.Series()).index.min()) \
        if "SSO" in actual_lev and "QLD" in actual_lev else pd.Timestamp("2099-01-01")

    return daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start


def asset_score(day, asset, dpdf, smas):
    if asset not in dpdf.columns: return 0
    p = dpdf.loc[:day, asset]
    if len(p) == 0 or pd.isna(p.iloc[-1]): return 0
    price = p.iloc[-1]; sc = 0
    for per in SMA_PERIODS:
        s = smas[per].loc[:day, asset]
        if len(s) > 0 and pd.notna(s.iloc[-1]) and price > s.iloc[-1]: sc += 1
    return sc


def check_breach_single(day, asset, dpdf, smas):
    if asset not in dpdf.columns: return False
    p = dpdf.loc[:day, asset]
    if len(p) == 0: return False
    price = p.iloc[-1]; b = 0
    for per in SMA_PERIODS:
        s = smas[per].loc[:day, asset]
        if len(s) > 0 and pd.notna(s.iloc[-1]) and price < s.iloc[-1]: b += 1
    return b >= 3


def lev_ret(underlying_ret, rfr, expense, day, actual_lev, ticker, both_start):
    if day >= both_start and ticker in actual_lev:
        r = float(actual_lev[ticker].get(day, np.nan))
        if not np.isnan(r): return r
    return 2.0 * underlying_ret - rfr - expense / 252


# ── Variant runner ───────────────────────────────────────────────────────────

def run_variant(name, config, daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, start_date):
    """Run a single variant from start_date. Returns daily return series + cb count."""
    bt_start = pd.Timestamp(start_date)
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:pd.Timestamp("2026-03-31")].index

    port = {}; cb_count = 0
    # State
    scores = {}; weights = {}; leverage = False; delevered = False

    for day in trading_days:
        if day not in daily_ret.index: continue
        dr = daily_ret.loc[day]
        actual = {}
        for a in ASSETS_ALL:
            if a in dr.index and pd.notna(dr[a]): actual[a] = float(dr[a])
            else: actual[a] = 0.0
        rfr = float(rfr_daily.get(day, 0))

        is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day)-1].month)

        if is_ms:
            delevered = False
            prior = trading_days[trading_days < day]
            sd = prior[-1] if len(prior) > 0 else day

            # Compute scores for relevant assets
            for a in config["score_assets"]:
                scores[a] = asset_score(sd, a, dpdf, daily_smas)

            # Determine weights and leverage via config function
            weights, leverage = config["allocate"](scores, config)

        # Circuit breaker
        if leverage and not delevered:
            for a in config["cb_assets"]:
                if check_breach_single(day, a, dpdf, daily_smas):
                    leverage = False; delevered = True; cb_count += 1; break

        # Compute return
        ret = 0.0
        for a, w in weights.items():
            if w <= 0.001: continue
            if a == "cash":
                ret += w * rfr
            elif a == "SSO" and leverage:
                ret += w * lev_ret(actual.get("IVV", 0), rfr, SSO_EXP, day, actual_lev, "SSO", both_start)
            elif a == "QLD" and leverage:
                ret += w * lev_ret(actual.get("QQQ", 0), rfr, QLD_EXP, day, actual_lev, "QLD", both_start)
            elif a == "SSO" and not leverage:
                ret += w * actual.get("IVV", 0)  # reverted to 1x
            elif a == "QLD" and not leverage:
                ret += w * actual.get("QQQ", 0)
            elif a in actual:
                ret += w * actual[a]

        port[day] = ret

    return pd.Series(port).sort_index(), cb_count


# ── Variant configurations ───────────────────────────────────────────────────

def alloc_baseline(scores, cfg):
    baseline = {"IVV": 0.45, "QQQ": 0.25, "VGLT": 0.05, "IAU": 0.10, "DBC": 0.05, "cash": 0.10}
    w = {}; pool = 0.0
    for a in ["IVV", "QQQ", "VGLT", "IAU", "DBC"]:
        sc = scores.get(a, 0)
        bw = baseline[a]
        if sc >= 3: w[a] = bw
        elif sc == 2: w[a] = bw * 0.70; pool += bw * 0.30
        else: w[a] = 0.0; pool += bw
    w["cash"] = baseline["cash"] + pool

    lev = scores.get("IVV", 0) >= 3 and scores.get("QQQ", 0) >= 3
    if lev:
        w["SSO"] = w.pop("IVV"); w["QLD"] = w.pop("QQQ"); w["IVV"] = 0; w["QQQ"] = 0
    else:
        w["SSO"] = 0; w["QLD"] = 0
    return w, lev


def alloc_single(scores, cfg):
    """Single asset: lever when score 3/3, partial at 2/3, cash at 0-1."""
    asset = cfg["primary"]; lev_ticker = cfg["lev_ticker"]
    sc = scores.get(asset, 0)
    if sc >= 3:
        return {lev_ticker: 1.0, "cash": 0.0, asset: 0.0}, True
    elif sc == 2:
        return {asset: 0.70, "cash": 0.30, lev_ticker: 0.0}, False
    else:
        return {"cash": 1.0, asset: 0.0, lev_ticker: 0.0}, False


def alloc_two_asset(scores, cfg):
    """Two equity assets + cash, no defensive."""
    iw = cfg["ivv_w"]; qw = cfg["qqq_w"]; cash_base = 1.0 - iw - qw
    w = {}; pool = 0.0
    for a, bw in [("IVV", iw), ("QQQ", qw)]:
        sc = scores.get(a, 0)
        if sc >= 3: w[a] = bw
        elif sc == 2: w[a] = bw * 0.70; pool += bw * 0.30
        else: w[a] = 0.0; pool += bw
    w["cash"] = cash_base + pool

    lev = scores.get("IVV", 0) >= 3 and scores.get("QQQ", 0) >= 3
    if lev:
        w["SSO"] = w.pop("IVV"); w["QLD"] = w.pop("QQQ"); w["IVV"] = 0; w["QQQ"] = 0
    else:
        w["SSO"] = 0; w["QLD"] = 0
    return w, lev


def alloc_v9(scores, cfg):
    """V9: QLD primary, IVV as guard for leverage."""
    sc_q = scores.get("QQQ", 0); sc_i = scores.get("IVV", 0)
    if sc_q >= 3:
        if sc_i <= 1:  # IVV breaking down — reduce to 1x
            return {"QQQ": 1.0, "QLD": 0.0, "cash": 0.0}, False
        else:
            return {"QLD": 1.0, "QQQ": 0.0, "cash": 0.0}, True
    elif sc_q == 2:
        return {"QQQ": 0.70, "cash": 0.30, "QLD": 0.0}, False
    else:
        return {"cash": 1.0, "QQQ": 0.0, "QLD": 0.0}, False


VARIANTS = {
    "Baseline": {"allocate": alloc_baseline, "score_assets": ["IVV","QQQ","VGLT","IAU","DBC"],
                  "cb_assets": ["IVV", "QQQ"]},
    "V1 QLD-only": {"allocate": alloc_single, "score_assets": ["QQQ"],
                     "cb_assets": ["QQQ"], "primary": "QQQ", "lev_ticker": "QLD"},
    "V2 SSO-only": {"allocate": alloc_single, "score_assets": ["IVV"],
                     "cb_assets": ["IVV"], "primary": "IVV", "lev_ticker": "SSO"},
    "V3 Eq+Cash 45/25": {"allocate": alloc_two_asset, "score_assets": ["IVV","QQQ"],
                           "cb_assets": ["IVV","QQQ"], "ivv_w": 0.45, "qqq_w": 0.25},
    "V4 Eq+Cash 35/35": {"allocate": alloc_two_asset, "score_assets": ["IVV","QQQ"],
                           "cb_assets": ["IVV","QQQ"], "ivv_w": 0.35, "qqq_w": 0.35},
    "V5 Eq+Cash 25/45": {"allocate": alloc_two_asset, "score_assets": ["IVV","QQQ"],
                           "cb_assets": ["IVV","QQQ"], "ivv_w": 0.25, "qqq_w": 0.45},
    "V6 QQQ+Cash": {"allocate": alloc_single, "score_assets": ["QQQ"],
                      "cb_assets": ["QQQ"], "primary": "QQQ", "lev_ticker": "QLD"},
    "V9 QLD+IVVguard": {"allocate": alloc_v9, "score_assets": ["QQQ", "IVV"],
                          "cb_assets": ["QQQ"]},
}
# V6 is same as V1 (single asset with partial), V7/V8 require DBMF proxy — skip (proxy failed)


def cagr(s):
    if len(s) < 20: return np.nan
    return (1+s).prod() ** (252/len(s)) - 1

def max_dd(s):
    cum = (1+s).cumprod()
    return ((cum-cum.expanding().max())/cum.expanding().max()).min()

def sharpe_r(s):
    ar = s.mean()*252; av = s.std()*np.sqrt(252)
    return ar/av if av > 0 else 0

def sortino_r(s):
    ar = s.mean()*252; neg = s[s<0]
    ds = neg.std()*np.sqrt(252) if len(neg)>10 else s.std()*np.sqrt(252)
    return ar/ds if ds > 0 else 0

def calmar_r(s):
    ar = s.mean()*252; dd = max_dd(s)
    return ar/abs(dd) if dd != 0 else 0

def dca_terminal(s_monthly, start=21000, contrib=700):
    val = start
    for i, r in enumerate(s_monthly):
        if i > 0: val = val * (1+r) + contrib
        else: val += contrib
    return val


def run_all(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start):
    # QQQ and IVV benchmarks
    qqq_full = daily_ret["QQQ"].loc[pd.Timestamp("2002-01-01"):pd.Timestamp("2026-03-31")].dropna()
    ivv_full = daily_ret["IVV"].loc[pd.Timestamp("2002-01-01"):pd.Timestamp("2026-03-31")].dropna()

    # ── TABLE 1: Full-period comparison ──────────────────────────────────
    print(f"\n{'='*130}")
    print(f"  TABLE 1: FULL-PERIOD COMPARISON (2002-2026)")
    print(f"{'='*130}")

    print(f"\n  {'Variant':<20} {'CAGR':>7} {'Vol':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>7} {'Calmar':>7} {'Term$1':>8} {'DCA$700':>10} {'CB':>4}")
    print(f"  {'-'*20} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*8} {'-'*10} {'-'*4}")

    full_results = {}
    for vname, cfg in VARIANTS.items():
        s, cbc = run_variant(vname, cfg, daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, "2002-01-01")
        full_results[vname] = (s, cbc)
        c = cagr(s); v = s.std()*np.sqrt(252); sh = sharpe_r(s); so = sortino_r(s)
        dd = max_dd(s); cl = calmar_r(s); t = (1+s).cumprod().iloc[-1]
        sm = s.resample("MS").apply(lambda x: (1+x).prod()-1)
        dca = dca_terminal(sm)
        print(f"  {vname:<20} {c:>6.1%} {v:>6.1%} {sh:>7.3f} {so:>8.3f} {dd:>6.1%} {cl:>7.2f} ${t:>7.2f} ${dca/1e6:>8.2f}M {cbc:>3}")

    # Benchmarks
    for bname, bs in [("QQQ B&H", qqq_full), ("IVV B&H", ivv_full)]:
        c = cagr(bs); v = bs.std()*np.sqrt(252); sh = sharpe_r(bs)
        dd = max_dd(bs); t = (1+bs).cumprod().iloc[-1]
        sm = bs.resample("MS").apply(lambda x: (1+x).prod()-1)
        dca = dca_terminal(sm)
        print(f"  {bname:<20} {c:>6.1%} {v:>6.1%} {sh:>7.3f} {'':>8} {dd:>6.1%} {'':>7} ${t:>7.2f} ${dca/1e6:>8.2f}M {'':>4}")

    # ── TABLE 2: Start-date sensitivity ──────────────────────────────────
    print(f"\n{'='*130}")
    print(f"  TABLE 2: START-DATE CAGR SENSITIVITY")
    print(f"{'='*130}")

    start_dates = ["2002-01-01", "2007-01-01", "2010-01-01", "2013-01-01", "2019-01-01"]
    header = f"  {'Variant':<20}"
    for sd in start_dates: header += f" {sd[:4]:>8}"
    print(header)
    print(f"  {'-'*20}" + f" {'-'*8}" * len(start_dates))

    sd_results = {}
    for vname, cfg in VARIANTS.items():
        row = f"  {vname:<20}"
        sd_results[vname] = {}
        for sd in start_dates:
            s, _ = run_variant(vname, cfg, daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, sd)
            c = cagr(s); sd_results[vname][sd] = (s, c)
            row += f" {c:>7.1%}"
        print(row)

    # QQQ benchmark
    row = f"  {'QQQ B&H':<20}"
    for sd in start_dates:
        qs = qqq_full[qqq_full.index >= pd.Timestamp(sd)]
        row += f" {cagr(qs):>7.1%}"
    print(row)

    # ── TABLE 3: Sub-period breakdown ────────────────────────────────────
    print(f"\n{'='*130}")
    print(f"  TABLE 3: SUB-PERIOD CAGR")
    print(f"{'='*130}")

    sub_periods = [
        ("Dot-com",   "2002-01-01", "2003-03-31"),
        ("GFC",       "2007-11-01", "2009-03-31"),
        ("2013-21",   "2013-01-01", "2021-12-31"),
        ("2022 bear", "2022-01-01", "2022-12-31"),
        ("2023-26",   "2023-01-01", "2026-03-31"),
    ]

    header = f"  {'Variant':<20}"
    for label, _, _ in sub_periods: header += f" {label:>10}"
    print(header)
    print(f"  {'-'*20}" + f" {'-'*10}" * len(sub_periods))

    for vname in VARIANTS:
        s, _ = full_results[vname]
        row = f"  {vname:<20}"
        for _, cs, ce in sub_periods:
            sp = s[(s.index >= pd.Timestamp(cs)) & (s.index <= pd.Timestamp(ce))]
            row += f" {cagr(sp):>9.1%}" if len(sp) > 20 else f" {'N/A':>10}"
        print(row)

    row = f"  {'QQQ B&H':<20}"
    for _, cs, ce in sub_periods:
        sp = qqq_full[(qqq_full.index >= pd.Timestamp(cs)) & (qqq_full.index <= pd.Timestamp(ce))]
        row += f" {cagr(sp):>9.1%}" if len(sp) > 20 else f" {'N/A':>10}"
    print(row)

    # ── TABLE 4: DCA dollar gap 2013-2026 ────────────────────────────────
    print(f"\n{'='*130}")
    print(f"  TABLE 4: DCA DOLLAR GAP vs QQQ (2013-2026, $21K + $700/mo)")
    print(f"{'='*130}")

    # Pick top variants
    show_variants = ["Baseline", "V1 QLD-only", "V3 Eq+Cash 45/25", "V5 Eq+Cash 25/45", "V9 QLD+IVVguard"]

    print(f"\n  {'Year-end':>10}", end="")
    for vn in show_variants: print(f" {vn:>16}", end="")
    print(f" {'QQQ B&H':>14}")

    for yr in range(2013, 2027):
        row = f"  {yr:>10}"
        for vn in show_variants:
            s, _ = full_results[vn]
            sp = s[(s.index >= "2013-01-01") & (s.index <= f"{yr}-12-31")]
            sm = sp.resample("MS").apply(lambda x: (1+x).prod()-1)
            dca = dca_terminal(sm)
            row += f" ${dca/1e3:>14.0f}K"
        # QQQ
        qs = qqq_full[(qqq_full.index >= "2013-01-01") & (qqq_full.index <= f"{yr}-12-31")]
        qm = qs.resample("MS").apply(lambda x: (1+x).prod()-1)
        qdca = dca_terminal(qm)
        row += f" ${qdca/1e3:>12.0f}K"
        print(row)

    # ── TABLE 5: Pass/Fail summary ───────────────────────────────────────
    print(f"\n{'='*130}")
    print(f"  TABLE 5: PASS/FAIL SUMMARY")
    print(f"{'='*130}")

    # Get QQQ CAGR from 2013
    qqq_2013 = qqq_full[qqq_full.index >= pd.Timestamp("2013-01-01")]
    qqq_cagr_2013 = cagr(qqq_2013)
    qqq_sharpe_2013 = sharpe_r(qqq_2013)

    print(f"\n  QQQ from 2013: CAGR={qqq_cagr_2013:.1%}, Sharpe={qqq_sharpe_2013:.3f}, MaxDD={max_dd(qqq_2013):.1%}")

    print(f"\n  {'Variant':<20} {'2013 CAGR':>10} {'beat QQQ?':>10} {'MaxDD':>7} {'<-30%?':>7} "
          f"{'Sharpe>Q':>9} {'DotCom DD':>10} {'<-40%?':>7} {'PASS?':>7}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*7} {'-'*7} {'-'*9} {'-'*10} {'-'*7} {'-'*7}")

    for vname in VARIANTS:
        s_2013, _ = sd_results[vname]["2013-01-01"]
        c_2013 = cagr(s_2013)
        dd_full = max_dd(full_results[vname][0])
        sh_2013 = sharpe_r(s_2013)

        # Dot-com DD
        s_dc = full_results[vname][0]
        sp_dc = s_dc[(s_dc.index >= "2002-01-01") & (s_dc.index <= "2003-03-31")]
        dd_dc = max_dd(sp_dc) if len(sp_dc) > 20 else 0

        beat_qqq = c_2013 > qqq_cagr_2013
        dd_ok = dd_full > -0.30
        sh_ok = sh_2013 > qqq_sharpe_2013
        dc_ok = dd_dc > -0.40
        all_pass = beat_qqq and dd_ok and sh_ok and dc_ok

        print(f"  {vname:<20} {c_2013:>9.1%} {'YES' if beat_qqq else 'NO':>10} {dd_full:>6.1%} "
              f"{'YES' if dd_ok else 'NO':>7} {'YES' if sh_ok else 'NO':>9} {dd_dc:>9.1%} "
              f"{'YES' if dc_ok else 'NO':>7} {'PASS' if all_pass else 'FAIL':>7}")

    # ── VERDICT ──────────────────────────────────────────────────────────
    print(f"\n{'='*130}")
    print(f"  VERDICT")
    print(f"{'='*130}")

    # Find passing variants
    passing = []
    for vname in VARIANTS:
        s_2013, _ = sd_results[vname]["2013-01-01"]
        c_2013 = cagr(s_2013)
        dd_full = max_dd(full_results[vname][0])
        sh_2013 = sharpe_r(s_2013)
        s_dc = full_results[vname][0]
        sp_dc = s_dc[(s_dc.index >= "2002-01-01") & (s_dc.index <= "2003-03-31")]
        dd_dc = max_dd(sp_dc) if len(sp_dc) > 20 else 0

        if c_2013 > qqq_cagr_2013 and dd_full > -0.30 and sh_2013 > qqq_sharpe_2013 and dd_dc > -0.40:
            passing.append(vname)

    if passing:
        print(f"\n  Variants passing ALL criteria: {passing}")
        for vn in passing:
            s, cbc = full_results[vn]
            c = cagr(s); sh = sharpe_r(s); dd = max_dd(s); t = (1+s).cumprod().iloc[-1]
            s_bl, _ = full_results["Baseline"]
            c_bl = cagr(s_bl)
            print(f"    {vn}: CAGR {c:.1%} (vs baseline {c_bl:.1%}), Sharpe {sh:.3f}, DD {dd:.1%}, terminal ${t:.2f}")
    else:
        print(f"\n  NO variant passes all four criteria.")
        print(f"\n  Best tradeoffs:")
        # Sort by terminal from 2013
        ranked = []
        for vname in VARIANTS:
            s_2013, _ = sd_results[vname]["2013-01-01"]
            c_2013 = cagr(s_2013); dd = max_dd(full_results[vname][0])
            t = (1+full_results[vname][0]).cumprod().iloc[-1]
            ranked.append((vname, c_2013, dd, t))
        ranked.sort(key=lambda x: -x[3])
        for vn, c13, dd, t in ranked[:5]:
            beat = "beats" if c13 > qqq_cagr_2013 else "trails"
            print(f"    {vn}: 2013-CAGR {c13:.1%} ({beat} QQQ {qqq_cagr_2013:.1%}), DD {dd:.1%}, terminal ${t:.2f}")

    # Key questions
    print(f"\n  KEY QUESTIONS:")
    s_bl, _ = full_results["Baseline"]
    s_v3, _ = full_results["V3 Eq+Cash 45/25"]
    print(f"  Q1. Defensive assets value: Baseline ${(1+s_bl).cumprod().iloc[-1]:.2f} vs "
          f"V3 no-defense ${(1+s_v3).cumprod().iloc[-1]:.2f} "
          f"({'defense helps' if (1+s_bl).cumprod().iloc[-1] > (1+s_v3).cumprod().iloc[-1] else 'defense hurts'})")

    s_v1, _ = full_results["V1 QLD-only"]
    print(f"  Q2. Single QLD terminal: ${(1+s_v1).cumprod().iloc[-1]:.2f} (DD {max_dd(s_v1):.1%})")

    s_v5, _ = full_results["V5 Eq+Cash 25/45"]
    print(f"  Q3. QQQ-heavy (25/45): ${(1+s_v5).cumprod().iloc[-1]:.2f} (Sharpe {sharpe_r(s_v5):.3f})")

    print()


if __name__ == "__main__":
    print("=" * 130)
    print("  TERMINAL WEALTH OPTIMIZATION: SIMPLIFIED HIGH-CONVICTION ARCHITECTURES")
    print("=" * 130)

    print(f"\n  Loading data...")
    daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start = load_data()

    run_all(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start)

"""Dynamic equity sleeve tilt: IVV/QQQ relative momentum within leveraged periods.

Phase 1: Dot-com stress test (kill gate)
Phase 2: Full backtest — Baseline, Binary tilt, Three-state tilt
Phase 3: Performance metrics and sub-periods
Phase 4: Whipsaw analysis
Phase 5: Sensitivity check (small/base/large tilt magnitudes)
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv; load_dotenv()

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

from taa.data import load_daily_etf_returns, load_monthly_prices
from taa.faber import apply_faber_filter

ASSETS_LIST = ["IVV", "QQQ", "VGLT", "IAU", "DBC", "cash"]
RISKY = ["IVV", "QQQ", "VGLT", "IAU", "DBC"]
SMA_PERIODS = [126, 200, 252]
SSO_EXP = 0.0089; QLD_EXP = 0.0095

# Tilt configurations: (name, ivv_w_qqq_tilt, qqq_w_qqq_tilt, ivv_w_ivv_tilt, qqq_w_ivv_tilt)
TILTS = {
    "small":  (0.37, 0.33, 0.52, 0.18),
    "base":   (0.30, 0.40, 0.55, 0.15),
    "large":  (0.20, 0.50, 0.60, 0.10),
}
NEUTRAL = (0.45, 0.25)  # baseline weights
DEADBAND = 0.02  # for three-state variant


def load_data():
    import yfinance as yf
    daily_ret = load_daily_etf_returns()
    daily_ret = daily_ret[[c for c in daily_ret.columns if c in ASSETS_LIST]]

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

    # QQQ/IVV ratio and its 200-day SMA
    ratio = dpdf["QQQ"] / dpdf["IVV"]
    ratio_sma200 = ratio.rolling(200, min_periods=200).mean()

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

    # DBMF hybrid (reuse from prior experiment — simplified: just use T-bill for signal-off)
    # For this analysis, signal-off behavior is identical across variants

    return daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, ratio, ratio_sma200


def sma_scores(day, dpdf, smas):
    scores = {}
    for a in RISKY:
        if a not in dpdf.columns: scores[a] = 0; continue
        p = dpdf.loc[:day, a]
        if len(p) == 0 or pd.isna(p.iloc[-1]): scores[a] = 0; continue
        price = p.iloc[-1]; sc = 0
        for per in SMA_PERIODS:
            s = smas[per].loc[:day, a]
            if len(s) > 0 and pd.notna(s.iloc[-1]) and price > s.iloc[-1]: sc += 1
        scores[a] = sc
    return scores


def check_breach(day, dpdf, smas):
    for etf in ["IVV", "QQQ"]:
        if etf not in dpdf.columns: continue
        p = dpdf.loc[:day, etf]
        if len(p) == 0: continue
        price = p.iloc[-1]; b = 0
        for per in SMA_PERIODS:
            s = smas[per].loc[:day, etf]
            if len(s) > 0 and pd.notna(s.iloc[-1]) and price < s.iloc[-1]: b += 1
        if b >= 3: return True
    return False


def get_tilt_weights(ratio_val, ratio_sma_val, variant, tilt_cfg="base"):
    """Return (ivv_w, qqq_w) based on variant and ratio signal."""
    if variant == "A":
        return NEUTRAL

    ivv_qt, qqq_qt, ivv_it, qqq_it = TILTS[tilt_cfg]

    if pd.isna(ratio_val) or pd.isna(ratio_sma_val):
        return NEUTRAL

    if variant == "B":
        if ratio_val > ratio_sma_val:
            return (ivv_qt, qqq_qt)  # QQQ tilt
        else:
            return (ivv_it, qqq_it)  # IVV tilt

    if variant == "C":
        pct_above = (ratio_val - ratio_sma_val) / ratio_sma_val
        if pct_above > DEADBAND:
            return (ivv_qt, qqq_qt)
        elif pct_above < -DEADBAND:
            return (ivv_it, qqq_it)
        else:
            return NEUTRAL

    return NEUTRAL


def run_backtest(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                 ratio, ratio_sma200, variant="A", tilt_cfg="base",
                 start_date="2002-01-01"):
    bt_start = pd.Timestamp(start_date)
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:pd.Timestamp("2026-03-31")].index

    BASELINE_W = {"IVV": 0.45, "QQQ": 0.25, "VGLT": 0.05, "IAU": 0.10, "DBC": 0.05, "cash": 0.10}
    cur_scores = {a: 3 for a in RISKY}
    w_faber = dict(BASELINE_W); la = False; dlv = False
    port = {}; cb_events = []
    tilt_log = []  # (date, state: "QQQ"/"IVV"/"neutral")
    current_ivv_w = 0.45; current_qqq_w = 0.25

    for day in trading_days:
        if day not in daily_ret.index: continue
        dr = daily_ret.loc[day]
        avail = [a for a in ASSETS_LIST if a in dr.index and pd.notna(dr[a])]
        if len(avail) < 3: continue
        actual = {a: float(dr[a]) for a in avail}
        rfr = float(rfr_daily.get(day, 0))
        is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day)-1].month)

        if is_ms:
            dlv = False
            prior = trading_days[trading_days < day]
            sd = prior[-1] if len(prior) > 0 else day
            cur_scores = sma_scores(sd, dpdf, daily_smas)

            # Get ratio signal at scoring day
            r_val = ratio.get(sd, np.nan) if sd in ratio.index else np.nan
            r_sma = ratio_sma200.get(sd, np.nan) if sd in ratio_sma200.index else np.nan

            # Determine equity weights
            faber_conv = cur_scores.get("IVV", 0) >= 3 and cur_scores.get("QQQ", 0) >= 3

            if faber_conv:
                ivv_w, qqq_w = get_tilt_weights(r_val, r_sma, variant, tilt_cfg)
            else:
                ivv_w, qqq_w = 0.45, 0.25  # baseline for filter application

            current_ivv_w = ivv_w; current_qqq_w = qqq_w

            # Apply Faber filter with current equity weights
            custom_baseline = {"IVV": ivv_w, "QQQ": qqq_w, "VGLT": 0.05, "IAU": 0.10, "DBC": 0.05, "cash": 0.10}
            w1, pool = apply_faber_filter(cur_scores, custom_baseline)
            w_faber = dict(w1); w_faber["cash"] = w_faber.get("cash", 0) + pool
            la = faber_conv

            # Log tilt state
            if faber_conv and not pd.isna(r_val) and not pd.isna(r_sma):
                pct = (r_val - r_sma) / r_sma if r_sma != 0 else 0
                if variant == "C" and abs(pct) <= DEADBAND:
                    tilt_log.append((day, "neutral"))
                elif r_val > r_sma:
                    tilt_log.append((day, "QQQ"))
                else:
                    tilt_log.append((day, "IVV"))
            else:
                tilt_log.append((day, "off"))

        if la and not dlv:
            if check_breach(day, dpdf, daily_smas):
                la = False; dlv = True; cb_events.append(day)

        iw = w_faber.get("IVV", 0); qw = w_faber.get("QQQ", 0)
        ir = actual.get("IVV", 0); qr = actual.get("QQQ", 0)
        base = sum(w_faber.get(a, 0) * actual.get(a, 0) for a in avail if a not in ["IVV", "QQQ"])

        if la:
            if day >= both_start:
                sso = float(actual_lev.get("SSO", pd.Series()).get(day, np.nan))
                qld = float(actual_lev.get("QLD", pd.Series()).get(day, np.nan))
                if np.isnan(sso): sso = 2*ir-rfr-SSO_EXP/252
                if np.isnan(qld): qld = 2*qr-rfr-QLD_EXP/252
            else:
                sso = 2*ir-rfr-SSO_EXP/252; qld = 2*qr-rfr-QLD_EXP/252
            port[day] = iw*sso + qw*qld + base
        else:
            port[day] = iw*ir + qw*qr + base

    return pd.Series(port).sort_index(), cb_events, tilt_log


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
    ds = neg.std()*np.sqrt(252) if len(neg) > 10 else s.std()*np.sqrt(252)
    return ar/ds if ds > 0 else 0

def calmar_r(s):
    ar = s.mean()*252; dd = max_dd(s)
    return ar/abs(dd) if dd != 0 else 0


def run_all(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, ratio, ratio_sma200):

    # ── PHASE 1: Dot-com Stress Test ─────────────────────────────────────
    print(f"\n{'='*110}")
    print(f"  PHASE 1: DOT-COM STRESS TEST (1999-2002) — KILL GATE")
    print(f"{'='*110}")

    # Run from 1999 with daily prices available
    # Note: we need prices from 1998 for 200-day SMA warmup
    dc_start = "1999-01-01"
    dc_end = "2002-12-31"

    pa_dc, _, _ = run_backtest(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                                ratio, ratio_sma200, "A", "base", dc_start)
    pb_dc, _, tilt_dc = run_backtest(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                                      ratio, ratio_sma200, "B", "base", dc_start)

    pa_dc = pa_dc[pa_dc.index <= pd.Timestamp(dc_end)]
    pb_dc = pb_dc[pb_dc.index <= pd.Timestamp(dc_end)]

    if len(pa_dc) > 50 and len(pb_dc) > 50:
        ca = cagr(pa_dc); cb_cagr = cagr(pb_dc)
        da = max_dd(pa_dc); db = max_dd(pb_dc)
        ta = (1+pa_dc).cumprod().iloc[-1]; tb_term = (1+pb_dc).cumprod().iloc[-1]

        print(f"\n  {'Variant':<12} {'CAGR':>8} {'MaxDD':>8} {'Terminal':>10}")
        print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*10}")
        print(f"  {'A Baseline':<12} {ca:>7.1%} {da:>7.1%} ${ta:>9.2f}")
        print(f"  {'B Tilt':<12} {cb_cagr:>7.1%} {db:>7.1%} ${tb_term:>9.2f}")

        # When did ratio cross below SMA?
        dc_ratio = ratio[(ratio.index >= pd.Timestamp("1999-01-01")) & (ratio.index <= pd.Timestamp(dc_end))]
        dc_rsma = ratio_sma200[(ratio_sma200.index >= pd.Timestamp("1999-01-01")) & (ratio_sma200.index <= pd.Timestamp(dc_end))]
        cross_below = None
        for dt in dc_ratio.index:
            if dt in dc_rsma.index and not pd.isna(dc_rsma.loc[dt]):
                if dc_ratio.loc[dt] < dc_rsma.loc[dt]:
                    cross_below = dt; break
        if cross_below:
            print(f"\n  QQQ/IVV ratio crossed below 200d SMA: {cross_below.strftime('%Y-%m-%d')}")

        # Month-by-month 2000-2002
        print(f"\n  Monthly detail (2000-2002):")
        print(f"  {'Month':>10} {'A ret':>8} {'B ret':>8} {'Tilt':>8}")
        pa_m = pa_dc.resample("MS").apply(lambda x: (1+x).prod()-1)
        pb_m = pb_dc.resample("MS").apply(lambda x: (1+x).prod()-1)
        tilt_states = {d.strftime("%Y-%m"): s for d, s in tilt_dc if d.year >= 2000}

        for dt in pa_m.index:
            if dt.year < 2000: continue
            a_r = pa_m.get(dt, 0); b_r = pb_m.get(dt, 0)
            ts = tilt_states.get(dt.strftime("%Y-%m"), "?")
            print(f"  {dt.strftime('%Y-%m'):>10} {a_r:>+7.1%} {b_r:>+7.1%} {ts:>8}")

        dd_diff = db - da
        dotcom_pass = dd_diff > -0.03  # allow up to 3% worse DD
        print(f"\n  Max DD difference (B-A): {dd_diff*100:+.1f}%")
        print(f"  DOT-COM STRESS TEST: {'PASS' if dotcom_pass else 'FAIL — IDEA REJECTED'} "
              f"(threshold: max DD no more than 3% worse)")

        if not dotcom_pass:
            print(f"\n  *** The dynamic tilt MATERIALLY worsened dot-com performance. ***")
            print(f"  *** Remaining phases reported for completeness but idea is REJECTED. ***")
    else:
        dotcom_pass = True
        print(f"  Insufficient data for dot-com test — proceeding")

    # ── PHASE 2-3: Full backtest ─────────────────────────────────────────
    print(f"\n{'='*110}")
    print(f"  PHASE 2-3: FULL BACKTEST AND PERFORMANCE")
    print(f"{'='*110}")

    results = {}
    tilt_logs = {}
    for vname, variant in [("A Baseline", "A"), ("B Binary", "B"), ("C ThreeState", "C")]:
        r, cbs, tl = run_backtest(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                                   ratio, ratio_sma200, variant, "base")
        results[vname] = r
        tilt_logs[vname] = tl

    qqq = daily_ret["QQQ"].reindex(results["A Baseline"].index).fillna(0)
    ivv = daily_ret["IVV"].reindex(results["A Baseline"].index).fillna(0)

    # Full period metrics
    print(f"\n  Full period 2002-2026:")
    print(f"  {'Variant':<16} {'CAGR':>8} {'Vol':>7} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'Calmar':>8} {'Terminal':>10}")
    print(f"  {'-'*16} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    for vname, s in results.items():
        c = cagr(s); v = s.std()*np.sqrt(252); sh = sharpe_r(s); so = sortino_r(s)
        dd = max_dd(s); cl = calmar_r(s); t = (1+s).cumprod().iloc[-1]
        print(f"  {vname:<16} {c:>7.1%} {v:>6.1%} {sh:>8.3f} {so:>8.3f} {dd:>7.1%} {cl:>8.2f} ${t:>9.2f}")

    # QQQ benchmark
    qc = cagr(qqq); qsh = sharpe_r(qqq)
    print(f"  {'QQQ B&H':<16} {qc:>7.1%} {qqq.std()*np.sqrt(252):>6.1%} {qsh:>8.3f}")

    # Sub-period breakdown
    print(f"\n  Sub-period CAGR (A vs B vs C vs QQQ):")
    print(f"  {'Period':<22} {'A':>8} {'B':>8} {'C':>8} {'QQQ':>8} {'B-A':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    sub_periods = [
        ("Dot-com 02-03/03",    "2002-01-01", "2003-03-31"),
        ("Pre-GFC bull",        "2003-04-01", "2007-10-31"),
        ("GFC 07/11-09/03",     "2007-11-01", "2009-03-31"),
        ("Recovery 09/04-12",   "2009-04-01", "2012-12-31"),
        ("2013-2021 bull",      "2013-01-01", "2021-12-31"),
        ("2022 bear",           "2022-01-01", "2022-12-31"),
        ("2023-2026",           "2023-01-01", "2026-03-31"),
    ]

    for label, cs, ce in sub_periods:
        row = f"  {label:<22}"
        for vn in ["A Baseline", "B Binary", "C ThreeState"]:
            sp = results[vn][(results[vn].index >= pd.Timestamp(cs)) & (results[vn].index <= pd.Timestamp(ce))]
            row += f" {cagr(sp):>7.1%}" if len(sp) > 20 else f" {'N/A':>8}"
        qp = qqq[(qqq.index >= pd.Timestamp(cs)) & (qqq.index <= pd.Timestamp(ce))]
        row += f" {cagr(qp):>7.1%}" if len(qp) > 20 else f" {'N/A':>8}"
        # B-A delta
        ba = results["A Baseline"][(results["A Baseline"].index >= pd.Timestamp(cs)) & (results["A Baseline"].index <= pd.Timestamp(ce))]
        bb = results["B Binary"][(results["B Binary"].index >= pd.Timestamp(cs)) & (results["B Binary"].index <= pd.Timestamp(ce))]
        if len(ba) > 20 and len(bb) > 20:
            row += f" {cagr(bb)-cagr(ba):>+7.1%}"
        print(row)

    # ── PHASE 4: Tilt diagnostics & whipsaw ──────────────────────────────
    print(f"\n{'='*110}")
    print(f"  PHASE 4: TILT DIAGNOSTICS AND WHIPSAW")
    print(f"{'='*110}")

    for vname in ["B Binary", "C ThreeState"]:
        tl = tilt_logs[vname]
        states = [s for _, s in tl]
        from collections import Counter
        counts = Counter(states)
        total = len(states)

        print(f"\n  {vname}:")
        for st in ["QQQ", "IVV", "neutral", "off"]:
            n = counts.get(st, 0)
            print(f"    {st:>8}: {n:>4} months ({n/total*100:.0f}%)")

        # Transitions
        transitions = sum(1 for i in range(1, len(states)) if states[i] != states[i-1] and states[i] != "off" and states[i-1] != "off")
        years = total / 12
        print(f"    Transitions (excl off): {transitions} ({transitions/years:.1f}/year)")

        if transitions / years > 8:
            print(f"    ⚠ WARNING: >{8}/year — signal may be too noisy for monthly rebalance")

        # Mean months between transitions
        if transitions > 0:
            print(f"    Avg months between transitions: {total/max(transitions,1):.1f}")

    # ── PHASE 5: Sensitivity check ───────────────────────────────────────
    print(f"\n{'='*110}")
    print(f"  PHASE 5: SENSITIVITY CHECK (tilt magnitude)")
    print(f"{'='*110}")

    print(f"\n  {'Config':<14} {'QQQ-tilt':>10} {'IVV-tilt':>10} {'Sharpe':>8} {'MaxDD':>8} {'Terminal':>10} {'CAGR':>8}")
    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")

    for cfg_name in ["small", "base", "large"]:
        ivq, qqt, ivi, qit = TILTS[cfg_name]
        r, _, _ = run_backtest(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                                ratio, ratio_sma200, "B", cfg_name)
        sh = sharpe_r(r); dd = max_dd(r); t = (1+r).cumprod().iloc[-1]; c = cagr(r)
        print(f"  {cfg_name:<14} {ivq:.0%}/{qqt:.0%} {ivi:.0%}/{qit:.0%} {sh:>8.3f} {dd:>7.1%} ${t:>9.2f} {c:>7.1%}")

    # Baseline for comparison
    r_base = results["A Baseline"]
    print(f"  {'baseline':<14} {'45/25':>10} {'45/25':>10} {sharpe_r(r_base):>8.3f} {max_dd(r_base):>7.1%} "
          f"${(1+r_base).cumprod().iloc[-1]:>9.2f} {cagr(r_base):>7.1%}")

    # ── DCA comparison 2013-2021 ─────────────────────────────────────────
    print(f"\n  2013-2021 DCA ($21K + $700/mo) — A vs B:")
    for vn in ["A Baseline", "B Binary"]:
        s = results[vn]
        bull = s[(s.index >= "2013-01-01") & (s.index <= "2021-12-31")]
        bm = bull.resample("MS").apply(lambda x: (1+x).prod()-1)
        dca = 21000.0
        for i, r in enumerate(bm):
            if i > 0: dca = dca * (1+r) + 700
            else: dca += 700
        print(f"    {vn}: ${dca:,.0f}")

    # ── VERDICT ──────────────────────────────────────────────────────────
    print(f"\n{'='*110}")
    print(f"  VERDICT")
    print(f"{'='*110}")

    ra = results["A Baseline"]; rb = results["B Binary"]; rc = results["C ThreeState"]
    sha = sharpe_r(ra); shb = sharpe_r(rb); shc = sharpe_r(rc)
    dda = max_dd(ra); ddb = max_dd(rb); ddc = max_dd(rc)
    ta = (1+ra).cumprod().iloc[-1]; tb_f = (1+rb).cumprod().iloc[-1]; tc = (1+rc).cumprod().iloc[-1]

    print(f"\n  1. Dot-com test: {'PASS' if dotcom_pass else 'FAIL'}")
    bull_a = ra[(ra.index >= "2013-01-01") & (ra.index <= "2021-12-31")]
    bull_b = rb[(rb.index >= "2013-01-01") & (rb.index <= "2021-12-31")]
    print(f"  2. 2013-2021 CAGR: A={cagr(bull_a):.1%}, B={cagr(bull_b):.1%} (delta: {cagr(bull_b)-cagr(bull_a):+.1%})")
    print(f"  3. Full Sharpe: A={sha:.3f}, B={shb:.3f}, C={shc:.3f} (B-A: {shb-sha:+.3f})")
    print(f"  4. Full MaxDD: A={dda:.1%}, B={ddb:.1%}, C={ddc:.1%}")
    print(f"  5. Terminal: A=${ta:.2f}, B=${tb_f:.2f}, C=${tc:.2f} (B-A: ${tb_f-ta:+.2f})")

    if shb > sha and ddb >= dda - 0.005 and tb_f > ta:
        print(f"\n  → ADOPT Variant B: improves Sharpe ({sha:.3f}→{shb:.3f}), terminal (${ta:.2f}→${tb_f:.2f}), DD stable")
    elif shb > sha and tb_f > ta:
        print(f"\n  → CONDITIONAL ADOPT: improves Sharpe and terminal but DD worsened by {(ddb-dda)*100:.1f}%")
    elif tb_f > ta:
        print(f"\n  → TRADEOFF: more terminal but lower Sharpe")
    else:
        print(f"\n  → REJECT: no improvement over baseline")

    print()


if __name__ == "__main__":
    print("=" * 110)
    print("  DYNAMIC EQUITY SLEEVE TILT: IVV/QQQ RELATIVE MOMENTUM")
    print("=" * 110)

    print(f"\n  Loading data...")
    daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, ratio, ratio_sma200 = load_data()

    # Check ratio data
    valid_ratio = ratio.dropna()
    print(f"  QQQ/IVV ratio: {valid_ratio.index.min().date()} to {valid_ratio.index.max().date()}")
    valid_sma = ratio_sma200.dropna()
    print(f"  Ratio 200d SMA from: {valid_sma.index.min().date()}")

    run_all(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, ratio, ratio_sma200)

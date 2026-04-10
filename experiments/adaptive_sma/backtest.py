"""Adaptive SMA lookbacks based on volatility regime.

HIGH VOL (>75th pct): 63/126/200-day SMAs (faster — trends reverse quickly)
NORMAL (25-75th pct): 126/200/252-day SMAs (baseline)
LOW VOL (<25th pct): 168/252/315-day SMAs (slower — trends persist)

Three strategies:
1. BASELINE: Fixed 126/200/252 + daily circuit breaker (current production)
2. ADAPTIVE: Adaptive lookbacks, circuit breaker uses fixed 126/200/252
3. ADAPTIVE-CB: Adaptive lookbacks + circuit breaker also uses adaptive periods
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv; load_dotenv()

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

from taa.data import load_daily_etf_returns, load_monthly_prices
from taa.faber import compute_trend_scores, apply_faber_filter

BASELINE_W = {"IVV": 0.45, "QQQ": 0.25, "VGLT": 0.05, "IAU": 0.10, "DBC": 0.05, "cash": 0.10}
ASSETS = list(BASELINE_W.keys())
SSO_EXP = 0.0089; QLD_EXP = 0.0095; FABER_SUB = 1.00

# Regime lookback sets
LOOKBACKS = {
    "HIGH": [63, 126, 200],
    "NORMAL": [126, 200, 252],
    "LOW": [168, 252, 315],
}
ALL_PERIODS = sorted(set(p for lbs in LOOKBACKS.values() for p in lbs))


def load_data():
    import yfinance as yf
    daily_ret = load_daily_etf_returns()
    daily_ret = daily_ret[[c for c in daily_ret.columns if c in ASSETS]]

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

    # Precompute ALL needed SMAs
    all_smas = {}
    for period in ALL_PERIODS:
        all_smas[period] = dpdf.rolling(period, min_periods=period).mean()

    # Compute IVV 21-day realized vol and 252-day percentile rank
    ivv_ret = dpdf["IVV"].pct_change().dropna() if "IVV" in dpdf.columns else pd.Series(dtype=float)
    rvol_21d = ivv_ret.rolling(21, min_periods=15).std() * np.sqrt(252)
    vol_pctile = rvol_21d.rolling(252, min_periods=126).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

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

    return daily_ret, dpdf, all_smas, rfr_daily, rvol_21d, vol_pctile, actual_lev, both_start


def classify_regime(vol_pctile_val):
    if pd.isna(vol_pctile_val): return "NORMAL"
    if vol_pctile_val > 0.75: return "HIGH"
    if vol_pctile_val < 0.25: return "LOW"
    return "NORMAL"


def sma_scores_with_periods(day, dpdf, all_smas, periods):
    """Compute Faber scores using specified SMA periods."""
    scores = {}
    for a in ["IVV", "QQQ", "VGLT", "IAU", "DBC"]:
        if a not in dpdf.columns: scores[a] = 0; continue
        p = dpdf.loc[:day, a]
        if len(p) == 0 or pd.isna(p.iloc[-1]): scores[a] = 0; continue
        price = p.iloc[-1]; sc = 0
        for per in periods:
            s = all_smas[per].loc[:day, a]
            if len(s) > 0 and pd.notna(s.iloc[-1]) and price > s.iloc[-1]: sc += 1
        scores[a] = sc
    return scores


def check_breach_with_periods(day, dpdf, all_smas, periods):
    """Circuit breaker using specified SMA periods."""
    for etf in ["IVV", "QQQ"]:
        if etf not in dpdf.columns: continue
        p = dpdf.loc[:day, etf]
        if len(p) == 0: continue
        price = p.iloc[-1]; b = 0
        for per in periods:
            s = all_smas[per].loc[:day, etf]
            if len(s) > 0 and pd.notna(s.iloc[-1]) and price < s.iloc[-1]: b += 1
        if b >= 3: return True
    return False


def lev_return(iw, qw, ir, qr, rfr, day, actual_lev, both_start, leveraged):
    base_non_eq = 0.0  # filled externally
    if not leveraged:
        return iw * ir + qw * qr
    if day >= both_start:
        sso = float(actual_lev.get("SSO", pd.Series()).get(day, 2*ir - rfr - SSO_EXP/252))
        qld = float(actual_lev.get("QLD", pd.Series()).get(day, 2*qr - rfr - QLD_EXP/252))
        if np.isnan(sso): sso = 2*ir - rfr - SSO_EXP/252
        if np.isnan(qld): qld = 2*qr - rfr - QLD_EXP/252
    else:
        sso = 2*ir - rfr - SSO_EXP/252; qld = 2*qr - rfr - QLD_EXP/252
    return iw * sso + qw * qld


def run_backtest(daily_ret, dpdf, all_smas, rfr_daily, vol_pctile, actual_lev, both_start):
    print(f"\n{'='*120}")
    print(f"  BACKTEST")
    print(f"{'='*120}")

    bt_start = pd.Timestamp("2002-01-01")
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:].index
    print(f"  {len(trading_days)} trading days ({common_start.date()} to {trading_days.max().date()})")

    # Strategies: (name, use_adaptive_alloc, use_adaptive_cb)
    strats = {
        "BASELINE":    (False, False),
        "ADAPTIVE":    (True,  False),
        "ADAPTIVE-CB": (True,  True),
    }

    results = {s: {} for s in strats}
    results["IVV B&H"] = {}

    state = {}
    for s in strats:
        state[s] = {
            "wf": dict(BASELINE_W), "la": False, "dlv": False,
            "scores": {a: 3 for a in ASSETS if a != "cash"},
        }

    regime_log = []; regime_counts = {"HIGH": 0, "NORMAL": 0, "LOW": 0}
    signal_agreement = {"total": 0, "agree_all": 0, "agree_leverage": 0}
    disagreement_returns = {"adaptive_better": 0, "baseline_better": 0}
    leveraged_regime = {"HIGH": 0, "NORMAL": 0, "LOW": 0, "total": 0}

    for day in trading_days:
        if day not in daily_ret.index: continue
        dr = daily_ret.loc[day]
        avail = [a for a in ASSETS if a in dr.index and pd.notna(dr[a])]
        if len(avail) < 3: continue
        actual = {a: float(dr[a]) for a in avail}
        rfr = float(rfr_daily.get(day, 0))

        is_ms = (day == trading_days[0] or
                 day.month != trading_days[trading_days.get_loc(day)-1].month)

        # Determine current vol regime
        vp = vol_pctile.get(day, np.nan) if day in vol_pctile.index else np.nan
        regime = classify_regime(vp)
        regime_counts[regime] += 1

        adaptive_periods = LOOKBACKS[regime]
        fixed_periods = LOOKBACKS["NORMAL"]

        if is_ms:
            regime_log.append({"date": day, "regime": regime,
                               "vol_pctile": vp if not pd.isna(vp) else None})

            # PIT: use prior day scores
            prior = trading_days[trading_days < day]
            sd = prior[-1] if len(prior) > 0 else day

            # Determine regime at scoring day
            vp_sd = vol_pctile.get(sd, np.nan) if sd in vol_pctile.index else np.nan
            regime_sd = classify_regime(vp_sd)
            adaptive_sd = LOOKBACKS[regime_sd]

            scores_fixed = sma_scores_with_periods(sd, dpdf, all_smas, fixed_periods)
            scores_adaptive = sma_scores_with_periods(sd, dpdf, all_smas, adaptive_sd)

            # Signal agreement tracking
            signal_agreement["total"] += 1
            all_agree = all(
                (scores_fixed.get(a, 0) >= 3) == (scores_adaptive.get(a, 0) >= 3) and
                (scores_fixed.get(a, 0) == 2) == (scores_adaptive.get(a, 0) == 2)
                for a in ["IVV", "QQQ", "VGLT", "IAU", "DBC"])
            if all_agree: signal_agreement["agree_all"] += 1

            fc_fixed = scores_fixed.get("IVV", 0) >= 3 and scores_fixed.get("QQQ", 0) >= 3
            fc_adaptive = scores_adaptive.get("IVV", 0) >= 3 and scores_adaptive.get("QQQ", 0) >= 3
            if fc_fixed == fc_adaptive: signal_agreement["agree_leverage"] += 1

            # Track leveraged regime
            if fc_fixed:
                leveraged_regime[regime_sd] += 1
                leveraged_regime["total"] += 1

            for s_name, (use_adapt_alloc, use_adapt_cb) in strats.items():
                st = state[s_name]
                st["dlv"] = False
                scores = scores_adaptive if use_adapt_alloc else scores_fixed
                st["scores"] = scores
                w1, pool = apply_faber_filter(scores, BASELINE_W)
                st["wf"] = dict(w1); st["wf"]["cash"] = st["wf"].get("cash", 0) + pool
                fc = scores.get("IVV", 0) >= 3 and scores.get("QQQ", 0) >= 3
                st["la"] = fc

        # Daily circuit breaker
        for s_name, (use_adapt_alloc, use_adapt_cb) in strats.items():
            st = state[s_name]
            if st["la"] and not st["dlv"]:
                cb_periods = adaptive_periods if use_adapt_cb else fixed_periods
                if check_breach_with_periods(day, dpdf, all_smas, cb_periods):
                    st["la"] = False; st["dlv"] = True

        # Compute returns
        iw_base = None
        for s_name in strats:
            st = state[s_name]
            iw = st["wf"].get("IVV", 0); qw = st["wf"].get("QQQ", 0)
            ir = actual.get("IVV", 0); qr = actual.get("QQQ", 0)
            base_ret = sum(st["wf"].get(a, 0) * actual.get(a, 0) for a in avail if a not in ["IVV", "QQQ"])
            eq_ret = lev_return(iw, qw, ir, qr, rfr, day, actual_lev, both_start, st["la"])
            results[s_name][day] = eq_ret + base_ret

        results["IVV B&H"][day] = actual.get("IVV", 0)

    ret = {s: pd.Series(d).sort_index() for s, d in results.items()}
    return ret, regime_counts, regime_log, signal_agreement, leveraged_regime


def metrics(s):
    s = s.dropna()
    if len(s) < 50: return None
    ar = s.mean()*252; av = s.std()*np.sqrt(252)
    sh = ar/av if av > 0 else 0
    neg = s[s < 0]; ds = neg.std()*np.sqrt(252) if len(neg) > 10 else av
    so = ar/ds if ds > 0 else 0
    cum = (1+s).cumprod()
    dd = ((cum-cum.expanding().max())/cum.expanding().max()).min()
    cal = ar/abs(dd) if dd != 0 else 0
    return {"ar": ar, "av": av, "sh": sh, "sortino": so, "dd": dd, "calmar": cal, "final": cum.iloc[-1]}


def report(ret, regime_counts, regime_log, signal_agreement, leveraged_regime, rvol_21d):
    strat_order = ["BASELINE", "ADAPTIVE", "ADAPTIVE-CB", "IVV B&H"]
    perfs = {s: metrics(ret[s]) for s in strat_order}
    p_base = perfs.get("BASELINE", {})

    total_days = sum(regime_counts.values())

    # ── Regime distribution ──────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  VOLATILITY REGIME ANALYSIS")
    print(f"{'='*120}")
    print(f"\n  Regime distribution (2002-2026, {total_days} trading days):")
    for r in ["HIGH", "NORMAL", "LOW"]:
        print(f"    {r} VOL ({'>75th' if r=='HIGH' else '<25th' if r=='LOW' else '25-75th'} pct): "
              f"{regime_counts[r]:>6} days ({regime_counts[r]/total_days*100:.0f}%)")

    # Average vol by regime (from regime_log month-ends)
    rl = pd.DataFrame(regime_log)
    if len(rl) > 0:
        print(f"\n  High vol periods concentrated around:")
        high_months = rl[rl["regime"] == "HIGH"]["date"]
        if len(high_months) > 0:
            # Group by year
            by_year = high_months.dt.year.value_counts().sort_index()
            for yr, n in by_year.items():
                if n >= 2:
                    print(f"    {yr}: {n} months")

    # Average vol by regime
    if len(rvol_21d) > 0:
        for r in ["HIGH", "NORMAL", "LOW"]:
            r_months = rl[rl["regime"] == r]["date"]
            if len(r_months) > 0:
                vols = [rvol_21d.get(d, np.nan) for d in r_months if d in rvol_21d.index]
                vols = [v for v in vols if not np.isnan(v)]
                if vols:
                    print(f"\n  Average vol in {r} regime: {np.mean(vols)*100:.1f}% annualized")

    # Leveraged months by regime
    if leveraged_regime["total"] > 0:
        print(f"\n  Month-end regime when leverage was ON:")
        for r in ["HIGH", "NORMAL", "LOW"]:
            n = leveraged_regime[r]
            pct = n / leveraged_regime["total"] * 100
            print(f"    {r}: {n} months ({pct:.0f}%)")

    # ── Performance table ────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  FULL PERFORMANCE TABLE")
    print(f"{'='*120}")
    print(f"\n  {'Strategy':<16} {'Return':>8} {'Vol':>7} {'Sharpe':>8} {'Sortino':>8} "
          f"{'MaxDD':>8} {'Calmar':>8} {'Terminal($1)':>13} {'vs Base':>10}")
    print(f"  {'-'*16} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*13} {'-'*10}")
    for s in strat_order:
        p = perfs.get(s)
        if not p: continue
        vs = p["final"] - p_base.get("final", 0) if s != "IVV B&H" else 0
        vs_str = f"{vs:>+9.2f}" if s != "IVV B&H" else f"{'—':>10}"
        print(f"  {s:<16} {p['ar']:>7.1%} {p['av']:>6.1%} {p['sh']:>8.3f} {p['sortino']:>8.3f} "
              f"{p['dd']:>7.1%} {p['calmar']:>8.2f} ${p['final']:>12.2f} {vs_str}")

    # ── Crisis analysis ──────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  CRISIS ANALYSIS")
    print(f"{'='*120}")
    for cname, cs, ce in [("GFC (2008-09)", "2008-09-01", "2009-03-31"),
                           ("COVID (Feb-Mar 2020)", "2020-02-19", "2020-03-23"),
                           ("2022 Bear", "2022-01-03", "2022-10-31")]:
        print(f"\n  {cname}:")
        print(f"  {'Strategy':<16} {'Return':>10} {'MaxDD':>10}")
        print(f"  {'-'*16} {'-'*10} {'-'*10}")
        for s in strat_order:
            sr = ret[s]; c = sr[(sr.index >= pd.Timestamp(cs)) & (sr.index <= pd.Timestamp(ce))]
            if len(c) > 0:
                cum = (1+c).cumprod(); mdd = ((cum-cum.expanding().max())/cum.expanding().max()).min()
                print(f"  {s:<16} {(1+c).prod()-1:>+9.1%} {mdd:>9.1%}")

    # ── Signal difference ────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  SIGNAL DIFFERENCE ANALYSIS")
    print(f"{'='*120}")
    total = signal_agreement["total"]
    agree_all = signal_agreement["agree_all"]
    agree_lev = signal_agreement["agree_leverage"]
    print(f"\n  Months where ADAPTIVE and BASELINE scores agree (all assets): "
          f"{agree_all}/{total} ({agree_all/total*100:.0f}%)")
    print(f"  Months where they disagree on at least one asset: "
          f"{total-agree_all}/{total} ({(total-agree_all)/total*100:.0f}%)")
    print(f"  Months where leverage decision differs: "
          f"{total-agree_lev}/{total} ({(total-agree_lev)/total*100:.0f}%)")

    # ── Key questions ────────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  KEY QUESTIONS")
    print(f"{'='*120}")

    pa = perfs.get("ADAPTIVE", {})
    pac = perfs.get("ADAPTIVE-CB", {})
    if pa and p_base:
        print(f"\n  Q1. Adaptive vs fixed Sharpe: {p_base['sh']:.3f} → {pa['sh']:.3f} ({pa['sh']-p_base['sh']:+.3f})")
        print(f"      Terminal: ${p_base['final']:.2f} → ${pa['final']:.2f} (${pa['final']-p_base['final']:+.2f})")
    if pac:
        print(f"\n  Q3. Adaptive CB incremental: {pa['sh']:.3f} → {pac['sh']:.3f} ({pac['sh']-pa['sh']:+.3f})")

    # Q4: COVID max DD comparison
    for s in ["BASELINE", "ADAPTIVE", "ADAPTIVE-CB"]:
        sr = ret[s]
        c = sr[(sr.index >= "2020-02-19") & (sr.index <= "2020-03-23")]
        if len(c) > 0:
            cum = (1+c).cumprod(); mdd = ((cum-cum.expanding().max())/cum.expanding().max()).min()
            print(f"\n  Q4. {s} COVID MaxDD: {mdd:.1%}")

    # Regime transition count
    if len(rl) > 0:
        transitions = (rl["regime"] != rl["regime"].shift(1)).sum()
        print(f"\n  Q5. Regime transitions: {transitions} over {len(rl)} months ({transitions/len(rl)*100:.0f}%)")

    # Verdict
    if pa and p_base:
        both_better = pa["sh"] > p_base["sh"] and pa["final"] > p_base["final"]
        if both_better:
            print(f"\n  VERDICT: Adaptive improves BOTH Sharpe and terminal → ADOPT")
        elif pa["sh"] > p_base["sh"]:
            print(f"\n  VERDICT: Adaptive improves Sharpe but costs terminal → TRADEOFF")
        else:
            print(f"\n  VERDICT: Adaptive does NOT improve Sharpe → KEEP BASELINE")

    print()
    return perfs


if __name__ == "__main__":
    print("=" * 120)
    print("  ADAPTIVE SMA LOOKBACKS: Vol-Regime-Conditional Periods")
    print("=" * 120)
    print(f"  HIGH VOL: {LOOKBACKS['HIGH']} | NORMAL: {LOOKBACKS['NORMAL']} | LOW VOL: {LOOKBACKS['LOW']}")

    print(f"\n  Loading data...")
    daily_ret, dpdf, all_smas, rfr_daily, rvol_21d, vol_pctile, actual_lev, both_start = load_data()

    ret, regime_counts, regime_log, signal_agreement, leveraged_regime = run_backtest(
        daily_ret, dpdf, all_smas, rfr_daily, vol_pctile, actual_lev, both_start)

    perfs = report(ret, regime_counts, regime_log, signal_agreement, leveraged_regime, rvol_21d)

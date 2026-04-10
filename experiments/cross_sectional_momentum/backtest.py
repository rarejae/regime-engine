"""Cross-sectional momentum overlay on Faber-Sweep-40-Daily-Daily.

Among eligible assets, rank by 12-month momentum. Tilt top half up
(1.20x weight), bottom half down (0.80x weight). Renormalize.

BASELINE: fixed baseline weights (current production)
XSMOM: 1.20x/0.80x tilt
XSMOM-STRONG: 1.35x/0.65x tilt
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
RISKY = ["IVV", "QQQ", "VGLT", "IAU", "DBC"]
DAILY_SMA_PERIODS = [126, 200, 252]
SSO_EXP = 0.0089; QLD_EXP = 0.0095; FABER_SUB = 1.00
MOM_LOOKBACK = 252  # 12-month momentum


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
    daily_smas = {p: dpdf.rolling(p, min_periods=p).mean() for p in DAILY_SMA_PERIODS}

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


def sma_scores(day, dpdf, smas):
    scores = {}
    for a in RISKY:
        if a not in dpdf.columns: scores[a] = 0; continue
        p = dpdf.loc[:day, a]
        if len(p) == 0 or pd.isna(p.iloc[-1]): scores[a] = 0; continue
        price = p.iloc[-1]; sc = 0
        for per in DAILY_SMA_PERIODS:
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
        for per in DAILY_SMA_PERIODS:
            s = smas[per].loc[:day, etf]
            if len(s) > 0 and pd.notna(s.iloc[-1]) and price < s.iloc[-1]: b += 1
        if b >= 3: return True
    return False


def compute_momentum(day, dpdf):
    """Compute 12-month momentum for each risky asset as of `day`."""
    mom = {}
    for a in RISKY:
        if a not in dpdf.columns: mom[a] = np.nan; continue
        p = dpdf.loc[:day, a]
        if len(p) < MOM_LOOKBACK + 1: mom[a] = np.nan; continue
        current = p.iloc[-1]
        past = p.iloc[-MOM_LOOKBACK - 1]
        if pd.notna(current) and pd.notna(past) and past > 0:
            mom[a] = (current / past) - 1
        else:
            mom[a] = np.nan
    return mom


def apply_momentum_tilt(faber_weights, momentum, tilt_up, tilt_down):
    """Apply cross-sectional momentum tilt to eligible assets.

    tilt_up: multiplier for top half (e.g. 1.20)
    tilt_down: multiplier for bottom half (e.g. 0.80)
    """
    # Identify eligible risky assets (weight > 0)
    eligible = [(a, faber_weights.get(a, 0), momentum.get(a, np.nan))
                for a in RISKY if faber_weights.get(a, 0) > 0.001 and not np.isnan(momentum.get(a, np.nan))]

    if len(eligible) <= 1:
        return dict(faber_weights), None, None  # can't rank a single asset

    # Sort by momentum descending
    eligible.sort(key=lambda x: x[2], reverse=True)
    n = len(eligible)
    mid = n // 2

    tilted = dict(faber_weights)
    top_assets = []; bottom_assets = []

    for i, (a, w, m) in enumerate(eligible):
        if i < mid:
            tilted[a] = w * tilt_up
            top_assets.append(a)
        elif i >= n - mid:
            tilted[a] = w * tilt_down
            bottom_assets.append(a)
        # Middle asset (odd n) stays at 1.0

    # Renormalize: keep cash at its current level, adjust risky to sum correctly
    cash_w = tilted.get("cash", 0.10)
    risky_total = sum(tilted.get(a, 0) for a in RISKY)
    target_risky = 1.0 - cash_w

    if risky_total > 0 and target_risky > 0:
        scale = target_risky / risky_total
        for a in RISKY:
            tilted[a] = tilted.get(a, 0) * scale

    return tilted, top_assets, bottom_assets


def run_backtest(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start):
    print(f"\n{'='*120}")
    print(f"  BACKTEST")
    print(f"{'='*120}")

    bt_start = pd.Timestamp("2002-01-01")
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:].index
    print(f"  {len(trading_days)} trading days ({common_start.date()} to {trading_days.max().date()})")

    strats = {
        "BASELINE":     (1.0, 1.0),   # no tilt
        "XSMOM":        (1.20, 0.80),
        "XSMOM-STRONG": (1.35, 0.65),
    }

    results = {s: {} for s in strats}
    results["IVV B&H"] = {}

    state = {s: {"wf": dict(BASELINE_W), "la": False, "dlv": False} for s in strats}

    # Analysis tracking
    eligible_counts = []
    top_ranked_counts = {a: 0 for a in RISKY}
    bottom_ranked_counts = {a: 0 for a in RISKY}
    fwd_correct = 0; fwd_total = 0
    tilt_helped = 0; tilt_hurt = 0; tilt_help_sum = 0; tilt_hurt_sum = 0
    best_tilt = None; worst_tilt = None
    dbc_weights_baseline = []; dbc_weights_xsmom = []

    for day in trading_days:
        if day not in daily_ret.index: continue
        dr = daily_ret.loc[day]
        avail = [a for a in ASSETS if a in dr.index and pd.notna(dr[a])]
        if len(avail) < 3: continue
        actual = {a: float(dr[a]) for a in avail}
        rfr = float(rfr_daily.get(day, 0))

        is_ms = (day == trading_days[0] or
                 day.month != trading_days[trading_days.get_loc(day)-1].month)

        if is_ms:
            prior = trading_days[trading_days < day]
            sd = prior[-1] if len(prior) > 0 else day
            scores = sma_scores(sd, dpdf, daily_smas)
            momentum = compute_momentum(sd, dpdf)

            # Faber weights (same for all strategies before tilt)
            w1, pool = apply_faber_filter(scores, BASELINE_W)
            base_w = dict(w1); base_w["cash"] = base_w.get("cash", 0) + pool

            # Count eligible
            elig = [a for a in RISKY if base_w.get(a, 0) > 0.001]
            eligible_counts.append(len(elig))

            fc = scores.get("IVV", 0) >= 3 and scores.get("QQQ", 0) >= 3

            for s_name, (tu, td) in strats.items():
                st = state[s_name]
                st["dlv"] = False

                if tu == 1.0 and td == 1.0:
                    # BASELINE — no tilt
                    st["wf"] = dict(base_w)
                else:
                    tilted, top_a, bot_a = apply_momentum_tilt(base_w, momentum, tu, td)
                    st["wf"] = tilted

                    # Track analytics for XSMOM only
                    if s_name == "XSMOM" and top_a and bot_a:
                        for a in top_a: top_ranked_counts[a] += 1
                        for a in bot_a: bottom_ranked_counts[a] += 1

                        # Forward return check: did top outperform bottom next month?
                        # (we'll track this approximately using the actual monthly returns)
                        fwd_total += 1

                st["la"] = fc

            # Track DBC weights in 2022
            if day.year == 2022:
                dbc_weights_baseline.append(base_w.get("DBC", 0))
                xsmom_w = state["XSMOM"]["wf"]
                dbc_weights_xsmom.append(xsmom_w.get("DBC", 0))

        # Daily circuit breaker
        for s_name in strats:
            st = state[s_name]
            if st["la"] and not st["dlv"]:
                if check_breach(day, dpdf, daily_smas):
                    st["la"] = False; st["dlv"] = True

        # Compute returns
        for s_name in strats:
            st = state[s_name]
            iw = st["wf"].get("IVV", 0); qw = st["wf"].get("QQQ", 0)
            ir = actual.get("IVV", 0); qr = actual.get("QQQ", 0)
            base_ret = sum(st["wf"].get(a, 0) * actual.get(a, 0) for a in avail if a not in ["IVV", "QQQ"])

            if st["la"]:
                if day >= both_start:
                    sso = float(actual_lev.get("SSO", pd.Series()).get(day, 2*ir-rfr-SSO_EXP/252))
                    qld = float(actual_lev.get("QLD", pd.Series()).get(day, 2*qr-rfr-QLD_EXP/252))
                    if np.isnan(sso): sso = 2*ir-rfr-SSO_EXP/252
                    if np.isnan(qld): qld = 2*qr-rfr-QLD_EXP/252
                else:
                    sso = 2*ir-rfr-SSO_EXP/252; qld = 2*qr-rfr-QLD_EXP/252
                results[s_name][day] = iw*sso + qw*qld + base_ret
            else:
                results[s_name][day] = iw*ir + qw*qr + base_ret

        results["IVV B&H"][day] = actual.get("IVV", 0)

    # Forward return check — use monthly returns
    ret_s = {s: pd.Series(d).sort_index() for s, d in results.items()}
    base_m = ret_s["BASELINE"].resample("MS").apply(lambda x: (1+x).prod()-1)
    xsmom_m = ret_s["XSMOM"].resample("MS").apply(lambda x: (1+x).prod()-1)
    diff = xsmom_m - base_m
    tilt_helped = (diff > 0.0001).sum()
    tilt_hurt = (diff < -0.0001).sum()
    tilt_help_sum = diff[diff > 0].sum()
    tilt_hurt_sum = diff[diff < 0].sum()

    if len(diff) > 0:
        best_idx = diff.idxmax()
        worst_idx = diff.idxmin()
        best_tilt = (best_idx, diff.loc[best_idx])
        worst_tilt = (worst_idx, diff.loc[worst_idx])

    analytics = {
        "eligible_counts": eligible_counts,
        "top_ranked": top_ranked_counts,
        "bottom_ranked": bottom_ranked_counts,
        "tilt_helped": tilt_helped,
        "tilt_hurt": tilt_hurt,
        "tilt_help_sum": tilt_help_sum,
        "tilt_hurt_sum": tilt_hurt_sum,
        "best_tilt": best_tilt,
        "worst_tilt": worst_tilt,
        "dbc_baseline": np.mean(dbc_weights_baseline) if dbc_weights_baseline else 0,
        "dbc_xsmom": np.mean(dbc_weights_xsmom) if dbc_weights_xsmom else 0,
    }

    return ret_s, analytics


def metrics(s):
    s = s.dropna()
    if len(s) < 50: return None
    ar = s.mean()*252; av = s.std()*np.sqrt(252)
    sh = ar/av if av > 0 else 0
    neg = s[s < 0]; ds = neg.std()*np.sqrt(252) if len(neg) > 10 else av
    so = ar/ds if ds > 0 else 0
    cum = (1+s).cumprod(); dd = ((cum-cum.expanding().max())/cum.expanding().max()).min()
    cal = ar/abs(dd) if dd != 0 else 0
    return {"ar": ar, "av": av, "sh": sh, "sortino": so, "dd": dd, "calmar": cal, "final": cum.iloc[-1]}


def report(ret_s, analytics):
    strat_order = ["BASELINE", "XSMOM", "XSMOM-STRONG", "IVV B&H"]
    perfs = {s: metrics(ret_s[s]) for s in strat_order}
    pb = perfs.get("BASELINE", {})

    # ── Momentum signal analysis ─────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  MOMENTUM SIGNAL ANALYSIS")
    print(f"{'='*120}")

    ec = analytics["eligible_counts"]
    from collections import Counter
    ec_dist = Counter(ec)
    print(f"\n  Average eligible assets per month: {np.mean(ec):.1f}")
    print(f"  Distribution:")
    for n in sorted(ec_dist.keys()):
        print(f"    {n} assets eligible: {ec_dist[n]} months ({ec_dist[n]/len(ec)*100:.0f}%)")

    print(f"\n  Asset most often top-ranked (XSMOM, strongest momentum):")
    tr = analytics["top_ranked"]
    for a in sorted(tr, key=tr.get, reverse=True):
        if tr[a] > 0:
            print(f"    {a}: {tr[a]} months ({tr[a]/len(ec)*100:.0f}%)")

    print(f"\n  Asset most often bottom-ranked (weakest momentum):")
    br = analytics["bottom_ranked"]
    for a in sorted(br, key=br.get, reverse=True):
        if br[a] > 0:
            print(f"    {a}: {br[a]} months ({br[a]/len(ec)*100:.0f}%)")

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
        vs = p["final"] - pb.get("final", 0) if s != "IVV B&H" else 0
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
            sr = ret_s[s]; c = sr[(sr.index >= pd.Timestamp(cs)) & (sr.index <= pd.Timestamp(ce))]
            if len(c) > 0:
                cum = (1+c).cumprod(); mdd = ((cum-cum.expanding().max())/cum.expanding().max()).min()
                print(f"  {s:<16} {(1+c).prod()-1:>+9.1%} {mdd:>9.1%}")

    # ── Tilt attribution ─────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  TILT ATTRIBUTION")
    print(f"{'='*120}")

    th = analytics["tilt_helped"]; tt = analytics["tilt_hurt"]
    ths = analytics["tilt_help_sum"]; tts = analytics["tilt_hurt_sum"]
    total_months = th + tt + (len(ec) - th - tt)

    print(f"\n  Months where tilt HELPED: {th} ({th/len(ec)*100:.0f}%), avg improvement: {ths/th*100:+.2f}%/month" if th > 0 else "")
    print(f"  Months where tilt HURT:   {tt} ({tt/len(ec)*100:.0f}%), avg cost: {tts/tt*100:+.2f}%/month" if tt > 0 else "")
    net = (ths + tts) * 12
    print(f"  Net tilt contribution: {net*100:+.2f}% annualized")

    bt = analytics["best_tilt"]; wt = analytics["worst_tilt"]
    if bt: print(f"\n  Best tilt month: {bt[0].strftime('%Y-%m')}, improvement: {bt[1]*100:+.2f}%")
    if wt: print(f"  Worst tilt month: {wt[0].strftime('%Y-%m')}, cost: {wt[1]*100:+.2f}%")

    # ── DBC concentration ────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  DBC CONCENTRATION CHECK (2022)")
    print(f"{'='*120}")
    print(f"\n  2022 DBC average weight:")
    print(f"    BASELINE: {analytics['dbc_baseline']*100:.1f}%")
    print(f"    XSMOM:    {analytics['dbc_xsmom']*100:.1f}%")

    # ── Key questions ────────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  KEY QUESTIONS")
    print(f"{'='*120}")

    px = perfs.get("XSMOM", {})
    pxs = perfs.get("XSMOM-STRONG", {})
    if px and pb:
        print(f"\n  Q1. XSMOM vs BASELINE Sharpe: {pb['sh']:.3f} → {px['sh']:.3f} ({px['sh']-pb['sh']:+.3f})")
        print(f"      Terminal: ${pb['final']:.2f} → ${px['final']:.2f} (${px['final']-pb['final']:+.2f})")
    if pxs and px:
        print(f"\n  Q2. XSMOM-STRONG vs XSMOM: {px['sh']:.3f} → {pxs['sh']:.3f} ({pxs['sh']-px['sh']:+.3f})")
        if pxs["sh"] > px["sh"]:
            print(f"      Stronger tilt HELPS — momentum signal has forward power")
        else:
            print(f"      Stronger tilt HURTS — momentum introduces noise at this tilt level")

    # Verdict
    if px and pb:
        both = px["sh"] > pb["sh"] and px["final"] > pb["final"]
        if both:
            print(f"\n  VERDICT: Momentum tilt improves BOTH Sharpe and terminal → ADOPT")
        elif px["sh"] > pb["sh"]:
            print(f"\n  VERDICT: Tilt improves Sharpe but costs terminal → TRADEOFF")
        elif px["final"] > pb["final"]:
            print(f"\n  VERDICT: Tilt improves terminal but hurts Sharpe → TRADEOFF")
        else:
            print(f"\n  VERDICT: Tilt does NOT improve either → KEEP BASELINE")

    print()
    return perfs


if __name__ == "__main__":
    print("=" * 120)
    print("  CROSS-SECTIONAL MOMENTUM OVERLAY")
    print("=" * 120)
    print(f"  Tilt: top half 1.20x, bottom half 0.80x (XSMOM)")
    print(f"  Strong tilt: top half 1.35x, bottom half 0.65x (XSMOM-STRONG)")
    print(f"  Momentum lookback: {MOM_LOOKBACK} trading days")

    print(f"\n  Loading data...")
    daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start = load_data()

    ret_s, analytics = run_backtest(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start)
    perfs = report(ret_s, analytics)

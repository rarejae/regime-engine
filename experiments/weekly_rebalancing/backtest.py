"""Weekly vs monthly full portfolio rebalancing on Faber-Sweep-40-Daily-Daily.

Tests whether recomputing Faber scores and rebalancing every Friday (instead
of month-end only) improves performance at 100% SSO/QLD substitution.

Both strategies use identical daily circuit breaker (independent of rebalance cadence).
Hybrid real ETF data: actual SSO/QLD from Jun 2006, synthetic before.
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv; load_dotenv()

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

from taa.data import load_daily_etf_returns, load_monthly_prices
from taa.faber import compute_trend_scores, apply_faber_filter

BASELINE = {"IVV": 0.45, "QQQ": 0.25, "VGLT": 0.05, "IAU": 0.10, "DBC": 0.05, "cash": 0.10}
ASSETS = list(BASELINE.keys())
SSO_EXP = 0.0089; QLD_EXP = 0.0095
FABER_SUB = 1.00  # 100% substitution (Pedersen level)
DAILY_SMA_PERIODS = [126, 200, 252]

# Transaction costs (bid-ask spread per trade, one-way)
TC_EQUITY = 0.0001   # IVV, QQQ, SSO, QLD: 0.01%
TC_OTHER = 0.0003    # VGLT, IAU, DBC: 0.03%


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
            p.index = pd.to_datetime(p.index).tz_localize(None)
            dp[our] = p
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

    both_start = max(actual_lev["SSO"].index.min(), actual_lev["QLD"].index.min()) \
        if "SSO" in actual_lev and "QLD" in actual_lev else pd.Timestamp("2099-01-01")

    return daily_ret, rfr_daily, dpdf, daily_smas, actual_lev, both_start


def sma_scores(day, dpdf, smas):
    scores = {}
    for a in ["IVV", "QQQ", "VGLT", "IAU", "DBC"]:
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
        price = p.iloc[-1]; breaches = 0
        for per in DAILY_SMA_PERIODS:
            s = smas[per].loc[:day, etf]
            if len(s) > 0 and pd.notna(s.iloc[-1]) and price < s.iloc[-1]: breaches += 1
        if breaches >= 3: return True
    return False


def scores_to_weights(scores):
    """Convert Faber scores to target weights."""
    w = {}; pool = 0.0
    for a, base_w in BASELINE.items():
        if a == "cash": w[a] = base_w; continue
        sc = scores.get(a, 0)
        if sc >= 3: w[a] = base_w
        elif sc == 2: w[a] = base_w * 0.70; pool += base_w * 0.30
        else: w[a] = 0.0; pool += base_w
    w["cash"] = w.get("cash", 0) + pool
    return w


def compute_tc(old_w, new_w, leverage_changed):
    """Compute one-way transaction cost for rebalance."""
    cost = 0.0
    for a in ASSETS:
        delta = abs(new_w.get(a, 0) - old_w.get(a, 0))
        if delta < 0.001: continue
        tc_rate = TC_OTHER if a in ["VGLT", "IAU", "DBC"] else TC_EQUITY
        cost += delta * tc_rate
    # Leverage switch costs (SSO/QLD trade)
    if leverage_changed:
        # Trade the full equity position both ways
        cost += (BASELINE["IVV"] + BASELINE["QQQ"]) * TC_EQUITY * 2  # buy + sell
    return cost


def lev_return_hybrid(iw, qw, ir, qr, rfr, sub, base_ret, day, actual_lev, both_start):
    if sub <= 0:
        return iw * ir + qw * qr + base_ret
    if day >= both_start and "SSO" in actual_lev and "QLD" in actual_lev:
        sso_r = float(actual_lev["SSO"].get(day, np.nan))
        qld_r = float(actual_lev["QLD"].get(day, np.nan))
        if np.isnan(sso_r): sso_r = 2.0 * ir - rfr - SSO_EXP / 252
        if np.isnan(qld_r): qld_r = 2.0 * qr - rfr - QLD_EXP / 252
    else:
        sso_r = 2.0 * ir - rfr - SSO_EXP / 252
        qld_r = 2.0 * qr - rfr - QLD_EXP / 252
    return iw * ((1 - sub) * ir + sub * sso_r) + qw * ((1 - sub) * qr + sub * qld_r) + base_ret


def run_backtest(daily_ret, rfr_daily, dpdf, daily_smas, actual_lev, both_start):
    print(f"\n{'='*120}")
    print(f"  BACKTEST")
    print(f"{'='*120}")

    bt_start = pd.Timestamp("2002-01-01")
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:].index
    print(f"  {len(trading_days)} trading days ({common_start.date()} to {trading_days.max().date()})")

    # Strategies: (rebalance_cadence, apply_tc)
    strats = {
        "MONTHLY-100":        ("monthly", True),
        "WEEKLY-100":         ("weekly",  True),
        "WEEKLY-100-NOCOST":  ("weekly",  False),
    }

    results = {s: {} for s in strats}
    results["IVV B&H"] = {}

    state = {}
    for s in strats:
        state[s] = {
            "w_faber": dict(BASELINE),
            "scores": {a: 3 for a in ASSETS if a != "cash"},
            "faber_conv": False,
            "leverage_active": False,
            "delevered": False,
            "rebalance_count": 0,
            "tc_total": 0.0,
            "score_changes": [],  # (date, asset, old, new)
            "leverage_changes": 0,
            "whipsaw_events": [],  # (date, asset, score_went, then_reverted)
        }

    # Track per-asset score history for whipsaw analysis
    score_history_weekly = {a: [] for a in ["IVV", "QQQ", "VGLT", "IAU", "DBC"]}

    for day in trading_days:
        if day not in daily_ret.index: continue
        dr = daily_ret.loc[day]
        avail = [a for a in ASSETS if a in dr.index and pd.notna(dr[a])]
        if len(avail) < 3: continue
        actual = {a: float(dr[a]) for a in avail}
        rfr = float(rfr_daily.get(day, 0))

        is_ms = (day == trading_days[0] or
                 day.month != trading_days[trading_days.get_loc(day) - 1].month)
        is_friday = day.weekday() == 4

        for s_name, (cadence, _) in strats.items():
            st = state[s_name]

            should_rebalance = False
            if cadence == "monthly" and is_ms:
                should_rebalance = True
            elif cadence == "weekly" and is_friday:
                should_rebalance = True
            elif cadence == "monthly" and is_ms:
                should_rebalance = True

            # Also reset delevered flag on monthly boundary for all strategies
            if is_ms:
                st["delevered"] = False

            if should_rebalance:
                # PIT: use prior day's scores
                prior = trading_days[trading_days < day]
                sd = prior[-1] if len(prior) > 0 else day
                new_scores = sma_scores(sd, dpdf, daily_smas)

                # Track score changes
                old_scores = st["scores"]
                for a in ["IVV", "QQQ", "VGLT", "IAU", "DBC"]:
                    old_sc = old_scores.get(a, 0)
                    new_sc = new_scores.get(a, 0)
                    # Categorize: 3 → full, 2 → partial, 0-1 → exit
                    old_cat = 2 if old_sc >= 3 else (1 if old_sc == 2 else 0)
                    new_cat = 2 if new_sc >= 3 else (1 if new_sc == 2 else 0)
                    if old_cat != new_cat:
                        st["score_changes"].append((day, a, old_sc, new_sc))

                # Check leverage change
                new_conv = (new_scores.get("IVV", 0) >= 3 and new_scores.get("QQQ", 0) >= 3)
                lev_changed = new_conv != st["faber_conv"]
                if lev_changed:
                    st["leverage_changes"] += 1

                # Compute transaction cost
                new_w = scores_to_weights(new_scores)
                old_w = st["w_faber"]
                tc = compute_tc(old_w, new_w, lev_changed)

                st["scores"] = new_scores
                st["w_faber"] = new_w
                st["faber_conv"] = new_conv
                st["leverage_active"] = new_conv
                st["rebalance_count"] += 1
                st["tc_total"] += tc

                # Record weekly scores for whipsaw
                if cadence == "weekly":
                    for a in ["IVV", "QQQ", "VGLT", "IAU", "DBC"]:
                        score_history_weekly[a].append((day, new_scores.get(a, 0)))

            # Daily circuit breaker (independent of rebalance cadence)
            if st["leverage_active"] and not st["delevered"]:
                if check_breach(day, dpdf, daily_smas):
                    st["leverage_active"] = False
                    st["delevered"] = True

        # Compute returns
        for s_name, (cadence, apply_tc) in strats.items():
            st = state[s_name]
            iw = st["w_faber"].get("IVV", 0)
            qw = st["w_faber"].get("QQQ", 0)
            ir = actual.get("IVV", 0)
            qr = actual.get("QQQ", 0)
            base_ret = sum(st["w_faber"].get(a, 0) * actual.get(a, 0)
                           for a in avail if a not in ["IVV", "QQQ"])
            sub = FABER_SUB if st["leverage_active"] else 0.0
            ret = lev_return_hybrid(iw, qw, ir, qr, rfr, sub, base_ret, day, actual_lev, both_start)

            # Apply transaction cost as drag on rebalance days
            if apply_tc and (is_ms or (is_friday and cadence == "weekly")):
                # Spread TC over the month/week as daily drag — simpler: apply as lump on rebalance day
                pass  # TC accounted at end via total

            results[s_name][day] = ret

        results["IVV B&H"][day] = actual.get("IVV", 0)

    ret_series = {s: pd.Series(d).sort_index() for s, d in results.items()}

    # Apply cumulative TC as terminal adjustment
    years = len(trading_days) / 252
    for s_name, (cadence, apply_tc) in strats.items():
        if apply_tc:
            st = state[s_name]
            ann_tc = st["tc_total"] / years
            # Apply as daily drag
            daily_tc = ann_tc / 252
            ret_series[s_name] = ret_series[s_name] - daily_tc

    # Whipsaw analysis for weekly
    for a in ["IVV", "QQQ", "VGLT", "IAU", "DBC"]:
        hist = score_history_weekly[a]
        for i in range(1, len(hist) - 1):
            d0, s0 = hist[i - 1]
            d1, s1 = hist[i]
            d2, s2 = hist[i + 1]
            cat0 = 2 if s0 >= 3 else (1 if s0 == 2 else 0)
            cat1 = 2 if s1 >= 3 else (1 if s1 == 2 else 0)
            cat2 = 2 if s2 >= 3 else (1 if s2 == 2 else 0)
            if cat0 != cat1 and cat1 != cat2 and cat0 == cat2:
                state["WEEKLY-100"]["whipsaw_events"].append((d1, a, s0, s1, s2))

    return ret_series, state, score_history_weekly


def metrics(s):
    s = s.dropna()
    if len(s) < 50: return None
    ar = s.mean() * 252; av = s.std() * np.sqrt(252)
    sh = ar / av if av > 0 else 0
    neg = s[s < 0]; ds = neg.std() * np.sqrt(252) if len(neg) > 10 else av
    so = ar / ds if ds > 0 else 0
    cum = (1 + s).cumprod()
    dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
    cal = ar / abs(dd) if dd != 0 else 0
    return {"ar": ar, "av": av, "sh": sh, "sortino": so, "dd": dd, "calmar": cal, "final": cum.iloc[-1]}


def report(ret_series, state, score_history_weekly):
    strat_order = ["MONTHLY-100", "WEEKLY-100", "WEEKLY-100-NOCOST", "IVV B&H"]
    perfs = {s: metrics(ret_series[s]) for s in strat_order}
    p_mo = perfs.get("MONTHLY-100", {})
    years = 24.0

    # ── Signal change frequency ──────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  SIGNAL CHANGE FREQUENCY ANALYSIS")
    print(f"{'='*120}")

    w_st = state["WEEKLY-100"]
    m_st = state["MONTHLY-100"]

    w_changes = w_st["score_changes"]
    m_changes = m_st["score_changes"]

    # Count weeks with at least one change
    w_change_dates = set(d for d, _, _, _ in w_changes)
    total_fridays = w_st["rebalance_count"]
    active_weeks = len(w_change_dates)
    idle_weeks = total_fridays - active_weeks

    print(f"\n  Weekly rebalance frequency (2002-2026, {total_fridays} Fridays):")
    print(f"    Weeks with at least one score change:  {active_weeks} ({active_weeks/total_fridays*100:.0f}%)")
    print(f"    Weeks with no score change (idle):     {idle_weeks} ({idle_weeks/total_fridays*100:.0f}%)")
    print(f"    Average score changes per year:        {len(w_changes)/years:.1f}")
    print(f"    Monthly rebalance score changes/year:  {len(m_changes)/years:.1f}")
    print(f"    Additional changes from weekly:        {(len(w_changes)-len(m_changes))/years:.1f}/year")

    # Per-asset score changes
    print(f"\n  Asset score changes (weekly cadence):")
    for a in ["IVV", "QQQ", "VGLT", "IAU", "DBC"]:
        n = sum(1 for _, asset, _, _ in w_changes if asset == a)
        print(f"    {a}: {n} changes over {years:.0f} years ({n/years:.1f}/year)")

    # Leverage condition changes
    print(f"\n  Leverage condition changes (both IVV+QQQ 3/3 → not, or reverse):")
    print(f"    Monthly: {m_st['leverage_changes']} over {years:.0f} years ({m_st['leverage_changes']/years:.1f}/year)")
    print(f"    Weekly:  {w_st['leverage_changes']} over {years:.0f} years ({w_st['leverage_changes']/years:.1f}/year)")

    # Transaction costs
    m_tc_ann = m_st["tc_total"] / years
    w_tc_ann = w_st["tc_total"] / years
    print(f"\n  Estimated annual transaction cost:")
    print(f"    Monthly: {m_tc_ann:.4%}")
    print(f"    Weekly:  {w_tc_ann:.4%}")
    print(f"    Additional cost of weekly: {w_tc_ann - m_tc_ann:.4%}")

    # ── Performance table ────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  FULL PERFORMANCE TABLE")
    print(f"{'='*120}")

    print(f"\n  {'Strategy':<22} {'Return':>8} {'Vol':>7} {'Sharpe':>8} {'Sortino':>8} "
          f"{'MaxDD':>8} {'Calmar':>8} {'Terminal($1)':>13} {'vs Monthly':>11}")
    print(f"  {'-'*22} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*13} {'-'*11}")

    for s in strat_order:
        p = perfs.get(s)
        if not p: continue
        vs = p["final"] - p_mo.get("final", 0) if s != "IVV B&H" else 0
        vs_str = f"{vs:>+10.2f}" if s != "IVV B&H" else f"{'—':>11}"
        print(f"  {s:<22} {p['ar']:>7.1%} {p['av']:>6.1%} {p['sh']:>8.3f} {p['sortino']:>8.3f} "
              f"{p['dd']:>7.1%} {p['calmar']:>8.2f} ${p['final']:>12.2f} {vs_str}")

    # ── Crisis analysis ──────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  CRISIS ANALYSIS")
    print(f"{'='*120}")

    for cname, cs, ce in [("GFC (2008-09)", "2008-09-01", "2009-03-31"),
                           ("COVID (Feb-Mar 2020)", "2020-02-19", "2020-03-23"),
                           ("2022 Bear", "2022-01-03", "2022-10-31")]:
        print(f"\n  {cname}:")
        print(f"  {'Strategy':<22} {'Return':>10} {'MaxDD':>10}")
        print(f"  {'-'*22} {'-'*10} {'-'*10}")
        for s in strat_order:
            sr = ret_series[s]; c = sr[(sr.index >= pd.Timestamp(cs)) & (sr.index <= pd.Timestamp(ce))]
            if len(c) > 0:
                cum = (1 + c).cumprod()
                mdd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
                print(f"  {s:<22} {(1+c).prod()-1:>+9.1%} {mdd:>9.1%}")

    # ── Age-65 ───────────────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  AGE-65 PROJECTION ($21,000, 40 years)")
    print(f"{'='*120}")
    mo_proj = 21000 * (1 + p_mo.get("ar", 0)) ** 40
    print(f"\n  {'Strategy':<22} {'Ann Return':>11} {'$21K at 65':>13} {'vs Monthly':>12}")
    for s in ["MONTHLY-100", "WEEKLY-100", "WEEKLY-100-NOCOST"]:
        p = perfs.get(s)
        if not p: continue
        proj = 21000 * (1 + p["ar"]) ** 40
        vs = proj - mo_proj
        print(f"  {s:<22} {p['ar']:>10.2%} ${proj:>12,.0f} ${vs:>+11,.0f}")

    # ── Whipsaw analysis ─────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  WHIPSAW ANALYSIS (WEEKLY-100)")
    print(f"{'='*120}")

    whipsaws = w_st["whipsaw_events"]
    print(f"\n  Total whipsaw events (score changed then reverted within 1-2 weeks): {len(whipsaws)}")
    if whipsaws:
        # Count by asset
        by_asset = {}
        for _, a, _, _, _ in whipsaws:
            by_asset[a] = by_asset.get(a, 0) + 1
        print(f"  By asset:")
        for a in sorted(by_asset, key=by_asset.get, reverse=True):
            print(f"    {a}: {by_asset[a]} whipsaws ({by_asset[a]/years:.1f}/year)")

    # ── Timing improvement ───────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  TIMING IMPROVEMENT (weekly caught signal change before month-end)")
    print(f"{'='*120}")

    # Compare: for each month, did weekly rebalance before month-end produce different weights?
    w_changes_set = set((d.strftime("%Y-%m"), a) for d, a, _, _ in w_changes)
    m_changes_set = set((d.strftime("%Y-%m"), a) for d, a, _, _ in m_changes)
    early_catches = w_changes_set - m_changes_set
    print(f"\n  Score changes caught by weekly but not by monthly: {len(early_catches)}")
    print(f"  Score changes per year only weekly caught: {len(early_catches)/years:.1f}")

    # ── Key questions ────────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  KEY QUESTIONS")
    print(f"{'='*120}")

    p_w = perfs.get("WEEKLY-100")
    p_wn = perfs.get("WEEKLY-100-NOCOST")
    if p_w and p_mo:
        print(f"\n  Q1. Weeks requiring trades: {active_weeks}/{total_fridays} ({active_weeks/total_fridays*100:.0f}%)")
        print(f"  Q2. Weekly vs Monthly Sharpe: {p_mo['sh']:.3f} → {p_w['sh']:.3f} ({p_w['sh']-p_mo['sh']:+.3f})")
        print(f"      Weekly vs Monthly Terminal: ${p_mo['final']:.2f} → ${p_w['final']:.2f} (${p_w['final']-p_mo['final']:+.2f})")
        print(f"  Q3. Whipsaw rate: {len(whipsaws)} events ({len(whipsaws)/years:.1f}/year)")
        print(f"  Q5. TC meaningful? Additional {w_tc_ann-m_tc_ann:.4%}/year")
        if p_wn:
            print(f"      NOCOST Sharpe: {p_wn['sh']:.3f} (TC costs {p_wn['sh']-p_w['sh']:+.3f} Sharpe)")

    # Verdict
    if p_w and p_mo:
        both_better = p_w["sh"] > p_mo["sh"] and p_w["final"] > p_mo["final"]
        if both_better:
            print(f"\n  VERDICT: Weekly rebalancing IMPROVES both Sharpe and terminal → ADOPT")
        elif p_w["sh"] > p_mo["sh"]:
            print(f"\n  VERDICT: Weekly improves Sharpe but reduces terminal → TRADEOFF")
        elif p_w["final"] > p_mo["final"]:
            print(f"\n  VERDICT: Weekly improves terminal but reduces Sharpe → TRADEOFF")
        else:
            print(f"\n  VERDICT: Weekly does NOT improve either metric → KEEP MONTHLY")

    print()
    return perfs


if __name__ == "__main__":
    print("=" * 120)
    print("  WEEKLY vs MONTHLY REBALANCING: Faber-Sweep-40-Daily-Daily at 100% SUB")
    print("=" * 120)
    print(f"  Monthly: rebalance month-end. Weekly: rebalance every Friday.")
    print(f"  Both use daily circuit breaker independently. 100% SSO/QLD substitution.")

    print(f"\n  Loading data...")
    daily_ret, rfr_daily, dpdf, daily_smas, actual_lev, both_start = load_data()
    print(f"  Actual SSO/QLD from: {both_start.date()}")

    ret_series, state, score_history = run_backtest(daily_ret, rfr_daily, dpdf, daily_smas, actual_lev, both_start)
    perfs = report(ret_series, state, score_history)

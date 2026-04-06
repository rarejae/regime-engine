"""Daily vs weekly leverage circuit breaker on Faber-Sweep-40.

Three strategies:
1. S40-Daily-Weekly     — weekly Friday-only circuit breaker (current production)
2. S40-Daily-Daily      — daily circuit breaker, re-entry at monthly rebalance only
3. S40-Daily-Daily-Reentry — daily circuit breaker, daily re-entry when price recovers above all 3 SMAs

All use identical daily SMAs (126/200/252-day), identical Faber allocation logic,
identical 40% SSO/QLD substitution. Only difference is circuit breaker frequency.
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
SSO_EXPENSE = 0.0089
QLD_EXPENSE = 0.0095
FABER_SUB = 0.40
DAILY_SMA_PERIODS = [126, 200, 252]


def load_data():
    import yfinance as yf
    daily_ret = load_daily_etf_returns()
    daily_ret = daily_ret[[c for c in daily_ret.columns if c in ASSETS]]
    monthly_prices = load_monthly_prices()
    trend_df_monthly = compute_trend_scores(monthly_prices)

    from fredapi import Fred
    rfr_daily = pd.Series(0.0, index=daily_ret.index)
    key = os.environ.get("FRED_API_KEY")
    if key:
        tb = Fred(api_key=key).get_series("DTB3", observation_start="1998-01-01")
        tb.index = pd.to_datetime(tb.index)
        rfr_daily = (tb / 100 / 252).reindex(daily_ret.index, method="ffill").fillna(0)

    ticker_map = {"IVV": "SPY", "QQQ": "QQQ", "VGLT": "TLT", "IAU": "GLD", "DBC": "DBC"}
    daily_prices = {}
    for our, ticker in ticker_map.items():
        d = yf.download(ticker, start="1998-01-01", progress=False)
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p, "columns"): p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None)
            daily_prices[our] = p
    daily_prices_df = pd.DataFrame(daily_prices).sort_index()

    daily_smas = {}
    for period in DAILY_SMA_PERIODS:
        daily_smas[period] = daily_prices_df.rolling(period, min_periods=period).mean()

    return daily_ret, trend_df_monthly, rfr_daily, daily_prices_df, daily_smas


def daily_sma_scores(day, daily_prices_df, daily_smas):
    scores = {}
    for asset in ["IVV", "QQQ", "VGLT", "IAU", "DBC"]:
        if asset not in daily_prices_df.columns:
            scores[asset] = 0; continue
        p = daily_prices_df.loc[:day, asset]
        if len(p) == 0 or pd.isna(p.iloc[-1]):
            scores[asset] = 0; continue
        price = p.iloc[-1]
        score = 0
        for period in DAILY_SMA_PERIODS:
            sma_to_day = daily_smas[period].loc[:day, asset]
            if len(sma_to_day) > 0 and pd.notna(sma_to_day.iloc[-1]) and price > sma_to_day.iloc[-1]:
                score += 1
        scores[asset] = score
    return scores


def check_breach(day, daily_prices_df, daily_smas):
    """Check if either IVV or QQQ is below ALL 3 daily SMAs. Returns (breach, detail)."""
    for etf in ["IVV", "QQQ"]:
        if etf not in daily_prices_df.columns: continue
        p = daily_prices_df.loc[:day, etf]
        if len(p) == 0: continue
        price = p.iloc[-1]
        breaches = 0
        for period in DAILY_SMA_PERIODS:
            sma = daily_smas[period].loc[:day, etf]
            if len(sma) > 0 and pd.notna(sma.iloc[-1]) and price < sma.iloc[-1]:
                breaches += 1
        if breaches >= 3:
            return True, f"{etf} below all 3 SMAs ({price:.1f})"
    return False, ""


def check_recovery(day, daily_prices_df, daily_smas):
    """Check if BOTH IVV and QQQ are above ALL 3 daily SMAs (for daily re-entry)."""
    for etf in ["IVV", "QQQ"]:
        if etf not in daily_prices_df.columns: return False
        p = daily_prices_df.loc[:day, etf]
        if len(p) == 0: return False
        price = p.iloc[-1]
        for period in DAILY_SMA_PERIODS:
            sma = daily_smas[period].loc[:day, etf]
            if len(sma) == 0 or pd.isna(sma.iloc[-1]) or price <= sma.iloc[-1]:
                return False
    return True


def lev_return(iw, qw, ir, qr, rfr, sub, base_ret):
    if sub <= 0:
        return iw * ir + qw * qr + base_ret
    sso = 2.0 * ir - rfr - SSO_EXPENSE / 252
    qld = 2.0 * qr - rfr - QLD_EXPENSE / 252
    return iw * ((1 - sub) * ir + sub * sso) + qw * ((1 - sub) * qr + sub * qld) + base_ret


def run_backtest(daily_ret, trend_df_monthly, rfr_daily, daily_prices_df, daily_smas):
    print(f"\n{'='*120}")
    print(f"  BACKTEST")
    print(f"{'='*120}")

    bt_start = pd.Timestamp("2002-01-01")
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:].index
    print(f"  {len(trading_days)} trading days ({common_start.date()} to {trading_days.max().date()})")

    # Strategy configs: (breaker_freq, reentry_daily)
    # breaker_freq: "weekly" = Friday only, "daily" = every day
    # reentry_daily: True = re-enter leverage as soon as both above all 3 SMAs
    strat_cfgs = {
        "S40-Daily-Weekly":        ("weekly", False),
        "S40-Daily-Daily":         ("daily",  False),
        "S40-Daily-Daily-Reentry": ("daily",  True),
    }

    results = {s: {} for s in strat_cfgs}
    results["IVV B&H"] = {}

    state = {}
    for s_name in strat_cfgs:
        state[s_name] = {
            "w_faber": dict(BASELINE),
            "faber_conv": False,          # monthly allocation says leverage ON
            "leverage_active": False,     # actual leverage state after breaker
            "delevered_this_month": False,
            "breaker_events": [],
            "reentry_events": [],
            "cur_trends": {a: 3 for a in ASSETS if a != "cash"},
        }

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

        # ── Monthly rebalance ────────────────────────────────────────────
        if is_ms:
            for s_name in strat_cfgs:
                st = state[s_name]
                st["delevered_this_month"] = False

                # Daily SMAs as of last trading day of prior month
                prior_days = trading_days[trading_days < day]
                score_day = prior_days[-1] if len(prior_days) > 0 else day
                st["cur_trends"] = daily_sma_scores(score_day, daily_prices_df, daily_smas)

                w1, pool = apply_faber_filter(st["cur_trends"], BASELINE)
                st["w_faber"] = dict(w1)
                st["w_faber"]["cash"] = st["w_faber"].get("cash", 0) + pool

                st["faber_conv"] = (st["cur_trends"].get("IVV", 0) >= 3 and
                                     st["cur_trends"].get("QQQ", 0) >= 3)
                st["leverage_active"] = st["faber_conv"]

        # ── Circuit breaker checks ───────────────────────────────────────
        for s_name, (breaker_freq, reentry_daily) in strat_cfgs.items():
            st = state[s_name]

            # Determine if we should check today
            should_check = False
            if breaker_freq == "daily":
                should_check = True
            elif breaker_freq == "weekly" and is_friday:
                should_check = True

            # EXIT check
            if should_check and st["leverage_active"] and not st["delevered_this_month"]:
                breach, detail = check_breach(day, daily_prices_df, daily_smas)
                if breach:
                    st["breaker_events"].append({"date": day, "detail": detail})
                    st["leverage_active"] = False
                    st["delevered_this_month"] = True

            # DAILY RE-ENTRY check (strategy 3 only)
            if reentry_daily and not st["leverage_active"] and st["faber_conv"] and st["delevered_this_month"]:
                if check_recovery(day, daily_prices_df, daily_smas):
                    st["leverage_active"] = True
                    st["delevered_this_month"] = False
                    st["reentry_events"].append({"date": day})

        # ── Compute daily returns ────────────────────────────────────────
        for s_name in strat_cfgs:
            st = state[s_name]
            iw = st["w_faber"].get("IVV", 0)
            qw = st["w_faber"].get("QQQ", 0)
            ir = actual.get("IVV", 0)
            qr = actual.get("QQQ", 0)
            base_ret = sum(st["w_faber"].get(a, 0) * actual.get(a, 0)
                           for a in avail if a not in ["IVV", "QQQ"])
            sub = FABER_SUB if st["leverage_active"] else 0.0
            results[s_name][day] = lev_return(iw, qw, ir, qr, rfr, sub, base_ret)

        results["IVV B&H"][day] = actual.get("IVV", 0)

    return {s: pd.Series(d).sort_index() for s, d in results.items()}, state


def compute_metrics(s):
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


def report(ret_series, state):
    strats = ["S40-Daily-Weekly", "S40-Daily-Daily", "S40-Daily-Daily-Reentry", "IVV B&H"]
    perfs = {s: compute_metrics(ret_series[s]) for s in strats}
    p_weekly = perfs.get("S40-Daily-Weekly", {})

    # ── Performance ──────────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  FULL PERFORMANCE TABLE")
    print(f"{'='*120}")
    print(f"  {'Strategy':<30} {'Return':>8} {'Vol':>7} {'Sharpe':>8} {'Sortino':>8} "
          f"{'MaxDD':>8} {'Calmar':>8} {'Terminal($1)':>13} {'vs Weekly':>10}")
    print(f"  {'-'*30} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*13} {'-'*10}")

    for s in strats:
        p = perfs.get(s)
        if not p: continue
        vs = p["final"] - p_weekly.get("final", 0) if s != "IVV B&H" else 0
        vs_str = f"{vs:>+9.2f}" if s != "IVV B&H" else f"{'—':>10}"
        print(f"  {s:<30} {p['ar']:>7.1%} {p['av']:>6.1%} {p['sh']:>8.3f} {p['sortino']:>8.3f} "
              f"{p['dd']:>7.1%} {p['calmar']:>8.2f} ${p['final']:>12.2f} {vs_str}")

    # ── Crisis analysis ──────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  CRISIS ANALYSIS")
    print(f"{'='*120}")
    for cname, cs, ce in [("GFC (2008-09)", "2008-09-01", "2009-03-31"),
                           ("COVID (Feb-Mar 2020)", "2020-02-19", "2020-03-23"),
                           ("2022 Bear", "2022-01-03", "2022-10-31")]:
        print(f"\n  {cname}:")
        print(f"  {'Strategy':<30} {'Return':>10} {'MaxDD':>10}")
        print(f"  {'-'*30} {'-'*10} {'-'*10}")
        for s in strats:
            sr = ret_series[s]
            c = sr[(sr.index >= pd.Timestamp(cs)) & (sr.index <= pd.Timestamp(ce))]
            if len(c) > 0:
                cum = (1 + c).cumprod()
                mdd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
                print(f"  {s:<30} {(1+c).prod()-1:>+9.1%} {mdd:>9.1%}")

    # ── Circuit breaker diagnostics ──────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  CIRCUIT BREAKER DIAGNOSTICS")
    print(f"{'='*120}")

    for s_name in ["S40-Daily-Weekly", "S40-Daily-Daily", "S40-Daily-Daily-Reentry"]:
        events = state[s_name]["breaker_events"]
        reentries = state[s_name].get("reentry_events", [])
        years = 24.0
        print(f"\n  {s_name}:")
        print(f"    Total events: {len(events)} over {years:.0f} years ({len(events)/years:.1f}/year)")
        if reentries:
            print(f"    Daily re-entries: {len(reentries)}")

        if events:
            print(f"\n    {'Date':>12} {'Detail'}")
            print(f"    {'-'*12} {'-'*50}")
            for e in events:
                print(f"    {e['date'].strftime('%Y-%m-%d'):>12} {e['detail']}")

    # COVID timing comparison
    print(f"\n  COVID TIMING COMPARISON:")
    for s_name in ["S40-Daily-Weekly", "S40-Daily-Daily"]:
        events = state[s_name]["breaker_events"]
        covid_events = [e for e in events if pd.Timestamp("2020-02-01") <= e["date"] <= pd.Timestamp("2020-04-01")]
        if covid_events:
            first = covid_events[0]
            print(f"    {s_name}: first trigger {first['date'].strftime('%Y-%m-%d')} ({first['detail']})")

            # Return from Feb 19 to trigger date
            sr = ret_series[s_name]
            pre_trigger = sr[(sr.index >= pd.Timestamp("2020-02-19")) & (sr.index <= first["date"])]
            if len(pre_trigger) > 0:
                ret = (1 + pre_trigger).prod() - 1
                print(f"      Return Feb 19 - trigger: {ret:+.1%} ({len(pre_trigger)} trading days)")
        else:
            print(f"    {s_name}: no COVID trigger")

    # 2022 timing comparison
    print(f"\n  2022 TIMING COMPARISON:")
    for s_name in ["S40-Daily-Weekly", "S40-Daily-Daily"]:
        events = state[s_name]["breaker_events"]
        events_2022 = [e for e in events if pd.Timestamp("2021-12-01") <= e["date"] <= pd.Timestamp("2022-06-01")]
        if events_2022:
            first = events_2022[0]
            print(f"    {s_name}: first trigger {first['date'].strftime('%Y-%m-%d')} ({first['detail']})")

    # ── Whipsaw analysis ─────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  WHIPSAW ANALYSIS")
    print(f"{'='*120}")

    for s_name in ["S40-Daily-Daily", "S40-Daily-Daily-Reentry"]:
        events = state[s_name]["breaker_events"]
        reentries = state[s_name].get("reentry_events", [])

        # For daily-no-reentry: count how many times price recovered before next monthly
        if s_name == "S40-Daily-Daily":
            recovery_count = 0
            sr = ret_series[s_name]
            for e in events:
                trigger_date = e["date"]
                # Find next month start
                next_month = trigger_date + pd.offsets.MonthBegin(1)
                # Check if price recovered before next month
                check_days = daily_prices_df.index[(daily_prices_df.index > trigger_date) &
                                                     (daily_prices_df.index < next_month)]
                for cd in check_days:
                    if check_recovery(cd, daily_prices_df, daily_smas):
                        recovery_count += 1
                        break

            print(f"\n  {s_name}:")
            print(f"    Events where price recovered before next monthly rebalance: {recovery_count}/{len(events)}")
            print(f"    These are potential whipsaws if re-entry were daily.")

        if s_name == "S40-Daily-Daily-Reentry":
            print(f"\n  {s_name}:")
            print(f"    Exit events: {len(events)}")
            print(f"    Daily re-entry events: {len(reentries)}")
            if reentries:
                # Check for rapid exit-reentry cycles (within 5 days)
                rapid = 0
                for e in events:
                    for r in reentries:
                        if 0 < (r["date"] - e["date"]).days <= 5:
                            rapid += 1
                            break
                print(f"    Rapid exit-reentry cycles (within 5 trading days): {rapid}")

            # Whipsaw cost: compare with daily-no-reentry
            p_no = perfs.get("S40-Daily-Daily")
            p_re = perfs.get("S40-Daily-Daily-Reentry")
            if p_no and p_re:
                print(f"\n    Impact of daily re-entry:")
                print(f"      Sharpe: {p_no['sh']:.3f} → {p_re['sh']:.3f} ({p_re['sh']-p_no['sh']:+.3f})")
                print(f"      Return: {p_no['ar']:.1%} → {p_re['ar']:.1%}")
                print(f"      Terminal: ${p_no['final']:.2f} → ${p_re['final']:.2f} (${p_re['final']-p_no['final']:+.2f})")
                print(f"      MaxDD: {p_no['dd']:.1%} → {p_re['dd']:.1%}")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  KEY QUESTIONS")
    print(f"{'='*120}")

    p_w = perfs.get("S40-Daily-Weekly")
    p_d = perfs.get("S40-Daily-Daily")
    p_r = perfs.get("S40-Daily-Daily-Reentry")

    if p_w and p_d:
        print(f"\n  Q1. Daily vs weekly Sharpe: {p_w['sh']:.3f} → {p_d['sh']:.3f} ({p_d['sh']-p_w['sh']:+.3f})")
        print(f"  Q2. Daily vs weekly MaxDD:  {p_w['dd']:.1%} → {p_d['dd']:.1%} ({(p_d['dd']-p_w['dd'])*100:+.1f}%)")
        n_w = len(state["S40-Daily-Weekly"]["breaker_events"])
        n_d = len(state["S40-Daily-Daily"]["breaker_events"])
        print(f"  Q3. Additional triggers: weekly={n_w}, daily={n_d} ({n_d-n_w:+d} more)")

    if p_r:
        print(f"  Q5. Daily re-entry value: Sharpe {p_r['sh']:.3f}, Terminal ${p_r['final']:.2f}")
        if p_d:
            print(f"      vs no-reentry: {p_r['sh']-p_d['sh']:+.3f} Sharpe, ${p_r['final']-p_d['final']:+.2f} terminal")

    # Final verdict
    if p_w and p_d:
        if p_d["sh"] > p_w["sh"] and p_d["final"] > p_w["final"]:
            print(f"\n  VERDICT: Daily circuit breaker IMPROVES both Sharpe and terminal → adopt")
        elif p_d["sh"] > p_w["sh"] and p_d["final"] <= p_w["final"]:
            print(f"\n  VERDICT: Daily breaker improves Sharpe but reduces terminal → tradeoff")
        elif p_d["sh"] <= p_w["sh"]:
            print(f"\n  VERDICT: Daily breaker does NOT improve Sharpe → keep weekly")

    print()
    return perfs


if __name__ == "__main__":
    print("=" * 120)
    print("  DAILY vs WEEKLY CIRCUIT BREAKER ON FABER-SWEEP-40")
    print("=" * 120)
    print(f"  Circuit breaker: price below ALL 3 daily SMAs ({DAILY_SMA_PERIODS}) → exit leverage")
    print(f"  Weekly: check Fridays only. Daily: check every close.")

    print(f"\n  Loading data...")
    daily_ret, trend_df_monthly, rfr_daily, daily_prices_df, daily_smas = load_data()

    ret_series, state = run_backtest(daily_ret, trend_df_monthly, rfr_daily, daily_prices_df, daily_smas)
    perfs = report(ret_series, state)

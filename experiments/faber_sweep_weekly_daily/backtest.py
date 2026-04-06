"""Faber-Sweep-40 with daily SMAs and weekly leverage circuit breaker.

Four strategies:
1. Faber-Sweep-40-Monthly     — monthly SMA signals, monthly rebalance (control)
2. Faber-Sweep-40-Daily-Monthly — daily SMA signals (126/200/252), monthly rebalance only
3. Faber-Sweep-40-Daily-Weekly  — daily SMA signals + weekly circuit breaker (target)
4. Faber-1x-Daily-Weekly        — same as #3 but no leverage (isolation test)

Daily SMAs: 126-day (~6mo), 200-day (~10mo), 252-day (~12mo)
Circuit breaker: Friday close, if either IVV or QQQ below ALL 3 daily SMAs → exit leverage Monday.
Re-entry: next monthly rebalance only.
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


# ── Data ─────────────────────────────────────────────────────────────────────

def load_data():
    import yfinance as yf

    daily_ret = load_daily_etf_returns()
    daily_ret = daily_ret[[c for c in daily_ret.columns if c in ASSETS]]
    monthly_prices = load_monthly_prices()
    # Monthly SMA trend scores (for Strategy 1 — the control)
    trend_df_monthly = compute_trend_scores(monthly_prices)

    from fredapi import Fred
    rfr_daily = pd.Series(0.0, index=daily_ret.index)
    key = os.environ.get("FRED_API_KEY")
    if key:
        tb = Fred(api_key=key).get_series("DTB3", observation_start="1998-01-01")
        tb.index = pd.to_datetime(tb.index)
        rfr_daily = (tb / 100 / 252).reindex(daily_ret.index, method="ffill").fillna(0)

    # Daily prices for all risky assets (for daily SMA computation)
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

    # Precompute daily SMAs for all assets
    daily_smas = {}
    for period in DAILY_SMA_PERIODS:
        daily_smas[period] = daily_prices_df.rolling(period, min_periods=period).mean()

    return daily_ret, trend_df_monthly, rfr_daily, daily_prices_df, daily_smas


# ── Daily SMA scoring ──────────────────────────────────────────────────��─────

def daily_sma_scores(day, daily_prices_df, daily_smas):
    """Compute trend score (0-3) for each risky asset using daily SMAs as of `day`.

    Score = number of daily SMAs the closing price is above.
    PIT safe: only uses data through `day`.
    """
    scores = {}
    for asset in ["IVV", "QQQ", "VGLT", "IAU", "DBC"]:
        if asset not in daily_prices_df.columns:
            scores[asset] = 0
            continue
        prices_to_day = daily_prices_df.loc[:day, asset]
        if len(prices_to_day) == 0 or pd.isna(prices_to_day.iloc[-1]):
            scores[asset] = 0
            continue
        price = prices_to_day.iloc[-1]
        score = 0
        for period in DAILY_SMA_PERIODS:
            sma_df = daily_smas[period]
            sma_to_day = sma_df.loc[:day, asset]
            if len(sma_to_day) == 0:
                continue
            sma_val = sma_to_day.iloc[-1]
            if pd.notna(sma_val) and price > sma_val:
                score += 1
        scores[asset] = score
    return scores


def check_weekly_breach_daily(day, daily_prices_df, daily_smas):
    """Check if either IVV or QQQ is below ALL 3 daily SMAs.

    Returns (breach, detail_string).
    """
    for etf in ["IVV", "QQQ"]:
        if etf not in daily_prices_df.columns:
            continue
        prices_to_day = daily_prices_df.loc[:day, etf]
        if len(prices_to_day) == 0:
            continue
        price = prices_to_day.iloc[-1]

        breaches = 0
        for period in DAILY_SMA_PERIODS:
            sma_df = daily_smas[period]
            sma_to_day = sma_df.loc[:day, etf]
            if len(sma_to_day) == 0:
                continue
            sma_val = sma_to_day.iloc[-1]
            if pd.notna(sma_val) and price < sma_val:
                breaches += 1

        if breaches >= 3:
            return True, f"{etf} below all 3 daily SMAs ({price:.1f})"

    return False, ""


# ── Leveraged return ─────────────────────────────────────────────────────────

def lev_return(iw, qw, ir, qr, rfr, sub, base_ret):
    if sub <= 0:
        return iw * ir + qw * qr + base_ret
    sso = 2.0 * ir - rfr - SSO_EXPENSE / 252
    qld = 2.0 * qr - rfr - QLD_EXPENSE / 252
    return iw * ((1 - sub) * ir + sub * sso) + qw * ((1 - sub) * qr + sub * qld) + base_ret


# ── Backtest ─────────────────────────────────────────────────────────────────

def run_backtest(daily_ret, trend_df_monthly, rfr_daily, daily_prices_df, daily_smas):
    print(f"\n{'='*120}")
    print(f"  BACKTEST")
    print(f"{'='*120}")

    bt_start = pd.Timestamp("2002-01-01")
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:].index
    print(f"  {len(trading_days)} trading days ({common_start.date()} to {trading_days.max().date()})")

    # State per strategy
    # S1: Monthly SMA, monthly rebalance (control)
    # S2: Daily SMA, monthly rebalance only
    # S3: Daily SMA, monthly + weekly circuit breaker
    # S4: Daily SMA + weekly breaker, no leverage (1x)

    strats = {
        "S40-Monthly": {"sub": FABER_SUB, "use_daily_sma": False, "weekly_breaker": False},
        "S40-Daily-Monthly": {"sub": FABER_SUB, "use_daily_sma": True, "weekly_breaker": False},
        "S40-Daily-Weekly": {"sub": FABER_SUB, "use_daily_sma": True, "weekly_breaker": True},
        "1x-Daily-Weekly": {"sub": 0.0, "use_daily_sma": True, "weekly_breaker": True},
    }

    results = {s: {} for s in strats}
    results["IVV B&H"] = {}
    results["60/40"] = {}

    state = {}
    for s_name, cfg in strats.items():
        state[s_name] = {
            "w_faber": dict(BASELINE),
            "faber_conv": False,
            "delevered_this_month": False,
            "breaker_events": [],
            "cur_trends": {a: 3 for a in ASSETS if a != "cash"},
            # Track signal agreement
            "monthly_signals": [],
        }

    total_months = 0

    for day in trading_days:
        if day not in daily_ret.index:
            continue
        dr = daily_ret.loc[day]
        avail = [a for a in ASSETS if a in dr.index and pd.notna(dr[a])]
        if len(avail) < 3:
            continue
        actual = {a: float(dr[a]) for a in avail}
        rfr = float(rfr_daily.get(day, 0))

        is_ms = (day == trading_days[0] or
                 day.month != trading_days[trading_days.get_loc(day) - 1].month)
        is_friday = day.weekday() == 4

        if is_ms:
            total_months += 1

            for s_name, cfg in strats.items():
                st = state[s_name]
                st["delevered_this_month"] = False

                if cfg["use_daily_sma"]:
                    # Use daily SMAs as of last trading day of prior month
                    # PIT: scores through day before month start
                    prior_day_idx = trading_days[trading_days < day]
                    if len(prior_day_idx) > 0:
                        score_day = prior_day_idx[-1]
                    else:
                        score_day = day
                    st["cur_trends"] = daily_sma_scores(score_day, daily_prices_df, daily_smas)
                else:
                    # Monthly SMAs (original Faber)
                    ts_c = trend_df_monthly.index[trend_df_monthly.index <= day]
                    if len(ts_c) > 0:
                        for a in ASSETS:
                            if a == "cash": continue
                            if a in trend_df_monthly.columns:
                                v = trend_df_monthly.loc[ts_c[-1], a]
                                st["cur_trends"][a] = int(v) if pd.notna(v) else 0

                # Apply Faber filter
                w1, pool = apply_faber_filter(st["cur_trends"], BASELINE)
                st["w_faber"] = dict(w1)
                st["w_faber"]["cash"] = st["w_faber"].get("cash", 0) + pool

                # Leverage condition
                st["faber_conv"] = (st["cur_trends"].get("IVV", 0) >= 3 and
                                     st["cur_trends"].get("QQQ", 0) >= 3)

                # Record signal for agreement analysis
                st["monthly_signals"].append({
                    "date": day,
                    "ivv_score": st["cur_trends"].get("IVV", 0),
                    "qqq_score": st["cur_trends"].get("QQQ", 0),
                    "faber_conv": st["faber_conv"],
                    "all_scores": dict(st["cur_trends"]),
                })

        # Weekly circuit breaker (Friday)
        if is_friday:
            for s_name, cfg in strats.items():
                st = state[s_name]
                if not cfg["weekly_breaker"]:
                    continue
                if not st["faber_conv"] or st["delevered_this_month"]:
                    continue

                breach, detail = check_weekly_breach_daily(day, daily_prices_df, daily_smas)
                if breach:
                    st["breaker_events"].append({
                        "date": day, "detail": detail,
                        "month": day.strftime("%Y-%m"),
                    })
                    st["faber_conv"] = False  # exit leverage
                    st["delevered_this_month"] = True

        # Compute daily returns
        for s_name, cfg in strats.items():
            st = state[s_name]
            iw = st["w_faber"].get("IVV", 0)
            qw = st["w_faber"].get("QQQ", 0)
            ir = actual.get("IVV", 0)
            qr = actual.get("QQQ", 0)
            base_ret = sum(st["w_faber"].get(a, 0) * actual.get(a, 0)
                           for a in avail if a not in ["IVV", "QQQ"])

            active_sub = cfg["sub"] if st["faber_conv"] else 0.0
            results[s_name][day] = lev_return(iw, qw, ir, qr, rfr, active_sub, base_ret)

        # Benchmarks
        results["IVV B&H"][day] = actual.get("IVV", 0)
        vglt_r = actual.get("VGLT", 0)
        results["60/40"][day] = 0.60 * actual.get("IVV", 0) + 0.40 * vglt_r

    ret_series = {s: pd.Series(d).sort_index() for s, d in results.items()}
    return ret_series, state, total_months


# ── Metrics ──────────────────────────────────────────────────────────────────

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


# ── Report ───────────────────────────────────────────────────────────────────

def report(ret_series, state, total_months):
    strat_order = ["S40-Monthly", "S40-Daily-Monthly", "S40-Daily-Weekly",
                    "1x-Daily-Weekly", "IVV B&H", "60/40"]
    labels = {
        "S40-Monthly": "Faber-Sweep-40-Monthly",
        "S40-Daily-Monthly": "Faber-Sweep-40-Daily-Monthly",
        "S40-Daily-Weekly": "Faber-Sweep-40-Daily-Weekly",
        "1x-Daily-Weekly": "Faber-1x-Daily-Weekly",
        "IVV B&H": "IVV B&H",
        "60/40": "60/40",
    }

    perfs = {s: compute_metrics(ret_series[s]) for s in strat_order}
    p_ctrl = perfs.get("S40-Monthly", {})

    # ── Performance table ────────────────────────────────────────────────
    print(f"\n{'='*130}")
    print(f"  FULL PERFORMANCE TABLE")
    print(f"{'='*130}")
    print(f"  {'Strategy':<34} {'Return':>8} {'Vol':>7} {'Sharpe':>8} {'Sortino':>8} "
          f"{'MaxDD':>8} {'Calmar':>8} {'Terminal($1)':>13} {'vs Monthly':>11}")
    print(f"  {'-'*34} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*13} {'-'*11}")

    for s in strat_order:
        p = perfs.get(s)
        if not p: continue
        vs = p["final"] - p_ctrl.get("final", 0) if s != "IVV B&H" and s != "60/40" else 0
        vs_str = f"{vs:>+10.2f}" if s not in ["IVV B&H", "60/40"] else f"{'—':>11}"
        print(f"  {labels[s]:<34} {p['ar']:>7.1%} {p['av']:>6.1%} {p['sh']:>8.3f} {p['sortino']:>8.3f} "
              f"{p['dd']:>7.1%} {p['calmar']:>8.2f} ${p['final']:>12.2f} {vs_str}")

    # ── Crisis analysis ──────────────────────────────────────────────────
    print(f"\n{'='*130}")
    print(f"  CRISIS ANALYSIS")
    print(f"{'='*130}")

    for cname, cs, ce in [("GFC (2008-09)", "2008-09-01", "2009-03-31"),
                           ("COVID (Feb-Mar 2020)", "2020-02-19", "2020-03-23"),
                           ("2022 Bear", "2022-01-03", "2022-10-31")]:
        print(f"\n  {cname}:")
        print(f"  {'Strategy':<34} {'Return':>10} {'MaxDD':>10}")
        print(f"  {'-'*34} {'-'*10} {'-'*10}")
        for s in strat_order:
            sr = ret_series[s]
            c = sr[(sr.index >= pd.Timestamp(cs)) & (sr.index <= pd.Timestamp(ce))]
            if len(c) > 0:
                cum = (1 + c).cumprod()
                mdd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
                print(f"  {labels[s]:<34} {(1+c).prod()-1:>+9.1%} {mdd:>9.1%}")

    # ── Circuit breaker diagnostics ──────────────────────────────────────
    print(f"\n{'='*130}")
    print(f"  CIRCUIT BREAKER DIAGNOSTICS (Faber-Sweep-40-Daily-Weekly)")
    print(f"{'='*130}")

    events = state["S40-Daily-Weekly"]["breaker_events"]
    years = (pd.Timestamp("2026-04-01") - pd.Timestamp("2002-01-01")).days / 365.25
    print(f"\n  Total de-lever events: {len(events)} over {years:.0f} years ({len(events)/years:.1f}/year)")

    if events:
        print(f"\n  {'Date':>12} {'Detail'}")
        print(f"  {'-'*12} {'-'*50}")
        for e in events:
            print(f"  {e['date'].strftime('%Y-%m-%d'):>12} {e['detail']}")

        # Crisis trigger behavior
        for cname, cs, ce in [("GFC 2008", "2007-10-01", "2009-06-30"),
                               ("COVID 2020", "2020-01-01", "2020-06-30"),
                               ("2022 Bear", "2021-12-01", "2023-01-01"),
                               ("2018 Q4", "2018-09-01", "2019-01-01")]:
            crisis_events = [e for e in events
                             if pd.Timestamp(cs) <= e["date"] <= pd.Timestamp(ce)]
            if crisis_events:
                print(f"\n  {cname}: triggered {len(crisis_events)} time(s)")
                for e in crisis_events:
                    print(f"    {e['date'].strftime('%Y-%m-%d')}: {e['detail']}")
            else:
                print(f"\n  {cname}: no trigger")

        # False trigger analysis
        crisis_dates = set()
        for cs, ce in [("2007-10-01", "2009-06-30"), ("2020-01-01", "2020-06-30"),
                        ("2021-12-01", "2023-01-01"), ("2018-09-01", "2019-01-01"),
                        ("2015-07-01", "2016-03-01")]:
            for e in events:
                if pd.Timestamp(cs) <= e["date"] <= pd.Timestamp(ce):
                    crisis_dates.add(e["date"])

        non_crisis = [e for e in events if e["date"] not in crisis_dates]
        print(f"\n  False triggers (outside major crisis periods): {len(non_crisis)}")
        if non_crisis:
            # Check return in month following trigger
            s_daily = ret_series["S40-Daily-Weekly"]
            for e in non_crisis[:5]:
                month_after = s_daily[(s_daily.index > e["date"]) &
                                       (s_daily.index <= e["date"] + pd.DateOffset(months=1))]
                if len(month_after) > 0:
                    ret_after = (1 + month_after).prod() - 1
                    print(f"    {e['date'].strftime('%Y-%m-%d')}: {e['detail']} → return next month: {ret_after:+.1%}")

    # ── Signal agreement analysis ────────────────────────────────────────
    print(f"\n{'='*130}")
    print(f"  DAILY vs MONTHLY SMA SIGNAL AGREEMENT")
    print(f"{'='*130}")

    monthly_sigs = state["S40-Monthly"]["monthly_signals"]
    daily_sigs = state["S40-Daily-Monthly"]["monthly_signals"]

    if len(monthly_sigs) == len(daily_sigs):
        agree_all = 0
        disagree_any = 0
        disagree_lev = 0

        for ms, ds in zip(monthly_sigs, daily_sigs):
            all_agree = True
            for a in ["IVV", "QQQ", "VGLT", "IAU", "DBC"]:
                m_score = ms["all_scores"].get(a, 0)
                d_score = ds["all_scores"].get(a, 0)
                # Compare: both >= 3, both == 2, both <= 1
                m_cat = 2 if m_score >= 3 else (1 if m_score == 2 else 0)
                d_cat = 2 if d_score >= 3 else (1 if d_score == 2 else 0)
                if m_cat != d_cat:
                    all_agree = False
                    break

            if all_agree:
                agree_all += 1
            else:
                disagree_any += 1

            if ms["faber_conv"] != ds["faber_conv"]:
                disagree_lev += 1

        total = len(monthly_sigs)
        print(f"\n  Months where daily and monthly SMAs agree on all assets: {agree_all}/{total} ({agree_all/total:.0%})")
        print(f"  Months where they disagree on at least one asset:        {disagree_any}/{total} ({disagree_any/total:.0%})")
        print(f"  Months where they disagree on leverage (IVV+QQQ 3/3):   {disagree_lev}/{total} ({disagree_lev/total:.0%})")

    # Return difference from SMA granularity
    p_monthly = perfs.get("S40-Monthly")
    p_daily_mo = perfs.get("S40-Daily-Monthly")
    if p_monthly and p_daily_mo:
        print(f"\n  Return difference from SMA granularity: {(p_daily_mo['ar']-p_monthly['ar'])*100:+.2f}% annualized")

    # ── Decomposition ────────────────────────────────────────────────────
    print(f"\n{'='*130}")
    print(f"  SHARPE AND RETURN DECOMPOSITION")
    print(f"{'='*130}")

    p_m = perfs.get("S40-Monthly")
    p_dm = perfs.get("S40-Daily-Monthly")
    p_dw = perfs.get("S40-Daily-Weekly")

    if p_m and p_dm:
        print(f"\n  Effect of daily SMAs (vs monthly SMAs):")
        print(f"    Sharpe change:   {p_dm['sh']-p_m['sh']:+.3f}")
        print(f"    Return change:   {(p_dm['ar']-p_m['ar'])*100:+.2f}%")
        print(f"    MaxDD change:    {(p_dm['dd']-p_m['dd'])*100:+.1f}%")
        print(f"    Terminal change:  ${p_dm['final']-p_m['final']:+.2f}")

    if p_dm and p_dw:
        print(f"\n  Effect of weekly circuit breaker (vs daily-SMA monthly-only):")
        print(f"    Sharpe change:   {p_dw['sh']-p_dm['sh']:+.3f}")
        print(f"    Return change:   {(p_dw['ar']-p_dm['ar'])*100:+.2f}%")
        print(f"    MaxDD change:    {(p_dw['dd']-p_dm['dd'])*100:+.1f}%")
        print(f"    Terminal change:  ${p_dw['final']-p_dm['final']:+.2f}")

    if p_m and p_dw:
        print(f"\n  Combined effect (daily SMAs + weekly breaker vs monthly baseline):")
        print(f"    Sharpe change:   {p_dw['sh']-p_m['sh']:+.3f}")
        print(f"    Return change:   {(p_dw['ar']-p_m['ar'])*100:+.2f}%")
        print(f"    MaxDD change:    {(p_dw['dd']-p_m['dd'])*100:+.1f}%")
        print(f"    Terminal change:  ${p_dw['final']-p_m['final']:+.2f}")

    # ── Calendar year returns ────────────────────────────────────────────
    print(f"\n{'='*130}")
    print(f"  CALENDAR YEAR RETURNS")
    print(f"{'='*130}")
    cal_strats = ["S40-Monthly", "S40-Daily-Monthly", "S40-Daily-Weekly", "1x-Daily-Weekly", "IVV B&H"]
    header = f"  {'Year':>6}"
    for s in cal_strats:
        header += f" {labels[s][-15:]:>16}"
    print(header)

    for yr in range(2002, 2027):
        row = f"  {yr:>6}"
        for s in cal_strats:
            sr = ret_series[s]
            y = sr[sr.index.year == yr]
            if len(y) > 20:
                row += f" {(1+y).prod()-1:>+15.1%}"
            else:
                row += f" {'--':>16}"
        print(row)

    print()
    return perfs


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 120)
    print("  FABER-SWEEP-40: DAILY SMAs + WEEKLY CIRCUIT BREAKER")
    print("=" * 120)
    print(f"  Daily SMA periods: {DAILY_SMA_PERIODS}")
    print(f"  Circuit breaker: Friday close, either IVV or QQQ below ALL 3 daily SMAs → exit leverage Monday")
    print(f"  Leverage: 40% SSO/QLD substitution when both IVV+QQQ at 3/3")

    print(f"\n  Loading data...")
    daily_ret, trend_df_monthly, rfr_daily, daily_prices_df, daily_smas = load_data()
    print(f"  Daily prices: {daily_prices_df.index.min().date()} to {daily_prices_df.index.max().date()}")
    print(f"  Daily returns: {daily_ret.index.min().date()} to {daily_ret.index.max().date()}")

    ret_series, state, total_months = run_backtest(
        daily_ret, trend_df_monthly, rfr_daily, daily_prices_df, daily_smas)

    perfs = report(ret_series, state, total_months)

"""Leverage calibration on simplified Faber-only architecture.

Recalibrates graduated leverage tiers for Faber-only (no Harvey/macro engine).
Faber-only runs at 7.4% vol vs ~9.5% with Harvey. Tier substitution percentages
are derived from risk tolerance targets, not optimized from backtest results.

Strategies:
1. Faber-1x: No leverage (control)
2. Faber-Tier1-Only: Tier 1 only, monthly rebalance
3. Faber-Graduated-Monthly: Tier 1 + Tier 2, monthly rebalance
4. Faber-Graduated-Weekly: Tier 1 + Tier 2 with weekly circuit breaker
5. Faber-Sweep-1.25x: Flat 1.25x when both equities at 3/3
6. Faber-Sweep-1.5x: Flat 1.5x when both equities at 3/3

Plus leverage sweep from 1.0x to 2.5x in 0.1x steps.
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv; load_dotenv()

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

from taa.data import (load_daily_etf_returns, load_monthly_prices,
                       load_monthly_asset_returns)
from taa.faber import compute_trend_scores, apply_faber_filter

BASELINE = {"IVV": 0.45, "QQQ": 0.25, "VGLT": 0.05, "IAU": 0.10, "DBC": 0.05, "cash": 0.10}
ASSETS = list(BASELINE.keys())

# Leveraged ETF expense ratios (current rates)
SSO_EXPENSE = 0.0089  # annualized
QLD_EXPENSE = 0.0095  # annualized


# ════════��═════════════════════════════════════════════════════════════════════
# STEP 1: TIER SUBSTITUTION MATH
# ���═════════════════════════════════════════════════════════════════════════════

def compute_tier_substitutions():
    """Derive substitution percentages from risk tolerance targets.

    When both IVV and QQQ are at 3/3 Faber:
      IVV weight = 45%, QQQ weight = 25%, total equity = 70%

    Effective equity exposure = equity_weight * (1 + sub)
    where sub = fraction of each equity position replaced by 2x leveraged ETF.

    Target A (Tier 1): ~98% effective equity exposure
    Target B (Tier 2): ~127% effective equity exposure
    """
    equity_weight = 0.45 + 0.25  # 70%

    # Tier 1: 98% effective equity
    target_t1 = 0.98
    sub_t1 = (target_t1 / equity_weight) - 1.0

    # Tier 2: 127% effective equity
    target_t2 = 1.27
    sub_t2 = (target_t2 / equity_weight) - 1.0

    return sub_t1, sub_t2, equity_weight


def print_tier_math():
    sub_t1, sub_t2, eq_w = compute_tier_substitutions()

    print(f"\n{'='*90}")
    print(f"  STEP 1: TIER SUBSTITUTION PERCENTAGES")
    print(f"{'='*90}")
    print(f"\n  Faber-only average equity weights when both at 3/3: IVV=45.0%, QQQ=25.0%")
    print(f"  Total equity weight: {eq_w:.0%}")
    print(f"\n  Effective equity = equity_weight * (1 + sub)")
    print(f"  where sub = fraction of each equity position replaced by 2x leveraged ETF")

    print(f"\n  Target effective equity (Tier 1): ~98%")
    print(f"  Required substitution for Tier 1: {sub_t1:.1%} of each equity position -> SSO/QLD")
    eff_t1 = eq_w * (1 + sub_t1)
    print(f"  Resulting effective equity check: {eq_w:.0%} * (1 + {sub_t1:.3f}) = {eff_t1:.1%}")
    print(f"    IVV: {0.45*(1-sub_t1):.1%} unleveraged + {0.45*sub_t1:.1%} in SSO (2x) = {0.45*(1+sub_t1):.1%} effective")
    print(f"    QQQ: {0.25*(1-sub_t1):.1%} unleveraged + {0.25*sub_t1:.1%} in QLD (2x) = {0.25*(1+sub_t1):.1%} effective")

    print(f"\n  Target effective equity (Tier 2): ~127%")
    print(f"  Required substitution for Tier 2: {sub_t2:.1%} of each equity position -> SSO/QLD")
    eff_t2 = eq_w * (1 + sub_t2)
    print(f"  Resulting effective equity check: {eq_w:.0%} * (1 + {sub_t2:.3f}) = {eff_t2:.1%}")
    print(f"    IVV: {0.45*(1-sub_t2):.1%} unleveraged + {0.45*sub_t2:.1%} in SSO (2x) = {0.45*(1+sub_t2):.1%} effective")
    print(f"    QQQ: {0.25*(1-sub_t2):.1%} unleveraged + {0.25*sub_t2:.1%} in QLD (2x) = {0.25*(1+sub_t2):.1%} effective")

    # Flat sweep reference points
    sub_125 = 0.25
    sub_150 = 0.50
    print(f"\n  Reference — Flat 1.25x: sub={sub_125:.0%}, eff equity = {eq_w*(1+sub_125):.1%}")
    print(f"  Reference — Flat 1.50x: sub={sub_150:.0%}, eff equity = {eq_w*(1+sub_150):.1%}")

    print(f"\n  Leveraged ETF simulation: 2.0 * daily_return - rfr/252 - expense/252")
    print(f"  SSO expense: {SSO_EXPENSE:.4f} ann ({SSO_EXPENSE/252*10000:.2f} bps/day)")
    print(f"  QLD expense: {QLD_EXPENSE:.4f} ann ({QLD_EXPENSE/252*10000:.2f} bps/day)")

    return sub_t1, sub_t2


# ═══════════════════════════════════════════════════════��══════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════��═══════════════════════

def load_data():
    """Load all required data."""
    import yfinance as yf

    daily_ret = load_daily_etf_returns()
    daily_ret = daily_ret[[c for c in daily_ret.columns if c in ASSETS]]
    monthly_prices = load_monthly_prices()
    asset_ret = load_monthly_asset_returns()
    trend_df = compute_trend_scores(monthly_prices)

    # Risk-free rate
    from fredapi import Fred
    rfr_daily = pd.Series(0.0, index=daily_ret.index)
    key = os.environ.get("FRED_API_KEY")
    if key:
        tb = Fred(api_key=key).get_series("DTB3", observation_start="1998-01-01")
        tb.index = pd.to_datetime(tb.index)
        rfr_daily = (tb / 100 / 252).reindex(daily_ret.index, method="ffill").fillna(0)

    # Daily prices for weekly SMA checks
    daily_prices = {}
    for our, ticker in [("IVV", "SPY"), ("QQQ", "QQQ")]:
        d = yf.download(ticker, start="1998-01-01", progress=False)
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p, "columns"):
                p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None)
            daily_prices[our] = p
    daily_prices_df = pd.DataFrame(daily_prices)

    # Monthly SMAs for weekly circuit breaker
    monthly_close = daily_prices_df.resample("MS").last()
    sma_monthly = {}
    for period in [6, 10, 12]:
        sma_monthly[period] = monthly_close.rolling(period, min_periods=period).mean()

    # Monthly returns for trailing 12-month momentum (Tier 2 condition)
    monthly_ret = asset_ret[["IVV", "QQQ"]].copy()

    return (daily_ret, monthly_prices, asset_ret, trend_df, rfr_daily,
            daily_prices_df, sma_monthly, monthly_ret)


# ════════════════��═════════════════════════════════════════════════════════════
# LEVERAGED RETURN COMPUTATION
# ═══════════════════════════════════════════════════════��══════════════════════

def compute_leveraged_daily_return(iw, qw, ir, qr, rfr, sub, base_ret):
    """Compute daily portfolio return with leverage substitution.

    sub: fraction of each equity position replaced by 2x leveraged ETF.
    """
    if sub <= 0:
        return iw * ir + qw * qr + base_ret

    # Synthetic leveraged returns
    sso_ret = 2.0 * ir - rfr - SSO_EXPENSE / 252
    qld_ret = 2.0 * qr - rfr - QLD_EXPENSE / 252

    # Portfolio: (1-sub) in unleveraged + sub in leveraged
    ivv_contrib = iw * ((1 - sub) * ir + sub * sso_ret)
    qqq_contrib = qw * ((1 - sub) * qr + sub * qld_ret)

    return ivv_contrib + qqq_contrib + base_ret


# ══════════════════════════════════════════════════════════════════════════════
# WEEKLY CIRCUIT BREAKER
# ═══════════════════════════���══════════════════════════════════════════════════

def check_weekly_breach(day, daily_prices_df, sma_monthly):
    """Check if either IVV or QQQ has broken ALL 3 monthly SMAs.

    Returns (breach, detail_string).
    """
    for etf in ["IVV", "QQQ"]:
        if etf not in daily_prices_df.columns:
            continue
        prices_to_day = daily_prices_df.loc[:day, etf]
        if len(prices_to_day) == 0:
            continue
        price = prices_to_day.iloc[-1]

        # Count how many SMAs the price is below
        breaches = 0
        ms = pd.Timestamp(f"{day.year}-{day.month:02d}-01")
        for period in [6, 10, 12]:
            sma_df = sma_monthly[period]
            sma_dates = sma_df.index[sma_df.index <= ms]
            if len(sma_dates) == 0:
                continue
            sma_val = sma_df.loc[sma_dates[-1], etf]
            if pd.notna(sma_val) and price < sma_val:
                breaches += 1

        if breaches >= 3:  # ALL 3 SMAs breached
            return True, f"{etf} below all 3 SMAs ({price:.1f})"

    return False, ""


# ══════════════════════════════════════════════════════════════════════════════
# TIER 2 MOMENTUM CONDITION
# ══════════════════════════════════════════════════════════════════════════════

def compute_trailing_12m_return(monthly_ret, dt):
    """Compute trailing 12-month return for IVV and QQQ as of date dt.

    Uses monthly returns. PIT safe — only uses returns through prior month.
    """
    # Get months prior to dt
    prior_months = monthly_ret.index[monthly_ret.index < dt]
    if len(prior_months) < 12:
        return None, None

    last_12 = prior_months[-12:]
    ivv_12m = (1 + monthly_ret.loc[last_12, "IVV"]).prod() - 1
    qqq_12m = (1 + monthly_ret.loc[last_12, "QQQ"]).prod() - 1
    return float(ivv_12m), float(qqq_12m)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(sub_t1, sub_t2, data):
    (daily_ret, monthly_prices, asset_ret, trend_df, rfr_daily,
     daily_prices_df, sma_monthly, monthly_ret) = data

    print(f"\n{'='*90}")
    print(f"  STEP 3: BACKTEST")
    print(f"{'='*90}")

    bt_start = pd.Timestamp("2002-01-01")
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:].index

    print(f"\n  Backtest: {len(trading_days)} days ({common_start.date()} to {trading_days.max().date()})")
    print(f"  Tier 1 sub: {sub_t1:.1%}, Tier 2 sub: {sub_t2:.1%}")

    # ── Pre-backtest seed for expanding median (trailing 12m returns) ────
    pre_bt_months = monthly_ret.index[monthly_ret.index < bt_start]
    seed_returns = {"IVV": [], "QQQ": []}
    for i in range(12, len(pre_bt_months)):
        window = pre_bt_months[i-12:i]
        for a in ["IVV", "QQQ"]:
            tr = (1 + monthly_ret.loc[window, a]).prod() - 1
            if pd.notna(tr):
                seed_returns[a].append(tr)

    seed_med = {
        "IVV": float(np.mean(seed_returns["IVV"])) if seed_returns["IVV"] else 0.10,
        "QQQ": float(np.mean(seed_returns["QQQ"])) if seed_returns["QQQ"] else 0.10,
    }
    print(f"  Tier 2 seed medians: IVV={seed_med['IVV']:.3f}, QQQ={seed_med['QQQ']:.3f}")
    print(f"    (from {len(seed_returns['IVV'])} pre-BT 12m return windows)")

    # ── Strategy definitions ─────────────────────────────────────────────
    # Each strategy: (name, sub_t1, sub_t2, use_weekly_check)
    strategies = {
        "Faber-1x":                 (0.0,    0.0,    False),
        "Faber-Tier1-Only":         (sub_t1, 0.0,    False),  # Tier 1 only, no Tier 2
        "Faber-Graduated-Monthly":  (sub_t1, sub_t2, False),
        "Faber-Graduated-Weekly":   (sub_t1, sub_t2, True),
        "Faber-Sweep-1.25x":       (0.25,   0.25,   False),  # Flat 1.25x
        "Faber-Sweep-1.5x":        (0.50,   0.50,   False),  # Flat 1.5x
    }

    strat_names = list(strategies.keys()) + ["IVV B&H"]

    # ── State per strategy ───────────────────────────────────────────────
    state = {}
    for name in strategies:
        state[name] = {
            "tier": 0,
            "delevered_this_month": False,
            "trailing_hist": {"IVV": list(seed_returns["IVV"]),
                              "QQQ": list(seed_returns["QQQ"])},
            "trailing_med": dict(seed_med),
            "weekly_events": [],
        }

    results = {s: {} for s in strat_names}
    tier_log = {s: [] for s in strategies}
    eff_equity_log = {s: [] for s in strategies}

    # Faber-only weights (no Harvey)
    cur_trends = {a: 3 for a in ASSETS if a != "cash"}
    w_faber = dict(BASELINE)
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

            # ── Update Faber signals (PIT safe) ──────────────────────────
            ts_c = trend_df.index[trend_df.index <= day]
            if len(ts_c) > 0:
                for a in ASSETS:
                    if a == "cash":
                        continue
                    if a in trend_df.columns:
                        v = trend_df.loc[ts_c[-1], a]
                        cur_trends[a] = int(v) if pd.notna(v) else 0

            # ── Faber-only weights: freed capital to cash ────────────────
            w1, pool = apply_faber_filter(cur_trends, BASELINE)
            w_faber = dict(w1)
            w_faber["cash"] = w_faber.get("cash", 0) + pool

            # ── Tier conditions ──────────────────────────────────────────
            faber_conv = (cur_trends.get("IVV", 0) >= 3 and
                          cur_trends.get("QQQ", 0) >= 3)

            # Trailing 12m return for Tier 2
            ivv_12m, qqq_12m = compute_trailing_12m_return(monthly_ret, day)

            for name, (s_t1, s_t2, use_weekly) in strategies.items():
                s = state[name]
                s["delevered_this_month"] = False

                if not faber_conv:
                    s["tier"] = 0
                else:
                    # Tier 1 always if faber_conv and strategy has leverage
                    if s_t1 > 0:
                        s["tier"] = 1
                    else:
                        s["tier"] = 0

                    # Tier 2 check (only if s_t2 > s_t1, i.e., graduated)
                    if s_t2 > s_t1 and ivv_12m is not None and qqq_12m is not None:
                        above_med = (ivv_12m > s["trailing_med"]["IVV"] and
                                     qqq_12m > s["trailing_med"]["QQQ"])
                        if above_med:
                            s["tier"] = 2

                    # For Tier1-Only: cap at tier 1
                    if name == "Faber-Tier1-Only":
                        s["tier"] = min(s["tier"], 1)

                    # Update expanding median (even for flat strategies)
                    if ivv_12m is not None:
                        s["trailing_hist"]["IVV"].append(ivv_12m)
                        s["trailing_med"]["IVV"] = float(np.median(
                            s["trailing_hist"]["IVV"]))
                    if qqq_12m is not None:
                        s["trailing_hist"]["QQQ"].append(qqq_12m)
                        s["trailing_med"]["QQQ"] = float(np.median(
                            s["trailing_hist"]["QQQ"]))

        # ── Weekly circuit breaker (Friday) ──────────────────────────────
        if is_friday:
            for name, (s_t1, s_t2, use_weekly) in strategies.items():
                s = state[name]
                if use_weekly and s["tier"] > 0 and not s["delevered_this_month"]:
                    breach, detail = check_weekly_breach(
                        day, daily_prices_df, sma_monthly)
                    if breach:
                        s["weekly_events"].append({
                            "date": day, "from_tier": s["tier"],
                            "detail": detail,
                        })
                        s["tier"] = 0
                        s["delevered_this_month"] = True

        # ── Compute daily returns ────────────────────────────────────────
        iw = w_faber.get("IVV", 0)
        qw = w_faber.get("QQQ", 0)
        ir = actual.get("IVV", 0)
        qr = actual.get("QQQ", 0)
        base_ret = sum(w_faber.get(a, 0) * actual.get(a, 0)
                       for a in avail if a not in ["IVV", "QQQ"])

        for name, (s_t1, s_t2, use_weekly) in strategies.items():
            s = state[name]
            tier = s["tier"]

            # Determine substitution for this tier
            if tier == 0:
                sub = 0.0
            elif tier == 1:
                sub = s_t1
            else:  # tier == 2
                sub = s_t2

            ret = compute_leveraged_daily_return(iw, qw, ir, qr, rfr, sub, base_ret)
            results[name][day] = ret
            tier_log[name].append((day, tier))

            # Log effective equity for this day
            eff_eq = (iw + qw) * (1 + sub)
            eff_equity_log[name].append((day, eff_eq, tier))

        # IVV B&H
        results["IVV B&H"][day] = actual.get("IVV", 0)

    ret_series = {s: pd.Series(d).sort_index() for s, d in results.items()}
    return ret_series, tier_log, eff_equity_log, state, strat_names, total_months


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: LEVERAGE SWEEP
# ══════════════════════════════════════════════════════════════════════════════

def run_leverage_sweep(data):
    """Sweep flat leverage from 1.0x to 2.5x in 0.1x steps."""
    (daily_ret, monthly_prices, asset_ret, trend_df, rfr_daily,
     daily_prices_df, sma_monthly, monthly_ret) = data

    print(f"\n{'='*90}")
    print(f"  STEP 4: LEVERAGE SWEEP (1.0x to 2.5x, flat, when IVV+QQQ both at 3/3)")
    print(f"{'='*90}")

    bt_start = pd.Timestamp("2002-01-01")
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:].index

    cur_trends = {a: 3 for a in ASSETS if a != "cash"}
    w_faber = dict(BASELINE)

    levels = np.arange(1.0, 2.55, 0.1)
    sweep_results = {L: {} for L in levels}

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

        if is_ms:
            ts_c = trend_df.index[trend_df.index <= day]
            if len(ts_c) > 0:
                for a in ASSETS:
                    if a == "cash":
                        continue
                    if a in trend_df.columns:
                        v = trend_df.loc[ts_c[-1], a]
                        cur_trends[a] = int(v) if pd.notna(v) else 0

            w1, pool = apply_faber_filter(cur_trends, BASELINE)
            w_faber = dict(w1)
            w_faber["cash"] = w_faber.get("cash", 0) + pool

        iw = w_faber.get("IVV", 0)
        qw = w_faber.get("QQQ", 0)
        ir = actual.get("IVV", 0)
        qr = actual.get("QQQ", 0)
        base_ret = sum(w_faber.get(a, 0) * actual.get(a, 0)
                       for a in avail if a not in ["IVV", "QQQ"])

        faber_conv = (cur_trends.get("IVV", 0) >= 3 and
                      cur_trends.get("QQQ", 0) >= 3)

        for L in levels:
            sub = L - 1.0 if faber_conv else 0.0
            ret = compute_leveraged_daily_return(iw, qw, ir, qr, rfr, sub, base_ret)
            sweep_results[L][day] = ret

    sweep_series = {L: pd.Series(d).sort_index() for L, d in sweep_results.items()}
    return sweep_series, levels


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(s):
    ar = s.mean() * 252
    av = s.std() * np.sqrt(252)
    sh = ar / av if av > 0 else 0
    neg = s[s < 0]
    ds = neg.std() * np.sqrt(252) if len(neg) > 10 else av
    sortino = ar / ds if ds > 0 else 0
    cum = (1 + s).cumprod()
    dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
    calmar = ar / abs(dd) if dd != 0 else 0
    final = cum.iloc[-1]
    return {"ar": ar, "av": av, "sh": sh, "sortino": sortino,
            "dd": dd, "calmar": calmar, "final": final}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def print_report(ret_series, tier_log, eff_equity_log, state,
                 strat_names, total_months, sub_t1, sub_t2,
                 sweep_series, sweep_levels):

    perfs = {s: compute_metrics(ret_series[s]) for s in strat_names}

    # ── Performance table ────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"  PERFORMANCE")
    print(f"{'='*100}")
    print(f"  {'Strategy':<28} {'Return':>8} {'Vol':>7} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'Calmar':>8} {'Terminal':>10}")
    print(f"  {'-'*28} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for s in strat_names:
        p = perfs[s]
        print(f"  {s:<28} {p['ar']:>7.1%} {p['av']:>6.1%} {p['sh']:>8.3f} {p['sortino']:>8.3f} {p['dd']:>7.1%} {p['calmar']:>8.2f} ${p['final']:>9.2f}")

    # ── Crisis analysis ──────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"  CRISIS ANALYSIS")
    print(f"{'='*100}")
    for cname, cs, ce in [("GFC (2008-2009)", "2008-09-01", "2009-03-31"),
                           ("COVID (2020)", "2020-02-19", "2020-03-23"),
                           ("2022 Bear", "2022-01-03", "2022-10-31")]:
        print(f"\n  {cname}:")
        print(f"  {'Strategy':<28} {'Return':>10} {'MaxDD':>10}")
        print(f"  {'-'*28} {'-'*10} {'-'*10}")
        for s in strat_names:
            sr = ret_series[s]
            c = sr[(sr.index >= pd.Timestamp(cs)) & (sr.index <= pd.Timestamp(ce))]
            if len(c) > 0:
                cum = (1 + c).cumprod()
                mdd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
                print(f"  {s:<28} {(1+c).prod()-1:>+9.1%} {mdd:>9.1%}")

    # ── Tier diagnostics ─────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"  TIER DIAGNOSTICS (Faber-Graduated-Weekly)")
    print(f"{'='*100}")

    for strat_label in ["Faber-Graduated-Monthly", "Faber-Graduated-Weekly"]:
        tl = tier_log.get(strat_label, [])
        if not tl:
            continue

        # Count months (first trading day of each month)
        month_tiers = {}
        for d, t in tl:
            ms = pd.Timestamp(f"{d.year}-{d.month:02d}-01")
            if ms not in month_tiers:
                month_tiers[ms] = t

        tier_counts = {0: 0, 1: 0, 2: 0}
        for ms, t in month_tiers.items():
            tier_counts[t] = tier_counts.get(t, 0) + 1

        total = sum(tier_counts.values())
        print(f"\n  {strat_label}:")
        print(f"    Tier 0 (1x):      {tier_counts[0]:>4} months ({tier_counts[0]/total:.0%})")
        print(f"    Tier 1 ({sub_t1:.0%} sub): {tier_counts[1]:>4} months ({tier_counts[1]/total:.0%})")
        print(f"    Tier 2 ({sub_t2:.0%} sub): {tier_counts[2]:>4} months ({tier_counts[2]/total:.0%})")

    # Weekly de-lever events
    weekly_events = state.get("Faber-Graduated-Weekly", {}).get("weekly_events", [])
    print(f"\n  Weekly de-lever events: {len(weekly_events)} across {total_months//12} years "
          f"({len(weekly_events)/(total_months/12):.1f}/year)")
    if weekly_events:
        print(f"\n  {'Date':>12} {'From Tier':>10} {'Detail'}")
        print(f"  {'-'*12} {'-'*10} {'-'*40}")
        for e in weekly_events:
            print(f"  {e['date'].strftime('%Y-%m-%d'):>12} {'Tier '+str(e['from_tier']):>10} {e['detail']}")

    # ── Effective equity exposure ────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"  EFFECTIVE EQUITY EXPOSURE CONFIRMATION")
    print(f"{'='*100}")

    for strat_label in ["Faber-1x", "Faber-Graduated-Monthly", "Faber-Graduated-Weekly"]:
        el = eff_equity_log.get(strat_label, [])
        if not el:
            continue
        eff_df = pd.DataFrame(el, columns=["date", "eff_equity", "tier"]).set_index("date")

        for tier_val, target in [(0, "N/A"), (1, "~98%"), (2, "~127%")]:
            tier_data = eff_df[eff_df["tier"] == tier_val]["eff_equity"]
            if len(tier_data) > 0:
                avg_eff = tier_data.mean()
                target_str = f"(target: {target})" if target != "N/A" else ""
                print(f"  {strat_label:<28} Tier {tier_val}: avg eff equity = {avg_eff:.1%} {target_str}")

    # ── Leverage sweep ───────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"  LEVERAGE SWEEP (flat, when IVV+QQQ both at 3/3)")
    print(f"{'='*100}")
    print(f"\n  {'Level':>7} {'Return':>8} {'Vol':>7} {'Sharpe':>8} {'MaxDD':>8} {'Terminal':>10}")
    print(f"  {'-'*7} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*10}")

    sweep_sharpes = []
    for L in sweep_levels:
        sm = compute_metrics(sweep_series[L])
        marker = " <--" if abs(L - 1.0) < 0.01 else ""
        print(f"  {L:>6.1f}x {sm['ar']:>7.1%} {sm['av']:>6.1%} {sm['sh']:>8.3f} {sm['dd']:>7.1%} ${sm['final']:>9.2f}{marker}")
        sweep_sharpes.append((L, sm['sh']))

    # CML check
    sharpe_1x = [sh for L, sh in sweep_sharpes if abs(L - 1.0) < 0.01][0]
    sharpe_2x = [sh for L, sh in sweep_sharpes if abs(L - 2.0) < 0.01][0]
    sharpe_min = min(sh for _, sh in sweep_sharpes)
    sharpe_max = max(sh for _, sh in sweep_sharpes)
    sharpe_range = sharpe_max - sharpe_min

    print(f"\n  CML check: Sharpe range across sweep = {sharpe_range:.3f}")
    print(f"  1.0x Sharpe: {sharpe_1x:.3f}, 2.0x Sharpe: {sharpe_2x:.3f}, delta: {sharpe_2x - sharpe_1x:+.3f}")
    if sharpe_range < 0.10:
        print(f"  APPROXIMATELY FLAT — consistent with CML theory")
    else:
        print(f"  NOT FLAT — indicates non-linear interaction between leverage and Faber timing")
        # Investigate: is Sharpe declining?
        if sharpe_2x < sharpe_1x - 0.05:
            print(f"  Sharpe DECLINING with leverage — leveraged ETF drag (volatility decay + expense)")
        elif sharpe_2x > sharpe_1x + 0.05:
            print(f"  Sharpe INCREASING with leverage — Faber timing creates positive convexity")

    # ── Calendar year returns ────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"  CALENDAR YEAR RETURNS")
    print(f"{'='*100}")
    cal_strats = ["Faber-1x", "Faber-Graduated-Weekly", "Faber-Sweep-1.25x", "Faber-Sweep-1.5x", "IVV B&H"]
    header = f"  {'Year':>6}"
    for s in cal_strats:
        label = s.replace("Faber-", "")[:12]
        header += f" {label:>13}"
    print(header)

    for yr in range(2002, 2027):
        row = f"  {yr:>6}"
        for s in cal_strats:
            sr = ret_series.get(s, pd.Series(dtype=float))
            y = sr[sr.index.year == yr]
            if len(y) > 20:
                row += f" {(1+y).prod()-1:>+12.1%}"
            else:
                row += f" {'--':>13}"
        print(row)

    print()
    return perfs


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 90)
    print("  LEVERAGE CALIBRATION: FABER-ONLY ARCHITECTURE")
    print("=" * 90)

    # Step 1: Tier math
    sub_t1, sub_t2 = print_tier_math()

    # Load data
    print(f"\n  Loading data...")
    data = load_data()

    # Step 3: Backtest
    ret_series, tier_log, eff_equity_log, state, strat_names, total_months = \
        run_backtest(sub_t1, sub_t2, data)

    # Step 4: Leverage sweep
    sweep_series, sweep_levels = run_leverage_sweep(data)

    # Step 5: Report
    perfs = print_report(
        ret_series, tier_log, eff_equity_log, state, strat_names,
        total_months, sub_t1, sub_t2, sweep_series, sweep_levels
    )

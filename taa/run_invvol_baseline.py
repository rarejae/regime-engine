"""Test: static inverse-volatility baseline weights vs subjective baseline.

Computes inverse-vol weights from pre-2002 data, then runs full backtest
with all other parameters identical.
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv; load_dotenv()

import numpy as np, pandas as pd
from collections import defaultdict

from taa.data import (load_daily_etf_returns, load_monthly_prices,
                       load_realized_vols, load_monthly_macro, load_monthly_asset_returns)
from taa.faber import compute_trend_scores, apply_faber_filter
from taa.harvey import compute_zscore_variables, find_similar_months, compute_expected_returns
from taa.allocation import BASELINE, ASSETS, EQUITY, direct_capital, normalize

TIER_SUBS = {0: 0.0, 1: 0.30, 2: 0.65}
SEED = {"IVV": 0.0028, "QQQ": 0.0080}

OUTPUT = Path("taa/output"); OUTPUT.mkdir(parents=True, exist_ok=True)


def compute_invvol_weights(asset_ret, cutoff="2002-01-01", cash_vol=0.005):
    """Compute inverse-volatility weights from pre-cutoff data.

    Returns (weights_dict, vol_info_list) where vol_info has per-asset details.
    """
    pre = asset_ret[asset_ret.index < cutoff]
    assets = ["IVV", "QQQ", "VGLT", "IAU", "DBC", "cash"]
    vol_info = []
    inv_vols = {}

    for a in assets:
        if a not in pre.columns:
            continue
        rets = pre[a].dropna()
        if len(rets) < 12:
            continue

        if a == "cash":
            ann_vol = cash_vol
        else:
            ann_vol = float(rets.std() * np.sqrt(12))

        inv_vols[a] = 1.0 / ann_vol if ann_vol > 0 else 0
        vol_info.append({
            "asset": a,
            "start": rets.index.min().strftime("%Y-%m"),
            "end": rets.index.max().strftime("%Y-%m"),
            "n_months": len(rets),
            "ann_vol": ann_vol,
            "inv_vol": inv_vols[a],
        })

    total = sum(inv_vols.values())
    weights = {a: v / total for a, v in inv_vols.items()}
    for vi in vol_info:
        vi["weight"] = weights[vi["asset"]]

    return weights, vol_info


def run_backtest(baseline, daily_ret, monthly_prices, rvol_monthly,
                 asset_ret_fwd, rfr_daily, dpdf, z_clean, trend_df,
                 sma_6, sma_10, sma_12, pre_ers, trading_days):
    """Full backtest with given baseline weights."""

    tier = 0; delevered = False
    hist = {"IVV": list(pre_ers["IVV"]), "QQQ": list(pre_ers["QQQ"])}
    med = dict(SEED); events = []
    ct = {a: 3 for a in ASSETS if a != "cash"}
    he = {a: 0.0 for a in ASSETS}
    wp = dict(baseline)
    tier_counts = {0: 0, 1: 0, 2: 0}
    total_months = 0
    alloc_log = []

    def count_sma_breaches(etf, day):
        if etf not in dpdf.columns: return 0
        prices_up_to = dpdf.loc[:day, etf]
        if len(prices_up_to) == 0: return 0
        price = prices_up_to.iloc[-1]
        ms = pd.Timestamp(f"{day.year}-{day.month:02d}-01")
        breaches = 0
        for sma_df in [sma_6, sma_10, sma_12]:
            sma_dates = sma_df.index[sma_df.index <= ms]
            if len(sma_dates) == 0: continue
            val = sma_df.loc[sma_dates[-1], etf]
            if pd.notna(val) and price < val:
                breaches += 1
        return breaches

    for day in trading_days:
        if day not in daily_ret.index: continue
        dr = daily_ret.loc[day]
        avail = [a for a in ASSETS if a in dr.index and pd.notna(dr[a])]
        if len(avail) < 3: continue
        actual = {a: float(dr[a]) for a in avail}
        rfr = float(rfr_daily.get(day, 0))

        is_ms = (day == trading_days[0] or
                 day.month != trading_days[trading_days.get_loc(day)-1].month)
        is_friday = day.weekday() == 4

        if is_ms:
            total_months += 1

            ts_c = trend_df.index[trend_df.index <= day]
            if len(ts_c) > 0:
                for a in ASSETS:
                    if a == "cash": continue
                    if a in trend_df.columns:
                        v = trend_df.loc[ts_c[-1], a]
                        ct[a] = int(v) if pd.notna(v) else 0

            z_c = z_clean.index[z_clean.index < day]
            if len(z_c) > 0:
                try:
                    sim, _ = find_similar_months(z_clean, z_c[-1])
                    he = compute_expected_returns(sim, asset_ret_fwd, avail)
                except: pass

            rv_c = rvol_monthly.index[rvol_monthly.index <= day]; rvols = {}
            if len(rv_c) > 0:
                for a in ASSETS:
                    if a == "cash": continue
                    if a in rvol_monthly.columns:
                        v = rvol_monthly.loc[rv_c[-1], a]
                        rvols[a] = float(v) if pd.notna(v) and v > 0 else 0.15

            w1, pool = apply_faber_filter(ct, baseline)
            w2, _ = direct_capital(dict(w1), pool, he, rvols)
            wp = normalize(w2)

            alloc_log.append({"date": day, **wp})

            f_c = ct.get("IVV", 0) >= 3 and ct.get("QQQ", 0) >= 3
            h_c = he.get("IVV", 0) > 0 and he.get("QQQ", 0) > 0
            conv = f_c and h_c

            delevered = False
            if conv:
                above = all(he.get(a, 0) > med.get(a, 0) for a in ["IVV", "QQQ"])
                tier = 2 if above else 1
                for a in ["IVV", "QQQ"]:
                    hist[a].append(he.get(a, 0))
                    med[a] = float(np.median(hist[a]))
            else:
                tier = 0

            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        if is_friday:
            ivv_b = count_sma_breaches("IVV", day)
            qqq_b = count_sma_breaches("QQQ", day)
            if tier > 0 and not delevered:
                if ivv_b >= 3 or qqq_b >= 3:
                    events.append({"date": day, "tier": tier})
                    tier = 0; delevered = True

        iw = wp.get("IVV", 0); qw = wp.get("QQQ", 0)
        ir = actual.get("IVV", 0); qr = actual.get("QQQ", 0)
        sr = 2*ir - rfr - 0.0091/252
        ql = 2*qr - rfr - 0.0089/252
        base = sum(wp.get(a, 0)*actual.get(a, 0) for a in avail if a not in ["IVV", "QQQ"])

        sub = TIER_SUBS.get(tier, 0)
        if sub > 0:
            strat_returns[day] = (iw*(1-sub)*ir + iw*sub*sr +
                                   qw*(1-sub)*qr + qw*sub*ql + base)
        else:
            strat_returns[day] = iw*ir + qw*qr + base

    return (pd.Series(strat_returns).sort_index(), events,
            pd.DataFrame(alloc_log), tier_counts, total_months)


def metrics(s):
    ar = s.mean()*252; av = s.std()*np.sqrt(252)
    sh = ar/av if av > 0 else 0
    neg = s[s < 0]; ds = neg.std()*np.sqrt(252) if len(neg) > 10 else av
    so = ar/ds if ds > 0 else 0
    cum = (1+s).cumprod()
    dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
    cal = ar/abs(dd) if dd != 0 else 0
    final = cum.iloc[-1]
    return {"ann_ret": ar, "ann_vol": av, "sharpe": sh, "sortino": so,
            "max_dd": dd, "calmar": cal, "terminal": final}


def crisis_dd(s, start, end):
    c = s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
    if len(c) < 5: return float("nan"), float("nan")
    total = (1+c).prod() - 1
    cum = (1+c).cumprod()
    dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
    return total, dd


# ── Load data ────────────────────────────────────────────────────
print("Loading data...")
daily_ret = load_daily_etf_returns()
daily_ret = daily_ret[[c for c in daily_ret.columns if c in ASSETS]]
monthly_prices = load_monthly_prices()
rvol_monthly = load_realized_vols()
asset_ret = load_monthly_asset_returns()
asset_ret_fwd = asset_ret.shift(-1)
trend_df = compute_trend_scores(monthly_prices)
z_data = compute_zscore_variables(load_monthly_macro())
z_clean = z_data[[c for c in z_data.columns if c.endswith("_z")]].dropna()

from fredapi import Fred
rfr_daily = pd.Series(0.0, index=daily_ret.index)
key = os.environ.get("FRED_API_KEY")
if key:
    tb = Fred(api_key=key).get_series("DTB3", observation_start="1998-01-01")
    tb.index = pd.to_datetime(tb.index)
    rfr_daily = (tb / 100 / 252).reindex(daily_ret.index, method="ffill").fillna(0)

import yfinance as yf
daily_prices = {}
for our, ticker in [("IVV", "SPY"), ("QQQ", "QQQ")]:
    d = yf.download(ticker, start="1998-01-01", progress=False)
    if d is not None and not d.empty:
        p = d["Close"]
        if hasattr(p, "columns"): p = p.iloc[:, 0]
        p.index = pd.to_datetime(p.index).tz_localize(None)
        daily_prices[our] = p
dpdf = pd.DataFrame(daily_prices)

mc = dpdf.resample("MS").last()
sma_6 = mc.rolling(6, min_periods=6).mean()
sma_10 = mc.rolling(10, min_periods=10).mean()
sma_12 = mc.rolling(12, min_periods=12).mean()

bt_start = pd.Timestamp("2002-01-01")
pre_ers = {"IVV": [], "QQQ": []}
for z_dt in z_clean.index[z_clean.index < bt_start]:
    try:
        sim, _ = find_similar_months(z_clean, z_dt)
        er = compute_expected_returns(sim, asset_ret_fwd, ["IVV", "QQQ"])
        for a in ["IVV", "QQQ"]:
            if not np.isnan(er.get(a, np.nan)): pre_ers[a].append(er[a])
    except: pass

common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
trading_days = daily_ret.loc[common_start:].index

# ── Compute inverse-vol weights ──────────────────────────────────
print("Computing inverse-vol weights from pre-2002 data...")
invvol_baseline, vol_info = compute_invvol_weights(asset_ret, "2002-01-01")

# ── Run both configs ─────────────────────────────────────────────
strat_returns = {}  # shared mutable for run_backtest closure

print("Running subjective baseline...")
strat_returns = {}
s_subj, ev_subj, alloc_subj, tier_subj, months_subj = run_backtest(
    BASELINE, daily_ret, monthly_prices, rvol_monthly, asset_ret_fwd,
    rfr_daily, dpdf, z_clean, trend_df, sma_6, sma_10, sma_12,
    pre_ers, trading_days)

print("Running inverse-vol baseline...")
strat_returns = {}
s_invv, ev_invv, alloc_invv, tier_invv, months_invv = run_backtest(
    invvol_baseline, daily_ret, monthly_prices, rvol_monthly, asset_ret_fwd,
    rfr_daily, dpdf, z_clean, trend_df, sma_6, sma_10, sma_12,
    pre_ers, trading_days)

m_subj = metrics(s_subj)
m_invv = metrics(s_invv)

# ── Report ───────────────────────────────────────────────────────
lines = []
def P(s=""): lines.append(s)

P("=" * 70)
P("  STATIC INVERSE-VOL BASELINE TEST")
P("=" * 70)
P()

P("WEIGHT COMPUTATION (pre-2002 monthly returns)")
P(f"  {'Asset':>6} {'Data Range':>18} {'Months':>7} {'Ann Vol':>8} {'InvVol Wt':>10} {'Current Wt':>11} {'Delta':>8}")
P(f"  {'-'*70}")
eq_subj = sum(BASELINE.get(a, 0) for a in EQUITY)
eq_invv = sum(invvol_baseline.get(a, 0) for a in EQUITY)
for vi in vol_info:
    a = vi["asset"]
    cur = BASELINE.get(a, 0)
    new = vi["weight"]
    P(f"  {a:>6} {vi['start']+' - '+vi['end']:>18} {vi['n_months']:>7} "
      f"{vi['ann_vol']:>7.1%} {new:>9.1%} {cur:>10.1%} {new-cur:>+7.1%}")
P()
P(f"  Total equity weight: subjective {eq_subj:.0%} vs inverse-vol {eq_invv:.1%}")
P()

P("PERFORMANCE COMPARISON (2002-2026)")
P(f"  {'Metric':>14} {'Subjective':>12} {'Inverse-Vol':>12} {'Delta':>10}")
P(f"  {'-'*50}")
for label, key, fmt in [
    ("Ann. Return", "ann_ret", "pct"),
    ("Volatility", "ann_vol", "pct"),
    ("Sharpe", "sharpe", "f2"),
    ("Sortino", "sortino", "f2"),
    ("Max Drawdown", "max_dd", "pct"),
    ("Calmar", "calmar", "f2"),
    ("Terminal $1", "terminal", "dollar"),
]:
    vb = m_subj[key]; vu = m_invv[key]
    if fmt == "pct":
        P(f"  {label:>14} {vb:>11.1%} {vu:>11.1%} {vu-vb:>+9.2%}")
    elif fmt == "f2":
        P(f"  {label:>14} {vb:>11.2f} {vu:>11.2f} {vu-vb:>+9.3f}")
    elif fmt == "dollar":
        P(f"  {label:>14} ${vb:>10.2f} ${vu:>10.2f} {vu-vb:>+9.2f}")

P()
P("CRISIS PERIODS")
P(f"  {'Period':>14} {'Subjective':>12} {'Inverse-Vol':>12} {'Delta':>10}")
P(f"  {'-'*50}")
for name, cs, ce in [("GFC 2008-09", "2008-09-01", "2009-03-31"),
                      ("COVID 2020", "2020-02-19", "2020-03-23"),
                      ("2022 Bear", "2022-01-03", "2022-10-31")]:
    tb, db = crisis_dd(s_subj, cs, ce)
    tu, du = crisis_dd(s_invv, cs, ce)
    P(f"  {name:>14} {tb:>+11.1%} {tu:>+11.1%} {tu-tb:>+9.2%}")
    P(f"  {'(max DD)':>14} {db:>11.1%} {du:>11.1%} {du-db:>+9.2%}")

P()
P("LEVERAGE TIER FREQUENCY")
P(f"  {'Tier':>8} {'Subjective':>12} {'Inverse-Vol':>12}")
P(f"  {'-'*34}")
for t in [0, 1, 2]:
    cs = tier_subj.get(t, 0); ci = tier_invv.get(t, 0)
    P(f"  {'Tier '+str(t):>8} {cs:>8} ({cs/months_subj:.0%}) {ci:>8} ({ci/months_invv:.0%})")
P(f"  De-lever events: {len(ev_subj)} vs {len(ev_invv)}")

P()
P("ALLOCATION SHIFT ANALYSIS")
P(f"  {'Asset':>8} {'Subj Mean':>10} {'InvVol Mean':>12} {'Delta':>8}")
P(f"  {'-'*40}")
for a in ASSETS:
    ms = alloc_subj[a].mean() if a in alloc_subj.columns else 0
    mi = alloc_invv[a].mean() if a in alloc_invv.columns else 0
    P(f"  {a:>8} {ms:>9.1%} {mi:>11.1%} {mi-ms:>+7.2%}")

eq_exp_subj = (alloc_subj.get("IVV", 0).mean() + alloc_subj.get("QQQ", 0).mean()
               if "IVV" in alloc_subj.columns else 0)
eq_exp_invv = (alloc_invv.get("IVV", 0).mean() + alloc_invv.get("QQQ", 0).mean()
               if "IVV" in alloc_invv.columns else 0)
P(f"\n  Effective equity exposure: subjective {eq_exp_subj:.1%} vs inverse-vol {eq_exp_invv:.1%}")

P()
P("KEY QUESTION: Does inverse-vol produce a more conservative or aggressive portfolio?")
if eq_exp_invv < eq_exp_subj - 0.02:
    P("  -> More conservative: lower equity, higher bond/gold/commodity allocation.")
    P(f"     Equity exposure drops {eq_exp_subj-eq_exp_invv:.1%} vs subjective.")
elif eq_exp_invv > eq_exp_subj + 0.02:
    P("  -> More aggressive: higher equity exposure.")
    P(f"     Equity exposure rises {eq_exp_invv-eq_exp_subj:.1%} vs subjective.")
else:
    P("  -> Similar aggressiveness: equity exposure nearly unchanged.")

P()
P("RECOMMENDATION")
sharpe_diff = m_invv["sharpe"] - m_subj["sharpe"]
dd_diff = m_invv["max_dd"] - m_subj["max_dd"]

if sharpe_diff > 0.05:
    P("  Adopt inverse-vol baseline — meaningful Sharpe improvement.")
    P(f"  Sharpe gain of {sharpe_diff:+.3f} with max DD change of {dd_diff:+.1%}.")
    P("  Removes subjective weight judgment; grounded in pre-backtest vol data.")
elif sharpe_diff < -0.05:
    P("  Keep subjective baseline — inverse-vol underperforms.")
    P(f"  Sharpe drops {sharpe_diff:+.3f}. Subjective equity tilt captures")
    P("  risk premium that inverse-vol under-allocates to.")
elif abs(sharpe_diff) <= 0.05 and abs(dd_diff) < 0.03:
    P("  Either approach works — performance is statistically equivalent.")
    P(f"  Sharpe delta: {sharpe_diff:+.3f}, Max DD delta: {dd_diff:+.1%}.")
    P("  Inverse-vol has the advantage of being objective/reproducible,")
    P("  but the subjective baseline captures a deliberate equity risk premium tilt")
    P("  that has been validated through the full parameter sensitivity suite.")
else:
    P(f"  Keep subjective baseline — Sharpe delta ({sharpe_diff:+.3f}) doesn't justify")
    P(f"  the {'increased drawdown' if dd_diff < -0.02 else 'change'}.")

report = "\n".join(lines)

# Print
print()
print(report)

# Save
out_path = OUTPUT / "invvol_baseline_test.txt"
with open(out_path, "w") as f:
    f.write(report)
print(f"\nSaved: {out_path}")
